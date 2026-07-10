from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
import zenoh
from loguru import logger

from litter_detector.config import Settings
from litter_detector.detector import model as model_mod
from litter_detector.detector.metrics import SERVICE_NAME, detector_metrics, tracer
from litter_detector.stability import StabilityGate
from litter_detector.telemetry import setup_telemetry, shutdown_telemetry
from litter_detector.tracker import (
    ByteTracker,
    ObjectRegistry,
    Track,
    clean_mask,
    mask_to_detections,
)


class LatestFrameSlot:
    """Single-slot holder backed by deque(maxlen=1): newest frame wins."""

    def __init__(self) -> None:
        self._slot: deque[tuple[zenoh.Sample, int]] = deque(maxlen=1)
        self._event = threading.Event()

    def put(self, sample: zenoh.Sample) -> bool:
        replaced = len(self._slot) == 1
        self._slot.append((sample, time.perf_counter_ns()))
        self._event.set()
        return replaced

    def take(self, timeout: float = 1.0) -> tuple[zenoh.Sample, int] | None:
        if not self._event.wait(timeout):
            return None
        self._event.clear()
        try:
            return self._slot.popleft()
        except IndexError:
            return None

    def stop(self) -> None:
        self._event.set()


class LitterDetector:
    @tracer.start_as_current_span("detector.init")
    def __init__(self, model_uri: str) -> None:
        self.settings = Settings()
        self.session = zenoh.open(Settings.zenoh_config())
        self.topics = Settings.topics()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.runner: model_mod.AnyRunner
        self.runner, resolved_uri = model_mod.load_model(model_uri, self.device)
        logger.info(f"Model loaded from {resolved_uri} on device={self.device}")

        self.frame_pub = self.session.declare_publisher(
            self.topics.detection.frame, encoding=zenoh.Encoding.IMAGE_JPEG
        )
        self.mask_pub = self.session.declare_publisher(
            self.topics.detection.mask, encoding=zenoh.Encoding.IMAGE_JPEG
        )
        self.masked_pub = self.session.declare_publisher(
            self.topics.detection.masked_frame, encoding=zenoh.Encoding.IMAGE_JPEG
        )
        self.detections_pub = self.session.declare_publisher(
            self.topics.detection.detections, encoding=zenoh.Encoding.APPLICATION_JSON
        )
        self.tracked_pub = self.session.declare_publisher(
            self.topics.detection.tracked, encoding=zenoh.Encoding.APPLICATION_JSON
        )

        self.tracker = ByteTracker(
            iou_threshold=self.settings.tracker_iou_threshold,
            iou_threshold_low=self.settings.tracker_iou_threshold_low,
            det_high_thresh=self.settings.tracker_det_high_thresh,
            max_age=self.settings.tracker_max_age,
            min_hits=self.settings.tracker_min_hits,
            appearance_weight=self.settings.tracker_appearance_weight,
            appearance_alpha=self.settings.tracker_appearance_alpha,
        )
        self.registry = ObjectRegistry(self.settings.registry_db_path)
        self._confirmed_ids: set[int] = set()

        self.stability = StabilityGate(
            session=self.session,
            topic=self.settings.stability_imu_topic,
            max_angular_velocity=self.settings.stability_max_angular_velocity,
        )

        self.slot = LatestFrameSlot()
        self._drop_count = 0
        # Rolling EWMA state for temporal smoothing of probability maps.
        self._prev_probs: np.ndarray | None = None
        self.subscriber = self.session.declare_subscriber(
            self.topics.camera.frame, self._on_frame
        )

    def _on_frame(self, sample: zenoh.Sample) -> None:
        detector_metrics.frames_received.add(1)
        if self.slot.put(sample):
            detector_metrics.frames_dropped.add(1)
            self._drop_count += 1
            if self._drop_count % 100 == 0:
                logger.warning(f"Dropped {self._drop_count} frames so far (inference slower than stream)")

    def _process(self, sample: zenoh.Sample, enqueued_at_ns: int) -> None:
        with tracer.start_as_current_span("detector.process_frame") as span:
            t_start = time.perf_counter()
            payload = bytes(sample.payload)
            span.set_attribute("frame.size_bytes", len(payload))
            span.set_attribute("inference.device", str(self.device))
            queue_age_ms = (time.perf_counter_ns() - enqueued_at_ns) / 1e6
            span.set_attribute("frame.queue_age_ms", queue_age_ms)

            # Skip frames captured while the robot is shaking too much for the
            # model to be useful. Fail-open if the IMU stream is dead.
            if not self.stability.is_stable():
                span.set_attribute("stability.skipped", True)
                span.set_attribute("stability.angular_velocity", self.stability.latest_magnitude)
                detector_metrics.frames_dropped.add(1)
                return

            with tracer.start_as_current_span("decode"):
                arr = np.frombuffer(payload, dtype=np.uint8)
                frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    logger.error("Failed to decode JPEG frame")
                    return

            h, w = frame_bgr.shape[:2]

            with tracer.start_as_current_span("inference"):
                inf_start = time.perf_counter()
                if isinstance(self.runner, model_mod.YoloRunner):
                    # YOLO handles its own preprocessing internally; returns
                    # (mask_uint8, probs_float) at full frame resolution.
                    mask, probs = self.runner.infer_frame(
                        frame_bgr, self.settings.detector_prob_threshold
                    )
                else:
                    tensor = model_mod.preprocess(frame_bgr, self.device)
                    logits = self.runner.infer(tensor)
                    probs = model_mod.probs_from_logits(logits, (h, w))
                    # Temporal EWMA smooths flicker from a shaking camera.
                    alpha = self.settings.detector_temporal_alpha
                    if alpha > 0.0 and self._prev_probs is not None and self._prev_probs.shape == probs.shape:
                        probs = (alpha * self._prev_probs + (1.0 - alpha) * probs).astype(np.float32)
                    self._prev_probs = probs
                    mask = model_mod.binarize(probs, self.settings.detector_prob_threshold)
                inf_ms = (time.perf_counter() - inf_start) * 1000
                detector_metrics.inference_time_ms.record(inf_ms)
                span.set_attribute("inference.time_ms", inf_ms)

            with tracer.start_as_current_span("postprocess"):
                mask = model_mod.morph_close(mask, self.settings.detector_morph_close_kernel)
                overlay_img = model_mod.overlay(frame_bgr, mask)
                coverage = float((mask > 0).mean())
                detector_metrics.mask_coverage_ratio.record(coverage)
                span.set_attribute("mask.coverage_ratio", coverage)

            frame_ts_ns = time.time_ns()
            with tracer.start_as_current_span("track"):
                cleaned = clean_mask(
                    mask, erode_kernel=self.settings.tracker_mask_erode_kernel
                )
                detections = mask_to_detections(
                    cleaned,
                    min_area_px=self.settings.tracker_min_area_px,
                    probs=probs,
                    min_confidence=self.settings.tracker_min_confidence,
                    # Only pay the histogram cost when the tracker will use it.
                    frame_bgr=frame_bgr if self.settings.tracker_appearance_weight > 0.0 else None,
                )
                tracks = self.tracker.update(detections, frame_ts_ns)
                self.registry.upsert_all(tracks)
                _draw_tracks(overlay_img, tracks)

                detector_metrics.detections_per_frame.record(len(detections))
                detector_metrics.confirmed_tracks_per_frame.record(len(tracks))
                detector_metrics.tracker_active_tracks.record(self.tracker.active_track_count)
                # Count an ID exactly once — the first frame it crosses the
                # `tracker_count_min_observations` bar. This sits on top of the
                # tracker's own `min_hits` gate: SORT confirms after `min_hits`
                # consecutive matches, but the unique-objects counter only
                # ticks once the track has been observed enough times to be
                # considered a real, persistent object (not a flickery blob).
                threshold = self.settings.tracker_count_min_observations
                newly_counted = [
                    t for t in tracks
                    if t.id not in self._confirmed_ids and t.n_observations >= threshold
                ]
                if newly_counted:
                    self._confirmed_ids.update(t.id for t in newly_counted)
                    detector_metrics.tracker_unique_ids.add(len(newly_counted))
                span.set_attribute("tracker.detections", len(detections))
                span.set_attribute("tracker.confirmed_tracks", len(tracks))

            with tracer.start_as_current_span("publish"):
                ok_mask, mask_jpeg = cv2.imencode(".jpg", mask)
                ok_overlay, overlay_jpeg = cv2.imencode(".jpg", overlay_img)
                if not (ok_mask and ok_overlay):
                    logger.error("Failed to JPEG-encode mask or overlay")
                    return
                # timestamp_ns attachment lets downstream consumers pair this
                # frame exactly with the tracked-objects message of the same
                # loop iteration (existing consumers ignore attachments).
                self.frame_pub.put(payload, attachment=str(frame_ts_ns).encode())
                self.mask_pub.put(mask_jpeg.tobytes())
                self.masked_pub.put(overlay_jpeg.tobytes())
                self.detections_pub.put(
                    json.dumps({
                        "timestamp_ns": frame_ts_ns,
                        "mask_coverage_ratio": coverage,
                        "inference_ms": inf_ms,
                    })
                )
                self.tracked_pub.put(
                    json.dumps({
                        "timestamp_ns": frame_ts_ns,
                        "tracks": [t.to_dict() for t in tracks],
                    })
                )

            total_ms = (time.perf_counter() - t_start) * 1000
            detector_metrics.end_to_end_time_ms.record(total_ms)
            detector_metrics.frames_processed.add(1)

    def run(self) -> None:
        logger.info("Detector running — waiting for frames")
        try:
            while True:
                taken = self.slot.take(timeout=1.0)
                if taken is None:
                    continue
                sample, enq = taken
                try:
                    self._process(sample, enq)
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
        except KeyboardInterrupt:
            logger.info("Stopping detector")

    def close(self) -> None:
        self.slot.stop()
        self.subscriber.undeclare()
        self.frame_pub.undeclare()
        self.mask_pub.undeclare()
        self.masked_pub.undeclare()
        self.detections_pub.undeclare()
        self.tracked_pub.undeclare()
        self.stability.close()
        self.session.close()
        self.registry.close()


def _draw_tracks(image: np.ndarray, tracks: list[Track]) -> None:
    """Annotate an image in-place with bbox + track ID for each confirmed track."""
    for t in tracks:
        x, y, w, h = t.bbox.x, t.bbox.y, t.bbox.w, t.bbox.h
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"#{t.id}"
        cv2.putText(image, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Litter segmentation detector")
    parser.add_argument(
        "--model",
        default=model_mod.resolve_default_uri(),
        help=(
            "Model source: MLflow URI ('models:/name/version', 'runs:/<id>/model'), "
            "a local .onnx file (U-Net), or a local .pt file (YOLO11-seg via ultralytics). "
            "Defaults to LITTER_MODEL_URI / MLFLOW_MODEL_URI env vars, "
            f"then {model_mod.DEFAULT_MODEL_URI!r}."
        ),
    )
    args = parser.parse_args()

    setup_telemetry(SERVICE_NAME)
    detector: LitterDetector | None = None
    try:
        detector = LitterDetector(model_uri=args.model)
        detector.run()
    finally:
        if detector is not None:
            detector.close()
        shutdown_telemetry()


if __name__ == "__main__":
    main()
