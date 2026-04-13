from __future__ import annotations

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
from litter_detector.telemetry import setup_telemetry, shutdown_telemetry


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
    def __init__(self) -> None:
        self.session = zenoh.open(Settings.zenoh_config())
        self.topics = Settings.topics()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, model_uri = model_mod.load_model(self.device)
        logger.info(f"Model loaded from {model_uri} on device={self.device}")

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

        self.slot = LatestFrameSlot()
        self._drop_count = 0
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

            with tracer.start_as_current_span("decode"):
                arr = np.frombuffer(payload, dtype=np.uint8)
                frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    logger.error("Failed to decode JPEG frame")
                    return

            with tracer.start_as_current_span("preprocess"):
                tensor = model_mod.preprocess(frame_bgr, self.device)

            with tracer.start_as_current_span("inference"):
                inf_start = time.perf_counter()
                with torch.inference_mode():
                    logits = self.model(tensor)
                inf_ms = (time.perf_counter() - inf_start) * 1000
                detector_metrics.inference_time_ms.record(inf_ms)
                span.set_attribute("inference.time_ms", inf_ms)

            with tracer.start_as_current_span("postprocess"):
                h, w = frame_bgr.shape[:2]
                mask = model_mod.postprocess(logits, (h, w))
                overlay_img = model_mod.overlay(frame_bgr, mask)
                coverage = float((mask > 0).mean())
                detector_metrics.mask_coverage_ratio.record(coverage)
                span.set_attribute("mask.coverage_ratio", coverage)

            with tracer.start_as_current_span("publish"):
                ok_mask, mask_jpeg = cv2.imencode(".jpg", mask)
                ok_overlay, overlay_jpeg = cv2.imencode(".jpg", overlay_img)
                if not (ok_mask and ok_overlay):
                    logger.error("Failed to JPEG-encode mask or overlay")
                    return
                self.frame_pub.put(payload)
                self.mask_pub.put(mask_jpeg.tobytes())
                self.masked_pub.put(overlay_jpeg.tobytes())
                self.detections_pub.put(
                    json.dumps({
                        "timestamp_ns": time.time_ns(),
                        "mask_coverage_ratio": coverage,
                        "inference_ms": inf_ms,
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
        self.session.close()


def main() -> None:
    setup_telemetry(SERVICE_NAME)
    detector: LitterDetector | None = None
    try:
        detector = LitterDetector()
        detector.run()
    finally:
        if detector is not None:
            detector.close()
        shutdown_telemetry()


if __name__ == "__main__":
    main()
