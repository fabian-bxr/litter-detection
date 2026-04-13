from __future__ import annotations

from opentelemetry import trace, metrics

SERVICE_NAME = "litter-detector-detector"

tracer = trace.get_tracer(SERVICE_NAME)
meter = metrics.get_meter(SERVICE_NAME)


class DetectorMetrics:
    """Metric instruments for the detector pipeline."""

    def __init__(self) -> None:
        self.frames_received = meter.create_counter(
            "detector.frames.received",
            description="Total frames received from the camera topic",
            unit="frames",
        )
        self.frames_dropped = meter.create_counter(
            "detector.frames.dropped",
            description="Frames overwritten in the latest-frame slot before inference",
            unit="frames",
        )
        self.frames_processed = meter.create_counter(
            "detector.frames.processed",
            description="Frames that completed inference and were published",
            unit="frames",
        )
        self.inference_time_ms = meter.create_histogram(
            "detector.inference.time_ms",
            description="Forward-pass time for a single frame",
            unit="ms",
        )
        self.end_to_end_time_ms = meter.create_histogram(
            "detector.frame.end_to_end_time_ms",
            description="Receive-to-publish latency for a single frame",
            unit="ms",
        )
        self.mask_coverage_ratio = meter.create_histogram(
            "detector.mask.coverage_ratio",
            description="Fraction of pixels classified as litter",
            unit="ratio",
        )


detector_metrics = DetectorMetrics()
