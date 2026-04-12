from __future__ import annotations

from opentelemetry import trace, metrics

SERVICE_NAME = "litter-detector-camera"

tracer = trace.get_tracer(SERVICE_NAME)
meter = metrics.get_meter(SERVICE_NAME)


class CameraMetrics:
    """Metric instruments for the camera pipeline."""

    def __init__(self) -> None:
        self.frames_captured = meter.create_counter(
            "camera.frames.captured",
            description="Total frames captured from source",
            unit="frames",
        )
        self.frames_published = meter.create_counter(
            "camera.frames.published",
            description="Total frames published to Zenoh",
            unit="frames",
        )
        self.frames_dropped = meter.create_counter(
            "camera.frames.dropped",
            description="Frames dropped due to queue overflow (Go2Source)",
            unit="frames",
        )
        self.frame_processing_time = meter.create_histogram(
            "camera.frame.processing_time",
            description="Time to postprocess and JPEG-encode a frame",
            unit="ms",
        )
        self.frame_publish_time = meter.create_histogram(
            "camera.frame.publish_time",
            description="Time to publish a frame to Zenoh",
            unit="ms",
        )
        self.frame_size_bytes = meter.create_histogram(
            "camera.frame.size",
            description="JPEG frame size in bytes",
            unit="bytes",
        )


camera_metrics = CameraMetrics()
