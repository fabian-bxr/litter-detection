from __future__ import annotations

import time

import zenoh
from loguru import logger

from litter_detector.config import Settings
from litter_detector.camera.camera_source import CameraSource
from litter_detector.camera.webcam_source import WebcamSource
from litter_detector.telemetry import setup_telemetry, shutdown_telemetry
from litter_detector.camera.metrics import SERVICE_NAME, tracer, camera_metrics


class CameraPublisher:
    @tracer.start_as_current_span("camera.publisher.init")
    def __init__(self, camera: CameraSource) -> None:
        self.session = zenoh.open(Settings.zenoh_config())
        self.topics = Settings.topics()
        self.camera = camera

        self.publisher = self.session.declare_publisher(
            key_expr=self.topics.camera.frame,
            encoding=zenoh.Encoding.IMAGE_JPEG,
        )

    def run(self) -> None:
        self.camera.start()
        logger.info(f"Starting camera {self.camera}")
        try:
            for frame in self.camera.frames():
                with tracer.start_as_current_span("camera.publish_frame") as span:
                    start = time.perf_counter()
                    self.publisher.put(frame)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    camera_metrics.frames_published.add(1)
                    camera_metrics.frame_publish_time.record(elapsed_ms)
                    span.set_attribute("frame.publish_time_ms", elapsed_ms)
        except KeyboardInterrupt:
            logger.info(f"Stopping camera {self.camera}")
        except Exception as e:
            logger.error(f"Error publishing frame: {e}")
        finally:
            self.camera.stop()


def main():
    setup_telemetry(SERVICE_NAME)
    try:
        # TODO: Source needs to be configurable in settings, optional ID as well
        camera = WebcamSource(framerate=30)
        # camera = Go2Source()  # Requires Robodog to publish to: robodog/sensors/go2_camera
        publisher = CameraPublisher(camera)
        publisher.run()
    finally:
        shutdown_telemetry()


if __name__ == "__main__":
    main()
