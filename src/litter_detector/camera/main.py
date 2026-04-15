from __future__ import annotations

import argparse
import time

import zenoh
from loguru import logger

from litter_detector.config import Settings
from litter_detector.camera.camera_source import CameraSource
from litter_detector.camera.webcam_source import WebcamSource
from litter_detector.camera.go2_source import Go2Source
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
    parser = argparse.ArgumentParser(description="Litter Detector Camera Publisher")
    parser.add_argument("--source", type=str, choices=["webcam", "go2"], help="Camera source to use")
    parser.add_argument("--id", type=int, help="Webcam ID to use (only for webcam source)")
    args = parser.parse_args()

    setup_telemetry(SERVICE_NAME)
    try:
        settings = Settings()
        
        # Override settings with command line arguments if provided
        source = args.source if args.source else settings.source
        cam_id = args.id if args.id is not None else settings.id

        if source.lower() == "go2":
            camera = Go2Source()
        else:
            camera = WebcamSource(camera_id=cam_id, framerate=30)

        publisher = CameraPublisher(camera)
        publisher.run()
    finally:
        shutdown_telemetry()


if __name__ == "__main__":
    main()
