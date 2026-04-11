from __future__ import annotations

import zenoh
from loguru import logger

from litter_detector.config import Settings
from litter_detector.camera.camera_source import CameraSource
from litter_detector.camera.webcam_source import WebcamSource
from litter_detector.camera.go2_source import Go2Source


class CameraPublisher:
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
                self.publisher.put(frame)
        except KeyboardInterrupt:
            logger.info(f"Stopping camera {self.camera}")
        except Exception as e:
            logger.error(f"Error publishing frame: {e}")
        finally:
            self.camera.stop()


def main():
    # TODO: Source needs to be configurable in settings, optional ID as well
    camera = WebcamSource(framerate=30)
    # camera = Go2Source()  # Requires Robodog to publish to: robodog/sensors/go2_camera
    publisher = CameraPublisher(camera)
    publisher.run()


if __name__ == "__main__":
    main()
