import time
import cv2

from typing import Iterator
from imutils.video import VideoStream

from litter_detector.camera.camera_source import CameraSource


class WebcamSource(CameraSource):
    def __init__(self, camera_id=0, framerate=30) -> None:
        super().__init__()
        self.camera_id = camera_id
        self.framerate = framerate
        self.vs: VideoStream | None = None

    def _capture_frames(self) -> Iterator[cv2.typing.MatLike]:
        if self.vs is None:
            raise RuntimeError("WebcamSource not started — call start() first")
        while True:
            yield self.vs.read()
            time.sleep(1 / self.framerate)

    def start(self) -> None:
        self.vs = VideoStream(src=self.camera_id)
        self.vs.start()

    def stop(self) -> None:
        if self.vs is not None:
            self.vs.stop()

