import time
import cv2

from typing import Iterator
from imutils.video import VideoStream

from litter_detector.camera.camera_source import CameraSource
from litter_detector.camera.metrics import tracer


class WebcamSource(CameraSource):
    def __init__(self, camera_id=None, framerate=30) -> None:
        super().__init__()
        self.camera_id = camera_id
        self.framerate = framerate
        self.vs: VideoStream | None = None

    @staticmethod
    def _get_first_camera_id() -> int:
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                cap.release()
                return i
        raise RuntimeError("No webcam found")

    def _capture_frames(self) -> Iterator[cv2.typing.MatLike]:
        if self.vs is None:
            raise RuntimeError("WebcamSource not started — call start() first")
        while True:
            yield self.vs.read()
            time.sleep(1 / self.framerate)

    @tracer.start_as_current_span("camera.webcam.start")
    def start(self) -> None:
        if self.camera_id is None:
            self.camera_id = self._get_first_camera_id()
        self.vs = VideoStream(src=self.camera_id)
        self.vs.start()

    @tracer.start_as_current_span("camera.webcam.stop")
    def stop(self) -> None:
        if self.vs is not None:
            self.vs.stop()
