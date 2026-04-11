import inspect
from abc import ABC, abstractmethod
from typing import Iterator
from litter_detector.config import Settings
import cv2


class CameraSource(ABC):
    def __init__(self) -> None:
        self.settings = Settings()
        self._frame_size = (self.settings.frame_width, self.settings.frame_height)

    def frames(self) -> Iterator[bytes]:
        for frame in self._capture_frames():
            processed = self.postprocess_frame(frame)
            _, buf = cv2.imencode(".jpg", processed)
            yield buf.tobytes()

    # TODO: Add proper postprocessing steps, like from the example example (lecture-ki-syteme-code)
    def postprocess_frame(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        return cv2.resize(frame, self._frame_size)

    @abstractmethod
    def _capture_frames(self) -> Iterator[cv2.typing.MatLike]:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    def __repr__(self) -> str:
        params = inspect.signature(self.__init__).parameters
        attrs = ", ".join(f"{k}={getattr(self, k)!r}" for k in params)
        return f"{self.__class__.__name__}({attrs})"
