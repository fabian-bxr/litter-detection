import inspect
import time
from abc import ABC, abstractmethod
from typing import Iterator
from litter_detector.config import Settings
from litter_detector.camera.metrics import tracer, camera_metrics
import cv2


class CameraSource(ABC):
    def __init__(self) -> None:
        self.settings = Settings()
        self._frame_size = (self.settings.frame_width, self.settings.frame_height)

    def frames(self) -> Iterator[bytes]:
        for frame in self._capture_frames():
            with tracer.start_as_current_span("camera.frame"):
                camera_metrics.frames_captured.add(1)
                with tracer.start_as_current_span("camera.process_frame") as span:
                    start = time.perf_counter()
                    processed = self.postprocess_frame(frame)
                    _, buf = cv2.imencode(".jpg", processed)
                    frame_bytes = buf.tobytes()
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    camera_metrics.frame_processing_time.record(elapsed_ms)
                    camera_metrics.frame_size_bytes.record(len(frame_bytes))
                    span.set_attribute("frame.size_bytes", len(frame_bytes))
                    span.set_attribute("frame.processing_time_ms", elapsed_ms)
                yield frame_bytes

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
