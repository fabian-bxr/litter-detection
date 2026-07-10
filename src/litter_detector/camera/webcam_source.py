from __future__ import annotations

import sys
import threading
import time
from typing import Iterator

import cv2

from litter_detector.camera.camera_source import CameraSource
from litter_detector.camera.metrics import tracer


def _default_backend() -> int:
    # MSMF (the Windows default) regularly opens cameras it cannot grab frames
    # from (error -2147483638 / MF_E_HW_MFT_FAILED_START_STREAMING). DirectShow
    # is far more reliable for typical USB / built-in webcams.
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


class _ThreadedCapture:
    """Minimal background-thread wrapper around cv2.VideoCapture.

    Replaces imutils.VideoStream so we can pin the backend (apiPreference).
    """

    def __init__(self, src: int, backend: int) -> None:
        self.cap = cv2.VideoCapture(src, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {src} (backend={backend})")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.release()
            raise RuntimeError(f"Camera index {src} opened but produced no frame")
        self._frame = frame
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "_ThreadedCapture":
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stopped:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._stopped = True
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.cap.release()


class WebcamSource(CameraSource):
    def __init__(self, camera_id=None, framerate=30) -> None:
        super().__init__()
        self.camera_id = camera_id
        self.framerate = framerate
        self.backend = _default_backend()
        self.vs: _ThreadedCapture | None = None

    def _probe_camera(self, src: int) -> bool:
        cap = cv2.VideoCapture(src, self.backend)
        try:
            if not cap.isOpened():
                return False
            # Some webcams need a couple of read attempts to warm up.
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None:
                    return True
                time.sleep(0.1)
            return False
        finally:
            cap.release()

    def _get_first_camera_id(self) -> int:
        for i in range(5):
            if self._probe_camera(i):
                return i
        raise RuntimeError(
            f"No webcam found (probed indices 0..4 with backend={self.backend}). "
            "Pass --id N to target a specific device."
        )

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
        self.vs = _ThreadedCapture(src=self.camera_id, backend=self.backend).start()

    @tracer.start_as_current_span("camera.webcam.stop")
    def stop(self) -> None:
        if self.vs is not None:
            self.vs.stop()
            self.vs = None