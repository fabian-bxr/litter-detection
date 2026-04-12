import queue
import cv2
import zenoh

from typing import Iterator
import numpy as np
from litter_detector.camera.camera_source import CameraSource
from litter_detector.camera.metrics import tracer, camera_metrics


class Go2Source(CameraSource):
    def __init__(self) -> None:
        super().__init__()
        self.session: zenoh.Session | None = None
        self.subscriber: zenoh.Subscriber | None = None
        self.frame_queue: queue.Queue[bytes] = queue.Queue(maxsize=1)

    def _on_frame(self, sample: zenoh.Sample) -> None:
        try:
            self.frame_queue.put_nowait(sample.payload.to_bytes())
        except queue.Full:
            self.frame_queue.get_nowait()  # drop oldest frame
            self.frame_queue.put_nowait(sample.payload.to_bytes())
            camera_metrics.frames_dropped.add(1)

    def _capture_frames(self) -> Iterator[cv2.typing.MatLike]:
        while True:
            try:
                frame_bytes = self.frame_queue.get(timeout=1)
                frame_array = cv2.imdecode(
                    np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame_array is None:
                    continue
                yield frame_array
            except queue.Empty:
                continue

    @tracer.start_as_current_span("camera.go2.start")
    def start(self) -> None:
        self.session = zenoh.open(self.settings.zenoh_config())
        self.subscriber = self.session.declare_subscriber(
            key_expr=self.settings.topics().camera.go2_camera,
            handler=self._on_frame,
        )

    @tracer.start_as_current_span("camera.go2.stop")
    def stop(self) -> None:
        if self.session:
            self.session.close()
