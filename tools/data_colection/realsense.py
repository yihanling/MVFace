import logging
from pathlib import Path

import numpy as np
import cv2

from camera import Camera

_log = logging.getLogger('realsense')

COLOR_SIZE = (1280, 720)
FPS = 30


class RealSense(Camera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe = None
        self.align = None
        self.depth_scale = None

    def start(self, color_size=COLOR_SIZE, fps=FPS):
        import pyrealsense2 as rs

        ir_w, ir_h = self.resolution or (480, 270)
        if self.type == 'd405':
            # the D405 streams depth and color off one sensor, so they must share a resolution
            color_size = (ir_w, ir_h)

        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.infrared, 1, ir_w, ir_h, rs.format.y8, fps)
        cfg.enable_stream(rs.stream.depth, ir_w, ir_h, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, *color_size, rs.format.bgr8, fps)

        profile = pipe.start(cfg)
        # only keep a pipeline that started, so stop() after a failed start is a no-op
        self.pipe = pipe
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        # keeps the calibrated robot->IR extrinsics, no separate color extrinsic needed. (Essential for D435 where color and IR cameras are physcially separated)
        self.align = rs.align(rs.stream.depth)

    def stop(self):
        if self.pipe is not None:
            self.pipe.stop()
            self.pipe = None
            self.align = None

    def _fresh_frames(self, warmup=0):
        # drop whatever the pipeline buffered while the arm was still moving
        while self.pipe.poll_for_frames():
            pass
        for _ in range(warmup):
            self.pipe.wait_for_frames()
        return self.pipe.wait_for_frames()

    def capture(self, out_dir, view_index=0, warmup=0):

        if self.pipe is None:
            raise RuntimeError(f'{self.name}: call start() before capture()')

        frames = self._fresh_frames(warmup)
        raw_color = frames.get_color_frame()

        aligned = self.align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth or not raw_color:
            raise RuntimeError(f'{self.name}: failed to capture a frame')

        # aligned color is in the calibrated IR frame at the depth resolution; pixels with no depth are black
        color_img = np.asanyarray(color.get_data())   # (H,W,3) BGR uint8
        # unaligned color at COLOR_SIZE, kept so pixels blacked out by alignment are recoverable
        raw_color_img = np.asanyarray(raw_color.get_data())

        # raw sensor units -> metric mm, float32; 0 means no depth, the same hole convention as training
        depth_mm = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale * 1000.0

        view_dir = Path(out_dir) / f'{view_index:03d}'
        view_dir.mkdir(parents=True, exist_ok=True)
        color_path = view_dir / f'{self.name}_color.png'
        raw_color_path = view_dir / f'{self.name}_color_raw.png'
        depth_path = view_dir / f'{self.name}_depth.npy'

        cv2.imwrite(str(color_path), color_img)
        cv2.imwrite(str(raw_color_path), raw_color_img)
        np.save(depth_path, depth_mm)

        _log.info('%-15s captured %s, %s, %s', self.name, color_path.name, raw_color_path.name, depth_path.name)
        return dict(
            color=color_path.name,
            color_raw=raw_color_path.name,
            depth=depth_path.name,
            serial=self.serial,
            frame_number=frames.get_frame_number(),
            timestamp_ms=frames.get_timestamp(),
            timestamp_domain=str(frames.get_frame_timestamp_domain()),
            depth_scale=self.depth_scale,
        )
