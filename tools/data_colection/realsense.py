import logging
from pathlib import Path

import numpy as np
import cv2

from camera import Camera

_log = logging.getLogger('realsense')

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / 'data_colection' / 'data'
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

        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.infrared, 1, ir_w, ir_h, rs.format.y8, fps)
        cfg.enable_stream(rs.stream.depth, ir_w, ir_h, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, *color_size, rs.format.bgr8, fps)

        profile = self.pipe.start(cfg)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        # keeps the calibrated robot->IR extrinsics, no separate color extrinsic needed. (Essential for D435 where color and IR cameras are physcially separated)
        self.align = rs.align(rs.stream.depth)

    def stop(self):
        if self.pipe is not None:
            self.pipe.stop()
            self.pipe = None
            self.align = None

    def capture(self, out_dir=DEFAULT_OUT_DIR, experiment_index=0):

        if self.pipe is None:
            raise RuntimeError(f'{self.name}: call start() before capture()')

        frames = self.align.process(self.pipe.wait_for_frames())
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError(f'{self.name}: failed to capture a frame')

        color_img = np.asanyarray(color.get_data())   # (H,W,3) BGR uint8

        # raw sensor units -> metric mm, float32
        depth_mm = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale * 1000.0

        exp_dir = Path(out_dir) / str(experiment_index)
        exp_dir.mkdir(parents=True, exist_ok=True)
        color_path = exp_dir / f'{self.name}_color.png'
        depth_path = exp_dir / f'{self.name}_depth.npy'

        cv2.imwrite(str(color_path), color_img)
        np.save(depth_path, depth_mm)

        _log.info('%-15s captured %s, %s', self.name, color_path.name, depth_path.name)
        return color_path, depth_path
