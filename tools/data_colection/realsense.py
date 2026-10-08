import logging
from pathlib import Path

import numpy as np
import cv2

from camera import Camera

_log = logging.getLogger('realsense')

COLOR_SIZE = (1280, 720)
# 15 fps with no IR stream keeps all 7 cameras at 1280x720 within the USB hubs' bandwidth
FPS = 15

VISUAL_PRESET = 'Default'

# each capture fuses N FRAMES
FRAMES = 15
# pixels with depth in at least k frames are marked as valid
MIN_VALID = 3
# crop background for anything further than 1 m
MAX_DEPTH_MM = 1000.0
# the largest z16 value is the camera's saturation code, not a distance
_SATURATED = 65535


def fuse(depth_mm, color, min_valid=MIN_VALID):
    '''Fuse a series of aligned frames.

    Returns: 
        the per-pixel median depth over its valid frames (0 where fewer than min_valid).
        the color from the frame holding that median.
        the number of valid frames per pixel.
    '''
    count = np.isfinite(depth_mm).sum(axis=0)
    # nan sorts last, so each pixel's valid readings come first, in order
    order = np.argsort(depth_mm, axis=0)
    lo = np.take_along_axis(order, ((np.maximum(count, 1) - 1) // 2)[None], axis=0)[0]
    hi = np.take_along_axis(order, (np.maximum(count, 1) // 2)[None], axis=0)[0]
    median = 0.5 * (np.take_along_axis(depth_mm, lo[None], axis=0)[0]
                    + np.take_along_axis(depth_mm, hi[None], axis=0)[0])

    keep = count >= min_valid
    depth = np.where(keep, median, 0).astype(np.float32)
    rows, cols = np.indices(lo.shape)
    fused_color = color[lo, rows, cols]
    fused_color[~keep] = 0
    return depth, fused_color, count.astype(np.uint8)


def _set_visual_preset(sensor, name):
    import pyrealsense2 as rs

    span = sensor.get_option_range(rs.option.visual_preset)
    for value in range(int(span.min), int(span.max) + 1):
        if sensor.get_option_value_description(rs.option.visual_preset, value) == name:
            sensor.set_option(rs.option.visual_preset, value)
            return
    raise RuntimeError(f'visual preset {name!r} is not available')


class RealSense(Camera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe = None
        self.align = None
        self.depth_scale = None

    def start(self, color_size=COLOR_SIZE, fps=FPS):
        import pyrealsense2 as rs

        depth_w, depth_h = self.resolution or (480, 270)
        if self.type == 'd405':
            # the D405 streams depth and color off one sensor, so they must share a resolution
            color_size = (depth_w, depth_h)

        # set the preset and imaging settings before streaming: changing them on a running stream
        # restarts it, which made starting the rig take ~20 s instead of ~4 s
        for device in rs.context().query_devices():
            if device.get_info(rs.camera_info.serial_number) == self.serial:
                _set_visual_preset(device.first_depth_sensor(), VISUAL_PRESET)
                # after the preset, which turns auto exposure back on
                self._apply_imaging(device)
                break

        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        # no IR stream: depth is computed on the camera, and IR frames were never saved
        cfg.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, *color_size, rs.format.bgr8, fps)

        profile = pipe.start(cfg)
        # only keep a pipeline that started, so stop() after a failed start is a no-op
        self.pipe = pipe
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        # keeps the calibrated robot->IR extrinsics, no separate color extrinsic needed. (Essential for D435 where color and IR cameras are physcially separated)
        self.align = rs.align(rs.stream.depth)

    def _apply_imaging(self, device):
        '''Set exposure, gain and white balance from the capture settings file; sensors without settings keep auto.'''
        import pyrealsense2 as rs

        for (kind, values) in self.imaging.items():
            if kind == 'stereo':
                sensor = device.first_depth_sensor()
            elif kind == 'rgb':
                sensor = device.first_color_sensor()
            else:
                raise ValueError(f'{self.name}: unknown sensor {kind!r} in the capture settings')

            if 'exposure' in values:
                sensor.set_option(rs.option.enable_auto_exposure, 0)
                sensor.set_option(rs.option.exposure, values['exposure'])
            if 'gain' in values:
                sensor.set_option(rs.option.gain, values['gain'])
            if 'white_balance' in values:
                sensor.set_option(rs.option.enable_auto_white_balance, 0)
                sensor.set_option(rs.option.white_balance, values['white_balance'])

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

    def _burst(self, first, count, max_depth_mm):
        '''count distinct aligned framesets starting with first, as (N,H,W) depth in mm with nan
        where unusable and (N,H,W,3) color.'''
        depths, colors, seen = [], [], set()
        frames = first
        while True:
            aligned = self.align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            if not depth or not color:
                raise RuntimeError(f'{self.name}: failed to capture a frame')

            # the pipeline can hand back the same frameset twice, which would count one reading twice
            if depth.get_frame_number() not in seen:
                seen.add(depth.get_frame_number())
                raw = np.asanyarray(depth.get_data())
                mm = raw.astype(np.float32) * self.depth_scale * 1000.0
                usable = (raw > 0) & (raw < _SATURATED) & (mm <= max_depth_mm)
                depths.append(np.where(usable, mm, np.nan))
                # aligned color is in the calibrated IR frame at the depth resolution
                colors.append(np.array(color.get_data()))
                if len(depths) == count:
                    return np.stack(depths), np.stack(colors)

            frames = self.pipe.wait_for_frames()

    def capture(self, out_dir, view_index=0, warmup=0,
                frames=FRAMES, min_valid=MIN_VALID, max_depth_mm=MAX_DEPTH_MM):

        if self.pipe is None:
            raise RuntimeError(f'{self.name}: call start() before capture()')

        first = self._fresh_frames(warmup)
        raw_color = first.get_color_frame()
        if not raw_color:
            raise RuntimeError(f'{self.name}: failed to capture a frame')
        # unaligned color at COLOR_SIZE, kept so pixels blacked out by alignment are recoverable
        raw_color_img = np.array(raw_color.get_data())

        depth_burst, color_burst = self._burst(first, frames, max_depth_mm)
        # float32 mm with 0 for no depth (missing, saturated, beyond max_depth_mm), the same hole
        # convention as training; color is black wherever depth is 0
        depth_mm, color_img, valid_count = fuse(depth_burst, color_burst, min_valid)

        view_dir = Path(out_dir) / f'{view_index:03d}'
        view_dir.mkdir(parents=True, exist_ok=True)
        color_path = view_dir / f'{self.name}_color.png'
        raw_color_path = view_dir / f'{self.name}_color_raw.png'
        depth_path = view_dir / f'{self.name}_depth.npy'
        count_path = view_dir / f'{self.name}_depth_count.npy'

        cv2.imwrite(str(color_path), color_img)
        cv2.imwrite(str(raw_color_path), raw_color_img)
        np.save(depth_path, depth_mm)
        # how many of the frames had depth at each pixel, so later analysis can weigh its reliability
        np.save(count_path, valid_count)

        _log.info('%-15s captured %s, %s, %s', self.name, color_path.name, raw_color_path.name, depth_path.name)
        return dict(
            color=color_path.name,
            color_raw=raw_color_path.name,
            depth=depth_path.name,
            depth_count=count_path.name,
            serial=self.serial,
            visual_preset=VISUAL_PRESET,
            imaging=self.imaging or 'auto',
            frames=frames,
            min_valid=min_valid,
            max_depth_mm=max_depth_mm,
            frame_number=first.get_frame_number(),
            timestamp_ms=first.get_timestamp(),
            timestamp_domain=str(first.get_frame_timestamp_domain()),
            depth_scale=self.depth_scale,
        )
