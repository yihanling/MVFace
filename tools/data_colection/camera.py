import logging
from pathlib import Path

import numpy as np
import yaml

_log = logging.getLogger('camera')

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIB = REPO_ROOT / 'src' / 'mvface' / 'assets' / 'camera_matrix.yaml'

# frames discarded after opening a camera, so auto-exposure settles (1 s at 30 fps)
WARMUP_FRAMES = 30


def _class_for_type(cam_type):
    if cam_type in ('d435', 'd405'):
        # same script-directory import as realsense.py uses for camera; deferred to avoid the import cycle
        from realsense import RealSense
        return RealSense
    
    raise ValueError(f'no Camera subclass registered for type {cam_type!r}')


class Camera:

    def __init__(self, name, serial, type, intrinsics, extrinsics,
                 distortion=None, resolution=None, rotate=0):
        self.name = name
        self.serial = serial
        self.type = type
        self.intrinsics = np.asarray(intrinsics, dtype=float)
        self.extrinsics = np.asarray(extrinsics, dtype=float)
        self.distortion = np.asarray(distortion if distortion is not None else [0, 0, 0, 0, 0], dtype=float)
        self.resolution = tuple(resolution) if resolution is not None else None
        self.rotate = rotate

    @classmethod
    def from_config(cls, name, cfg):
        if cls is Camera:
            # build based on subclass, so callers like CamerRig can build cameras
            return _class_for_type(cfg['type']).from_config(name, cfg)

        return cls(
            name=name,
            serial=str(cfg['serial']),
            type=cfg['type'],
            intrinsics=cfg['intrinsics'],
            extrinsics=cfg['extrinsics'],
            distortion=cfg.get('distortion'),
            resolution=cfg.get('resolution'),
            rotate=cfg.get('rotate', 0),
        )

    def start(self):
        raise NotImplementedError(f'{type(self).__name__}.start is not implemented')

    def stop(self):
        raise NotImplementedError(f'{type(self).__name__}.stop is not implemented')

    def capture(self, out_dir, view_index=0, warmup=0):
        raise NotImplementedError(f'{type(self).__name__}.capture is not implemented')


class CameraRig:
    def __init__(self, config_file=DEFAULT_CALIB):
        self.config_file = Path(config_file)
        cfg = yaml.safe_load(open(self.config_file))

        cameras = cfg.get('cameras')
        if not cameras:
            raise RuntimeError(f'no cameras in {config_file}, re-check is camera_matrix complies with the default format')

        self.cameras = [Camera.from_config(name, c) for (name, c) in cameras.items()]

        # True once the USB bus has failed to carry every camera at once
        self.sequential = False

    def start(self):
        '''Stream every camera at once, falling back to one camera at a time if they do not all fit.'''
        try:
            for camera in self.cameras:
                camera.start()
        except RuntimeError as e:
            self._fall_back(e)

            # still fail here, before the robot moves, if a camera is missing outright
            for camera in self.cameras:
                camera.start()
                camera.stop()

    def stop(self):
        for camera in self.cameras:
            camera.stop()

    def _fall_back(self, error):
        _log.warning('cannot stream all %d cameras at once (%s); capturing one camera at a time',
                     len(self.cameras), error)
        self.stop()
        self.sequential = True

    def capture_all(self, out_dir, view_index=0):
        if not self.sequential:
            try:
                return {camera.name: camera.capture(out_dir, view_index) for camera in self.cameras}
            except RuntimeError as e:
                # a starved bus often shows up as frame timeouts rather than a failed start
                self._fall_back(e)

        captured = {}
        for camera in self.cameras:
            camera.start()
            try:
                captured[camera.name] = camera.capture(out_dir, view_index, warmup=WARMUP_FRAMES)
            finally:
                camera.stop()
        return captured
