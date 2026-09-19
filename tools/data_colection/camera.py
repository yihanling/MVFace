from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIB = REPO_ROOT / 'src' / 'mvface' / 'assets' / 'camera_matrix.yaml'


def _class_for_type(cam_type):
    if cam_type in ('d435', 'd405'):
        from tools.data_colection.realsense import RealSense  # deferred: avoids a camera.py <-> realsense.py import cycle
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

    def capture(self, out_dir, experiment_index=0):
        raise NotImplementedError(f'{type(self).__name__}.capture is not implemented')


class CameraRig:
    def __init__(self, config_file=DEFAULT_CALIB):
        cfg = yaml.safe_load(open(config_file))

        cameras = cfg.get('cameras')
        if not cameras:
            raise RuntimeError(f'no cameras in {config_file}, re-check is camera_matrix complies with the default format')

        self.cameras = [Camera.from_config(name, c) for (name, c) in cameras.items()]

    def start(self):
        for camera in self.cameras:
            camera.start()

    def stop(self):
        for camera in self.cameras:
            camera.stop()

    def capture_all(self, out_dir, experiment_index=0):
        return {camera.name: camera.capture(out_dir, experiment_index) for camera in self.cameras}
