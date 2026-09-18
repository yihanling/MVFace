import logging
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_log = logging.getLogger('ir_to_color')

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIB = REPO_ROOT / 'src' / 'mvface' / 'assets' / 'camera.yaml'


SERIALS = {
    'd435_top':       '238222074486',
    'd435_below':     '238222070823',
    'd405_center':    '128422271548',
    'd405_right':     '218622273459',
    'd405_left':      '230322272744',
    'd405_top_left':  '218622272247',
    'd405_top_right': '218622270886',
}

IR_SIZE = (480, 270)
COLOR_SIZE = (1280, 720)


def load_yaml(path):
    try:
        from igmr_robotics_toolkit.util import yaml
        return yaml.safe_load(open(path))
    except ImportError:
        import yaml
        return yaml.safe_load(open(path))


def dump_yaml(obj, path):
    try:
        from igmr_robotics_toolkit.util import yaml
        yaml.safe_dump(obj, open(path, 'w'))
    except ImportError:
        import yaml
        yaml.safe_dump(obj, open(path, 'w'), default_flow_style=None, sort_keys=False)


def pose(R, t):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t
    return H


def check_rotation(R, what):
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-5):
        raise RuntimeError(f'{what}: rotation is not orthonormal')
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-5):
        raise RuntimeError(f'{what}: rotation has determinant {np.linalg.det(R):.6f}, expected +1')


def query_device(serial, ir_size, color_size):
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.infrared, 1, *ir_size, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.color, *color_size, rs.format.bgr8, 30)

    profile = pipe.start(cfg)
    try:
        ir = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        color = profile.get_stream(rs.stream.color).as_video_stream_profile()

        ext = ir.get_extrinsics_to(color)

        R = np.asarray(ext.rotation, dtype=float).reshape(3, 3).T
        t = np.asarray(ext.translation, dtype=float)
        check_rotation(R, f'{serial} IR->color')

        ci = color.get_intrinsics()
        K = np.array([[ci.fx, 0.0, ci.ppx],
                      [0.0, ci.fy, ci.ppy],
                      [0.0, 0.0, 1.0]])
        return pose(R, t), K, list(ci.coeffs), str(ci.model)
    finally:
        pipe.stop()


def update(name, cam, ir_size, color_size):
    serial = str(cam.get('serial') or SERIALS.get(name) or '')
    if not serial:
        raise RuntimeError(f'{name}: no serial in camera.yaml and none known for this name')

    E_ir = np.asarray(cam['extrinsics'], dtype=float)
    check_rotation(E_ir[:3, :3], f'{name} calibrated extrinsics')

    (T, K, coeffs, model) = query_device(serial, ir_size, color_size)
    E_color = T @ E_ir

    shift = np.linalg.norm(np.linalg.inv(E_color)[:3, 3] - np.linalg.inv(E_ir)[:3, 3])
    _log.info('%-15s %s  IR->color baseline %5.1f mm, camera center moves %5.1f mm',
              name, serial, 1e3 * np.linalg.norm(T[:3, 3]), 1e3 * shift)

    cam['serial'] = serial
    cam.setdefault('resolution', list(ir_size))
    cam['color'] = dict(
        intrinsics=K.tolist(),
        distortion=[float(c) for c in coeffs],
        extrinsics=E_color.tolist(),
        resolution=list(color_size),
        distortion_model=model,
        source=f'factory IR->color extrinsic composed onto the calibrated '
               f'flange->IR extrinsic, {datetime.now(timezone.utc).date().isoformat()}',
    )
    return shift


def main():
    parser = ArgumentParser(description='Compose the factory IR->color extrinsic into camera.yaml.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--calib', '-c', type=str, default=str(DEFAULT_CALIB),
                        help='rig calibration file to read and update')
    parser.add_argument('--only', '-o', nargs='+', metavar='NAME',
                        help='restrict to these cameras (default: every camera in the file)')
    parser.add_argument('--ir-size', nargs=2, type=int, default=IR_SIZE, metavar=('W', 'H'),
                        help='IR resolution the rig was calibrated at')
    parser.add_argument('--color-size', nargs=2, type=int, default=COLOR_SIZE, metavar=('W', 'H'),
                        help='color resolution the pipeline will stream; intrinsics are tied to it')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='query the devices and report, but do not write')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    path = Path(args.calib)
    cfg = load_yaml(path)
    cameras = cfg.get('cameras') or cfg.get('camera') or {}
    if not cameras:
        raise SystemExit(f'no cameras in {path}')

    names = args.only or list(cameras)
    unknown = [n for n in names if n not in cameras]
    if unknown:
        raise SystemExit(f'not in {path.name}: {", ".join(unknown)}')

    failed = []
    for name in names:
        try:
            update(name, cameras[name], tuple(args.ir_size), tuple(args.color_size))
        except Exception as e:
            _log.error('%-15s FAILED: %s', name, e)
            failed.append(name)

    if failed:
        raise SystemExit(f'{len(failed)} camera(s) not updated: {", ".join(failed)}; nothing written')

    if args.dry_run:
        _log.info('dry run: %s not modified', path)
        return

    dump_yaml(cfg, path)
    _log.info('wrote %s', path)


if __name__ == '__main__':
    sys.exit(main())
