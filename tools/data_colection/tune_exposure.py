import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from camera import CameraRig, DEFAULT_CALIB, DEFAULT_SETTINGS
from realsense import FPS, MAX_DEPTH_MM, fuse, _set_visual_preset

_log = logging.getLogger('tune_exposure')

OUTPUT_ROOT = Path(__file__).resolve().parent / 'output'

# exposure candidates in each sensor's own units: the D405 stereo module takes microseconds,
# the D435 RGB camera 100 us steps; all stay well under the 66 ms frame time at 15 fps
EXPOSURES = dict(stereo=[2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000],
                 rgb=[10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140, 160])
WHITE_BALANCES = range(3000, 5001, 100)
# a pixel is clipped when any channel reaches this; the chosen exposure is the longest that
# keeps fewer than MAX_CLIPPED percent of the head clipped
CLIP_LEVEL = 250
MAX_CLIPPED = 1.0
# below this share of the head having colour, the face is not fully in the colour view and the
# head pixels are not representative of the face (the D435 colour window is narrower than depth)
MIN_COLOR_COVERAGE = 0.9


class TuningFailure(RuntimeError):
    pass


class Session:
    '''One camera streaming alone at its calibrated resolution, Default preset, everything auto.'''

    def __init__(self, camera):
        import pyrealsense2 as rs
        self.rs = rs
        self.camera = camera
        self.device = next(d for d in rs.context().query_devices()
                           if d.get_info(rs.camera_info.serial_number) == camera.serial)
        self.advanced = rs.rs400_advanced_mode(self.device)
        self.saved = self.advanced.serialize_json()
        _set_visual_preset(self.device.first_depth_sensor(), 'Default')

        (w, h) = camera.resolution
        cfg = rs.config()
        cfg.enable_device(camera.serial)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, FPS)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, FPS)
        self.pipe = rs.pipeline()
        profile = self.pipe.start(cfg)
        self.scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.depth)

        # the D405 makes colour on its stereo imagers; the D435 has a separate RGB camera
        self.kind = 'stereo' if camera.type == 'd405' else 'rgb'
        dev = profile.get_device()
        self.sensor = dev.first_depth_sensor() if self.kind == 'stereo' else dev.first_color_sensor()

    def close(self):
        rs = self.rs
        self.pipe.stop()
        self.advanced.load_json(self.saved)
        for sensor in self.device.query_sensors():
            for option in (rs.option.enable_auto_exposure, rs.option.enable_auto_white_balance):
                if sensor.supports(option):
                    sensor.set_option(option, 1)

    def settle(self, frames):
        for _ in range(frames):
            self.pipe.wait_for_frames()

    def burst(self, count=15):
        depths, colors, seen = [], [], set()
        while len(depths) < count:
            aligned = self.align.process(self.pipe.wait_for_frames())
            depth = aligned.get_depth_frame()
            if depth.get_frame_number() in seen:
                continue
            seen.add(depth.get_frame_number())
            raw = np.asanyarray(depth.get_data())
            mm = raw.astype(np.float32) * self.scale * 1000.0
            depths.append(np.where((raw > 0) & (raw < 65535) & (mm <= MAX_DEPTH_MM), mm, np.nan))
            colors.append(np.array(aligned.get_color_frame().get_data()))
        return fuse(np.stack(depths), np.stack(colors))

    def set(self, **values):
        rs = self.rs
        if 'exposure' in values:
            self.sensor.set_option(rs.option.enable_auto_exposure, 0)
            self.sensor.set_option(rs.option.exposure, values['exposure'])
        if 'gain' in values:
            self.sensor.set_option(rs.option.gain, values['gain'])
        if 'white_balance' in values:
            self.sensor.set_option(rs.option.enable_auto_white_balance, 0)
            self.sensor.set_option(rs.option.white_balance, values['white_balance'])
        self.settle(10)

    @property
    def min_gain(self):
        return self.sensor.get_option_range(self.rs.option.gain).min


def head_mask(depth):
    '''The largest connected region nearer than MAX_DEPTH_MM, with the holes inside it filled.'''
    near = ((depth > 0) & (depth < MAX_DEPTH_MM)).astype(np.uint8)
    (count, labels, stats, _) = cv2.connectedComponentsWithStats(near)
    if count < 2:
        raise TuningFailure('nothing within %.0f mm' % MAX_DEPTH_MM)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = cv2.morphologyEx((labels == largest).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    outside = mask.copy()
    cv2.floodFill(outside, np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8), (0, 0), 1)
    return (mask | (1 - outside)).astype(bool)


def measure(depth, color, head):
    '''Head holes (%), clipped head pixels among those with colour (%), mean head brightness.'''
    has_color = head & (color.max(axis=-1) > 0)
    clipped = 100 * (color.max(axis=-1)[has_color] >= CLIP_LEVEL).mean()
    brightness = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)[has_color].mean()
    return 100 * (depth[head] == 0).mean(), clipped, brightness


def chroma(color, head):
    '''Mean R/G and B/G over unclipped, not-too-dark head pixels.'''
    peak = color.max(axis=-1)
    (b, g, r) = color[head & (peak < CLIP_LEVEL) & (peak > 20)].astype(np.float64).mean(axis=0)
    return np.array([r / g, b / g])


def tune(camera, out_dir):
    session = Session(camera)
    try:
        session.settle(45)
        (depth, auto_color, _) = session.burst()
        head = head_mask(depth)
        auto = measure(depth, auto_color, head)

        coverage = (head & (auto_color.max(axis=-1) > 0)).sum() / max((head & (depth > 0)).sum(), 1)
        (ys, xs) = np.nonzero(head)
        (h, w) = head.shape
        cut = [side for (side, hit) in (('left', xs.min() == 0), ('right', xs.max() == w - 1), ('top', ys.min() == 0)) if hit]
        if coverage < MIN_COLOR_COVERAGE or cut:
            raise TuningFailure(f'face not fully in the colour view (colour on {100 * coverage:.0f}% of the head'
                                + (f', cut off at the {"/".join(cut)} edge' if cut else '') + ')')

        # white balance first, at a short exposure where nothing clips, matched to the camera's own
        # auto white balance; a wrong balance makes one channel clip early and forces a darker exposure
        reference = chroma(auto_color, head)
        gain = session.min_gain
        session.set(exposure=EXPOSURES[session.kind][2], gain=gain)
        errors = {}
        for wb in WHITE_BALANCES:
            session.set(white_balance=wb)
            (_, color, _) = session.burst(5)
            errors[wb] = float(np.abs(chroma(color, head) - reference).sum())
        white_balance = min(errors, key=errors.get)
        session.set(white_balance=white_balance)

        # then the longest exposure that keeps the head unclipped
        results = {}
        for exposure in EXPOSURES[session.kind]:
            session.set(exposure=exposure)
            (d, color, _) = session.burst()
            results[exposure] = (measure(d, color, head), color)
        passing = [e for (e, (m, _)) in results.items() if m[1] < MAX_CLIPPED]
        if not passing:
            raise TuningFailure(f'head clips even at the shortest exposure ({results[EXPOSURES[session.kind][0]][0][1]:.1f}%)')
        exposure = max(passing)
        (chosen, color) = results[exposure]

        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        tiles = [cv2.resize(im[y0:y1, x0:x1], (int((x1 - x0) * 300 / (y1 - y0)), 300)) for im in (auto_color, color)]
        cv2.imwrite(str(out_dir / f'{camera.name}_auto_vs_manual.png'), np.hstack(tiles))

        settings = {session.kind: dict(exposure=int(exposure), gain=int(gain), white_balance=int(white_balance))}
        return settings, auto, chosen
    finally:
        session.close()


def main():
    parser = ArgumentParser(description='Choose fixed exposure, gain and white balance for each camera under the '
                                        'current lighting, with the subject in view, and store them in the '
                                        'capture settings file that RealSense.start() applies.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--calib', default=str(DEFAULT_CALIB), help='camera calibration file')
    parser.add_argument('--settings', default=str(DEFAULT_SETTINGS), help='capture settings file to update')
    parser.add_argument('--only', nargs='+', help='tune only these cameras')
    parser.add_argument('--dry-run', action='store_true', help='report without updating the settings file')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # tuning starts from auto, so build the rig without applying any stored settings
    rig = CameraRig(args.calib, settings_file=None)
    cameras = [c for c in rig.cameras if not args.only or c.name in args.only]
    out_dir = OUTPUT_ROOT / datetime.now().strftime('%m-%d-%Y') / f'exposure_tuning_{datetime.now():%H%M}'
    out_dir.mkdir(parents=True, exist_ok=True)

    tuned, failed = {}, {}
    for camera in cameras:
        _log.info('tuning %s', camera.name)
        try:
            (settings, auto, chosen) = tune(camera, out_dir)
            tuned[camera.name] = settings
            _log.info('%-15s %s | auto: clipped %.1f%%, brightness %.0f, holes %.1f%% -> manual: clipped %.1f%%, '
                      'brightness %.0f, holes %.1f%%', camera.name, settings, auto[1], auto[2], auto[0],
                      chosen[1], chosen[2], chosen[0])
        except (TuningFailure, RuntimeError) as e:
            failed[camera.name] = str(e)
            _log.warning('%-15s not tuned, stays on auto: %s', camera.name, e)

    _log.info('comparison images in %s', out_dir)
    if args.dry_run or not tuned:
        return

    path = Path(args.settings)
    current = yaml.safe_load(open(path)) if path.exists() else None
    current = current or {}
    entries = current.get('cameras') or {}
    entries.update(tuned)
    with open(path, 'w') as f:
        f.write(SETTINGS_HEADER.format(date=datetime.now().strftime('%Y-%m-%d')))
        yaml.safe_dump(dict(cameras=entries), f, sort_keys=False, default_flow_style=None)
    _log.info('updated %s for %s', path, ', '.join(tuned))


SETTINGS_HEADER = '''\
# Fixed imaging settings per camera for this lab's lighting, chosen by
# tools/data_colection/tune_exposure.py (last run {date}) and applied by RealSense.start().
# Auto exposure / white balance is turned off for the listed sensor; a camera or sensor not
# listed stays on auto. Values are raw librealsense option values:
#   stereo: the stereo module (on the D405 it also makes the colour image); exposure in us
#   rgb:    the D435's separate RGB camera; exposure in 100 us steps
# Re-run the tuning whenever the lighting or the subject's distance changes.
'''

if __name__ == '__main__':
    main()
