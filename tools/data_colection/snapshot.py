import logging
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime, timezone
from time import sleep

from tqdm import tqdm

from igmr_robotics_toolkit.util.yaml import safe_dump, denumpy

from camera import CameraRig, DEFAULT_CALIB
from face_scan import OUTPUT_ROOT, prompt_output

_log = logging.getLogger('face_scan')


def main():
    parser = ArgumentParser(description='Capture still sets from every camera with the arm stationary.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--count', '-n', type=int, default=1, help='number of sets to capture')
    parser.add_argument('--interval', type=float, default=1.0, help='seconds between sets')
    parser.add_argument('--settle', type=float, default=2.0,
                        help='seconds to stream before the first set, so auto-exposure settles')
    parser.add_argument('--calib', default=str(DEFAULT_CALIB), help='camera calibration file')
    args = parser.parse_args()

    output = prompt_output()
    rig = CameraRig(args.calib)
    rig.start()

    try:
        output.mkdir(parents=True)
        shutil.copy(rig.config_file, output / 'camera_matrix.yaml')
        # and the fixed exposure / white balance the cameras ran with
        if rig.settings_file is not None:
            shutil.copy(rig.settings_file, output / 'capture_settings.yaml')

        sleep(args.settle)
        with tqdm(total=args.count * len(rig.cameras), desc=str(output.relative_to(OUTPUT_ROOT)),
                  unit='image') as bar:
            for idx in range(args.count):
                if idx:
                    sleep(args.interval)

                # a fall back re-captures every camera, so count each camera once per set
                done = set()
                def captured(name):
                    if name not in done:
                        done.add(name)
                        bar.update()
                    bar.set_postfix_str(f'set {idx + 1}/{args.count}, {name}')

                entry = dict(view=idx, timestamp=datetime.now(timezone.utc).isoformat())
                try:
                    entry['cameras'] = rig.capture_all(output, idx, on_captured=captured)
                except Exception as e:
                    _log.error('set %d: capture failed: %s', idx, e)
                    entry['capture_error'] = str(e)
                entry['one_camera_at_a_time'] = rig.sequential

                with open(output / f'view_{idx:03d}.yaml', 'w') as f:
                    safe_dump(denumpy(entry), f)
    finally:
        rig.stop()


if __name__ == '__main__':
    main()
