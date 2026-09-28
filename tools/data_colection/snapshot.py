import logging
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime, timezone
from time import sleep

from igmr_robotics_toolkit.util.yaml import safe_dump, denumpy

from camera import CameraRig, DEFAULT_CALIB
from face_scan import prompt_output

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

        sleep(args.settle)
        for idx in range(args.count):
            if idx:
                sleep(args.interval)

            entry = dict(view=idx, timestamp=datetime.now(timezone.utc).isoformat())
            try:
                entry['cameras'] = rig.capture_all(output, idx)
            except Exception as e:
                _log.error('set %d: capture failed: %s', idx, e)
                entry['capture_error'] = str(e)
            entry['one_camera_at_a_time'] = rig.sequential

            with open(output / f'view_{idx:03d}.yaml', 'w') as f:
                safe_dump(denumpy(entry), f)

        _log.info('captured %d set(s) to %s', args.count, output)
    finally:
        rig.stop()


if __name__ == '__main__':
    main()
