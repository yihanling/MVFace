import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import ceil, cos, sin, degrees, radians
from pathlib import Path
from time import monotonic, sleep
from typing import Callable, List, Optional, Tuple

import numpy as np
import coloredlogs

import igmr_robotics_toolkit.util.default_logging

from igmr_robotics_toolkit.control.simple import PointToPoint, PointToPointFailure
from igmr_robotics_toolkit.math import hinv, norm, unit
from igmr_robotics_toolkit.motion.path import check_joint_path
from igmr_robotics_toolkit.robot.loader import load_robot
from igmr_robotics_toolkit.util import parse_xform
from igmr_robotics_toolkit.util.yaml import safe_load, safe_dump, denumpy

_log = logging.getLogger('face_scan')
coloredlogs.install(level=logging.INFO, logger=_log)

DEFAULT_CONFIG = Path(__file__).with_suffix('.yaml')


@dataclass
class ScanConfig:
    model: str = 'LBRmed7'
    home_q: np.ndarray = field(default_factory=lambda: np.radians([0, 30, 0, -60, 0, 90, 0]))

    joint_speed: float = radians(10)
    tool_speed: float = 20e-3

    face: np.ndarray = field(default_factory=lambda: parse_xform('trans(0.65, 0, 0.10) rot(0, 0, -90d)'))
    camera: np.ndarray = field(default_factory=lambda: np.eye(4))

    radius: float = 0.20
    azimuths: np.ndarray = field(default_factory=lambda: np.radians(np.linspace(-30, 30, 5)))
    elevations: np.ndarray = field(default_factory=lambda: np.radians(np.linspace(-15, 15, 3)))

    serpentine: bool = True
    arc_step: float = radians(5)

    approach_offset: float = 0.05
    retreat_offset: float = 0.15
    settle_time: float = 0.25

    max_joint_step: float = radians(30)

    @property
    def motion_limits(self) -> dict:
        return dict(qd_limits=self.joint_speed, tcp_linear_limit=self.tool_speed)

def load_config(path: Path) -> ScanConfig:
    with open(path) as f:
        raw = safe_load(f)

    def grid(spec):
        (low, high, count) = spec
        return np.radians(np.linspace(low, high, int(count)))

    (robot, face, camera, scan, limits) = (raw['robot'], raw['face'], raw['camera'], raw['scan'], raw['limits'])

    return ScanConfig(
        model=robot['model'],
        home_q=np.radians(robot['home']),
        joint_speed=radians(robot['joint_speed']),
        tool_speed=1e-3 * robot['tool_speed'],

        face=parse_xform(face['xform']),
        camera=parse_xform(camera['xform']),

        radius=scan['radius'],
        azimuths=grid(scan['azimuth']),
        elevations=grid(scan['elevation']),
        serpentine=scan['serpentine'],
        arc_step=radians(scan['arc_step']),
        approach_offset=scan['approach_offset'],
        retreat_offset=scan['retreat_offset'],
        settle_time=scan['settle_time'],

        max_joint_step=radians(limits['max_joint_step']),
    )


def look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
 
    eye = np.asanyarray(eye, dtype=float)
    z = unit(np.asanyarray(target, dtype=float) - eye)

   
    x = np.cross(z, np.asanyarray(up, dtype=float))
    if norm(x) < 1e-9:
        x = np.cross(z, [1, 0, 0] if abs(z[0]) < 0.9 else [0, 1, 0])
    x = unit(x)

    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, unit(np.cross(z, x)), z], axis=1)
    pose[:3, 3] = eye
    return pose

def view_pose(radius: float, azimuth: float, elevation: float) -> np.ndarray:
    eye = radius * np.array([
        cos(elevation) * sin(azimuth),
        sin(elevation),
        cos(elevation) * cos(azimuth),
    ])

    return look_at(eye, [0, 0, 0], up=[0, 1, 0])

def view_angles(config: ScanConfig) -> List[Tuple[float, float]]:
    angles = []
    for (row, elevation) in enumerate(config.elevations):
        azimuths = config.azimuths
        if config.serpentine and row % 2:
            azimuths = azimuths[::-1]
        angles.extend((azimuth, elevation) for azimuth in azimuths)
    return angles

def subdivide(start: Tuple[float, float], end: Tuple[float, float], step: float) -> List[Tuple[float, float]]:
    span = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    n = max(2, ceil(span / step) + 1)
    return [(start[0] + r * (end[0] - start[0]), start[1] + r * (end[1] - start[1]))
            for r in np.linspace(0, 1, n)[1:]]

# ----------------------------------------------------------------------------
# planning
# ----------------------------------------------------------------------------

@dataclass
class ScanPlan:
    config: ScanConfig
    angles: List[Tuple[float, float]]

    camera_poses: List[np.ndarray]

    # joint waypoints leading to each view; the last entry of each segment is
    # the view itself, so the robot comes to rest there
    segments: List[np.ndarray]

    approach_q: np.ndarray
    retreat_q: np.ndarray

    @property
    def view_count(self) -> int:
        return len(self.angles)

    @property
    def path(self) -> np.ndarray:
        '''The whole scan as one joint path, for validation and preview.'''
        return np.concatenate([[self.approach_q]] + self.segments + [[self.retreat_q]])

    @property
    def max_joint_step(self) -> float:
        return np.linalg.norm(np.diff(self.path, axis=0), axis=1).max()

class PlanFailure(RuntimeError):
    pass

def flange_pose(config: ScanConfig, radius: float, angle: Tuple[float, float]) -> np.ndarray:
    '''Flange pose in base coordinates that puts the camera at a viewpoint.'''
    return config.face @ view_pose(radius, *angle) @ hinv(config.camera)

def solve(kinematics, pose: np.ndarray, reference: np.ndarray, what: str) -> np.ndarray:
    try:
        return kinematics.inverse_nearest(pose, reference)
    except RuntimeError as e:
        raise PlanFailure(f'no inverse kinematics solution for {what} '
                          f'at {np.round(pose[:3, 3], 3)} m: {e}') from e

def solve_standoff(kinematics, config: ScanConfig, angle, offset: float,
                   reference: np.ndarray, what: str) -> np.ndarray:
    for scale in [1, 0.75, 0.5, 0.25]:
        try:
            q = kinematics.inverse_nearest(flange_pose(config, config.radius + scale * offset, angle), reference)
        except RuntimeError:
            continue

        if scale < 1:
            _log.warning('%s shortened to %.0f mm to stay in the workspace', what, 1e3 * scale * offset)
        return q

    raise PlanFailure(f'{what} is unreachable at any standoff up to {1e3 * offset:.0f} mm')

def chain(kinematics, config: ScanConfig, angles, seed: np.ndarray):
    q = seed
    segments = []

    for (idx, (previous, angle)) in enumerate(zip([angles[0]] + angles[:-1], angles)):
        # walk along the sphere from the previous view to this one; the first
        # view is simply where the seed already is
        steps = [angle] if idx == 0 else subdivide(previous, angle, config.arc_step)

        segment = []
        for step in steps:
            q = solve(kinematics, flange_pose(config, config.radius, step), q, f'view {idx}')
            segment.append(q)

        segments.append(np.asanyarray(segment))

    return segments

def plan_scan(model, config: ScanConfig) -> ScanPlan:
    kinematics = model.kinematics
    angles = view_angles(config)
    if not angles:
        raise PlanFailure('scan has no viewpoints')

    first = flange_pose(config, config.radius, angles[0])
    try:
        seeds = kinematics.inverse(first, config.home_q)
    except RuntimeError as e:
        raise PlanFailure(f'first viewpoint at {np.round(first[:3, 3], 3)} m is unreachable: {e}') from e
    if not seeds:
        raise PlanFailure(f'first viewpoint at {np.round(first[:3, 3], 3)} m is unreachable')

    best: Optional[ScanPlan] = None
    failures = []

    for seed in seeds:
        try:
            segments = chain(kinematics, config, angles, seed)
        except PlanFailure as e:
            failures.append(str(e))
            continue

        plan = ScanPlan(
            config, angles,
            camera_poses=[config.face @ view_pose(config.radius, *angle) for angle in angles],
            segments=segments,
            approach_q=solve_standoff(kinematics, config, angles[0], config.approach_offset,
                                      seed, 'approach standoff'),
            retreat_q=solve_standoff(kinematics, config, angles[-1], config.retreat_offset,
                                     segments[-1][-1], 'retreat standoff'),
        )

        if best is None or plan.max_joint_step < best.max_joint_step:
            best = plan

    if best is None:
        raise PlanFailure('no reachable branch for this scan:\n  ' + '\n  '.join(failures))

    _log.info('chose the smoothest of %d inverse kinematics branches', len(seeds))
    validate(model, best)
    return best

def validate(model, plan: ScanPlan) -> None:
    try:
        check_joint_path(plan.path, tolerance=plan.config.max_joint_step, limits=model.kinematics.limits())
    except RuntimeError as e:
        raise PlanFailure(str(e)) from e

def report(model, plan: ScanPlan) -> None:
    config = plan.config

    _log.info('face at %s m, looking along %s, crown along %s',
              np.round(config.face[:3, 3], 3), np.round(config.face[:3, 2], 2), np.round(config.face[:3, 1], 2))
    _log.info('%d views on a %.0f mm sphere, %d joint waypoints',
              plan.view_count, 1e3 * config.radius, len(plan.path))

    for (idx, (angle, pose)) in enumerate(zip(plan.angles, plan.camera_poses)):
        _log.debug('view %2d: azimuth %+6.1f deg, elevation %+6.1f deg, camera at %s m',
                   idx, degrees(angle[0]), degrees(angle[1]), np.round(pose[:3, 3], 3))

    (lb, ub) = model.kinematics.limits()
    margin = np.minimum(plan.path - lb, ub - plan.path).min(axis=0)
    _log.info('largest joint step %.1f deg (limit %.1f)',
              degrees(plan.max_joint_step), degrees(config.max_joint_step))
    _log.info('joint limit margin %s deg', np.round(np.degrees(margin), 1))

def preview(model, plan: ScanPlan) -> None:
    '''Animate the planned path with the viewpoint frames drawn in place.'''
    from igmr_robotics_toolkit.viewer.motion import PlanViewer
    from igmr_robotics_toolkit.viewer.widget import TransformListWidget, LineWidget

    viewer = PlanViewer(title='IRTk - Face Scan Preview')
    viewer.add_path(model, plan.path, duration=max(5.0, 0.5 * plan.view_count))

    # the face frame plus every viewpoint, and the arc joining them
    TransformListWidget(parent=viewer.root).load([plan.config.face] + list(plan.camera_poses))
    LineWidget(parent=viewer.root).load([pose[:3, 3] for pose in plan.camera_poses])

    viewer.show()


def create_controller(model, config: ScanConfig, args):
    if args.simulate:
        from igmr_robotics_toolkit.control.simulator import Simulator
        return Simulator(model, q=config.home_q)
    else:
        return model.Controller(args.robot, model, payload=args.payload)

@contextmanager
def connected(model, controller, config: ScanConfig):
    ptp = PointToPoint(model, controller, **config.motion_limits)
    ptp.connect()
    
    try:
        wait_until_ready(ptp)
        yield ptp
    finally:
        ptp.disconnect()

def wait_until_ready(ptp, timeout=10):
    '''Block until the control loop has ticked once and reported a state.'''
    deadline = monotonic() + timeout
    while ptp.motion_state is None:
        if monotonic() > deadline:
            raise TimeoutError('no robot state after connecting - is the robot reachable and in AUT mode?')
        sleep(0.01)

def execute(model, ptp, plan: ScanPlan, capture: Callable[[int, np.ndarray], None]) -> None:
    config = plan.config

    _log.info('moving to home')
    ptp.move_joint(config.home_q)

    _log.info('approaching the first viewpoint')
    ptp.move_joint(plan.approach_q)

    for (idx, segment) in enumerate(plan.segments):
        (azimuth, elevation) = plan.angles[idx]
        _log.info('view %2d/%d: azimuth %+6.1f deg, elevation %+6.1f deg',
                  idx + 1, plan.view_count, degrees(azimuth), degrees(elevation))

        ptp.move_joint_waypoints(list(segment))

        # let the arm settle before recording, so the image is not smeared
        sleep(config.settle_time)
        capture(idx, ptp.motion_state.q)

    _log.info('retreating from the face')
    ptp.move_joint(plan.retreat_q)

    _log.info('returning home')
    ptp.move_joint(config.home_q)

def make_recorder(model, plan: ScanPlan, output: Optional[Path]) -> Callable[[int, np.ndarray], None]:
    if output is None:
        def log_only(idx, q):
            _log.info('   at %s m (not recording)',
                      np.round(model.kinematics.forward(q)[:3, 3], 4))
        return log_only

    output.mkdir(parents=True, exist_ok=True)
    _log.info('recording to %s', output)

    def record(idx, q):
        (azimuth, elevation) = plan.angles[idx]
        flange = model.kinematics.forward(q)

        entry = dict(
            view=idx,
            timestamp=datetime.now(timezone.utc).isoformat(),
            azimuth_deg=degrees(azimuth),
            elevation_deg=degrees(elevation),
            actual_q=q,
            flange_pose=flange,
            camera_pose=flange @ plan.config.camera,
            planned_camera_pose=plan.camera_poses[idx],
        )

        with open(output / f'view_{idx:03d}.yaml', 'w') as f:
            safe_dump(denumpy(entry), f)

    return record

def write_manifest(output: Optional[Path], plan: ScanPlan) -> None:
    if output is None:
        return

    output.mkdir(parents=True, exist_ok=True)
    with open(output / 'scan.yaml', 'w') as f:
        safe_dump(denumpy(dict(
            started=datetime.now(timezone.utc).isoformat(),
            model=plan.config.model,
            face_pose=plan.config.face,
            camera_in_flange=plan.config.camera,
            radius=plan.config.radius,
            views=[dict(view=i, azimuth_deg=degrees(a), elevation_deg=degrees(e))
                   for (i, (a, e)) in enumerate(plan.angles)],
        )), f)

def run(args):
    config = load_config(args.config)
    model = load_robot(config.model)

    plan = plan_scan(model, config)
    report(model, plan)

    if args.preview:
        preview(model, plan)

    if args.plan_only:
        return

    output = Path(args.output) if args.output else None
    write_manifest(output, plan)

    controller = create_controller(model, config, args)
    with connected(model, controller, config) as ptp:
        mark = monotonic()
        execute(model, ptp, plan, make_recorder(model, plan, output))
        _log.info('scan finished in %.0f s', monotonic() - mark)

def main():
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0],
                            formatter_class=ArgumentDefaultsHelpFormatter)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--simulate', '-s', action='store_true', help='run against the built-in simulator')
    group.add_argument('--robot', '-r', type=str, help='hostname or address of the robot')
    group.add_argument('--plan-only', action='store_true', help='plan the scan and exit without connecting')

    parser.add_argument('--config', '-c', type=Path, default=DEFAULT_CONFIG, help='scan configuration file')
    parser.add_argument('--preview', action='store_true', help='animate the planned path in the 3D viewer')
    parser.add_argument('--output', '-o', type=str, help='directory to record per-view data into')
    parser.add_argument('--payload', type=float, default=0, help='tool payload in kg (hardware only)')
    parser.add_argument('--verbose', '-v', action='store_true', help='log every viewpoint')

    args = parser.parse_args()
    if args.verbose:
        _log.setLevel(logging.DEBUG)
        coloredlogs.install(level=logging.DEBUG, logger=_log)

    try:
        run(args)
    except (PlanFailure, PointToPointFailure) as e:
        _log.error('%s', e)
        raise SystemExit(1)

if __name__ == '__main__':
    main()
