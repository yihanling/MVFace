import logging
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import ceil, cos, pi, sin, degrees, radians
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import coloredlogs
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import igmr_robotics_toolkit.util.default_logging

from igmr_robotics_toolkit.control.program import ProgramBase
from igmr_robotics_toolkit.math import hinv, norm, unit
from igmr_robotics_toolkit.motion.path import check_joint_path
from igmr_robotics_toolkit.robot.loader import load_robot
from igmr_robotics_toolkit.util import parse_xform
from igmr_robotics_toolkit.util.yaml import safe_dump, denumpy

if TYPE_CHECKING:
    from igmr_robotics_toolkit.control.controller import ControllerBase, RobotState
    from camera import CameraRig

_log = logging.getLogger('face_scan')
coloredlogs.install(level=logging.INFO, logger=_log)

CONTROL_RATE = 100

SIM_SPEED_SCALE = 3.0

# captures land in OUTPUT_ROOT/<MM-DD-YYYY>/<experiment id>
OUTPUT_ROOT = Path(__file__).resolve().parent / 'output'
DATE_FORMAT = '%m-%d-%Y'


def look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    '''Pose at eye whose +z axis points at target, +y as close to up as possible.'''
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

def serpentine_grid(columns, rows, serpentine: bool = True) -> List[tuple]:
    out = []
    for (index, row) in enumerate(rows):
        line = columns[::-1] if (serpentine and index % 2) else columns
        out.extend((column, row) for column in line)
    return out

def interpolate(start, end, span: float, max_step: float) -> List[tuple]:
    '''Views evenly spaced from start to end, end inclusive, start excluded.'''
    n = max(2, ceil(span / max_step) + 1)
    return [tuple(a + r * (b - a) for (a, b) in zip(start, end))
            for r in np.linspace(0, 1, n)[1:]]


class ScanPath:
    name = 'path'

    def views(self) -> List[tuple]:
        '''Every viewpoint, in the order they are captured.'''
        raise NotImplementedError

    def pose(self, view, standoff: float = 0) -> np.ndarray:
        '''Camera pose in face coordinates, pulled back along its own optical
        axis by standoff metres.'''
        raise NotImplementedError

    def subdivide(self, start, end) -> List[tuple]:
        raise NotImplementedError

    def describe(self, view) -> str:
        raise NotImplementedError

    def record(self, view) -> dict:
        '''Path-specific fields written into each view\'s record.'''
        raise NotImplementedError

    def summary(self) -> str:
        raise NotImplementedError


@dataclass
class SpherePath(ScanPath):
    name = 'sphere'

    radius: float = 0.5

    azimuths: np.ndarray = field(default_factory=lambda: np.radians(np.linspace(-20, 20, 7)))
    elevations: np.ndarray = field(default_factory=lambda: np.radians(np.linspace(-10, 10, 4)))

    serpentine: bool = True

    arc_step: float = radians(5)

    def views(self):
        return serpentine_grid(list(self.azimuths), list(self.elevations), self.serpentine)

    def pose(self, view, standoff: float = 0):
        (azimuth, elevation) = view
        radius = self.radius + standoff
        eye = radius * np.array([
            cos(elevation) * sin(azimuth),
            sin(elevation),
            cos(elevation) * cos(azimuth),
        ])
        return look_at(eye, [0, 0, 0], up=[0, 1, 0])

    def subdivide(self, start, end):
        span = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
        return interpolate(start, end, span, self.arc_step)

    def describe(self, view):
        return f'azimuth {degrees(view[0]):+6.1f} deg, elevation {degrees(view[1]):+6.1f} deg'

    def record(self, view):
        return dict(azimuth_deg=degrees(view[0]), elevation_deg=degrees(view[1]))

    def summary(self):
        return (f'{len(self.views())} views on a {1e3 * self.radius:.0f} mm sphere, '
                f'azimuth {degrees(self.azimuths[0]):+.0f}..{degrees(self.azimuths[-1]):+.0f} deg '
                f'x elevation {degrees(self.elevations[0]):+.0f}..{degrees(self.elevations[-1]):+.0f} deg')


@dataclass
class PlanePath(ScanPath):
    name = 'plane'
    
    # 0.30 m puts the grid corners past the arm's reach with the face 0.5 m ahead of home
    distance: float = 0.35
    width: float = 0.36
    height: float = 0.30

    columns: int = 7
    rows: int = 6

    serpentine: bool = True

    # maximum spacing between interpolated poses along the grid, metres
    step: float = 0.03

    @property
    def us(self) -> np.ndarray:
        return np.linspace(-self.width / 2, self.width / 2, self.columns)

    @property
    def vs(self) -> np.ndarray:
        return np.linspace(-self.height / 2, self.height / 2, self.rows)

    @property
    def orientation(self) -> np.ndarray:
        return look_at([0, 0, 1], [0, 0, 0], up=[0, 1, 0])[:3, :3]

    def views(self):
        return serpentine_grid(list(self.us), list(self.vs), self.serpentine)

    def pose(self, view, standoff: float = 0):
        (u, v) = view
        pose = np.eye(4)
        pose[:3, :3] = self.orientation
        pose[:3, 3] = [u, v, self.distance + standoff]
        return pose

    def subdivide(self, start, end):
        span = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
        return interpolate(start, end, span, self.step)

    def describe(self, view):
        return f'u {1e3 * view[0]:+6.0f} mm, v {1e3 * view[1]:+6.0f} mm'

    def record(self, view):
        return dict(u_mm=1e3 * view[0], v_mm=1e3 * view[1],
                    off_axis_deg=degrees(np.arctan2(np.hypot(*view), self.distance)))

    def summary(self):
        baseline = 1e3 * self.width / max(self.columns - 1, 1)
        corner = degrees(np.arctan2(np.hypot(self.width / 2, self.height / 2), self.distance))
        return (f'{len(self.views())} views on a {1e3 * self.width:.0f} x {1e3 * self.height:.0f} mm '
                f'grid {1e3 * self.distance:.0f} mm from the face, {baseline:.0f} mm baseline, '
                f'face up to {corner:.1f} deg off axis')


PATHS = dict(sphere=SpherePath, plane=PlanePath)


@dataclass
class ScanConfig:
    model: str = 'LBRmed7'
    home_q: np.ndarray = field(default_factory=lambda: np.radians([0, 55, 0, -90, 0, -56, 90]))

    # the servo loop below moves joint space only and does not limit Cartesian
    # tool speed
    joint_speed: float = radians(10)

    # 0.5 m ahead of the flange at home_q and 10 cm below it
    face: np.ndarray = field(default_factory=lambda: parse_xform(
        'trans(1.182996, 0.000000, 0.143969) aa(-1.148309, -1.148309, 1.247021)'))

    camera: np.ndarray = field(default_factory=lambda: np.eye(4))

    path: ScanPath = field(default_factory=SpherePath)

    approach_offset: float = 0.05
    retreat_offset: float = 0.15
    settle_time: float = 0.25

    max_joint_step: float = radians(30)

def chain_joint_path(q0: np.ndarray, q1: np.ndarray, max_step: float) -> List[np.ndarray]:
    distance = np.linalg.norm(q1 - q0)
    n = max(2, ceil(distance / max_step) + 1)
    return [q0 + r * (q1 - q0) for r in np.linspace(0, 1, n)[1:]]

@dataclass
class ScanPlan:
    config: ScanConfig
    views: List[tuple]

    camera_poses: List[np.ndarray]

    segments: List[np.ndarray]

    approach_q: np.ndarray
    retreat_q: np.ndarray

    @property
    def view_count(self) -> int:
        return len(self.views)

    @property
    def path(self) -> np.ndarray:
        home_to_approach = chain_joint_path(self.config.home_q, self.approach_q, self.config.max_joint_step)
        retreat_to_home = chain_joint_path(self.retreat_q, self.config.home_q, self.config.max_joint_step)
        return np.concatenate(
            [[self.config.home_q], home_to_approach] + self.segments + [[self.retreat_q], retreat_to_home])

    @property
    def max_joint_step(self) -> float:
        return np.linalg.norm(np.diff(self.path, axis=0), axis=1).max()

    @property
    def reconfiguration(self) -> float:
        return (np.linalg.norm(self.approach_q - self.config.home_q)
                + np.linalg.norm(self.config.home_q - self.retreat_q))

    @property
    def score(self) -> Tuple[float, float]:
        '''Branch ranking key: reach the scan without reconfiguring first,
        and among branches that manage it, sweep as smoothly as possible.'''
        return (self.reconfiguration, self.max_joint_step)

class PlanFailure(RuntimeError):
    pass

def camera_pose(config: ScanConfig, view, standoff: float = 0) -> np.ndarray:
    '''Camera pose in base coordinates for a viewpoint.'''
    return config.face @ config.path.pose(view, standoff)

def flange_pose(config: ScanConfig, view, standoff: float = 0) -> np.ndarray:
    '''Flange pose in base coordinates that puts the camera at a viewpoint.'''
    return camera_pose(config, view, standoff) @ hinv(config.camera)

def solve(kinematics, pose: np.ndarray, reference: np.ndarray, what: str) -> np.ndarray:
    try:
        return kinematics.inverse_nearest(pose, reference)
    except RuntimeError as e:
        raise PlanFailure(f'no inverse kinematics solution for {what} '
                          f'at {np.round(pose[:3, 3], 3)} m: {e}') from e

def solve_standoff(kinematics, config: ScanConfig, view, offset: float,
                   reference: np.ndarray, what: str) -> np.ndarray:
    for scale in [1, 0.75, 0.5, 0.25]:
        try:
            q = kinematics.inverse_nearest(flange_pose(config, view, scale * offset), reference)
        except RuntimeError:
            continue

        # reject standoffs whose nearest IK solution requires too large a joint step,
        # even though reachable, so the plan doesn't jump through a wrist/elbow flip
        if np.linalg.norm(q - reference) > config.max_joint_step:
            continue

        if scale < 1:
            _log.warning('%s shortened to %.0f mm to stay in the workspace', what, 1e3 * scale * offset)
        return q

    raise PlanFailure(f'{what} is unreachable at any standoff up to {1e3 * offset:.0f} mm')

def chain(kinematics, config: ScanConfig, views, seed: np.ndarray):
    q = seed
    segments = []

    for (idx, (previous, view)) in enumerate(zip([views[0]] + views[:-1], views)):
        # walk along the path from the previous view to this one; the first
        # view is simply where the seed already is
        steps = [view] if idx == 0 else config.path.subdivide(previous, view)

        segment = []
        for step in steps:
            q = solve(kinematics, flange_pose(config, step), q, f'view {idx}')
            segment.append(q)

        segments.append(np.asanyarray(segment))

    return segments

def plan_scan(model, config: ScanConfig) -> ScanPlan:
    kinematics = model.kinematics
    views = config.path.views()
    if not views:
        raise PlanFailure('scan has no viewpoints')

    first = flange_pose(config, views[0])
    try:
        seeds = kinematics.inverse(first, config.home_q)
    except RuntimeError as e:
        raise PlanFailure(f'first viewpoint at {np.round(first[:3, 3], 3)} m is unreachable: {e}') from e
    if not seeds:
        raise PlanFailure(f'first viewpoint at {np.round(first[:3, 3], 3)} m is unreachable')

    best: Optional[ScanPlan] = None
    failures = []

    for (branch, seed) in enumerate(seeds):
        try:
            segments = chain(kinematics, config, views, seed)
        except PlanFailure as e:
            failures.append(f'branch {branch}: {e}')
            continue

        plan = ScanPlan(
            config, views,
            camera_poses=[camera_pose(config, view) for view in views],
            segments=segments,
            approach_q=solve_standoff(kinematics, config, views[0], config.approach_offset,
                                      seed, 'approach standoff'),
            retreat_q=solve_standoff(kinematics, config, views[-1], config.retreat_offset,
                                     segments[-1][-1], 'retreat standoff'),
        )

        # validate here rather than once on the winner: a branch that fails the
        # joint-path check should lose to one that passes, not take the whole
        # plan down with it after being picked
        try:
            validate(model, plan)
        except PlanFailure as e:
            failures.append(f'branch {branch}: {e}')
            continue

        if best is None or plan.score < best.score:
            best = plan

    if best is None:
        raise PlanFailure('no usable branch for this scan:\n  ' + '\n  '.join(failures))

    _log.info('chose the nearest-to-home of %d inverse kinematics branches '
              '(%.0f deg to reach the scan, largest step %.1f deg)',
              len(seeds), degrees(best.reconfiguration), degrees(best.max_joint_step))
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
    _log.info('%s path: %s', config.path.name, config.path.summary())
    _log.info('%d joint waypoints', len(plan.path))

    for (idx, (view, pose)) in enumerate(zip(plan.views, plan.camera_poses)):
        _log.debug('view %2d: %s, camera at %s m',
                   idx, config.path.describe(view), np.round(pose[:3, 3], 3))

    (lb, ub) = model.kinematics.limits()
    margin = np.minimum(plan.path - lb, ub - plan.path).min(axis=0)
    _log.info('largest joint step %.1f deg (limit %.1f)',
              degrees(plan.max_joint_step), degrees(config.max_joint_step))
    _log.info('joint limit margin %s deg', np.round(np.degrees(margin), 1))

# called once per viewpoint with the view index and joint angles; a returned Future
# holds the arm at the view until it completes, None advances immediately
Capture = Callable[[int, np.ndarray], Optional[Future]]

class ScanProgram(ProgramBase):
    def __init__(self, plan: ScanPlan, capture: Capture, **kwargs):
        self._plan = plan
        self._capture = capture

        super().__init__(**kwargs)
        self.reset(None)

    def reset(self, ctrl: 'ControllerBase'):
        self._moves = self._build_moves()
        self._index: Optional[int] = None
        self._seg_t0 = None
        self._move_qs = None
        self._move_cum = None
        self._move_total = None
        self._move_duration = None
        self._settle_until = None
        self._pending: Optional[Future] = None

        self.finished = False
        self.state: Optional[RobotState] = None

    def _build_waypoints(self) -> List[dict]:
        config = self._plan.config
        waypoints = [dict(q=config.home_q, label='moving to home', capture=None)]

        home_to_approach = chain_joint_path(config.home_q, self._plan.approach_q, config.max_joint_step)
        for q in home_to_approach[:-1]:
            waypoints.append(dict(q=q, label=None, capture=None))
        waypoints.append(dict(q=home_to_approach[-1], label='approaching the first viewpoint', capture=None))

        for (idx, segment) in enumerate(self._plan.segments):
            view = self._plan.views[idx]
            for q in segment[:-1]:
                waypoints.append(dict(q=q, label=None, capture=None))
            waypoints.append(dict(q=segment[-1], capture=idx, label=(
                f'view {idx + 1:2d}/{self._plan.view_count}: '
                f'{self._plan.config.path.describe(view)}')))

        waypoints.append(dict(q=self._plan.retreat_q, label='retreating from the face', capture=None))

        retreat_to_home = chain_joint_path(self._plan.retreat_q, config.home_q, config.max_joint_step)
        for q in retreat_to_home[:-1]:
            waypoints.append(dict(q=q, label=None, capture=None))
        waypoints.append(dict(q=retreat_to_home[-1], label='returning home', capture=None))
        return waypoints

    def _build_moves(self) -> List[List[dict]]:
        '''Group flat waypoints into runs that stop only at the last (labeled) entry.'''
        moves = []
        current = []
        for wp in self._build_waypoints():
            current.append(wp)
            if wp['label'] is not None:
                moves.append(current)
                current = []
        return moves

    def _enter(self, index: int, state: 'RobotState'):
        self._index = index
        self._seg_t0 = state.timestamp
        self._settle_until = None

        move = self._moves[index]
        self._move_qs = [state.actual_q] + [wp['q'] for wp in move]
        step_dist = np.max(np.abs(np.diff(self._move_qs, axis=0)), axis=1)
        self._move_cum = np.concatenate([[0.0], np.cumsum(step_dist)])
        self._move_total = self._move_cum[-1]
        self._move_duration = max(2 * self._move_total / self._plan.config.joint_speed, 1e-6)

        label = move[-1]['label']
        if label:
            # views are shown on the progress bar, so only list them with --verbose
            (_log.debug if move[-1]['capture'] is not None else _log.info)(label)

    def _advance(self, state: 'RobotState'):
        if self._index + 1 >= len(self._moves):
            self.finished = True
            _log.info('scan finished')
        else:
            self._enter(self._index + 1, state)

    def _interpolate(self, distance: float) -> np.ndarray:
        '''Position along this move's polyline at the given cumulative path distance.'''
        if self._move_total <= 0:
            return self._move_qs[-1]

        idx = min(max(int(np.searchsorted(self._move_cum, distance)), 1), len(self._move_cum) - 1)
        (lo, hi) = (self._move_cum[idx - 1], self._move_cum[idx])
        local_s = 0.0 if hi <= lo else (distance - lo) / (hi - lo)
        return self._move_qs[idx - 1] + local_s * (self._move_qs[idx] - self._move_qs[idx - 1])

    def update(self, ctrl: 'ControllerBase', state: 'RobotState'):
        if self._index is None:
            self._enter(0, state)

        move = self._moves[self._index]
        final = move[-1]

        if self.finished:
            ctrl.servo(q=final['q'])
            goal_q = final['q']

        elif self._settle_until is not None:
            # let the arm settle before recording, so the image is not smeared
            ctrl.servo(q=final['q'])
            goal_q = final['q']

            if self._pending is not None:
                # capture runs off this control thread; keep servoing the view until it is done
                if self._pending.done():
                    self._pending = None
                    self._advance(state)
            elif state.timestamp >= self._settle_until:
                self._pending = self._capture(final['capture'], state.actual_q)
                if self._pending is None:
                    self._advance(state)
        else:
            t = state.timestamp - self._seg_t0
            s = min(t / self._move_duration, 1.0)
            ease = s - sin(2 * pi * s) / (2 * pi)
            goal_q = self._interpolate(ease * self._move_total)

            ctrl.servo(q=goal_q)

            if s >= 1.0:
                if final['capture'] is not None:
                    self._settle_until = state.timestamp + self._plan.config.settle_time
                else:
                    self._advance(state)

        self.state = state
        self.state.goal_q = goal_q

def preview(model, ctrl: 'ControllerBase', plan: ScanPlan, capture: Capture) -> None:
    from igmr_robotics_toolkit.viewer.core import create_simple_viewer
    from igmr_robotics_toolkit.viewer.widget import ControlledRobotWidget, TransformListWidget, LineWidget

    (window, root) = create_simple_viewer(title='IRTk - Face Scan Preview')

    model.pose(plan.config.home_q)
    ControlledRobotWidget(model=model, controller=ctrl, parent=root, frames=[0, model.dof])

    # the face frame plus every viewpoint, and the arc joining them
    TransformListWidget(parent=root).load([plan.config.face] + list(plan.camera_poses))
    LineWidget(parent=root).load([pose[:3, 3] for pose in plan.camera_poses])

    def run_motion():
        try:
            execute(model, ctrl, plan, capture)
        finally:
            window.userExit()

    Thread(target=run_motion, daemon=True).start()
    window.run()


def create_controller(model, config: ScanConfig, args):
    if args.simulate:
        from igmr_robotics_toolkit.control.simulator import Simulator
        return Simulator(model, q=config.home_q, control_rate=CONTROL_RATE)
    else:
        return model.Controller(args.robot, model)

def execute(model, ctrl: 'ControllerBase', plan: ScanPlan, capture: Capture) -> None:
    prog = ScanProgram(plan, capture)

    thread = Thread(target=lambda: ctrl.run(prog), daemon=True)
    thread.start()

    while not prog.finished:
        sleep(0.01)

    ctrl.stop()
    thread.join()

class DryRunController:
    '''Stands in for a controller when checking a plan: tracks every servo
    command exactly and advances a simulated clock without sleeping.'''

    def __init__(self, q: np.ndarray):
        self.q = np.array(q, dtype=float)

    def servo(self, q: np.ndarray) -> None:
        self.q = np.array(q, dtype=float)

def dry_run(model, plan: ScanPlan, position_tolerance: float = 1e-3,
            angle_tolerance: float = radians(0.1)) -> None:
    '''Step the scan program to completion on a simulated clock and check that
    every viewpoint is captured with the camera where the plan put it.'''
    config = plan.config
    captured = {}

    def capture(idx, q):
        captured[idx] = np.array(q)

    prog = ScanProgram(plan, capture)
    ctrl = DryRunController(config.home_q)
    dt = 1 / CONTROL_RATE

    # every move is eased over twice its joint distance at joint_speed, so
    # twice the whole path plus settling is a generous bound on the scan time
    distance = np.linalg.norm(np.diff(plan.path, axis=0), axis=1).sum()
    budget = 2 * (2 * distance / config.joint_speed + plan.view_count * config.settle_time) + 10

    tick = 0
    while not prog.finished:
        if tick * dt > budget:
            raise PlanFailure(f'dry run did not finish within {budget:.0f} s of simulated time')
        prog.update(ctrl, SimpleNamespace(timestamp=tick * dt, actual_q=ctrl.q))
        tick += 1

    missing = sorted(set(range(plan.view_count)) - set(captured))
    if missing:
        raise PlanFailure(f'dry run never captured views {missing}')

    position_error = []
    angle_error = []
    for (idx, q) in sorted(captured.items()):
        actual = model.kinematics.forward(q) @ config.camera
        planned = plan.camera_poses[idx]
        position_error.append(norm(actual[:3, 3] - planned[:3, 3]))
        cos_angle = (np.trace(planned[:3, :3].T @ actual[:3, :3]) - 1) / 2
        angle_error.append(np.arccos(np.clip(cos_angle, -1, 1)))

    worst = int(np.argmax(position_error))
    if max(position_error) > position_tolerance or max(angle_error) > angle_tolerance:
        raise PlanFailure(f'dry run missed viewpoints: worst {1e3 * max(position_error):.3f} mm '
                          f'(view {worst}), {degrees(max(angle_error)):.3f} deg')

    if norm(ctrl.q - config.home_q) > 1e-6:
        raise PlanFailure(f'dry run ended at {np.round(np.degrees(ctrl.q), 1)} deg instead of home')

    _log.info('dry run passed: %d/%d views reached, worst error %.3f mm / %.3f deg, '
              'scan takes %.0f s at %.0f deg/s',
              len(captured), plan.view_count, 1e3 * max(position_error), degrees(max(angle_error)),
              tick * dt, degrees(config.joint_speed))

def make_recorder(model, plan: ScanPlan, rig: Optional['CameraRig'] = None,
                  output: Optional[Path] = None, pool: Optional[ThreadPoolExecutor] = None,
                  progress: Optional[tqdm] = None) -> Capture:
    '''progress, if given, advances one step per view once that view is recorded.'''

    def view_started(idx):
        if progress is not None:
            progress.set_postfix_str(f'view {idx + 1}: {plan.config.path.describe(plan.views[idx])}')

    def view_done():
        if progress is not None:
            progress.update()

    if rig is None:
        def log_only(idx, q):
            view_started(idx)
            _log.debug('   at %s m (not capturing)',
                       np.round(model.kinematics.forward(q)[:3, 3], 4))
            view_done()
        return log_only

    _log.info('capturing to %s', output)

    def capture_view(idx, entry):
        cameras = tqdm(total=len(rig.cameras), desc='  cameras', unit='image', leave=False, position=1,
                       disable=progress is None)
        # a fall back re-captures every camera, so count each camera once
        done = set()

        def captured(name):
            if name not in done:
                done.add(name)
                cameras.update()
            cameras.set_postfix_str(name)

        try:
            entry['cameras'] = rig.capture_all(output, idx, on_captured=captured)
        except Exception as e:
            # log and move on: a failed view must not leave the arm parked at the face
            _log.error('view %d: capture failed: %s', idx, e)
            entry['capture_error'] = str(e)
        finally:
            cameras.close()
        entry['one_camera_at_a_time'] = rig.sequential

        with open(output / f'view_{idx:03d}.yaml', 'w') as f:
            safe_dump(denumpy(entry), f)
        view_done()

    def record(idx, q):
        view_started(idx)
        flange = model.kinematics.forward(q)

        entry = dict(
            view=idx,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **plan.config.path.record(plan.views[idx]),
            actual_q=q,
            flange_pose=flange,
            camera_pose=flange @ plan.config.camera,
            planned_camera_pose=plan.camera_poses[idx],
        )

        # the cameras are slow next to the control loop, so capture on the pool while the arm holds the view
        return pool.submit(capture_view, idx, entry)

    return record

def write_manifest(output: Path, plan: ScanPlan, rig: 'CameraRig') -> None:
    output.mkdir(parents=True)

    # snapshot the calibration, so the scan stays usable if camera_matrix.yaml is recalibrated later
    shutil.copy(rig.config_file, output / 'camera_matrix.yaml')

    with open(output / 'scan.yaml', 'w') as f:
        safe_dump(denumpy(dict(
            experiment=output.name,
            started=datetime.now(timezone.utc).isoformat(),
            model=plan.config.model,
            path=plan.config.path.name,
            path_summary=plan.config.path.summary(),
            path_parameters=denumpy(vars(plan.config.path)),
            face_pose=plan.config.face,
            camera_in_flange=plan.config.camera,
            cameras={camera.name: dict(type=camera.type, serial=camera.serial) for camera in rig.cameras},
            images=('unrotated; <cam>_color.png is aligned to depth (the calibrated IR frame, '
                    'intrinsics in camera_matrix.yaml); <cam>_color_raw.png is unaligned; '
                    '<cam>_depth.npy is float32 mm with 0 for no depth'),
            views=[dict(view=i, **plan.config.path.record(v))
                   for (i, v) in enumerate(plan.views)],
        )), f)

def prompt_output(root: Path = OUTPUT_ROOT) -> Path:
    '''Ask for the experiment id; its captures go to root/<MM-DD-YYYY>/<id>.'''
    day = root / datetime.now().strftime(DATE_FORMAT)

    while True:
        try:
            experiment = input('experiment id: ').strip()
        except EOFError:
            raise SystemExit('no experiment id given')

        if not experiment or experiment in ('.', '..') or any(c in experiment for c in '/\\'):
            print('experiment id must be a non-empty name without slashes')
            continue

        output = day / experiment
        if output.exists():
            print(f'{output} already exists, pick another experiment id')
            continue

        return output

def run(args):
    capture = args.capture
    if capture and not args.robot:
        _log.warning('--capture only applies on the robot; %s continues without capturing',
                     'the simulation' if args.simulate else 'the plan-only dry run')
        capture = False

    output = prompt_output() if capture else None

    config = ScanConfig(path=PATHS[args.path]())

    # only the arm has to move at the real joint_speed; planning is purely
    # geometric and does not read it, so scaling here affects timing alone
    if args.simulate:
        config.joint_speed *= SIM_SPEED_SCALE
        _log.info('simulating at %.0fx speed (%.0f deg/s)',
                  SIM_SPEED_SCALE, degrees(config.joint_speed))

    model = load_robot(config.model)

    plan = plan_scan(model, config)
    report(model, plan)

    if args.plan_only:
        dry_run(model, plan)
        return

    rig = None
    if capture:
        # only needed with --capture, so the robot-only modes run without pyrealsense2
        from camera import CameraRig

        rig = CameraRig()
        # start the cameras before connecting, so a missing camera aborts before the arm moves
        rig.start()

    try:
        controller = create_controller(model, config, args)
        if rig is not None:
            write_manifest(output, plan, rig)

        mark = monotonic()
        # print log lines above the progress bars instead of through them
        loggers = [logging.getLogger(name) for name in ('face_scan', 'igmr_robotics_toolkit', 'camera', 'realsense')]
        with logging_redirect_tqdm(loggers=loggers), \
                tqdm(total=plan.view_count, desc=f'{config.path.name} scan', unit='view', mininterval=0.2) as progress, \
                ThreadPoolExecutor(max_workers=1, thread_name_prefix='capture') as pool:
            recorder = make_recorder(model, plan, rig, output, pool, progress)
            if args.simulate:
                preview(model, controller, plan, recorder)
            else:
                execute(model, controller, plan, recorder)
        _log.info('scan finished in %.0f s', monotonic() - mark)

    finally:
        if rig is not None:
            rig.stop()

def main():
    parser = ArgumentParser(description='Move a KUKA LBR arm around a phantom head to capture multi-view scan data for MVFace.',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--simulate', '-s', action='store_true',
                       help='run the scan on the built-in simulator in the 3D viewer')
    group.add_argument('--robot', '-r', type=str, help='hostname or address of the robot')
    group.add_argument('--plan-only', action='store_true',
                       help='plan the scan and dry-run it headless to check every viewpoint is reached, '
                            'without connecting or opening the viewer')

    parser.add_argument('--path', '-p', choices=sorted(PATHS), default='sphere',
                        help='scan path: "sphere" orbits the face at a constant standoff with every '
                             'view aimed at it; "plane" is a rectified grid in a vertical plane with '
                             'every view sharing one orientation')
    parser.add_argument('--capture', action='store_true',
                        help='capture every camera at each viewpoint (robot only; ignored with a warning '
                             'under --simulate and --plan-only), asking for an experiment id and saving to '
                             f'{OUTPUT_ROOT}/<MM-DD-YYYY>/<experiment id>')
    parser.add_argument('--verbose', '-v', action='store_true', help='log every viewpoint')

    args = parser.parse_args()
    if args.verbose:
        _log.setLevel(logging.DEBUG)
        coloredlogs.install(level=logging.DEBUG, logger=_log)

    try:
        run(args)
    except PlanFailure as e:
        _log.error('%s', e)
        raise SystemExit(1)

if __name__ == '__main__':
    main()
