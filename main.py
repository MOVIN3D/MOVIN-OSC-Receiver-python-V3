import argparse
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
import pythonosc.parsing.osc_types as _osc_types
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_ARRAY,
    GL_COLOR_BUFFER_BIT,
    GL_CURRENT_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_ENABLE_BIT,
    GL_FLOAT,
    GL_LINE_SMOOTH,
    GL_LINES,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINT_SMOOTH,
    GL_POINTS,
    GL_PROJECTION,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_ARRAY,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor3f,
    glColorPointer,
    glDisable,
    glDisableClientState,
    glDrawArrays,
    glDrawPixels,
    glEnable,
    glEnableClientState,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glPointSize,
    glPopAttrib,
    glPushAttrib,
    glVertexPointer,
    glWindowPos2d,
)
from OpenGL.GLU import gluLookAt, gluPerspective
from pygame.locals import DOUBLEBUF, K_DOWN, K_ESCAPE, K_r, K_UP, KEYDOWN, MOUSEBUTTONDOWN
from pygame.locals import MOUSEBUTTONUP, MOUSEMOTION, OPENGL, QUIT
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_server import BlockingOSCUDPServer


_orig_get_string = _osc_types.get_string


def _get_string_safe(dgram: bytes, start_index: int) -> Tuple[str, int]:
    try:
        return _orig_get_string(dgram, start_index)
    except UnicodeDecodeError:
        offset = 0
        while dgram[start_index + offset] != 0:
            offset += 1
        data_str = dgram[start_index : start_index + offset]
        decoded = ""
        for encoding in ("cp949", "euc-kr", "latin-1"):
            try:
                decoded = data_str.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        offset += 1
        if offset % 4 != 0:
            offset += 4 - (offset % 4)
        return decoded, start_index + offset


_osc_types.get_string = _get_string_safe


BONE_FIELD_COUNT = 17
POINT_FIELD_COUNT = 3
MAX_BONES = 4096
MAX_POINTS = 100_000
MAX_CHUNKS = 4096
MAX_ACTORS = 8
MAX_PENDING_FRAMES = 3
ASSEMBLY_TIMEOUT = 0.5
RESTART_WAIT = 1.0
logger = logging.getLogger("movin.osc")
ACTOR_COLORS = [
    (0.8, 0.8, 0.8),
    (1.0, 0.5, 0.3),
    (0.3, 0.8, 1.0),
    (1.0, 0.3, 0.8),
    (0.5, 1.0, 0.3),
    (0.8, 0.5, 1.0),
    (1.0, 1.0, 0.3),
    (0.3, 1.0, 0.8),
]


@dataclass
class BoneRecord:
    bone_index: int
    parent_index: int
    bone_name: str
    local_position: np.ndarray
    rest_rotation: np.ndarray
    local_rotation: np.ndarray
    local_scale: np.ndarray
    world_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    world_rotation: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )


@dataclass
class SkeletonFrameAssembly:
    timestamp: str
    actor_name: str
    frame_idx: int
    num_chunks: int
    total_bone_count: int
    created_at: float
    chunks: Dict[int, List[BoneRecord]] = field(default_factory=dict)
    received: int = 0

    def add_chunk(self, chunk_index: int, bones: List[BoneRecord]) -> None:
        self.chunks[chunk_index] = bones
        self.received += len(bones)

    def is_complete(self) -> bool:
        return len(self.chunks) == self.num_chunks

    def to_bones(self) -> List[BoneRecord]:
        bones = [bone for chunk in self.chunks.values() for bone in chunk]
        if len(bones) != self.total_bone_count:
            raise ValueError("Motion frame bone count does not match its header")
        return bones


@dataclass
class PointCloudAssembly:
    frame_idx: int
    total_points: int
    num_chunks: int
    created_at: float
    chunks: Dict[int, np.ndarray] = field(default_factory=dict)
    received: int = 0

    def add_chunk(self, chunk_idx: int, points: np.ndarray) -> None:
        self.chunks[chunk_idx] = points
        self.received += len(points)

    def is_complete(self) -> bool:
        return len(self.chunks) == self.num_chunks

    def to_points(self) -> np.ndarray:
        points = np.concatenate([self.chunks[idx] for idx in range(self.num_chunks)])
        if len(points) != self.total_points:
            raise ValueError("Point cloud count does not match its header")
        points.setflags(write=False)
        return points


@dataclass
class FrameStream:
    latest: int = -1
    progressed_at: float = 0.0
    heard_at: float = 0.0
    frames: Dict[int, SkeletonFrameAssembly | PointCloudAssembly] = field(default_factory=dict)

    def accept(self, frame: int, now: float) -> bool:
        self.frames = {f: a for f, a in self.frames.items() if now - a.created_at < ASSEMBLY_TIMEOUT}
        # Older/duplicate packets never refresh the restart timer.
        if frame < self.latest and now - self.progressed_at >= RESTART_WAIT:
            self.frames.clear()
            self.latest = -1
        accepted = frame > self.latest
        if accepted:
            self.heard_at = now
            if frame not in self.frames and len(self.frames) >= MAX_PENDING_FRAMES:
                oldest = min(self.frames)
                if frame > oldest:
                    del self.frames[oldest]
                else:
                    accepted = False
        return accepted

    def publish(self, frame: int, now: float) -> None:
        self.latest = frame
        self.progressed_at = now
        self.frames = {f: a for f, a in self.frames.items() if f > frame}


class FrameRate:
    def __init__(self) -> None:
        self.times = deque(maxlen=4096)

    def add(self, now: float) -> None:
        if self.times and now - self.times[-1] >= 1:
            self.times.clear()
        self.times.append(now)

    def read(self, now: float) -> Tuple[float, float]:
        while len(self.times) > 1 and self.times[0] < now - 1:
            self.times.popleft()
        elapsed = now - self.times[0] if self.times else 0
        fps = (len(self.times) - 1) / elapsed if len(self.times) > 1 and elapsed > 0 else 0.0
        age = now - self.times[-1] if self.times else -1.0
        return fps, age


class SharedState:
    def __init__(self, timeout: float) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be a positive finite number")
        self.lock = threading.Lock()
        self.timeout = timeout
        self.source: Optional[Tuple[str, int]] = None
        self.source_heard_at = 0.0
        self.motion_streams: Dict[str, FrameStream] = {}
        self.point_stream = FrameStream()
        self.latest_skeletons: Dict[str, List[BoneRecord]] = {}
        self.latest_points: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.last_update: Dict[str, float] = {}
        self.last_points_at = 0.0
        self.last_error_at = float("-inf")
        self.motion_rates: Dict[str, FrameRate] = {}
        self.cloud_rate = FrameRate()
        self.viewer_rate = FrameRate()

    def accept_source(self, source: Tuple[str, int], now: float) -> bool:
        accepted = source == self.source or self.source is None or now - self.source_heard_at >= RESTART_WAIT
        if accepted:
            if source != self.source:
                self.source = source
                self.motion_streams.clear()
                self.point_stream = FrameStream()
                with self.lock:
                    self.latest_skeletons.clear()
                    self.last_update.clear()
                    self.latest_points = np.empty((0, 3), dtype=np.float32)
                    self.last_points_at = 0.0
                    self.motion_rates.clear()
                    self.cloud_rate = FrameRate()
            self.source_heard_at = now
        return accepted

    def report_invalid(self, error: Exception) -> None:
        now = time.monotonic()
        if now - self.last_error_at >= 1:
            logger.warning("Discarded invalid OSC packet: %s", error)
            self.last_error_at = now

    def expire(self, now: float) -> None:
        self.motion_streams = {name: stream for name, stream in self.motion_streams.items()
                               if now - stream.heard_at < max(self.timeout, RESTART_WAIT)}
        for stream in [*self.motion_streams.values(), self.point_stream]:
            stream.frames = {f: a for f, a in stream.frames.items() if now - a.created_at < ASSEMBLY_TIMEOUT}
        with self.lock:
            for name in [name for name, at in self.last_update.items() if now - at >= self.timeout]:
                self.latest_skeletons.pop(name, None)
                del self.last_update[name]
                self.motion_rates.pop(name, None)
            if self.latest_points.size and now - self.last_points_at >= self.timeout:
                self.latest_points = np.empty((0, 3), dtype=np.float32)

    def add_motion_chunk(
        self,
        source: Tuple[str, int],
        timestamp: str,
        actor_name: str,
        frame_idx: int,
        num_chunks: int,
        chunk_index: int,
        total_bone_count: int,
        bones: List[BoneRecord],
    ) -> None:
        now = time.monotonic()
        if self.accept_source(source, now):
            self.expire(now)
            if actor_name not in self.motion_streams:
                if len(self.motion_streams) >= MAX_ACTORS:
                    raise ValueError("Too many active actors")
                self.motion_streams[actor_name] = FrameStream()
            stream = self.motion_streams[actor_name]
            if not stream.accept(frame_idx, now):
                return
            assembly = stream.frames.get(frame_idx)
            if assembly is None:
                assembly = SkeletonFrameAssembly(
                    timestamp=timestamp,
                    actor_name=actor_name,
                    frame_idx=frame_idx,
                    num_chunks=num_chunks,
                    total_bone_count=total_bone_count,
                    created_at=now,
                )
                stream.frames[frame_idx] = assembly
            try:
                if (assembly.timestamp, assembly.num_chunks, assembly.total_bone_count) != (timestamp, num_chunks, total_bone_count):
                    raise ValueError("Motion chunks disagree about their frame header")
                if chunk_index not in assembly.chunks:
                    if assembly.received + len(bones) > total_bone_count:
                        raise ValueError("Motion chunks exceed the declared bone count")
                    assembly.add_chunk(chunk_index, bones)
                    if assembly.is_complete():
                        skeleton = compute_world_pose(assembly.to_bones())
                        stream.publish(frame_idx, now)
                        with self.lock:
                            self.latest_skeletons[actor_name] = skeleton
                            self.last_update[actor_name] = now
                            if actor_name not in self.motion_rates:
                                self.motion_rates[actor_name] = FrameRate()
                            self.motion_rates[actor_name].add(now)
            except ValueError:
                stream.frames.pop(frame_idx, None)
                raise

    def add_point_chunk(
        self,
        source: Tuple[str, int],
        frame_idx: int,
        total_points: int,
        chunk_idx: int,
        num_chunks: int,
        points: np.ndarray,
    ) -> None:
        now = time.monotonic()
        if self.accept_source(source, now) and self.point_stream.accept(frame_idx, now):
            stream = self.point_stream
            assembly = stream.frames.get(frame_idx)
            if assembly is None:
                assembly = PointCloudAssembly(
                    frame_idx=frame_idx,
                    total_points=total_points,
                    num_chunks=num_chunks,
                    created_at=now,
                )
                stream.frames[frame_idx] = assembly
            try:
                if (assembly.num_chunks, assembly.total_points) != (num_chunks, total_points):
                    raise ValueError("Point cloud chunks disagree about their frame header")
                if chunk_idx not in assembly.chunks:
                    if assembly.received + len(points) > total_points:
                        raise ValueError("Point cloud chunks exceed the declared point count")
                    assembly.add_chunk(chunk_idx, points)
                    if assembly.is_complete():
                        cloud = assembly.to_points()
                        stream.publish(frame_idx, now)
                        with self.lock:
                            self.latest_points = cloud
                            self.last_points_at = now
                            self.cloud_rate.add(now)
            except ValueError:
                stream.frames.pop(frame_idx, None)
                raise

    def snapshot(self) -> Tuple[Dict[str, List[BoneRecord]], np.ndarray]:
        now = time.monotonic()
        with self.lock:
            stale_actors = [actor for actor, updated_at in self.last_update.items() if now - updated_at > self.timeout]
            for actor in stale_actors:
                self.latest_skeletons.pop(actor, None)
                self.last_update.pop(actor, None)
                self.motion_rates.pop(actor, None)
            if self.latest_points.size and now - self.last_points_at >= self.timeout:
                self.latest_points = np.empty((0, 3), dtype=np.float32)
            # Published frames are not mutated; rendering only needs their references.
            skeletons = self.latest_skeletons.copy()
            points = self.latest_points
            return skeletons, points


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat.astype(np.float64))
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = lhs
    x2, y2, z2, w2 = rhs
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def quaternion_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion(quat)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def unity_to_opengl_pos(position: np.ndarray) -> np.ndarray:
    return np.array([-position[0], position[1], position[2]], dtype=np.float32)


def unity_to_opengl_rot(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float32)


def trs_matrix(position: np.ndarray, rotation: np.ndarray, scale: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = quaternion_matrix(rotation) @ np.diag(scale.astype(np.float32))
    matrix[:3, 3] = position.astype(np.float32)
    return matrix


def compute_world_pose(bones: List[BoneRecord]) -> List[BoneRecord]:
    ordered = sorted(bones, key=lambda bone: bone.bone_index)
    bone_map = {bone.bone_index: bone for bone in ordered}
    if len(bone_map) != len(bones) or len({bone.bone_name for bone in bones}) != len(bones):
        raise ValueError("Motion frame contains duplicate bone indices or names")
    children = {bone.bone_index: [] for bone in ordered}
    pending = deque()
    for bone in ordered:
        if bone.parent_index == -1:
            pending.append(bone.bone_index)
        elif bone.parent_index not in bone_map:
            raise ValueError("Motion frame is missing a parent bone")
        else:
            children[bone.parent_index].append(bone.bone_index)
    unity_matrices: Dict[int, np.ndarray] = {}
    unity_rotations: Dict[int, np.ndarray] = {}
    while pending:
        bone_index = pending.popleft()
        bone = bone_map[bone_index]
        local_rotation = normalize_quaternion(bone.local_rotation)
        local_matrix = trs_matrix(bone.local_position, local_rotation, bone.local_scale)
        if bone.parent_index == -1:
            unity_matrices[bone_index] = local_matrix
            unity_rotations[bone_index] = local_rotation
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                unity_matrices[bone_index] = unity_matrices[bone.parent_index] @ local_matrix
            unity_rotations[bone_index] = normalize_quaternion(
                quaternion_multiply(unity_rotations[bone.parent_index], local_rotation)
            )
        if not np.isfinite(unity_matrices[bone_index]).all():
            raise ValueError("Motion hierarchy produces a non-finite world transform")
        bone.world_position = unity_to_opengl_pos(unity_matrices[bone_index][:3, 3])
        bone.world_rotation = unity_to_opengl_rot(unity_rotations[bone_index])
        pending.extend(children[bone_index])
    if len(unity_matrices) != len(bones):
        raise ValueError("Motion hierarchy contains a cycle")
    return ordered


def osc_int(value: object, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Expected an OSC integer between {minimum} and {maximum}")
    return value


def osc_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256 or '\0' in value:
        raise ValueError("Expected a non-empty name of at most 256 characters")
    return value


def parse_motion(address: str, *osc_args: object, state: SharedState, source: Tuple[str, int]) -> None:
    del address
    if len(osc_args) < 7:
        raise ValueError("Incomplete motion header")

    timestamp = osc_name(osc_args[0])
    actor_name = osc_name(osc_args[1])
    frame_idx = osc_int(osc_args[2], 0, 2**31 - 1)
    total_bone_count = osc_int(osc_args[5], 1, MAX_BONES)
    num_chunks = osc_int(osc_args[3], 1, min(total_bone_count, MAX_CHUNKS))
    chunk_index = osc_int(osc_args[4], 0, num_chunks - 1)
    chunk_bone_count = osc_int(osc_args[6], 1, total_bone_count)
    payload = osc_args[7:]
    expected_len = chunk_bone_count * BONE_FIELD_COUNT
    if len(payload) != expected_len:
        raise ValueError("Motion payload length does not match its header")

    bones: List[BoneRecord] = []
    for i in range(chunk_bone_count):
        base = i * BONE_FIELD_COUNT
        values = payload[base + 3 : base + BONE_FIELD_COUNT]
        if not all(type(v) is float for v in values):
            raise ValueError("Bone transforms must contain OSC floats")
        with np.errstate(over="ignore"):
            values = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Bone transforms must be finite")
        for rotation in (values[3:7], values[7:11]):
            norm = np.linalg.norm(rotation.astype(np.float64))
            if not math.isfinite(norm) or norm < 1e-6:
                raise ValueError("Bone rotations must be non-zero finite quaternions")
        index = osc_int(payload[base], 0, MAX_BONES - 1)
        parent = osc_int(payload[base + 1], -1, MAX_BONES - 1)
        if index == parent:
            raise ValueError("A bone cannot be its own parent")
        bones.append(
            BoneRecord(
                bone_index=index,
                parent_index=parent,
                bone_name=osc_name(payload[base + 2]),
                local_position=values[:3],
                rest_rotation=values[3:7],
                local_rotation=values[7:11],
                local_scale=values[11:14],
            )
        )

    state.add_motion_chunk(
        source=source,
        timestamp=timestamp,
        actor_name=actor_name,
        frame_idx=frame_idx,
        num_chunks=num_chunks,
        chunk_index=chunk_index,
        total_bone_count=total_bone_count,
        bones=bones,
    )


def parse_point_cloud(address: str, *osc_args: object, state: SharedState, source: Tuple[str, int]) -> None:
    del address
    if len(osc_args) < 5:
        raise ValueError("Incomplete point cloud header")

    frame_idx = osc_int(osc_args[0], 0, 2**31 - 1)
    total_points = osc_int(osc_args[1], 0, MAX_POINTS)
    num_chunks = osc_int(osc_args[3], 1, min(max(1, total_points), MAX_CHUNKS))
    chunk_idx = osc_int(osc_args[2], 0, num_chunks - 1)
    chunk_point_count = osc_int(osc_args[4], 1 if total_points else 0, total_points)
    payload = osc_args[5:]
    expected_len = chunk_point_count * POINT_FIELD_COUNT
    if len(payload) != expected_len:
        raise ValueError("Point cloud payload length does not match its header")
    if not all(type(v) is float for v in payload):
        raise ValueError("Point cloud coordinates must contain OSC floats")

    with np.errstate(over="ignore"):
        points = np.asarray(payload, dtype=np.float32).reshape((-1, 3))
    if not np.isfinite(points).all():
        raise ValueError("Point cloud coordinates must be finite")
    points[:, 0] *= -1.0
    state.add_point_chunk(
        source=source,
        frame_idx=frame_idx,
        total_points=total_points,
        chunk_idx=chunk_idx,
        num_chunks=num_chunks,
        points=points,
    )


def create_dispatcher(state: SharedState) -> Dispatcher:
    dispatcher = Dispatcher()
    def receive(source, address, *args):
        try:
            if address == "/MOVIN/Frame":
                parse_motion(address, *args, state=state, source=source)
            else:
                parse_point_cloud(address, *args, state=state, source=source)
        except (ValueError, TypeError, OverflowError) as error:
            state.report_invalid(error)
    dispatcher.map("/MOVIN/Frame", receive, needs_reply_address=True)
    dispatcher.map("/MOVIN/PointCloud", receive, needs_reply_address=True)
    return dispatcher


class ReceiverServer(BlockingOSCUDPServer):
    max_packet_size = 65535

    def __init__(self, address: Tuple[str, int], state: SharedState) -> None:
        self.state = state
        super().__init__(address, create_dispatcher(state))
        self.dispatcher.map("/MOVIN/OSC/Status/Request", self.reply_status, needs_reply_address=True)

    def reply_status(self, source, address, *args) -> None:
        try:
            if len(args) != 3:
                raise ValueError("Invalid OSC status request argument count")
            token, port, actor = args
            if (not isinstance(token, str) or len(token) != 32
                    or any(c not in '0123456789abcdefABCDEF' for c in token)
                    or type(port) is not int or not 1 <= port <= 65535
                    or not isinstance(actor, str) or len(actor) > 256 or '\0' in actor):
                raise ValueError("Invalid OSC status request")
            with self.state.lock:
                now = time.monotonic()
                same_source = source == self.state.source
                rate = self.state.motion_rates.get(actor) if same_source else None
                motion, motion_age = rate.read(now) if rate else (0.0, -1.0)
                cloud, cloud_age = self.state.cloud_rate.read(now) if same_source else (0.0, -1.0)
                viewer, viewer_age = self.state.viewer_rate.read(now)
            message = OscMessageBuilder(address="/MOVIN/OSC/Status")
            values = (token, 1, actor, motion, cloud, viewer, motion_age, cloud_age, viewer_age, int(same_source))
            for value, kind in zip(values, "sisffffffi"):
                message.add_arg(value, kind)
            self.socket.sendto(message.build().dgram, (source[0], port))
        except (ValueError, TypeError, OverflowError, OSError) as error:
            self.state.report_invalid(error)

    def service_actions(self) -> None:
        self.state.expire(time.monotonic())


class ViewerApp:
    def __init__(self, state: SharedState, fps: float, point_size: float, axis_size: float) -> None:
        self.state = state
        self.tick_interval = max(1.0 / fps, 0.001)
        self.axis_size = axis_size
        self.point_size = point_size
        self.draw_joint_axes = True
        self.cam_dist = 4.0
        self.cam_rx = 15.0
        self.cam_ry = 0.0
        self.cam_target = np.array([0.0, 0.9, 0.0], dtype=np.float32)
        self.dragging = False
        self.last_mouse = (0, 0)
        self.clock: Optional[pygame.time.Clock] = None
        self._grid_verts: Optional[np.ndarray] = None
        self.info_at = float("-inf")
        self.info_lines = ()
        self.info_pixels = b""
        self.info_size = (0, 0)

    def _build_grid(self) -> None:
        verts = []
        for i in range(26):
            x = -2.5 + i * 0.2
            verts.extend([x, 0, -2.5, x, 0, 2.5, -2.5, 0, x, 2.5, 0, x])
        self._grid_verts = np.array(verts, dtype=np.float32)

    def init(self) -> None:
        pygame.init()
        pygame.display.set_mode((1280, 720), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("MOVIN OSC Viewer")
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.12, 0.12, 0.18, 1.0)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, 1280 / 720, 0.1, 50)
        glEnableClientState(GL_VERTEX_ARRAY)
        self._build_grid()
        self.clock = pygame.time.Clock()
        self.info_font = pygame.font.SysFont("malgungothic,notosans,sans", 18)
        print("Controls: Drag=rotate, Scroll=zoom, R=reset, Arrows=move, ESC=exit")

    def draw_grid(self) -> None:
        if self._grid_verts is None:
            return
        glColor3f(0.3, 0.3, 0.35)
        glVertexPointer(3, GL_FLOAT, 0, self._grid_verts)
        glDrawArrays(GL_LINES, 0, len(self._grid_verts) // 3)

    def draw_world_axes(self) -> None:
        axes = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5],
            ],
            dtype=np.float32,
        )
        colors = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        glEnableClientState(GL_COLOR_ARRAY)
        glLineWidth(3)
        glVertexPointer(3, GL_FLOAT, 0, axes)
        glColorPointer(3, GL_FLOAT, 0, colors)
        glDrawArrays(GL_LINES, 0, len(axes))
        glDisableClientState(GL_COLOR_ARRAY)
        glLineWidth(1)

    def draw_skeleton(self, joints: List[BoneRecord], color_idx: int) -> None:
        if not joints:
            return

        color = ACTOR_COLORS[color_idx % len(ACTOR_COLORS)]
        joint_map = {joint.bone_index: joint for joint in joints}

        bone_verts: List[np.ndarray] = []
        for joint in joints:
            if joint.parent_index >= 0 and joint.parent_index in joint_map:
                bone_verts.append(joint_map[joint.parent_index].world_position)
                bone_verts.append(joint.world_position)
        if bone_verts:
            arr = np.asarray(bone_verts, dtype=np.float32)
            glLineWidth(3)
            glColor3f(*color)
            glVertexPointer(3, GL_FLOAT, 0, arr)
            glDrawArrays(GL_LINES, 0, len(arr))

        point_verts = np.asarray([joint.world_position for joint in joints], dtype=np.float32)
        glPointSize(8)
        glColor3f(1.0, 0.9, 0.2)
        glVertexPointer(3, GL_FLOAT, 0, point_verts)
        glDrawArrays(GL_POINTS, 0, len(point_verts))

        if not self.draw_joint_axes:
            glLineWidth(1)
            return

        axis_verts: List[np.ndarray] = []
        axis_colors: List[List[float]] = []
        for joint in joints:
            rot = quaternion_matrix(joint.world_rotation)
            origin = joint.world_position
            basis = (
                (np.array([self.axis_size, 0.0, 0.0], dtype=np.float32), [1.0, 0.2, 0.2]),
                (np.array([0.0, self.axis_size, 0.0], dtype=np.float32), [0.2, 1.0, 0.2]),
                (np.array([0.0, 0.0, self.axis_size], dtype=np.float32), [0.2, 0.2, 1.0]),
            )
            for axis, axis_color in basis:
                axis_verts.append(origin)
                axis_verts.append(origin + rot @ axis)
                axis_colors.extend([axis_color, axis_color])

        if axis_verts:
            arr = np.asarray(axis_verts, dtype=np.float32)
            carr = np.asarray(axis_colors, dtype=np.float32)
            glEnableClientState(GL_COLOR_ARRAY)
            glLineWidth(2)
            glVertexPointer(3, GL_FLOAT, 0, arr)
            glColorPointer(3, GL_FLOAT, 0, carr)
            glDrawArrays(GL_LINES, 0, len(arr))
            glDisableClientState(GL_COLOR_ARRAY)

        glLineWidth(1)

    def draw_point_cloud(self, points: np.ndarray) -> None:
        if points is None or len(points) == 0:
            return
        pts = np.ascontiguousarray(points, dtype=np.float32)
        glPointSize(self.point_size)
        glColor3f(0.2, 0.8, 1.0)
        glVertexPointer(3, GL_FLOAT, 0, pts)
        glDrawArrays(GL_POINTS, 0, len(pts))

    def setup_camera(self) -> None:
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        rx = math.radians(self.cam_rx)
        ry = math.radians(self.cam_ry)
        camera_pos = self.cam_target + self.cam_dist * np.array(
            [
                math.cos(rx) * math.sin(ry),
                math.sin(rx),
                math.cos(rx) * math.cos(ry),
            ],
            dtype=np.float32,
        )
        gluLookAt(*camera_pos, *self.cam_target, 0, 1, 0)

    def draw_info(self, skeletons: Dict[str, List[BoneRecord]], point_count: int) -> None:
        now = time.monotonic()
        if now - self.info_at >= 0.1:
            self.info_at = now
            with self.state.lock:
                now = time.monotonic()
                rates = {name: rate.read(now)[0] for name, rate in self.state.motion_rates.items()}
                cloud_fps = self.state.cloud_rate.read(now)[0]
                viewer_fps = self.state.viewer_rate.read(now)[0]
            lines = [("OSC Stream", (240, 243, 247))]
            if skeletons:
                for index, (name, bones) in enumerate(sorted(skeletons.items())):
                    motion_fps = rates.get(name, 0.0)
                    name = name.replace('\r', ' ').replace('\n', ' ')
                    suffix = f" | {len(bones):,} bones"
                    if self.info_font.size(name + suffix)[0] > 470:
                        while name and self.info_font.size(name + "..." + suffix)[0] > 470:
                            name = name[:-1]
                        name += "..."
                    color = tuple(round(c * 255) for c in ACTOR_COLORS[index % len(ACTOR_COLORS)])
                    lines.append((name + suffix, color))
                    lines.append((f"Received Motion: {motion_fps:.1f} fps", (200, 205, 215)))
            else:
                lines.append(("Waiting for motion...", (255, 190, 80)))
            lines.append((f"Point cloud: {point_count:,} points", (100, 210, 245)))
            lines.append((f"Received Pointcloud: {cloud_fps:.1f} fps", (100, 210, 245)))
            lines.append((f"Viewer: {viewer_fps:.1f} fps", (240, 243, 247)))
            # Rebuild text only when it changes, at most ten times per second.
            if tuple(lines) != self.info_lines:
                self.info_lines = tuple(lines)
                labels = [self.info_font.render(text, True, color) for text, color in lines]
                row_height = self.info_font.get_linesize() + 5
                self.info_size = (max(label.get_width() for label in labels) + 28, len(labels) * row_height + 18)
                panel = pygame.Surface(self.info_size, pygame.SRCALPHA)
                panel.fill((14, 17, 23, 225))
                for row, label in enumerate(labels):
                    panel.blit(label, (14, 9 + row * row_height))
                self.info_pixels = pygame.image.tobytes(panel, "RGBA", True)
        glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT)
        glDisable(GL_DEPTH_TEST)
        glWindowPos2d(16, pygame.display.get_window_size()[1] - 16 - self.info_size[1])
        glDrawPixels(*self.info_size, GL_RGBA, GL_UNSIGNED_BYTE, self.info_pixels)
        glPopAttrib()

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                return False
            if event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.dragging = True
                    self.last_mouse = event.pos
                elif event.button == 4:
                    self.cam_dist = max(1.0, self.cam_dist - 0.3)
                elif event.button == 5:
                    self.cam_dist = min(15.0, self.cam_dist + 0.3)
            elif event.type == MOUSEBUTTONUP and event.button == 1:
                self.dragging = False
            elif event.type == MOUSEMOTION and self.dragging:
                dx = event.pos[0] - self.last_mouse[0]
                dy = event.pos[1] - self.last_mouse[1]
                self.cam_ry += dx * 0.5
                self.cam_rx = max(-89.0, min(89.0, self.cam_rx + dy * 0.5))
                self.last_mouse = event.pos
            elif event.type == KEYDOWN:
                if event.key == K_r:
                    self.cam_dist = 4.0
                    self.cam_rx = 15.0
                    self.cam_ry = 0.0
                    self.cam_target = np.array([0.0, 0.9, 0.0], dtype=np.float32)
                elif event.key == K_UP:
                    self.cam_target[1] += 0.1
                elif event.key == K_DOWN:
                    self.cam_target[1] -= 0.1
        return True

    def render(self) -> None:
        skeletons, points = self.state.snapshot()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.setup_camera()
        self.draw_grid()
        self.draw_world_axes()

        for idx, (_, joints) in enumerate(sorted(skeletons.items())):
            self.draw_skeleton(joints, color_idx=idx)

        self.draw_point_cloud(points)
        self.draw_info(skeletons, len(points))
        pygame.display.flip()
        with self.state.lock:
            self.state.viewer_rate.add(time.monotonic())
        if self.clock is not None:
            self.clock.tick(max(1, int(round(1.0 / self.tick_interval))))

    def run(self) -> None:
        self.init()
        try:
            while self.handle_events():
                self.render()
        finally:
            pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive and visualize MOVIN OSC data.")
    parser.add_argument("--host", default="0.0.0.0", help="OSC listen host")
    parser.add_argument("--port", type=int, default=11235, help="OSC listen port")
    parser.add_argument("--fps", type=float, default=60.0, help="Visualizer refresh rate")
    parser.add_argument("--point-size", type=float, default=3.0, help="Rendered point size")
    parser.add_argument("--axis-size", type=float, default=0.08, help="Per-joint axis length")
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Remove skeleton after N seconds without data",
    )
    args = parser.parse_args()

    state = SharedState(timeout=args.timeout)
    server = ReceiverServer((args.host, args.port), state)

    print(f"Listening for OSC on {args.host}:{args.port}")
    server_thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    server_thread.start()

    try:
        ViewerApp(
            state=state,
            fps=args.fps,
            point_size=args.point_size,
            axis_size=args.axis_size,
        ).run()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()


if __name__ == "__main__":
    main()
