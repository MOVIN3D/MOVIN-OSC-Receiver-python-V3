import argparse
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
import pythonosc.parsing.osc_types as _osc_types
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_ARRAY,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FLOAT,
    GL_LINE_SMOOTH,
    GL_LINES,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINT_SMOOTH,
    GL_POINTS,
    GL_PROJECTION,
    GL_SRC_ALPHA,
    GL_VERTEX_ARRAY,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor3f,
    glColorPointer,
    glDisableClientState,
    glDrawArrays,
    glEnable,
    glEnableClientState,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glPointSize,
    glVertexPointer,
)
from OpenGL.GLU import gluLookAt, gluPerspective
from pygame.locals import DOUBLEBUF, K_DOWN, K_ESCAPE, K_r, K_UP, KEYDOWN, MOUSEBUTTONDOWN
from pygame.locals import MOUSEBUTTONUP, MOUSEMOTION, OPENGL, QUIT
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer


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
    chunks: Dict[int, List[BoneRecord]] = field(default_factory=dict)

    def add_chunk(self, chunk_index: int, bones: List[BoneRecord]) -> None:
        self.chunks[chunk_index] = bones

    def is_complete(self) -> bool:
        return len(self.chunks) == self.num_chunks

    def to_bones(self) -> List[BoneRecord]:
        ordered: List[Optional[BoneRecord]] = [None] * self.total_bone_count
        for chunk_idx in sorted(self.chunks):
            for bone in self.chunks[chunk_idx]:
                if 0 <= bone.bone_index < self.total_bone_count:
                    ordered[bone.bone_index] = bone
        return [bone for bone in ordered if bone is not None]


@dataclass
class PointCloudAssembly:
    frame_idx: int
    total_points: int
    num_chunks: int
    chunks: Dict[int, np.ndarray] = field(default_factory=dict)

    def add_chunk(self, chunk_idx: int, points: np.ndarray) -> None:
        self.chunks[chunk_idx] = points

    def is_complete(self) -> bool:
        return len(self.chunks) == self.num_chunks

    def to_points(self) -> np.ndarray:
        if not self.chunks:
            return np.empty((0, 3), dtype=np.float32)
        ordered = [self.chunks[idx] for idx in sorted(self.chunks)]
        return np.vstack(ordered).astype(np.float32, copy=False)


class SharedState:
    def __init__(self, timeout: float) -> None:
        self.lock = threading.Lock()
        self.timeout = timeout
        self.motion_assemblies: Dict[Tuple[str, int], SkeletonFrameAssembly] = {}
        self.point_assemblies: Dict[int, PointCloudAssembly] = {}
        self.latest_skeletons: Dict[str, List[BoneRecord]] = {}
        self.latest_points: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.last_update: Dict[str, float] = {}
        self.last_points_at = 0.0
        self.latest_motion_frame_idx: Dict[str, int] = {}
        self.latest_point_frame_idx = -1

    def add_motion_chunk(
        self,
        timestamp: str,
        actor_name: str,
        frame_idx: int,
        num_chunks: int,
        chunk_index: int,
        total_bone_count: int,
        bones: List[BoneRecord],
    ) -> None:
        with self.lock:
            previous_frame = self.latest_motion_frame_idx.get(actor_name, -1)
            if frame_idx < previous_frame - 1:
                if previous_frame - frame_idx > 100:
                    self.latest_motion_frame_idx[actor_name] = -1
                    self.motion_assemblies = {
                        key: value
                        for key, value in self.motion_assemblies.items()
                        if key[0] != actor_name
                    }
                else:
                    return

            key = (actor_name, frame_idx)
            assembly = self.motion_assemblies.get(key)
            if assembly is None:
                assembly = SkeletonFrameAssembly(
                    timestamp=timestamp,
                    actor_name=actor_name,
                    frame_idx=frame_idx,
                    num_chunks=num_chunks,
                    total_bone_count=total_bone_count,
                )
                self.motion_assemblies[key] = assembly

            assembly.add_chunk(chunk_index, bones)
            if not assembly.is_complete():
                return

            skeleton = compute_world_pose(assembly.to_bones())
            self.latest_skeletons[actor_name] = skeleton
            self.last_update[actor_name] = time.time()
            self.latest_motion_frame_idx[actor_name] = frame_idx
            self.motion_assemblies = {
                motion_key: motion_value
                for motion_key, motion_value in self.motion_assemblies.items()
                if motion_key[0] != actor_name or motion_key[1] >= frame_idx - 1
            }

    def add_point_chunk(
        self,
        frame_idx: int,
        total_points: int,
        chunk_idx: int,
        num_chunks: int,
        points: np.ndarray,
    ) -> None:
        with self.lock:
            if frame_idx < self.latest_point_frame_idx:
                if self.latest_point_frame_idx - frame_idx > 100:
                    self.latest_point_frame_idx = -1
                    self.point_assemblies.clear()
                else:
                    return

            if frame_idx != self.latest_point_frame_idx:
                self.latest_point_frame_idx = frame_idx
                self.point_assemblies.clear()

            assembly = self.point_assemblies.get(frame_idx)
            if assembly is None:
                assembly = PointCloudAssembly(
                    frame_idx=frame_idx,
                    total_points=total_points,
                    num_chunks=num_chunks,
                )
                self.point_assemblies[frame_idx] = assembly

            assembly.add_chunk(chunk_idx, points)
            if assembly.is_complete():
                self.latest_points = assembly.to_points()
                self.last_points_at = time.time()

    def snapshot(self) -> Tuple[Dict[str, List[BoneRecord]], np.ndarray]:
        now = time.time()
        with self.lock:
            stale_actors = [actor for actor, updated_at in self.last_update.items() if now - updated_at > self.timeout]
            for actor in stale_actors:
                self.latest_skeletons.pop(actor, None)
                self.last_update.pop(actor, None)
                self.motion_assemblies = {
                    key: value for key, value in self.motion_assemblies.items() if key[0] != actor
                }

            skeletons = {
                actor: [clone_bone(bone) for bone in bones]
                for actor, bones in self.latest_skeletons.items()
            }
            points = self.latest_points.copy()
            return skeletons, points


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


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


def clone_bone(bone: BoneRecord) -> BoneRecord:
    return BoneRecord(
        bone_index=bone.bone_index,
        parent_index=bone.parent_index,
        bone_name=bone.bone_name,
        local_position=bone.local_position.copy(),
        rest_rotation=bone.rest_rotation.copy(),
        local_rotation=bone.local_rotation.copy(),
        local_scale=bone.local_scale.copy(),
        world_position=bone.world_position.copy(),
        world_rotation=bone.world_rotation.copy(),
    )


def compute_world_pose(bones: List[BoneRecord]) -> List[BoneRecord]:
    ordered = sorted(bones, key=lambda bone: bone.bone_index)
    bone_map = {bone.bone_index: bone for bone in ordered}

    unity_matrices: Dict[int, np.ndarray] = {}
    unity_rotations: Dict[int, np.ndarray] = {}
    def resolve_pose(bone_index: int) -> Tuple[np.ndarray, np.ndarray]:
        if bone_index in unity_matrices:
            return unity_matrices[bone_index], unity_rotations[bone_index]

        bone = bone_map[bone_index]
        local_rotation = normalize_quaternion(bone.local_rotation)
        local_scale = bone.local_scale.astype(np.float32)
        local_matrix = trs_matrix(bone.local_position, local_rotation, local_scale)
        if bone.parent_index < 0 or bone.parent_index not in bone_map:
            unity_matrices[bone_index] = local_matrix
            unity_rotations[bone_index] = local_rotation
        else:
            parent_matrix, parent_rotation = resolve_pose(bone.parent_index)
            unity_matrices[bone_index] = parent_matrix @ local_matrix
            unity_rotations[bone_index] = normalize_quaternion(
                quaternion_multiply(parent_rotation, local_rotation)
            )

        bone.world_position = unity_to_opengl_pos(unity_matrices[bone_index][:3, 3])
        bone.world_rotation = unity_to_opengl_rot(unity_rotations[bone_index])
        return unity_matrices[bone_index], unity_rotations[bone_index]

    for bone in ordered:
        resolve_pose(bone.bone_index)

    return ordered


def parse_motion(address: str, *osc_args: object, state: SharedState) -> None:
    del address
    if len(osc_args) < 7:
        return

    timestamp = str(osc_args[0])
    actor_name = str(osc_args[1])
    frame_idx = int(osc_args[2])
    num_chunks = int(osc_args[3])
    chunk_index = int(osc_args[4])
    total_bone_count = int(osc_args[5])
    chunk_bone_count = int(osc_args[6])
    payload = osc_args[7:]
    expected_len = chunk_bone_count * BONE_FIELD_COUNT
    if len(payload) < expected_len:
        return

    bones: List[BoneRecord] = []
    for i in range(chunk_bone_count):
        base = i * BONE_FIELD_COUNT
        bones.append(
            BoneRecord(
                bone_index=int(payload[base + 0]),
                parent_index=int(payload[base + 1]),
                bone_name=str(payload[base + 2]),
                local_position=np.array(payload[base + 3 : base + 6], dtype=np.float32),
                rest_rotation=np.array(payload[base + 6 : base + 10], dtype=np.float32),
                local_rotation=np.array(payload[base + 10 : base + 14], dtype=np.float32),
                local_scale=np.array(payload[base + 14 : base + 17], dtype=np.float32),
            )
        )

    state.add_motion_chunk(
        timestamp=timestamp,
        actor_name=actor_name,
        frame_idx=frame_idx,
        num_chunks=num_chunks,
        chunk_index=chunk_index,
        total_bone_count=total_bone_count,
        bones=bones,
    )


def parse_point_cloud(address: str, *osc_args: object, state: SharedState) -> None:
    del address
    if len(osc_args) < 5:
        return

    frame_idx = int(osc_args[0])
    total_points = int(osc_args[1])
    chunk_idx = int(osc_args[2])
    num_chunks = int(osc_args[3])
    chunk_point_count = int(osc_args[4])
    payload = osc_args[5:]
    expected_len = chunk_point_count * POINT_FIELD_COUNT
    if len(payload) < expected_len:
        return

    points = np.asarray(payload[:expected_len], dtype=np.float32).reshape((-1, 3))
    points[:, 0] *= -1.0
    state.add_point_chunk(
        frame_idx=frame_idx,
        total_points=total_points,
        chunk_idx=chunk_idx,
        num_chunks=num_chunks,
        points=points,
    )


def create_dispatcher(state: SharedState) -> Dispatcher:
    dispatcher = Dispatcher()
    dispatcher.map("/MOVIN/Frame", lambda addr, *args: parse_motion(addr, *args, state=state))
    dispatcher.map("/MOVIN/PointCloud", lambda addr, *args: parse_point_cloud(addr, *args, state=state))
    return dispatcher


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
        pygame.display.flip()
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
    dispatcher = create_dispatcher(state)
    server = ThreadingOSCUDPServer((args.host, args.port), dispatcher)

    print(f"Listening for OSC on {args.host}:{args.port}")
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
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


if __name__ == "__main__":
    main()
