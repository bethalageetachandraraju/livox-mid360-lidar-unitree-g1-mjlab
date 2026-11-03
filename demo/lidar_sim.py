"""Braced G1 with its head Livox Mid-360, walking through a furnished room.

Physics and LiDAR run in mjlab on MuJoCo Warp. The Mid-360 is modelled by
``livox_mid360_mjlab`` on the real head mount (``add_g1_mid360``: pitched
2.3 deg, upside down): the non-repetitive scan table from MuJoCo-LiDAR fired at
200k points/s (1000 rays every 5 ms physics step, so walking distorts a frame as
on the robot), 10 Hz frames, 0.1 m blind zone, reflectivity-dependent range,
2-3 cm / 0.15 deg noise, false alarms, 1 mm coordinates, 4 laser lines, and the
built-in 200 Hz IMU. The rough/42998 walk policy drives the robot.

Two browser tabs (viser):
  http://localhost:8080  robot, objects and the live scan; drive / LiDAR controls
  http://localhost:8081  LiDAR map: voxelized scans registered with the true
                         sensor pose (what an ideal FAST-LIO would build)

Run from the repository root:
  uv run python demo/lidar_sim.py
  uv run python demo/lidar_sim.py --manual          # sliders instead of tour
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from pathlib import Path

import matplotlib
import mujoco
import numpy as np
import torch
import trimesh
import viser
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import ObjRef
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.utils.lab_api.math import quat_apply
from mjlab.viewer.viser.scene import MjlabViserScene

from g1_braced import JOINTS, STAND_POSE, build_spec, quat_rotate_inverse
from livox_mid360_mjlab import (
  G1_MID360_SITE,
  Mid360,
  Mid360Cfg,
  Mid360Frame,
  Mid360Imu,
  add_g1_mid360,
  mid360_raycast_cfg,
)
from walk_policy import WALK_CMD_HI, WALK_CMD_LO, MLPActor, RobotState, WalkPolicy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WALK = ROOT / "demo/checkpoints/g1_walk_rough_42998.pt"

SIM_DT = 0.005
DECIMATION = 4  # 50 Hz policy.
CONTROL_DT = SIM_DT * DECIMATION
VIEW_EVERY = 2  # Control steps per viewer update (25 Hz).
SELF_CROP = 0.5  # m, map ignores hits this close to the sensor axis (robot).
FALL_HEIGHT = 0.45  # Pelvis height that counts as a fall.

# Tour through the room (x forward from the start, y left), visited in a loop.
TOUR = np.array(
  [
    (2.0, 0.0),
    (5.2, -0.2),
    (9.2, 0.0),
    (9.6, -2.6),
    (1.0, -2.6),
    (-1.5, -1.0),
    (-1.5, 3.2),
    (4.6, 3.2),
    (5.4, 1.6),
    (9.0, 0.8),
    (4.0, -0.3),
    (0.0, 0.0),
  ]
)

_TURBO = (matplotlib.colormaps["turbo"](np.linspace(0, 1, 256))[:, :3] * 255).astype(
  np.uint8
)


_LINE_COLORS = np.array(
  [(230, 80, 80), (80, 200, 90), (70, 130, 240), (240, 200, 60)], np.uint8
)


def colormap(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
  idx = np.clip((v - lo) / (hi - lo) * 255, 0, 255).astype(np.int64)
  return _TURBO[idx]


# Points are drawn as instanced meshes: viser's point-cloud shader does not link
# in Chromium/ANGLE (mediump vs highp declarations of one uniform).
_OCTA = trimesh.creation.icosphere(subdivisions=0, radius=0.5)
_CUBE = trimesh.creation.box(extents=(1.0, 1.0, 1.0))


def add_points(
  server: viser.ViserServer,
  name: str,
  pts: np.ndarray,
  colors: np.ndarray | tuple[int, int, int],
  size: float,
  mesh: trimesh.Trimesh = _OCTA,
  visible: bool = True,
) -> viser.BatchedMeshHandle:
  n = len(pts)
  wxyz = np.zeros((n, 4), np.float32)
  wxyz[:, 0] = 1.0
  return server.scene.add_batched_meshes_simple(
    name,
    mesh.vertices,
    mesh.faces,
    batched_wxyzs=wxyz,
    batched_positions=pts.astype(np.float32),
    batched_scales=np.full(n, size, np.float32),
    batched_colors=colors,
    lod="off",
    flat_shading=True,
    cast_shadow=False,
    receive_shadow=False,
    visible=visible,
  )


# ---------------------------------------------------------------------------
# Scene: braced G1 with the Mid-360 site, and a furnished room.
# ---------------------------------------------------------------------------


def robot_spec() -> mujoco.MjSpec:
  spec = build_spec(SIM_DT)
  spec.delete(spec.geom("floor"))  # The arena brings the floor.
  add_g1_mid360(spec)
  return spec


def _yaw_pitch(yaw: float = 0.0, pitch: float = 0.0) -> list[float]:
  q = np.zeros(4)
  mujoco.mju_euler2Quat(q, np.array([0.0, pitch, yaw]), "xyz")
  return q.tolist()


def arena_spec() -> mujoco.MjSpec:
  """A 14 x 10 m room (door in the east wall) with furniture; all static."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="arena")
  box, cyl, sph = (
    mujoco.mjtGeom.mjGEOM_BOX,
    mujoco.mjtGeom.mjGEOM_CYLINDER,
    mujoco.mjtGeom.mjGEOM_SPHERE,
  )
  # Walls are see-through in the viewer only; rays ignore alpha.
  wall, wood, metal = (0.82, 0.8, 0.76, 0.3), (0.6, 0.42, 0.25, 1), (0.45, 0.5, 0.55, 1)

  def add(name, kind, pos, size, rgba, quat=(1.0, 0.0, 0.0, 0.0)):
    body.add_geom(name=name, type=kind, pos=pos, size=size, rgba=rgba, quat=quat)

  add(
    "floor", mujoco.mjtGeom.mjGEOM_PLANE, (4, 0, 0), (20, 20, 0.1), (0.3, 0.32, 0.35, 1)
  )
  # Walls, 2.5 m high; east wall has a 1.2 m door with a lintel.
  add("wall_n", box, (4, 5.05, 1.25), (7.1, 0.05, 1.25), wall)
  add("wall_s", box, (4, -5.05, 1.25), (7.1, 0.05, 1.25), wall)
  add("wall_w", box, (-3.05, 0, 1.25), (0.05, 5.1, 1.25), wall)
  add("wall_e1", box, (11.05, 2.8, 1.25), (0.05, 2.2, 1.25), wall)
  add("wall_e2", box, (11.05, -2.8, 1.25), (0.05, 2.2, 1.25), wall)
  add("lintel", box, (11.05, 0, 2.25), (0.05, 0.6, 0.25), wall)
  add("partition", box, (6.0, 3.55, 1.0), (0.06, 1.35, 1.0), (0.7, 0.75, 0.8, 1))
  # Furniture and clutter.
  add("crate1", box, (2.5, 1.8, 0.3), (0.3, 0.3, 0.3), wood)
  add("crate2", box, (2.5, 1.8, 0.75), (0.15, 0.15, 0.15), wood)
  add("crate3", box, (2.0, 2.3, 0.2), (0.2, 0.2, 0.2), wood, _yaw_pitch(0.5))
  add("cabinet", box, (3.5, -1.4, 0.6), (0.3, 0.4, 0.6), (0.2, 0.35, 0.65, 1))
  add("pillar1", cyl, (4.5, 0.9, 1.25), (0.2, 1.25, 0), metal)
  add("pillar2", cyl, (8.0, -1.5, 1.25), (0.25, 1.25, 0), metal)
  add("table_top", box, (8.5, 2.5, 0.74), (0.6, 0.4, 0.02), wood)
  for i, (dx, dy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
    add(
      f"table_leg{i}",
      cyl,
      (8.5 + 0.55 * dx, 2.5 + 0.35 * dy, 0.36),
      (0.03, 0.36, 0),
      metal,
    )
  add("platform", box, (7.5, -4.0, 0.075), (0.8, 0.6, 0.075), (0.5, 0.5, 0.45, 1))
  add(
    "ramp",
    box,
    (1.0, -4.2, 0.18),
    (1.0, 0.5, 0.04),
    (0.55, 0.5, 0.4, 1),
    _yaw_pitch(0, -0.25),
  )
  add("ball1", sph, (1.5, -1.5, 0.25), (0.25, 0, 0), (0.85, 0.2, 0.15, 1))
  add("ball2", sph, (9.5, 1.3, 0.15), (0.15, 0, 0), (0.95, 0.75, 0.1, 1))
  add("bin", cyl, (-2.2, -3.8, 0.4), (0.3, 0.4, 0), (0.25, 0.55, 0.3, 1))
  add("shelf", box, (-2.7, 2.0, 0.9), (0.2, 0.8, 0.9), wood)
  # Outside, seen through the door.
  add("outside_box", box, (13.5, 0.8, 0.5), (0.5, 0.5, 0.5), (0.6, 0.3, 0.6, 1))
  add("outside_post", cyl, (14.0, -1.5, 1.0), (0.15, 1.0, 0), metal)
  return spec


def build(device: str, cfg: Mid360Cfg) -> tuple[Scene, Simulation]:
  # One sense() per physics step; the head (parent body) is excluded, groups
  # 0-2 are the arena and the robot's visual meshes.
  lidar = mid360_raycast_cfg(
    "mid360",
    ObjRef(type="site", name=G1_MID360_SITE, entity="robot"),
    SIM_DT,
    cfg,
  )
  scene_cfg = SceneCfg(
    num_envs=1,
    entities={
      "robot": EntityCfg(spec_fn=robot_spec),
      "arena": EntityCfg(spec_fn=arena_spec),
    },
    sensors=(lidar,),
  )
  scene = Scene(scene_cfg, device)
  sim_cfg = SimulationCfg(
    njmax=1500,
    mujoco=MujocoCfg(timestep=SIM_DT, iterations=10, ls_iterations=20),
  )
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=scene.compile(), device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  assert scene.sensor_context is not None
  sim.set_sensor_context(scene.sensor_context)
  return scene, sim


# ---------------------------------------------------------------------------
# The robot as the walk policy sees it.
# ---------------------------------------------------------------------------


class Robot:
  def __init__(self, sim: Simulation) -> None:
    m = sim.mj_model
    self.sim = sim
    jids = [m.joint(f"robot/{j}").id for j in JOINTS]
    self.qadr = torch.tensor([m.jnt_qposadr[j] for j in jids], device=sim.device)
    self.dadr = torch.tensor([m.jnt_dofadr[j] for j in jids], device=sim.device)
    self.act = torch.tensor(
      [m.actuator(f"robot/{j}").id for j in JOINTS], device=sim.device
    )
    free = next(
      j
      for j in range(m.njnt)
      if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
      and m.body(m.jnt_bodyid[j]).name.startswith("robot/")
    )
    self.root = int(m.jnt_qposadr[free])
    self.quat_adr = int(m.sensor_adr[m.sensor("robot/imu_quat").id])
    self.gyro_adr = int(m.sensor_adr[m.sensor("robot/imu_ang_vel").id])
    self.stand_height = self._stand_height()

  def _stand_height(self) -> float:
    """Pelvis height that puts the lowest robot collision geom on the floor."""
    m = self.sim.mj_model
    d = mujoco.MjData(m)
    d.qpos[self.root : self.root + 7] = (0, 0, 1, 1, 0, 0, 0)
    d.qpos[self.qadr.cpu().numpy()] = STAND_POSE
    mujoco.mj_kinematics(m, d)
    lowest = np.inf
    for g in range(m.ngeom):
      if not m.geom_contype[g] or not m.body(m.geom_bodyid[g]).name.startswith(
        "robot/"
      ):
        continue
      center, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
      rot = d.geom_xmat[g].reshape(3, 3)
      z = d.geom_xpos[g, 2] + rot[2] @ center - np.abs(rot[2]) @ half
      lowest = min(lowest, z)
    return 1.0 - lowest + 0.002

  def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
    sim, dev = self.sim, self.sim.device
    sim.reset()
    qpos = sim.data.qpos[0]
    qpos[self.root : self.root + 3] = torch.tensor(
      [x, y, self.stand_height], device=dev
    )
    qpos[self.root + 3 : self.root + 7] = torch.tensor(
      [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], device=dev
    )
    stand = torch.tensor(STAND_POSE, device=dev, dtype=qpos.dtype)
    qpos[self.qadr] = stand
    sim.data.qvel[0] = 0.0
    sim.data.ctrl[0, self.act] = stand
    sim.forward()

  def state(self) -> tuple[RobotState, np.ndarray]:
    """Policy inputs and the base pose (x, y, z, qw, qx, qy, qz)."""
    d = self.sim.data
    qpos, qvel, sd = d.qpos[0], d.qvel[0], d.sensordata[0]
    flat = torch.cat(
      [
        qpos[self.qadr],
        qvel[self.dadr],
        sd[self.quat_adr : self.quat_adr + 4],
        sd[self.gyro_adr : self.gyro_adr + 3],
        qpos[self.root : self.root + 7],
      ]
    )
    v = flat.double().cpu().numpy()
    n = len(JOINTS)
    quat = v[2 * n : 2 * n + 4]
    gravity = quat_rotate_inverse(quat / np.linalg.norm(quat), np.array([0, 0, -1.0]))
    s = RobotState(
      q=v[:n], dq=v[n : 2 * n], gravity_b=gravity, gyro_b=v[2 * n + 4 : 2 * n + 7]
    )
    return s, v[2 * n + 7 :]

  def set_target(self, target: np.ndarray) -> None:
    d = self.sim.data
    d.ctrl[0, self.act] = torch.as_tensor(
      target, device=self.sim.device, dtype=d.ctrl.dtype
    )


def autopilot(base: np.ndarray, goal: np.ndarray) -> np.ndarray:
  """[vx, vy, wz] toward a waypoint: turn on the spot when facing away."""
  w, z = base[3], base[6]
  yaw = 2 * math.atan2(z, w)
  dx, dy = goal - base[:2]
  err = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
  if abs(err) > 0.6:
    return np.array([0.0, 0.0, 0.5 * math.copysign(1, err)])
  return np.array([0.6, 0.0, float(np.clip(1.5 * err, -0.6, 0.6))])


# ---------------------------------------------------------------------------
# LiDAR map (voxel hash of world hits).
# ---------------------------------------------------------------------------


class VoxelMap:
  def __init__(self, voxel: float) -> None:
    self.voxel = voxel
    self.keys: set[int] = set()
    self.pending: list[np.ndarray] = []
    self.total = 0

  def insert(self, pts: np.ndarray) -> None:
    ijk = np.floor(pts / self.voxel).astype(np.int64) + (1 << 20)
    keys = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
    keys, first = np.unique(keys, return_index=True)
    new = np.fromiter((k not in self.keys for k in keys.tolist()), bool, len(keys))
    self.keys.update(keys[new].tolist())
    if new.any():
      centers = (np.floor(pts[first[new]] / self.voxel) + 0.5) * self.voxel
      self.pending.append(centers.astype(np.float32))
      self.total += int(new.sum())

  def take(self) -> np.ndarray | None:
    if not self.pending:
      return None
    out = np.concatenate(self.pending)
    self.pending = []
    return out


# ---------------------------------------------------------------------------
# Viewer tabs.
# ---------------------------------------------------------------------------


class App:
  def __init__(self, args: argparse.Namespace) -> None:
    self.args = args
    self.mcfg = Mid360Cfg(frame_rate=args.frame_rate, noise=not args.no_noise)
    self.scene, self.sim = build("cuda:0", self.mcfg)
    self.mid360 = Mid360(self.scene["mid360"], self.sim.mj_model, SIM_DT, self.mcfg)
    self.imu = Mid360Imu(self.sim.mj_model, f"robot/{G1_MID360_SITE}", self.mcfg)
    self.imu_last = (torch.zeros(1, 3), torch.zeros(1, 3))
    self.robot = Robot(self.sim)
    actor = MLPActor(args.checkpoint)
    print("walk:", actor.describe())
    self.walk = WalkPolicy(actor, STAND_POSE)
    self.lock = threading.Lock()
    self.reset_requested = False
    self.tour_idx = 0
    self.fall_t: float | None = None
    self.t_sim = 0.0
    self.n_steps = 0
    self.base = np.zeros(7)
    self.map = VoxelMap(args.voxel)
    self.map_chunks: list[viser.BatchedMeshHandle] = []
    self.path: list[np.ndarray] = []
    self._build_live_tab(args.port)
    self._build_map_tab(args.port + 1)

  # -- tab 1: live -----------------------------------------------------------

  def _build_live_tab(self, port: int) -> None:
    server = viser.ViserServer(port=port, label="G1 Mid-360 (live)")
    self.live = server
    self.vscene = MjlabViserScene(server, self.sim.mj_model, num_envs=1)
    server.gui.add_markdown(
      f"**G1 + Livox Mid-360** in mjlab. Map: "
      f"[http://localhost:{port + 1}](http://localhost:{port + 1})"
    )
    with server.gui.add_folder("Drive"):
      self.gui_mode = server.gui.add_dropdown(
        "Mode",
        ("Tour", "Manual"),
        initial_value="Manual" if self.args.manual else "Tour",
      )
      lo, hi = WALK_CMD_LO, WALK_CMD_HI
      self.gui_vx = server.gui.add_slider("vx (m/s)", lo[0], hi[0], 0.05, 0.0)
      self.gui_vy = server.gui.add_slider("vy (m/s)", lo[1], hi[1], 0.05, 0.0)
      self.gui_wz = server.gui.add_slider("wz (rad/s)", lo[2], hi[2], 0.05, 0.0)
      stop = server.gui.add_button("Stop")
      reset = server.gui.add_button("Reset robot")
      self.gui_pause = server.gui.add_checkbox("Pause", False)

      @stop.on_click
      def _(_) -> None:
        self.gui_mode.value = "Manual"
        for s in (self.gui_vx, self.gui_vy, self.gui_wz):
          s.value = 0.0

      @reset.on_click
      def _(_) -> None:
        self.reset_requested = True

    with server.gui.add_folder("LiDAR"):
      self.gui_scan = server.gui.add_checkbox("Show scan", True)
      self.gui_frame = server.gui.add_dropdown(
        "Frame",
        ("raw", "deskewed"),
        initial_value="raw",
        hint="raw: /livox/lidar drawn at the frame's start pose, motion distortion"
        " included. deskewed: every point through its own true pose.",
      )
      self.gui_color = server.gui.add_dropdown(
        "Color by",
        ("height", "distance", "reflectivity", "laser line"),
        initial_value="height",
      )
      self.gui_noise = server.gui.add_checkbox("Sensor noise", self.mcfg.noise)

      @self.gui_noise.on_update
      def _(_) -> None:
        self.mcfg.noise = self.gui_noise.value

      self.gui_psize = server.gui.add_slider("Point size", 0.005, 0.06, 0.005, 0.025)
    self.gui_status = server.gui.add_markdown("")
    self.vscene.create_scene_gui(
      camera_distance=5.0, camera_azimuth=200, camera_elevation=35
    )
    self.scan_frame = server.scene.add_frame("/lidar", show_axes=False)
    self.n_hits = 0

  # -- tab 2: map ------------------------------------------------------------

  def _build_map_tab(self, port: int) -> None:
    server = viser.ViserServer(port=port, label="G1 Mid-360 (map)")
    self.mapsrv = server
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
      "/grid", width=24, height=16, cell_size=1.0, position=(4, 0, 0)
    )
    server.gui.add_markdown(
      f"**LiDAR map**, {self.args.voxel * 100:.0f} cm voxels, scans placed with the "
      f"true sensor pose. Live view: [http://localhost:{port - 1}]"
      f"(http://localhost:{port - 1})"
    )
    self.gui_map_live = server.gui.add_checkbox("Show current scan", True)
    self.gui_map_path = server.gui.add_checkbox("Show path", True)
    self.gui_map_psize = server.gui.add_slider(
      "Voxel draw size", 0.01, 0.15, 0.005, self.args.voxel
    )
    clear = server.gui.add_button("Clear map")
    save = server.gui.add_button("Save map (.ply)")
    self.gui_map_status = server.gui.add_markdown("")
    self.map_pose = server.scene.add_frame("/sensor", axes_length=0.4, axes_radius=0.02)
    self.map_path = None

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
      client.camera.position = (4.0, -9.0, 13.0)
      client.camera.look_at = (4.0, 0.0, 0.0)

    @clear.on_click
    def _(_) -> None:
      with self.lock:
        for h in self.map_chunks:
          h.remove()
        self.map_chunks = []
        self.map = VoxelMap(self.args.voxel)

    @save.on_click
    def _(_) -> None:
      with self.lock:
        pts = [h.batched_positions for h in self.map_chunks]
      out = ROOT / "maps" / f"lidar_map_{time.strftime('%Y%m%d_%H%M%S')}.ply"
      out.parent.mkdir(parents=True, exist_ok=True)
      write_ply(out, np.concatenate(pts) if pts else np.zeros((0, 3), np.float32))
      print("map saved:", out)

    @self.gui_map_psize.on_update
    def _(_) -> None:
      for h in self.map_chunks:
        h.batched_scales = np.full(
          len(h.batched_positions), self.gui_map_psize.value, np.float32
        )

  # -- loop ------------------------------------------------------------------

  def command(self, base: np.ndarray) -> np.ndarray:
    if self.gui_mode.value == "Manual":
      return np.array([self.gui_vx.value, self.gui_vy.value, self.gui_wz.value])
    goal = TOUR[self.tour_idx]
    if np.linalg.norm(goal - base[:2]) < 0.4:
      self.tour_idx = (self.tour_idx + 1) % len(TOUR)
      goal = TOUR[self.tour_idx]
    return autopilot(base, goal)

  def on_frame(self, f: Mid360Frame) -> None:
    valid = f.valid[0]
    raw = f.points[0][valid]  # Sensor frame, as the driver publishes it.
    if self.gui_frame.value == "raw":
      q = f.quat_w[0].expand(len(raw), 4)
      pts_t = f.pos_w[0] + quat_apply(q, raw)
    else:
      pts_t = f.points_w[0][valid]
    pts = pts_t.cpu().numpy()
    origin = f.pos_w[0].cpu().numpy()
    quat = f.quat_w[0].cpu().numpy()

    # Re-adding a cloud replaces it in one message (points and colors together).
    self.n_hits = len(pts)
    mode = self.gui_color.value
    if mode == "height":
      color = colormap(pts[:, 2], -0.05, 2.5)
    elif mode == "distance":
      color = colormap(raw.norm(dim=-1).cpu().numpy(), 0.5, 12.0)
    elif mode == "reflectivity":
      color = colormap(f.reflectivity[0][valid].cpu().numpy(), 0, 150)
    else:
      color = _LINE_COLORS[f.line[valid].long().cpu().numpy()]
    add_points(
      self.live,
      "/lidar/scan",
      pts,
      color,
      self.gui_psize.value,
      visible=self.gui_scan.value,
    )

    # Map: deskewed points minus self hits (vertical cylinder around the sensor).
    pts_w = f.points_w[0][valid].cpu().numpy()
    keep = np.linalg.norm(pts_w[:, :2] - origin[:2], axis=1) > SELF_CROP
    with self.lock:
      self.map.insert(pts_w[keep])
    self.map_pose.position = origin
    self.map_pose.wxyz = quat
    add_points(
      self.mapsrv,
      "/scan",
      pts_w[keep],
      (255, 255, 255),
      0.04,
      visible=self.gui_map_live.value,
    )
    self.path.append(origin.copy())

  def flush_map(self) -> None:
    with self.lock:
      pts = self.map.take()
      if pts is not None:
        h = add_points(
          self.mapsrv,
          f"/map/c{len(self.map_chunks)}",
          pts,
          colormap(pts[:, 2], -0.05, 2.5),
          self.gui_map_psize.value,
          mesh=_CUBE,
        )
        self.map_chunks.append(h)
    if len(self.path) > 1 and self.gui_map_path.value:
      p = np.array(self.path[::5] + [self.path[-1]], np.float32)
      p[:, 2] = 0.1
      self.map_path = self.mapsrv.scene.add_line_segments(
        "/path", np.stack([p[:-1], p[1:]], axis=1), (255, 80, 200), line_width=3.0
      )
    elif self.map_path is not None:
      self.map_path.visible = False

  def reset(self) -> None:
    self.robot.reset()
    self.walk.reset()
    self.tour_idx = 0
    self.fall_t = None

  def step(self) -> Mid360Frame | None:
    """One 50 Hz control step (4 physics steps). Returns a frame if one ended."""
    state, self.base = self.robot.state()
    if self.reset_requested or (
      self.fall_t is not None and self.t_sim - self.fall_t > 1.0
    ):
      self.reset_requested = False
      self.reset()
      return None
    if self.fall_t is None and self.base[2] < FALL_HEIGHT:
      self.fall_t = self.t_sim
      print(f"fell at t={self.t_sim:.1f}s, resetting")
    self.walk.cmd = self.command(self.base)
    self.robot.set_target(self.walk.compute(state))
    out = None
    for _ in range(DECIMATION):
      self.sim.step()
      self.t_sim += SIM_DT
      # One physics step = one Mid-360 batch (1000 rays) and one IMU sample.
      self.sim.sense()
      self.imu_last = self.imu.read(self.sim.data.sensordata)
      frame = self.mid360.update(self.t_sim)
      if frame is not None:
        self.on_frame(frame)
        out = frame
    self.n_steps += 1
    return out

  def run(self) -> None:
    self.reset()
    t0 = time.perf_counter()
    rtf_t, rtf_sim = t0, 0.0
    print(f"live view: http://localhost:{self.args.port}")
    print(f"LiDAR map: http://localhost:{self.args.port + 1}")
    while self.args.duration <= 0 or self.t_sim < self.args.duration:
      if self.gui_pause.value:
        time.sleep(0.05)
        t0 = time.perf_counter() - self.t_sim
        continue
      self.step()
      if self.n_steps % VIEW_EVERY == 0:
        self.vscene.update(self.sim.data)
        self.scan_frame.position = self.vscene._scene_offset
      if self.n_steps % 50 == 0:
        self.flush_map()
        now = time.perf_counter()
        rtf = (self.t_sim - rtf_sim) / (now - rtf_t)
        rtf_t, rtf_sim = now, self.t_sim
        self.show_status(rtf)
      ahead = self.t_sim - (time.perf_counter() - t0)
      if ahead > 0:
        time.sleep(ahead)

  def show_status(self, rtf: float) -> None:
    c, m, b = self.walk.cmd, self.mcfg, self.base
    gyro, acc = (x[0].cpu().numpy() for x in self.imu_last)
    self.gui_status.content = (
      f"t {self.t_sim:.0f} s, real-time x{rtf:.2f}  \n"
      f"cmd vx {c[0]:+.2f} vy {c[1]:+.2f} wz {c[2]:+.2f}  \n"
      f"Mid-360 {m.point_rate / 1000:.0f}k pts/s, {m.frame_rate:g} Hz: "
      f"{self.mid360.p} pts/frame, {self.n_hits} returns  \n"
      f"IMU {m.imu_rate:g} Hz: acc [{acc[0]:+.2f} {acc[1]:+.2f} {acc[2]:+.2f}] g, "
      f"gyro [{gyro[0]:+.2f} {gyro[1]:+.2f} {gyro[2]:+.2f}] rad/s"
    )
    self.gui_map_status.content = (
      f"{self.map.total:,} voxels in {len(self.map_chunks)} chunks"
    )
    if self.args.duration > 0:
      print(
        f"t {self.t_sim:5.1f}  pos ({b[0]:+.2f}, {b[1]:+.2f}, {b[2]:.2f})  "
        f"rtf {rtf:.2f}  hits {self.n_hits}  voxels {self.map.total}"
      )


def write_ply(path: Path, pts: np.ndarray) -> None:
  header = (
    "ply\nformat binary_little_endian 1.0\n"
    f"element vertex {len(pts)}\n"
    "property float x\nproperty float y\nproperty float z\nend_header\n"
  )
  path.write_bytes(header.encode() + pts.astype("<f4").tobytes())


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--checkpoint", default=str(DEFAULT_WALK))
  p.add_argument("--port", type=int, default=8080, help="live tab; map is port+1")
  p.add_argument("--voxel", type=float, default=0.05, help="map voxel size (m)")
  p.add_argument("--manual", action="store_true", help="start with sliders")
  p.add_argument(
    "--frame-rate", type=float, default=10.0, help="Mid-360 publish_freq (Hz)"
  )
  p.add_argument("--no-noise", action="store_true", help="ideal ranges, no IMU noise")
  p.add_argument("--duration", type=float, default=0.0, help="sim seconds, 0 = run")
  App(p.parse_args()).run()


if __name__ == "__main__":
  main()
