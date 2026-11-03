"""Livox Mid-360 measurement model on top of ``LivoxRayCastSensor``.

Numbers from the Mid-360 datasheet (livoxtech.com/mid-360/specs), the Mid-360
Ethernet protocol and the ICM-40609-D datasheet:

- 200,000 points/s (first return), fired in the time order of the scan table.
  Four lasers fire interleaved, so a point's laser ``line`` is its index mod 4.
  Frames are cut every 1 / ``frame_rate`` s (driver ``publish_freq``, 10 Hz).
- Blind zone 0.1 m. Range 40 m on a 10 % target, 70 m on an 80 % target.
- Range precision (1 sigma) 3 cm at 0.2 m, 2 cm at 10 m. Angular precision
  (1 sigma) 0.15 deg. False alarm rate 0.01 %.
- Coordinates are sent as int32 millimetres; no-return points are all zero.
- Built-in ICM-40609 IMU at 200 Hz, axes of the point cloud frame, origin at
  (11.0, 23.29, -44.12) mm in it. Gyro in rad/s, accelerometer in g. Noise
  density 4.5 mdps/sqrt(Hz) and 100 ug/sqrt(Hz).

Modelling choices where the datasheet is silent:

- Rays are cast once per ``sense()``, so ray times are exact but each batch of
  rays uses the pose at the end of its physics step (5 ms batches at 200 Hz).
- Target reflectivity is the luminance of the geom's colour (material colour
  for meshes), clipped to 5-95 %. The return strength scales with reflectivity
  times the cosine of the incidence angle, and the maximum range follows the
  power law through the two datasheet points: 40 m * (rho / 0.1) ** 0.269.
- The ``reflectivity`` byte is 0-150 for 0-100 % diffuse reflectivity, the
  Livox convention (151-255 for retro-reflectors is not modelled).
- ``tag`` is 0: rain/fog/dust and edge "glue" points are not modelled.
- IMU noise is white only; bias and bias drift are not modelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.sensor import ObjRef
from mjlab.utils.lab_api.math import matrix_from_quat

from livox_mid360_mjlab.pattern import (
  LivoxPatternCfg,
  LivoxRayCastSensor,
  LivoxRayCastSensorCfg,
)

MID360_IMU_POS = (0.011, 0.02329, -0.04412)
"""ICM-40609 position in the Mid-360 point cloud frame (m)."""

G = 9.80665
"""Standard gravity, the accelerometer's unit."""


@dataclass
class Mid360Cfg:
  point_rate: float = 200_000.0
  """Points per second."""
  frame_rate: float = 10.0
  """Frames per second (driver ``publish_freq``: 5, 10, 20, 50, ...)."""
  blind: float = 0.1
  """Minimum range (m)."""
  range_10: float = 40.0
  """Maximum range on a 10 % reflectivity target at normal incidence (m)."""
  range_80: float = 70.0
  """Maximum range on an 80 % reflectivity target at normal incidence (m)."""
  range_sigma_near: tuple[float, float] = (0.2, 0.03)
  """(range m, 1-sigma m) of the range noise, interpolated to ``_far``."""
  range_sigma_far: tuple[float, float] = (10.0, 0.02)
  angle_sigma: float = math.radians(0.15)
  """1-sigma angular noise per axis (rad)."""
  false_alarm_rate: float = 1e-4
  """Fraction of rays that report a spurious return."""
  resolution: float = 0.001
  """Coordinate quantization (m)."""
  imu_rate: float = 200.0
  """IMU sample rate (Hz)."""
  gyro_noise_density: float = math.radians(4.5e-3)
  """Gyro white noise (rad/s/sqrt(Hz))."""
  accel_noise_density: float = 100e-6
  """Accelerometer white noise (g/sqrt(Hz))."""
  noise: bool = True
  """Range/angle noise, false alarms and IMU noise. Limits apply regardless."""


def add_mid360(
  spec: mujoco.MjSpec,
  body: str,
  pos: tuple[float, float, float],
  quat: tuple[float, float, float, float],
  name: str = "mid360",
) -> mujoco.MjsSite:
  """Add a Mid-360 to ``body``: the point cloud frame site ``name`` and the
  built-in IMU (site ``{name}_imu`` with ``{name}_gyro`` / ``{name}_accel``)."""
  parent = spec.body(body)
  site = parent.add_site(name=name, pos=pos, quat=quat, group=5)
  imu_pos = np.zeros(3)
  mujoco.mju_rotVecQuat(imu_pos, np.array(MID360_IMU_POS), np.array(quat))
  imu = parent.add_site(
    name=f"{name}_imu",
    pos=np.array(pos) + imu_pos,
    quat=quat,
    group=5,
  )
  for kind, suffix in (
    (mujoco.mjtSensor.mjSENS_GYRO, "gyro"),
    (mujoco.mjtSensor.mjSENS_ACCELEROMETER, "accel"),
  ):
    spec.add_sensor(
      name=f"{name}_{suffix}",
      type=kind,
      objtype=mujoco.mjtObj.mjOBJ_SITE,
      objname=imu.name,
    )
  return site


def mid360_raycast_cfg(
  name: str,
  frame: ObjRef,
  sense_dt: float,
  cfg: Mid360Cfg | None = None,
  include_geom_groups: tuple[int, ...] | None = (0, 1, 2),
) -> LivoxRayCastSensorCfg:
  """Ray sensor that fires the rays of one ``sense_dt`` per ``sense()`` call."""
  cfg = cfg or Mid360Cfg()
  return LivoxRayCastSensorCfg(
    name=name,
    frame=frame,
    pattern=LivoxPatternCfg(model="mid360", samples=round(cfg.point_rate * sense_dt)),
    max_distance=cfg.range_80,
    exclude_parent_body=True,
    include_geom_groups=include_geom_groups,
  )


def geom_reflectivity(mj_model: mujoco.MjModel) -> torch.Tensor:
  """[ngeom] Diffuse reflectivity from colour luminance, clipped to 5-95 %."""
  rgb = torch.as_tensor(mj_model.geom_rgba[:, :3]).clone()
  matid = torch.as_tensor(mj_model.geom_matid)
  has_mat = matid >= 0
  rgb[has_mat] = torch.as_tensor(mj_model.mat_rgba[:, :3])[matid[has_mat]]
  lum = rgb @ torch.tensor([0.2126, 0.7152, 0.0722])
  return lum.clamp(0.05, 0.95).float()


@dataclass
class Mid360Frame:
  """One published frame, laid out like livox_ros_driver2's CustomMsg."""

  timebase: float
  """Time of the first point (s)."""
  offset_time: torch.Tensor
  """[P] int64 ns from ``timebase``."""
  points: torch.Tensor
  """[B, P, 3] Sensor frame at each point's own time, as reported; 0 = no return."""
  reflectivity: torch.Tensor
  """[B, P] uint8."""
  tag: torch.Tensor
  """[B, P] uint8."""
  line: torch.Tensor
  """[P] uint8 laser 0-3."""
  valid: torch.Tensor
  """[B, P] Point has a return."""
  points_w: torch.Tensor
  """[B, P, 3] World frame through each point's true pose (perfect deskew)."""
  pos_w: torch.Tensor
  """[B, 3] Sensor position at ``timebase``."""
  quat_w: torch.Tensor
  """[B, 4] Sensor orientation at ``timebase`` (w, x, y, z)."""


class Mid360:
  """Turns per-step Livox ray batches into Mid-360 frames."""

  def __init__(
    self,
    sensor: LivoxRayCastSensor,
    mj_model: mujoco.MjModel,
    sense_dt: float,
    cfg: Mid360Cfg | None = None,
  ) -> None:
    self.cfg = cfg or Mid360Cfg()
    self.sensor = sensor
    self.n = sensor.num_rays_per_frame
    if self.n != round(self.cfg.point_rate * sense_dt) or self.n % 4:
      raise ValueError(
        f"sensor fires {self.n} rays per sense(); a Mid-360 fires "
        f"{self.cfg.point_rate * sense_dt:g} in {sense_dt} s (use mid360_raycast_cfg)"
      )
    self.p = round(self.cfg.point_rate / self.cfg.frame_rate)
    if self.p % self.n:
      raise ValueError(f"frame of {self.p} points is not a whole number of steps")
    device = sensor.local_directions.device
    self.reflectivity = geom_reflectivity(mj_model).to(device)
    self._exponent = math.log(self.cfg.range_80 / self.cfg.range_10) / math.log(8.0)
    self.offset_time = (
      (
        torch.arange(self.p, device=device, dtype=torch.float64)
        * 1e9
        / self.cfg.point_rate
      )
      .round()
      .long()
    )
    self.line = (torch.arange(self.p, device=device) % 4).to(torch.uint8)
    self._slot = 0
    self._buf: dict[str, torch.Tensor] = {}
    self._timebase = 0.0

  def update(self, t: float) -> Mid360Frame | None:
    """Ingest the rays of the last ``sense()``, which ended at sim time ``t``.

    Returns the frame when it is complete, else None.
    """
    cfg, data = self.cfg, self.sensor.data
    d = data.distances  # [B, n]
    B, dev = d.shape[0], d.device
    if not self._buf:
      self._buf = {
        "points": torch.zeros(B, self.p, 3, device=dev),
        "points_w": torch.zeros(B, self.p, 3, device=dev),
        "reflectivity": torch.zeros(B, self.p, dtype=torch.uint8, device=dev),
        "valid": torch.zeros(B, self.p, dtype=torch.bool, device=dev),
        "pos_w": torch.zeros(B, 3, device=dev),
        "quat_w": torch.zeros(B, 4, device=dev),
      }
    if self._slot == 0:
      self._timebase = t - self.n / cfg.point_rate
      self._buf["pos_w"].copy_(data.pos_w)
      self._buf["quat_w"].copy_(data.quat_w)

    rot = matrix_from_quat(data.quat_w)  # [B, 3, 3]
    dirs = self.sensor.local_directions.expand(B, -1, -1)  # [B, n, 3]
    dirs_w = torch.einsum("bij,bnj->bni", rot, dirs)
    cos = (data.normals_w * dirs_w).sum(-1).abs()
    gid = self.sensor.hit_geom_ids
    rho = self.reflectivity[gid.clamp(min=0)] * cos
    max_range = cfg.range_10 * (rho / 0.1).clamp(min=1e-9) ** self._exponent
    valid = (d >= cfg.blind) & (d <= max_range)
    refl = (rho * 150).round()

    if cfg.noise:
      fa = torch.rand(B, self.n, device=dev) < cfg.false_alarm_rate
      far = torch.where(valid, d, torch.full_like(d, cfg.range_10))
      d = torch.where(fa, cfg.blind + torch.rand_like(d) * (far - cfg.blind), d)
      refl = torch.where(fa, torch.randint_like(refl, 0, 10), refl)
      valid = valid | fa
      (r0, s0), (r1, s1) = cfg.range_sigma_near, cfg.range_sigma_far
      sigma = s0 + (s1 - s0) * ((d - r0) / (r1 - r0)).clamp(0, 1)
      d = d + torch.randn_like(d) * sigma
      e1 = torch.nn.functional.normalize(
        torch.cross(
          dirs, torch.tensor([0.0, 0.0, 1.0], device=dev).expand_as(dirs), dim=-1
        ),
        dim=-1,
      )
      e2 = torch.cross(dirs, e1, dim=-1)
      jitter = torch.randn(B, self.n, 2, 1, device=dev) * cfg.angle_sigma
      dirs = torch.nn.functional.normalize(
        dirs + jitter[:, :, 0] * e1 + jitter[:, :, 1] * e2, dim=-1
      )

    pts = d.unsqueeze(-1) * dirs
    pts = (pts / cfg.resolution).round() * cfg.resolution
    pts = torch.where(valid.unsqueeze(-1), pts, 0.0)
    pts_w = data.pos_w.unsqueeze(1) + torch.einsum("bij,bnj->bni", rot, pts)

    s = slice(self._slot, self._slot + self.n)
    self._buf["points"][:, s] = pts
    self._buf["points_w"][:, s] = pts_w
    self._buf["reflectivity"][:, s] = refl.clamp(0, 150).to(torch.uint8)
    self._buf["valid"][:, s] = valid
    self._slot += self.n
    if self._slot < self.p:
      return None

    self._slot = 0
    b = self._buf
    return Mid360Frame(
      timebase=self._timebase,
      offset_time=self.offset_time,
      points=b["points"].clone(),
      reflectivity=b["reflectivity"].clone(),
      tag=torch.zeros_like(b["reflectivity"]),
      line=self.line,
      valid=b["valid"].clone(),
      points_w=b["points_w"].clone(),
      pos_w=b["pos_w"].clone(),
      quat_w=b["quat_w"].clone(),
    )


class Mid360Imu:
  """The Mid-360's ICM-40609: gyro (rad/s) and accelerometer (g) with noise."""

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    name: str = "mid360",
    cfg: Mid360Cfg | None = None,
  ) -> None:
    self.cfg = cfg or Mid360Cfg()
    self._gyro = int(mj_model.sensor_adr[mj_model.sensor(f"{name}_gyro").id])
    self._accel = int(mj_model.sensor_adr[mj_model.sensor(f"{name}_accel").id])
    root_fs = math.sqrt(self.cfg.imu_rate)
    self._gyro_sigma = self.cfg.gyro_noise_density * root_fs
    self._accel_sigma = self.cfg.accel_noise_density * root_fs

  def read(self, sensordata: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(gyro [B, 3] rad/s, accel [B, 3] g) from ``sim.data.sensordata``."""
    gyro = sensordata[:, self._gyro : self._gyro + 3].clone()
    accel = sensordata[:, self._accel : self._accel + 3] / G
    if self.cfg.noise:
      gyro += torch.randn_like(gyro) * self._gyro_sigma
      accel = accel + torch.randn_like(accel) * self._accel_sigma
    return gyro, accel
