"""Livox non-repetitive scan pattern for mjlab's ``RayCastSensor``.

The ray directions come from the recorded scan tables shipped with MuJoCo-LiDAR
(``mujoco_lidar/scan_mode/<model>.npy``), stored in firing order. Every
``sense()`` casts the next ``samples`` rays of the table, so consecutive scans
cover different directions like the real sensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import torch
import warp as wp
from mjlab.sensor import RayCastSensor, RayCastSensorCfg
from mujoco_lidar.scan_gen import LivoxGenerator


@dataclass
class LivoxPatternCfg:
  """Livox scan table (Mid-360, Avia, HAP, ...) consumed in firing order."""

  model: str = "mid360"
  """Livox model, the name of a table in ``mujoco_lidar/scan_mode``."""

  samples: int = 20000
  """Rays per ``sense()``."""

  def load_directions(self, device: str) -> torch.Tensor:
    """All unit directions of the scan table, in firing order [M, 3]."""
    angles = torch.as_tensor(LivoxGenerator(self.model).ray_angles, device=device)
    theta, phi = angles[:, 0], angles[:, 1]
    return torch.stack(
      [phi.cos() * theta.cos(), phi.cos() * theta.sin(), phi.sin()], dim=-1
    ).float()

  def generate_rays(
    self, mj_model: mujoco.MjModel | None, device: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """First scan of the table: (local_offsets [N, 3], local_directions [N, 3])."""
    del mj_model
    directions = self.load_directions(device)[: self.samples]
    return torch.zeros_like(directions), directions


@dataclass(kw_only=True)
class LivoxRayCastSensorCfg(RayCastSensorCfg):
  """``RayCastSensorCfg`` whose rays walk through a Livox scan table."""

  pattern: LivoxPatternCfg = field(default_factory=LivoxPatternCfg)  # type: ignore[assignment]

  def build(self) -> LivoxRayCastSensor:
    return LivoxRayCastSensor(self)


class LivoxRayCastSensor(RayCastSensor):
  """Ray sensor that advances through the Livox scan table on every scan."""

  cfg: LivoxRayCastSensorCfg

  def initialize(self, mj_model, model, data, device) -> None:
    super().initialize(mj_model, model, data, device)
    self._scan_table = self.cfg.pattern.load_directions(device)
    self._scan_index = torch.arange(self.num_rays_per_frame, device=device)

  def prepare_rays(self) -> None:
    self._local_directions = self._scan_table[self._scan_index]
    self._scan_index = (self._scan_index + self.num_rays_per_frame) % len(
      self._scan_table
    )
    super().prepare_rays()

  @property
  def local_directions(self) -> torch.Tensor:
    """[rays_per_frame, 3] Directions of the last scan, in the sensor frame."""
    assert self._local_directions is not None
    return self._local_directions

  @property
  def hit_geom_ids(self) -> torch.Tensor:
    """[B, N] Geom hit by each ray of the last scan, -1 if no hit."""
    assert self._ray_geomid is not None and self._distances is not None
    ids = wp.to_torch(self._ray_geomid)
    return torch.where(self._distances >= 0, ids, -1)
