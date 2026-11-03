"""Mid-360 model: timing, frame layout, range limits, noise, IMU."""

from __future__ import annotations

import math

import pytest
import torch
from conftest import make_scene_and_sim
from mjlab.sensor import ObjRef
from mjlab.sim import MujocoCfg, SimulationCfg

from livox_mid360_mjlab import Mid360, Mid360Cfg, Mid360Imu, mid360_raycast_cfg

DT = 0.005

# Upside-down sensor (like the G1 head) on a fixed post, `height` m over a
# floor of the given grey level, with the built-in IMU sensors.
_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="100 100 0.1" rgba="{grey} {grey} {grey} 1"/>
    <body name="post" pos="0 0 {height}">
      <geom type="box" size="0.03 0.03 0.03" contype="0" conaffinity="0"/>
      <site name="mid360" quat="0 1 0 0"/>
      <site name="mid360_imu" quat="0 1 0 0"/>
    </body>
  </worldbody>
  <sensor>
    <gyro name="mid360_gyro" site="mid360_imu"/>
    <accelerometer name="mid360_accel" site="mid360_imu"/>
  </sensor>
</mujoco>
"""


def _setup(device, cfg: Mid360Cfg, height: float = 1.9, grey: float = 0.5):
  ray_cfg = mid360_raycast_cfg(
    "mid360", ObjRef(type="site", name="mid360", entity="robot"), DT, cfg
  )
  scene, sim = make_scene_and_sim(
    device,
    _XML.format(height=height, grey=grey),
    (ray_cfg,),
    sim_cfg=SimulationCfg(mujoco=MujocoCfg(timestep=DT), njmax=20),
  )
  return sim, Mid360(scene["mid360"], sim.mj_model, DT, cfg)


def _frame(sim, lidar):
  t = 0.0
  while True:
    sim.step()
    t += DT
    sim.sense()
    frame = lidar.update(t)
    if frame is not None:
      return frame


def test_frame_timing_and_layout(device):
  """200k points/s in 5 ms batches, 50 Hz frames of 4000 points, 4 lines."""
  sim, lidar = _setup(device, Mid360Cfg(frame_rate=50.0, noise=False))
  assert lidar.n == 1000 and lidar.p == 4000
  for k in range(3):
    sim.step()
    sim.sense()
    assert lidar.update((k + 1) * DT) is None
  sim.step()
  sim.sense()
  frame = lidar.update(4 * DT)
  assert frame is not None
  assert frame.timebase == pytest.approx(0.0, abs=1e-9)
  assert torch.equal(
    frame.offset_time, torch.arange(4000, device=frame.offset_time.device) * 5000
  )
  assert frame.line[:8].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
  assert frame.points.shape == (1, 4000, 3)
  # No-return points are all zero, like the Mid-360's Cartesian packets.
  assert torch.all(frame.points[~frame.valid] == 0)


def test_noise_and_quantization(device):
  """Range noise has the datasheet sigma; coordinates are whole millimetres."""
  sim, ideal = _setup(device, Mid360Cfg(noise=False))
  exact = _frame(sim, ideal)
  sim, noisy = _setup(device, Mid360Cfg(noise=True))
  meas = _frame(sim, noisy)

  both = exact.valid[0] & meas.valid[0]
  err = meas.points[0][both].norm(dim=-1) - exact.points[0][both].norm(dim=-1)
  dist = exact.points[0][both].norm(dim=-1)
  sigma = 0.03 + (0.02 - 0.03) * ((dist - 0.2) / 9.8).clamp(0, 1)
  z = err / sigma
  # False alarms (0.01 % of rays) land far off; the rest is the range noise.
  outliers = z.abs() > 6
  assert outliers.sum().item() <= 10
  assert z[~outliers].std().item() == pytest.approx(1.0, abs=0.05)
  mm = meas.points[0] * 1000
  assert torch.allclose(mm, mm.round(), atol=1e-2)


def test_range_depends_on_reflectivity(device):
  """At 50 m a 95 % floor still returns, a 5 % floor does not (limit 33 m)."""
  cfg = Mid360Cfg(noise=False)
  sim, lidar = _setup(device, cfg, height=50.0, grey=0.95)
  bright = _frame(sim, lidar)
  sim, lidar = _setup(device, cfg, height=50.0, grey=0.05)
  dark = _frame(sim, lidar)
  # Near-vertical rays only: incidence angle is small.
  assert bright.valid.sum() > 0
  assert dark.valid.sum() == 0
  assert bright.reflectivity[bright.valid].max() <= 150


def test_blind_zone(device):
  """Nothing is reported closer than 0.1 m."""
  sim, lidar = _setup(device, Mid360Cfg(noise=False), height=0.08)
  frame = _frame(sim, lidar)
  dist = frame.points[0][frame.valid[0]].norm(dim=-1)
  assert dist.numel() == 0 or dist.min() >= 0.1


def test_imu_at_rest(device):
  """Upside down at rest: -1 g on z, noise at the datasheet density."""
  cfg = Mid360Cfg()
  sim, _ = _setup(device, cfg)
  imu = Mid360Imu(sim.mj_model, "robot/mid360", cfg)
  samples = []
  for _ in range(400):
    sim.step()
    samples.append(imu.read(sim.data.sensordata))
  gyro = torch.cat([g for g, _ in samples])
  acc = torch.cat([a for _, a in samples])
  assert acc.mean(0).tolist() == pytest.approx([0.0, 0.0, -9.81 / 9.80665], abs=2e-3)
  expected = 100e-6 * math.sqrt(200)
  assert acc.std(0).mean().item() == pytest.approx(expected, rel=0.15)
  expected = math.radians(4.5e-3) * math.sqrt(200)
  assert gyro.std(0).mean().item() == pytest.approx(expected, rel=0.15)
