"""Livox scan pattern on stock mjlab's RayCastSensor, and the G1 mount."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch
from conftest import make_scene_and_sim
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from mjlab.sensor import ObjRef
from mjlab.sim import MujocoCfg, SimulationCfg

from livox_mid360_mjlab import (
  MID360_IMU_POS,
  LivoxPatternCfg,
  LivoxRayCastSensorCfg,
  add_g1_mid360,
)

# Sensor upside down 1.9 m over the floor, like the G1 head LiDAR.
_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1"/>
    <body name="base" pos="0 0 2">
      <freejoint/>
      <geom type="box" size="0.2 0.2 0.1" mass="5.0"/>
      <site name="lidar" pos="0 0 -0.1" quat="0 1 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_scans_advance_through_the_table(device):
  cfg = LivoxRayCastSensorCfg(
    name="mid360",
    frame=ObjRef(type="site", name="lidar", entity="robot"),
    pattern=LivoxPatternCfg(samples=1000),
    max_distance=50.0,
  )
  sim_cfg = SimulationCfg(mujoco=MujocoCfg(gravity=(0, 0, 0)), njmax=20)
  scene, sim = make_scene_and_sim(device, _XML, (cfg,), sim_cfg)
  sensor = scene["mid360"]
  assert sensor.num_rays == 1000

  sim.step()
  sim.sense()
  first = sensor.data.distances.clone()
  sim.sense()
  second = sensor.data.distances.clone()

  # Local +z points down, so exactly the rays above the sensor's horizon hit.
  table = cfg.pattern.load_directions(device)
  for scan, dirs in ((first, table[:1000]), (second, table[1000:2000])):
    hit = scan[0] >= 0
    assert hit.any()
    assert torch.equal(hit, dirs[:, 2] > 0)
    assert torch.allclose(scan[0, hit], 1.9 / dirs[hit, 2], rtol=1e-3)
  assert not torch.equal(first, second)
  assert torch.equal(sensor.local_directions, table[1000:2000])


def test_g1_mount():
  """Head sensor: x forward pitched 2.3 deg down, z down, IMU at the spec offset."""
  spec = get_spec()
  add_g1_mid360(spec)
  m = spec.compile()
  d = mujoco.MjData(m)
  mujoco.mj_kinematics(m, d)
  lidar, imu = m.site("mid360").id, m.site("mid360_imu").id
  rot = d.site_xmat[lidar].reshape(3, 3)
  assert rot[:, 0] == pytest.approx([0.9992, 0.0, -0.0401], abs=1e-3)
  assert rot[:, 2] == pytest.approx([-0.0401, 0.0, -0.9992], abs=1e-3)
  offset = rot.T @ (d.site_xpos[imu] - d.site_xpos[lidar])
  np.testing.assert_allclose(offset, MID360_IMU_POS, atol=1e-6)
