"""Shared helpers: a one-entity mjlab scene with sensors wired up."""

from __future__ import annotations

import mujoco
import pytest
import torch
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg


@pytest.fixture(scope="session")
def device() -> str:
  return "cuda" if torch.cuda.is_available() else "cpu"


def make_scene_and_sim(
  device: str, xml: str, sensors: tuple, sim_cfg: SimulationCfg
) -> tuple[Scene, Simulation]:
  scene = Scene(
    SceneCfg(
      num_envs=1,
      entities={"robot": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))},
      sensors=sensors,
    ),
    device,
  )
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=scene.compile(), device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  assert scene.sensor_context is not None
  sim.set_sensor_context(scene.sensor_context)
  return scene, sim
