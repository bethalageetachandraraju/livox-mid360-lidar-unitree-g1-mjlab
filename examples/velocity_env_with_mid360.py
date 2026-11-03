"""Add the head Mid-360 to mjlab's G1 velocity task and read frames + IMU.

mjlab calls ``sim.sense()`` once per environment step (after the physics
substeps), so the sensor fires one env step of rays per call: 4000 rays every
20 ms at the stock 50 Hz env rate. Five env steps make one 10 Hz frame.

  uv run python examples/velocity_env_with_mid360.py --num-envs 4 --steps 50
"""

from __future__ import annotations

import argparse

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ObjRef
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from livox_mid360_mjlab import (
  G1_MID360_SITE,
  Mid360,
  Mid360Cfg,
  Mid360Imu,
  add_g1_mid360,
  mid360_raycast_cfg,
)


def make_cfg(num_envs: int, mid360: Mid360Cfg):
  cfg = unitree_g1_flat_env_cfg()
  cfg.scene.num_envs = num_envs

  # 1. Put the sensor (point cloud frame + IMU) on the robot's head.
  robot = cfg.scene.entities["robot"]
  base_spec_fn = robot.spec_fn

  def spec_with_mid360():
    spec = base_spec_fn()
    add_g1_mid360(spec)
    return spec

  robot.spec_fn = spec_with_mid360

  # 2. Ray sensor that fires one env step of Mid-360 rays per sense().
  step_dt = cfg.sim.mujoco.timestep * cfg.decimation
  lidar = mid360_raycast_cfg(
    "mid360",
    ObjRef(type="site", name=G1_MID360_SITE, entity="robot"),
    sense_dt=step_dt,
    cfg=mid360,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (lidar,)
  return cfg, step_dt


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--num-envs", type=int, default=4)
  p.add_argument("--steps", type=int, default=50)
  args = p.parse_args()

  mcfg = Mid360Cfg()
  cfg, step_dt = make_cfg(args.num_envs, mcfg)
  env = ManagerBasedRlEnv(
    cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu"
  )

  # 3. Measurement model + IMU on top of the env's sensor.
  mid360 = Mid360(env.scene["mid360"], env.sim.mj_model, step_dt, mcfg)
  imu = Mid360Imu(env.sim.mj_model, f"robot/{G1_MID360_SITE}", mcfg)

  env.reset()
  actions = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )
  t = 0.0
  for _ in range(args.steps):
    env.step(actions)
    t += step_dt
    gyro, accel = imu.read(env.sim.data.sensordata)  # One sample per env step here.
    frame = mid360.update(t)
    if frame is not None:
      n = frame.valid.sum(dim=1).tolist()
      print(
        f"t {frame.timebase:5.2f}s  frame {tuple(frame.points.shape)}  "
        f"returns per env {n}  accel env0 {accel[0].cpu().numpy().round(2)} g"
      )
  env.close()


if __name__ == "__main__":
  main()
