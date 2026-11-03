"""Walk policy for the braced G1: RSL-RL actor evaluated in numpy.

obs = [gravity_b(3), gyro_b(3), cmd(3), q - STAND(25), dq(25), last_action(25)]
target = STAND + WALK_ACTION_SCALE * clip(action, -5, 5)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from g1_braced import NUM_JOINTS, WALK_ACTION_SCALE

# Command ranges the policy was trained on: vx, vy (m/s), wz (rad/s).
WALK_CMD_LO = np.array([-0.6, -0.5, -0.8])
WALK_CMD_HI = np.array([1.0, 0.5, 0.8])


class MLPActor:
  """Deterministic RSL-RL actor (MLPModel) evaluated in numpy."""

  def __init__(self, checkpoint: str | Path) -> None:
    import torch  # Only needed to unpickle the checkpoint.

    ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"]
    idx = sorted(
      {int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and "weight" in k}
    )
    self.layers = [
      (
        sd[f"mlp.{i}.weight"].numpy().astype(np.float32),
        sd[f"mlp.{i}.bias"].numpy().astype(np.float32),
      )
      for i in idx
    ]
    self.obs_dim = self.layers[0][0].shape[1]
    self.act_dim = self.layers[-1][0].shape[0]
    if "obs_normalizer._mean" in sd:
      self.mean = sd["obs_normalizer._mean"].numpy().reshape(-1).astype(np.float32)
      self.std = sd["obs_normalizer._std"].numpy().reshape(-1).astype(np.float32)
    else:
      self.mean = None
      self.std = None
    self.iteration = ck.get("iter")
    self.path = Path(checkpoint)

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    x = obs.astype(np.float32)
    if self.mean is not None:
      x = (x - self.mean) / (self.std + 1e-2)
    for i, (w, b) in enumerate(self.layers):
      x = w @ x + b
      if i < len(self.layers) - 1:
        x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))  # ELU
    return x.astype(np.float64)

  def describe(self) -> str:
    hidden = "-".join(str(w.shape[0]) for w, _ in self.layers[:-1])
    norm = "normalizer" if self.mean is not None else "no normalizer"
    return (
      f"{self.path.name} (iter {self.iteration}, {self.obs_dim} in, {hidden}, {norm})"
    )


@dataclass
class RobotState:
  q: np.ndarray  # (25,) joint positions, policy order
  dq: np.ndarray  # (25,) joint velocities
  gravity_b: np.ndarray  # (3,) gravity unit vector in pelvis frame
  gyro_b: np.ndarray  # (3,) pelvis angular velocity, pelvis frame


class _Base:
  clip: float
  expected_obs: int

  def __init__(self, actor: MLPActor) -> None:
    if actor.obs_dim != self.expected_obs or actor.act_dim != NUM_JOINTS:
      raise ValueError(
        f"{type(self).__name__}: checkpoint has {actor.obs_dim} inputs / "
        f"{actor.act_dim} outputs, expected {self.expected_obs} / {NUM_JOINTS}"
      )
    self.actor = actor
    self.last_action = np.zeros(NUM_JOINTS)
    self.target = np.zeros(NUM_JOINTS)

  def reset(self) -> None:
    self.last_action[:] = 0.0

  def _act(self, obs: np.ndarray) -> np.ndarray:
    a = np.clip(self.actor(obs), -self.clip, self.clip)
    self.last_action = a
    return a


class WalkPolicy(_Base):
  """obs = [g(3), w(3), cmd(3), q-STAND, dq, last_a] -> STAND + scale*a."""

  clip = 5.0
  expected_obs = 84

  def __init__(self, actor: MLPActor, stand_pose: np.ndarray) -> None:
    super().__init__(actor)
    self.stand = stand_pose
    self.cmd = np.zeros(3)  # vx (m/s), vy (m/s, + left), wz (rad/s, + CCW)

  def compute(self, s: RobotState) -> np.ndarray:
    cmd = np.clip(self.cmd, WALK_CMD_LO, WALK_CMD_HI)
    obs = np.concatenate(
      [s.gravity_b, s.gyro_b, cmd, s.q - self.stand, s.dq, self.last_action]
    )
    a = self._act(obs)
    self.target = self.stand + WALK_ACTION_SCALE * a
    return self.target
