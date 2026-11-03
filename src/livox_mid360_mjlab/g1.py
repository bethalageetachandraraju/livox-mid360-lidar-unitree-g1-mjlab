"""Livox Mid-360 in the head of the Unitree G1."""

from __future__ import annotations

import math

import mujoco

from livox_mid360_mjlab.mid360 import add_mid360

# ``mid360_joint`` of Unitree's g1_29dof_rev_1_0.urdf: on torso_link, pitched
# 0.0401 rad nose down. The sensor sits upside down in the head, so its frame is
# that pose rolled by pi (x forward, z down): the Mid-360's -7..52 deg elevation
# window then looks from 7 deg above the horizon down to the feet.
G1_MID360_BODY = "torso_link"
G1_MID360_SITE = "mid360"
G1_MID360_POS = (0.0002835, 0.00003, 0.41618)
_HALF_PITCH = 0.5 * 0.04014257279586953
G1_MID360_QUAT = (0.0, math.cos(_HALF_PITCH), 0.0, -math.sin(_HALF_PITCH))


def add_g1_mid360(spec: mujoco.MjSpec) -> mujoco.MjsSite:
  """Add the head Mid-360 (point cloud frame site + built-in IMU) to a G1 spec."""
  return add_mid360(spec, G1_MID360_BODY, G1_MID360_POS, G1_MID360_QUAT, G1_MID360_SITE)
