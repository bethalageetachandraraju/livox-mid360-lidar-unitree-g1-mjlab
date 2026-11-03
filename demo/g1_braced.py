"""Braced Unitree G1 29-DoF, built from mjlab's stock G1 MJCF.

The robot used in the demo:

- rubber hands removed (149 g taken out of each ``*_wrist_yaw_link``);
- wrist pitch and wrist yaw joints deleted (the brace welds them at 0);
- brace: cylinder r 40 mm on ``*_wrist_roll_link`` from x 10 mm to 125.5 mm, 0.15 kg;
- forearm extension: capsule r 25 mm, 100 mm long from the wrist-yaw flange,
  0.10 kg, domed tip site ``*_hand_contact`` (elbow axis to tip 325.5 mm);
- 25 position actuators in policy order with the trained kp / kd / peak torque.

Total mass 33.54 kg.
"""

from __future__ import annotations

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

# Policy order = MuJoCo order.
JOINTS = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
)
NUM_JOINTS = len(JOINTS)

# Hardware geometry.
BRACE_RADIUS = 0.040
BRACE_X = (0.010, 0.1255)
BRACE_MASS = 0.15
EXTENSION_RADIUS = 0.025
EXTENSION_LENGTH = 0.100
EXTENSION_MASS = 0.10
FLANGE_X = 0.0415  # Hand mounting flange on *_wrist_yaw_link.
HAND_MASS = 0.149


# Actuators: armature from the rotor inertias (mjlab g1_constants), gains from
# armature * (2 pi 10 Hz)^2 with damping ratio 2.
def _reflected_inertia(rotor: tuple[float, ...], gear: tuple[float, ...]) -> float:
  return rotor[0] * (gear[1] * gear[2]) ** 2 + rotor[1] * gear[2] ** 2 + rotor[2]


_ARM_5020 = _reflected_inertia((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 4.5))
_ARM_7520_14 = _reflected_inertia((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22))
_ARM_7520_22 = _reflected_inertia((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))
_OMEGA = 10 * 2.0 * 3.1415926535
# (armature, peak torque) per motor class.
_MOTORS = {
  "7520_14": (_ARM_7520_14, 88.0),
  "7520_22": (_ARM_7520_22, 139.0),
  "linkage": (2 * _ARM_5020, 50.0),
  "5020": (_ARM_5020, 25.0),
}


def _motor_class(joint: str) -> str:
  if any(k in joint for k in ("hip_pitch", "hip_yaw", "waist_yaw")):
    return "7520_14"
  if any(k in joint for k in ("hip_roll", "knee")):
    return "7520_22"
  if any(k in joint for k in ("ankle", "waist_roll", "waist_pitch")):
    return "linkage"
  return "5020"


ARMATURE = np.array([_MOTORS[_motor_class(j)][0] for j in JOINTS])
KP = ARMATURE * _OMEGA**2
KD = 2.0 * 2.0 * ARMATURE * _OMEGA
EFFORT_LIMIT = np.array([_MOTORS[_motor_class(j)][1] for j in JOINTS])
# Walk action scale = 0.25 * peak / kp (mjlab G1_ACTION_SCALE).
WALK_ACTION_SCALE = 0.25 * EFFORT_LIMIT / KP

# Stand pose = mjlab KNEES_BENT keyframe.
STAND_POSE = np.zeros(NUM_JOINTS)
for _name, _val in {
  "left_hip_pitch_joint": -0.312,
  "right_hip_pitch_joint": -0.312,
  "left_knee_joint": 0.669,
  "right_knee_joint": 0.669,
  "left_ankle_pitch_joint": -0.363,
  "right_ankle_pitch_joint": -0.363,
  "left_shoulder_pitch_joint": 0.2,
  "right_shoulder_pitch_joint": 0.2,
  "left_shoulder_roll_joint": 0.2,
  "right_shoulder_roll_joint": -0.2,
  "left_elbow_joint": 0.6,
  "right_elbow_joint": 0.6,
}.items():
  STAND_POSE[JOINTS.index(_name)] = _val
STAND_HEIGHT = 0.76

_LIMB_BODIES = (
  "elbow_link",
  "wrist_roll_link",
  "wrist_pitch_link",
  "wrist_yaw_link",
  "wrist_brace",
  "forearm_extension",
)


def _collision_geom(geom: mujoco.MjsGeom) -> None:
  geom.contype = 1
  geom.conaffinity = 1
  geom.condim = 3
  geom.priority = 1
  geom.friction[:] = (0.6, 0.005, 0.0001)
  geom.group = 3
  geom.rgba[:] = (0.2, 0.6, 0.2, 0.3)


def _brace_arm(spec: mujoco.MjSpec, side: str) -> None:
  for j in ("wrist_pitch", "wrist_yaw"):
    spec.delete(spec.joint(f"{side}_{j}_joint"))

  # Hand off: drop the rubber hand mesh and its collision capsule.
  yaw = spec.body(f"{side}_wrist_yaw_link")
  for g in list(yaw.geoms):
    if g.meshname == f"{side}_rubber_hand" or g.name == f"{side}_hand_collision":
      spec.delete(g)
  for s in list(yaw.sites):
    if s.name == f"{side}_palm":
      spec.delete(s)
  # Motor link keeps the rest of the mass, centred near the yaw motor.
  m_old = yaw.mass
  yaw.mass = m_old - HAND_MASS
  yaw.ipos[:] = (0.02, 0.0, 0.0)
  yaw.inertia[:] = np.asarray(yaw.inertia) * (yaw.mass / m_old)

  # The wrist-pitch capsule sits inside the brace; the brace replaces it.
  spec.delete(spec.geom(f"{side}_wrist_collision"))

  # Brace on the wrist-roll link, spanning the welded pitch / yaw motors.
  roll = spec.body(f"{side}_wrist_roll_link")
  brace = roll.add_body(name=f"{side}_wrist_brace")
  length = BRACE_X[1] - BRACE_X[0]
  brace.mass = BRACE_MASS
  brace.ipos[:] = (0.5 * (BRACE_X[0] + BRACE_X[1]), 0.0, 0.0)
  r2 = BRACE_RADIUS**2
  brace.inertia[:] = (
    0.5 * BRACE_MASS * r2,
    BRACE_MASS * (3 * r2 + length**2) / 12,
    BRACE_MASS * (3 * r2 + length**2) / 12,
  )
  brace.explicitinertial = True
  fromto = (BRACE_X[0], 0.0, 0.0, BRACE_X[1], 0.0, 0.0)
  g = brace.add_geom(
    name=f"{side}_wrist_brace_collision",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=(BRACE_RADIUS, 0, 0),
    fromto=fromto,
  )
  _collision_geom(g)
  brace.add_geom(
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=(BRACE_RADIUS, 0, 0),
    fromto=fromto,
    group=2,
    contype=0,
    conaffinity=0,
    density=0,
    rgba=(0.95, 0.55, 0.15, 1.0),
  )

  # Forearm extension on the hand flange. Domed tip at x = EXTENSION_LENGTH.
  ext = yaw.add_body(name=f"{side}_forearm_extension", pos=(FLANGE_X, 0.0, 0.0))
  ext.mass = EXTENSION_MASS
  ext.ipos[:] = (0.5 * EXTENSION_LENGTH, 0.0, 0.0)
  r2 = EXTENSION_RADIUS**2
  ext.inertia[:] = (
    0.5 * EXTENSION_MASS * r2,
    EXTENSION_MASS * (3 * r2 + EXTENSION_LENGTH**2) / 12,
    EXTENSION_MASS * (3 * r2 + EXTENSION_LENGTH**2) / 12,
  )
  ext.explicitinertial = True
  seg = (0.0, 0.0, 0.0, EXTENSION_LENGTH - EXTENSION_RADIUS, 0.0, 0.0)
  g = ext.add_geom(
    name=f"{side}_extension_collision",
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    size=(EXTENSION_RADIUS, 0, 0),
    fromto=seg,
  )
  _collision_geom(g)
  ext.add_geom(
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    size=(EXTENSION_RADIUS, 0, 0),
    fromto=seg,
    group=2,
    contype=0,
    conaffinity=0,
    density=0,
    rgba=(0.9, 0.9, 0.9, 1.0),
  )
  ext.add_site(
    name=f"{side}_hand_contact",
    pos=(EXTENSION_LENGTH, 0.0, 0.0),
    size=(0.012, 0, 0),
    group=5,
    rgba=(1, 0, 0, 1),
  )

  # The welded forearm chain is one rigid piece: no internal contacts.
  names = [f"{side}_{b}" for b in _LIMB_BODIES]
  for i, a in enumerate(names):
    for b in names[i + 1 :]:
      spec.add_exclude(bodyname1=a, bodyname2=b)


def build_spec(sim_dt: float = 0.005) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  spec.modelname = "g1_29dof_braced"

  # Same solver settings as the training envs (mjlab velocity env cfg).
  spec.option.timestep = sim_dt
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  spec.option.impratio = 1.0
  spec.option.iterations = 10
  spec.option.ls_iterations = 20

  for side in ("left", "right"):
    _brace_arm(spec, side)

  # Robot-ground contacts: frictional, robot friction wins (priority).
  for g in spec.geoms:
    if g.name.endswith("_collision"):
      _collision_geom(g)

  # Actuators in policy order: ctrl[i] drives JOINTS[i].
  for name, arm, kp, kd, tau in zip(
    JOINTS, ARMATURE, KP, KD, EFFORT_LIMIT, strict=True
  ):
    spec.joint(name).armature = float(arm)
    act = spec.add_actuator(name=name, target=name)
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.gainprm[0] = kp
    act.biasprm[1] = -kp
    act.biasprm[2] = -kd
    act.inheritrange = 0.0
    act.ctrllimited = False
    act.forcelimited = True
    act.forcerange[:] = (-tau, tau)

  # IMU orientation (gyro "imu_ang_vel" already exists on imu_in_pelvis).
  spec.add_sensor(
    name="imu_quat",
    type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu_in_pelvis",
  )

  # Flat ground and a bit of scenery.
  spec.visual.headlight.ambient[:] = (0.3, 0.3, 0.3)
  spec.visual.headlight.diffuse[:] = (0.6, 0.6, 0.6)
  spec.visual.global_.offwidth = 1280
  spec.visual.global_.offheight = 720
  spec.add_texture(
    name="grid",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=(0.25, 0.28, 0.32),
    rgb2=(0.18, 0.2, 0.23),
    width=512,
    height=512,
    mark=mujoco.mjtMark.mjMARK_EDGE,
    markrgb=(0.5, 0.5, 0.5),
  )
  mat = spec.add_material(name="grid", texrepeat=(1, 1), texuniform=True)
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
  spec.add_texture(
    name="sky",
    type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
    rgb1=(0.4, 0.55, 0.7),
    rgb2=(0.05, 0.05, 0.1),
    width=256,
    height=256,
  )
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(0, 0, 0.05),
    material="grid",
  )
  spec.worldbody.add_light(
    pos=(0, 0, 4), dir=(0, 0, -1), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
  )
  return spec


def build_model(sim_dt: float = 0.005) -> mujoco.MjModel:
  return build_spec(sim_dt).compile()


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate v by the inverse of unit quaternion q (w, x, y, z)."""
  w, xyz = q[0], q[1:]
  t = 2.0 * np.cross(xyz, v)
  return v - w * t + np.cross(xyz, t)
