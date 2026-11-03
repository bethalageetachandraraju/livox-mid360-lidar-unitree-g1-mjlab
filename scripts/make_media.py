"""Render the README images and GIFs offscreen (MuJoCo renderer, no browser).

Runs the demo's tour faster than real time, draws the Mid-360 frames and the
map as MuJoCo user geoms, and writes media/*.png and media/*.gif.

  uv run python scripts/make_media.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))

from mjlab.utils.lab_api.math import quat_apply  # noqa: E402

from lidar_sim import _LINE_COLORS, App, colormap  # noqa: E402
from livox_mid360_mjlab import LivoxPatternCfg, Mid360Frame  # noqa: E402

MEDIA = ROOT / "media"
W, H = 960, 540
EYE = np.eye(3).ravel()
_FONT = ImageFont.load_default(size=20)


def scan_pattern_figure() -> None:
  """Mid-360 coverage after 0.1 s, 0.5 s and 1 s (non-repetitive pattern)."""
  dirs = LivoxPatternCfg().load_directions("cpu").numpy()
  az = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0]))
  el = np.degrees(np.arcsin(dirs[:, 2]))
  fig, axes = plt.subplots(3, 1, figsize=(10, 6.4), sharex=True)
  for ax, secs in zip(axes, (0.1, 0.5, 1.0), strict=True):
    n = int(200_000 * secs)
    ax.scatter(az[:n], el[:n], s=0.05 if n > 50_000 else 0.2, c="#1f4e79", lw=0)
    ax.set_ylim(-8, 53)
    ax.set_xlim(-180, 180)
    ax.set_ylabel("elevation (deg)")
    ax.set_title(f"{secs:g} s  ({n:,} points)", fontsize=10, loc="left")
  axes[-1].set_xlabel("azimuth (deg), sensor frame")
  fig.suptitle("Livox Mid-360 scan pattern fills in over time", fontsize=12)
  fig.tight_layout()
  fig.savefig(MEDIA / "scan_pattern.png", dpi=110)
  plt.close(fig)


def label(img: Image.Image, text: str) -> Image.Image:
  draw = ImageDraw.Draw(img)
  w = draw.textlength(text, font=_FONT)
  draw.rectangle((0, 0, w + 24, 36), fill=(20, 20, 20))
  draw.text((12, 7), text, fill=(255, 255, 255), font=_FONT)
  return img


def save_gif(
  frames: list[Image.Image], path: Path, fps: float, width: int = 720
) -> None:
  size = (width, round(frames[0].height * width / frames[0].width))
  pal = [
    f.resize(size, Image.Resampling.LANCZOS).convert(
      "P", palette=Image.Palette.ADAPTIVE, colors=128
    )
    for f in frames
  ]
  pal[0].save(
    path, save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0
  )


def frame_points(f: Mid360Frame, raw: bool) -> tuple[np.ndarray, np.ndarray]:
  """World points of a frame (raw: at the frame's start pose) and validity."""
  valid = f.valid[0]
  if raw:
    pts = f.pos_w[0] + quat_apply(
      f.quat_w[0].expand(int(valid.sum()), 4), f.points[0][valid]
    )
  else:
    pts = f.points_w[0][valid]
  return pts.cpu().numpy(), valid


def colors_for(f: Mid360Frame, pts: np.ndarray, valid, mode: str) -> np.ndarray:
  if mode == "height":
    return colormap(pts[:, 2], -0.05, 2.5)
  if mode == "distance":
    return colormap(f.points[0][valid].norm(dim=-1).cpu().numpy(), 0.5, 12.0)
  if mode == "reflectivity":
    return colormap(f.reflectivity[0][valid].cpu().numpy(), 0, 150)
  return _LINE_COLORS[f.line[valid].long().cpu().numpy()]


class Painter:
  def __init__(self, app: App, width: int = W, height: int = H) -> None:
    self.app = app
    self.model = app.sim.mj_model
    # Outer walls off in the renders: their LiDAR points show where they are.
    for g in range(self.model.ngeom):
      name = self.model.geom(g).name
      if name.startswith(("arena/wall_", "arena/lintel")):
        self.model.geom_rgba[g, 3] = 0.0
    self.data = mujoco.MjData(self.model)
    self.renderer = mujoco.Renderer(self.model, height, width, max_geom=400_000)
    self.cam = mujoco.MjvCamera()
    self.opt = mujoco.MjvOption()

  def sync(self) -> None:
    self.data.qpos[:] = self.app.sim.data.qpos[0].cpu().numpy()
    mujoco.mj_kinematics(self.model, self.data)

  def chase(self, distance=4.2, az_offset=150.0, elevation=-28.0) -> None:
    b = self.app.base
    yaw = math.degrees(2 * math.atan2(b[6], b[3]))
    self.cam.lookat[:] = (b[0], b[1], 0.6)
    self.cam.distance, self.cam.azimuth, self.cam.elevation = (
      distance,
      yaw + az_offset,
      elevation,
    )

  def top(self, center, distance) -> None:
    self.cam.lookat[:] = center
    self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, 90.0, -89.9

  def render(self, pts, colors, size, box=False, arena=True, path=None) -> Image.Image:
    self.sync()
    self.opt.geomgroup[0] = int(arena)
    r = self.renderer
    r.update_scene(self.data, camera=self.cam, scene_option=self.opt)
    scn = r.scene
    kind = mujoco.mjtGeom.mjGEOM_BOX if box else mujoco.mjtGeom.mjGEOM_SPHERE
    half = np.array([size, size, size]) if box else np.array([size, 0, 0])
    rgba = np.ones(4)
    for p, c in zip(pts[: scn.maxgeom - scn.ngeom], colors, strict=False):
      rgba[:3] = c / 255.0
      mujoco.mjv_initGeom(scn.geoms[scn.ngeom], kind, half, p, EYE, rgba)
      scn.ngeom += 1
    for p in [] if path is None else path:
      mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.05, 0, 0]),
        p,
        EYE,
        np.array([1.0, 0.3, 0.8, 1.0]),
      )
      scn.ngeom += 1
    return Image.fromarray(r.render())


def main() -> None:
  MEDIA.mkdir(exist_ok=True)
  scan_pattern_figure()

  app = App(
    argparse.Namespace(
      checkpoint=str(ROOT / "demo/checkpoints/g1_walk_rough_42998.pt"),
      port=8092,
      voxel=0.05,
      manual=False,
      frame_rate=10.0,
      no_noise=False,
      duration=0.0,
    )
  )
  paint = Painter(app)
  big_paint = Painter(app, 1280, 720)
  app.reset()

  def run_until(t: float):
    frame = None
    while app.t_sim < t - 1e-9:
      frame = app.step() or frame
    return frame

  # Live scan GIF (10 Hz frames, real time).
  run_until(4.0)
  gif = []
  while app.t_sim < 10.0:
    f = run_until(app.t_sim + 0.1)
    pts, valid = frame_points(f, raw=True)
    paint.chase()
    gif.append(paint.render(pts, colors_for(f, pts, valid, "height"), 0.012))
  save_gif(gif, MEDIA / "live_scan.gif", fps=10)

  # The colour modes, all from one frame.
  f = run_until(20.0)
  pts, valid = frame_points(f, raw=True)
  tiles = []
  for mode in ("height", "reflectivity", "laser line", "distance"):
    paint.chase(distance=5.0, az_offset=160, elevation=-24)
    img = paint.render(pts, colors_for(f, pts, valid, mode), 0.012)
    tiles.append(label(img, f"color by {mode}"))
  grid = Image.new("RGB", (2 * W + 8, 2 * H + 8), (255, 255, 255))
  for i, t in enumerate(tiles):
    grid.paste(t, ((i % 2) * (W + 8), (i // 2) * (H + 8)))
  grid.save(MEDIA / "color_modes.png")

  # Map building from a fresh start, plus raw vs deskewed while turning.
  for h in app.map_chunks:
    h.remove()
  app.map_chunks, app.path = [], []
  app.map = type(app.map)(app.map.voxel)
  app.reset()
  start = app.t_sim
  map_gif = []
  turned = False
  while app.t_sim - start < 100.0:
    run_until(app.t_sim + (0.5 if app.t_sim - start < 10.0 else 3.0))
    app.flush_map()
    vox = np.concatenate([h.batched_positions for h in app.map_chunks])
    paint.cam.lookat[:] = (4.0, 0.0, 0.0)
    paint.cam.distance, paint.cam.azimuth, paint.cam.elevation = 17.0, 90.0, -58.0
    trail = np.array(app.path[::3])
    trail[:, 2] = 0.1
    map_gif.append(
      paint.render(
        vox,
        colormap(vox[:, 2], -0.05, 2.5),
        0.025,
        box=True,
        arena=False,
        path=trail,
      )
    )
    if not turned and app.t_sim - start > 62.0:
      turned = True
      app.gui_mode.value, app.gui_wz.value = "Manual", 0.8
      f = run_until(app.t_sim + 2.0)
      b = app.base
      side = []
      for raw, text in (
        (True, "raw frame (as /livox/lidar), turning at 0.8 rad/s"),
        (False, "deskewed with each point's true pose"),
      ):
        pts, valid = frame_points(f, raw=raw)
        paint.top((b[0], b[1], 0.0), 11.0)
        side.append(
          label(paint.render(pts, colors_for(f, pts, valid, "height"), 0.03), text)
        )
      both = Image.new("RGB", (2 * W + 8, H), (255, 255, 255))
      both.paste(side[0], (0, 0))
      both.paste(side[1], (W + 8, 0))
      both.save(MEDIA / "raw_vs_deskewed.png")
      app.gui_wz.value, app.gui_mode.value = 0.0, "Tour"
  save_gif(map_gif, MEDIA / "map_build.gif", fps=4)

  app.flush_map()
  vox = np.concatenate([h.batched_positions for h in app.map_chunks])
  big_paint.cam.lookat[:] = (4.0, 0.0, 0.0)
  big_paint.cam.distance = 16.0
  big_paint.cam.azimuth, big_paint.cam.elevation = 115.0, -45.0
  big_paint.render(
    vox, colormap(vox[:, 2], -0.05, 2.5), 0.025, box=True, arena=False
  ).save(MEDIA / "map.png")
  print("saved:", sorted(p.name for p in MEDIA.iterdir()))


if __name__ == "__main__":
  main()
