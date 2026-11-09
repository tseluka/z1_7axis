"""Shared MuJoCo helpers for the mjt CLI: loading, state setup, naming, rendering."""

import json
import os
import sys
import time
from pathlib import Path

# Untouched environment for launching the GUI viewer (importing mujoco below sets MUJOCO_GL,
# PYOPENGL_PLATFORM and SDL_VIDEODRIVER, which break a GLFW window if inherited).
CLEAN_ENV = dict(os.environ)

# The GL backend is picked when mujoco is imported, so this must run first.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "out"

OBJ = mujoco.mjtObj
JNT_TYPE = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
JNT_NQ = {0: 7, 1: 4, 2: 1, 3: 1}


def die(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load(path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    p = Path(path)
    if not p.exists():
        die(f"model file not found: {p}")
    try:
        m = mujoco.MjModel.from_binary_path(str(p)) if p.suffix == ".mjb" else mujoco.MjModel.from_xml_path(str(p))
    except ValueError as e:
        die(f"MuJoCo failed to compile {p}:\n{e}")
    return m, mujoco.MjData(m)


def name(m, objtype, i: int) -> str:
    """Object name, or a readable fallback like 'geom7@link02' for unnamed objects."""
    n = mujoco.mj_id2name(m, objtype, i)
    if n:
        return n
    if objtype == OBJ.mjOBJ_GEOM:
        return f"geom{i}@{name(m, OBJ.mjOBJ_BODY, m.geom_bodyid[i])}"
    if objtype == OBJ.mjOBJ_BODY and i == 0:
        return "world"
    return f"{objtype.name.removeprefix('mjOBJ_').lower()}{i}"


def obj_id(m, objtype, n: str) -> int:
    i = mujoco.mj_name2id(m, objtype, n)
    if i < 0:
        kind = objtype.name.removeprefix("mjOBJ_").lower()
        names = [name(m, objtype, k) for k in range(_count(m, objtype))]
        die(f"no {kind} named '{n}'. Available: {', '.join(names)}")
    return i


def _count(m, objtype) -> int:
    return {OBJ.mjOBJ_BODY: m.nbody, OBJ.mjOBJ_JOINT: m.njnt, OBJ.mjOBJ_SITE: m.nsite, OBJ.mjOBJ_ACTUATOR: m.nu,
            OBJ.mjOBJ_KEY: m.nkey, OBJ.mjOBJ_CAMERA: m.ncam, OBJ.mjOBJ_GEOM: m.ngeom}[objtype]


def parse_assign(s: str) -> list[tuple[str, float]]:
    """'a=1,b=-0.5' -> [('a', 1.0), ('b', -0.5)]"""
    out = []
    for part in filter(None, (x.strip() for x in s.split(","))):
        if "=" not in part:
            die(f"expected name=value, got '{part}'")
        k, v = part.split("=", 1)
        out.append((k.strip(), float(v)))
    return out


def set_state(m, d, key: str | None = None, qpos: str | None = None):
    """Reset, optionally load a keyframe, then apply a qpos override, then mj_forward.

    qpos is 'joint2=0.5,joint3=-1' (hinge/slide joints by name), a raw list of nq numbers, or 'live'
    (the pose currently shown in a running `mjt view` window).
    """
    mujoco.mj_resetData(m, d)
    if key:
        mujoco.mj_resetDataKeyframe(m, d, obj_id(m, OBJ.mjOBJ_KEY, key))
    if qpos == "live":
        qpos = ",".join(f"{k}={v}" for k, v in live_state()["qpos"].items())
    if qpos:
        if "=" in qpos:
            for jn, v in parse_assign(qpos):
                j = obj_id(m, OBJ.mjOBJ_JOINT, jn)
                if m.jnt_type[j] not in (2, 3):
                    die(f"joint '{jn}' is {JNT_TYPE[m.jnt_type[j]]}; use a raw qpos list for it")
                d.qpos[m.jnt_qposadr[j]] = v
        else:
            vals = [float(x) for x in qpos.replace(",", " ").split()]
            if len(vals) != m.nq:
                die(f"raw qpos needs nq={m.nq} values, got {len(vals)}")
            d.qpos[:] = vals
    mujoco.mj_forward(m, d)


def live_state() -> dict:
    p = OUT / "viewer_state.json"
    if not p.exists():
        die(f"{p} not found; start the viewer with `uv run mjt view MODEL`")
    st = json.loads(p.read_text())
    if time.time() - st["wall_time"] > 2:
        print(f"note: viewer state is {time.time() - st['wall_time']:.0f}s old (viewer closed?)", file=sys.stderr)
    return st


def qpos_string(m, q) -> str:
    """Format qpos as a reusable --qpos 'j=v,...' string (hinge/slide joints only)."""
    return ",".join(f"{name(m, OBJ.mjOBJ_JOINT, j)}={q[m.jnt_qposadr[j]]:.4f}"
                    for j in range(m.njnt) if m.jnt_type[j] in (2, 3))


def contacts(m, d) -> list[dict]:
    out = []
    for c in d.contact[: d.ncon]:
        out.append(dict(g1=name(m, OBJ.mjOBJ_GEOM, c.geom1), g2=name(m, OBJ.mjOBJ_GEOM, c.geom2),
                        b1=name(m, OBJ.mjOBJ_BODY, m.geom_bodyid[c.geom1]),
                        b2=name(m, OBJ.mjOBJ_BODY, m.geom_bodyid[c.geom2]), dist=float(c.dist), pos=c.pos.copy()))
    return out


# ---------------------------------------------------------------- rendering

# Free-camera presets: (azimuth, elevation). Camera sits opposite its viewing direction,
# so az=180 places it on +x looking back toward -x ("front").
VIEWS = {"front": (180, -10), "back": (0, -10), "left": (270, -10), "right": (90, -10),
         "top": (90, -89.9), "iso": (135, -25), "iso2": (-45, -25)}


def make_renderer(m, w: int, h: int) -> mujoco.Renderer:
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
    m.vis.global_.offheight = max(m.vis.global_.offheight, h)
    try:
        return mujoco.Renderer(m, h, w)
    except Exception as e:  # noqa: BLE001
        die(f"could not create renderer with MUJOCO_GL={os.environ['MUJOCO_GL']}: {e}\n"
            "Try MUJOCO_GL=osmesa (needs mesa/libosmesa).")


def scene_option(frames: str = "none", contacts: bool = False, joints: bool = False, collision: bool = False,
                 transparent: bool = False, com: bool = False, labels: str = "none") -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.frame = getattr(mujoco.mjtFrame, f"mjFRAME_{frames.upper()}")
    opt.label = getattr(mujoco.mjtLabel, f"mjLABEL_{labels.upper()}")
    f = mujoco.mjtVisFlag
    opt.flags[f.mjVIS_CONTACTPOINT] = contacts
    opt.flags[f.mjVIS_CONTACTFORCE] = contacts
    opt.flags[f.mjVIS_JOINT] = joints
    opt.flags[f.mjVIS_TRANSPARENT] = transparent
    opt.flags[f.mjVIS_COM] = com
    if collision:  # Menagerie convention: group 2 = visual meshes, group 3 = collision primitives
        opt.geomgroup[3] = 1
        opt.geomgroup[2] = 0
    return opt


def camera_for(m, view: str, zoom: float = 1.0):
    """'iso' / 'front' / ... free camera, or 'cam:NAME' / 'cam:ID' fixed model camera."""
    if view.startswith("cam:"):
        ref = view[4:]
        cid = int(ref) if ref.isdigit() else obj_id(m, OBJ.mjOBJ_CAMERA, ref)
        return cid
    if view not in VIEWS:
        die(f"unknown view '{view}'. Use one of {', '.join(VIEWS)} or cam:NAME")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = m.stat.center
    cam.distance = 2.2 * m.stat.extent / zoom
    cam.azimuth, cam.elevation = VIEWS[view]
    return cam


def render_views(m, d, r: mujoco.Renderer, views: list[str], opt, zoom: float = 1.0) -> list[Image.Image]:
    imgs = []
    for v in views:
        r.update_scene(d, camera=camera_for(m, v, zoom), scene_option=opt)
        imgs.append(label(Image.fromarray(r.render()), v))
    return imgs


def label(img: Image.Image, text: str) -> Image.Image:
    dr = ImageDraw.Draw(img)
    dr.rectangle((0, 0, 8 + 7 * len(text), 18), fill=(0, 0, 0))
    dr.text((4, 3), text, fill=(255, 255, 0))
    return img


def tile(imgs: list[Image.Image], cols: int | None = None) -> Image.Image:
    cols = cols or min(len(imgs), 3)
    rows = -(-len(imgs) // cols)
    w, h = imgs[0].size
    sheet = Image.new("RGB", (cols * w, rows * h), (30, 30, 30))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    return sheet


def save(img: Image.Image, out: str) -> Path:
    p = Path(out) if "/" in out else OUT / out
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(p)
    return p


def fmt(v, p: int = 3) -> str:
    return " ".join(f"{x:.{p}f}" for x in np.atleast_1d(v))
