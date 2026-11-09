"""Classic MuJoCo viewer with a file bridge to Claude / the mjt CLI, and hot reload.

Launched by `mjt view MODEL` (run directly it needs MUJOCO_GL/PYOPENGL_PLATFORM unset).

  out/viewer_state.json  <- written ~5x/s: model, time, pause state, named qpos/ctrl, contacts.
                            `mjt render ... -q live` renders exactly the pose on screen.
  out/viewer_cmd.json    -> applied next frame, then deleted (`mjt pose` writes it). Any of:
                            {"key": "home", "qpos": {"joint2": 0.5}, "ctrl": {"motor1": 0.3}, "pause": true}

Keys (on top of the viewer's own): Space = pause/play our sim loop, Backspace = reset to first keyframe.
Saving any .xml / mesh under the model's directory reloads the model (the window reopens).
"""

import json
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

OUT = Path(__file__).resolve().parent.parent / "out"
STATE = OUT / "viewer_state.json"
CMD = OUT / "viewer_cmd.json"
WATCH_EXT = {".xml", ".stl", ".obj", ".msh", ".png"}
KEY_SPACE, KEY_BACKSPACE = 32, 259


def _id(m, objtype, n):
    i = mujoco.mj_name2id(m, objtype, n)
    if i < 0:
        raise KeyError(f"no {objtype.name[6:].lower()} named {n!r}")
    return i


def _names(m, objtype, n):
    return [mujoco.mj_id2name(m, objtype, i) or f"{i}" for i in range(n)]


def _snapshot(root: Path) -> dict:
    return {p: p.stat().st_mtime for p in root.rglob("*") if p.suffix.lower() in WATCH_EXT}


class Bridge:
    def __init__(self, path: Path):
        self.path = path
        self.paused = False
        self.reset_requested = False
        self.last_cmd = None
        self.reload_error = None

    def on_key(self, key: int):
        if key == KEY_SPACE:
            self.paused = not self.paused
        elif key == KEY_BACKSPACE:
            self.reset_requested = True

    def apply(self, m, d):
        cmd, cid = {}, None
        try:
            cmd = json.loads(CMD.read_text())
            CMD.unlink()
            cid = cmd.pop("id", None)
            if "key" in cmd:
                mujoco.mj_resetDataKeyframe(m, d, _id(m, mujoco.mjtObj.mjOBJ_KEY, cmd["key"]))
            for jn, v in cmd.get("qpos", {}).items():
                d.qpos[m.jnt_qposadr[_id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]] = v
            if cmd.get("qpos") or "key" in cmd:
                d.qvel[:] = 0
            for an, v in cmd.get("ctrl", {}).items():
                d.ctrl[_id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, an)] = v
            if "pause" in cmd:
                self.paused = bool(cmd["pause"])
            mujoco.mj_forward(m, d)
            self.last_cmd = {"ok": True, "cmd": cmd, "id": cid}
        except Exception as e:  # noqa: BLE001 - report bad commands instead of killing the viewer
            self.last_cmd = {"ok": False, "error": f"{type(e).__name__}: {e}", "id": cid or cmd.get("id")}

    def write(self, m, d):
        joints = _names(m, mujoco.mjtObj.mjOBJ_JOINT, m.njnt)
        state = {
            "model_path": str(self.path),
            "wall_time": time.time(),
            "sim_time": d.time,
            "paused": self.paused,
            "qpos": {joints[j]: d.qpos[m.jnt_qposadr[j]] for j in range(m.njnt) if m.jnt_type[j] in (2, 3)},
            "qpos_raw": d.qpos.tolist(),
            "ctrl": dict(zip(_names(m, mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu), d.ctrl.tolist(), strict=True)),
            "actuator_force": d.actuator_force.tolist(),
            "contacts": [{"geom1": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}",
                          "geom2": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}",
                          "dist": c.dist} for c in d.contact[: d.ncon]],
            "last_cmd": self.last_cmd,
            "reload_error": self.reload_error,
        }
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        os.replace(tmp, STATE)


def run(path: Path):
    OUT.mkdir(exist_ok=True)
    bridge = Bridge(path)
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    mtimes = _snapshot(path.parent)

    while True:
        reload_to = None
        with mujoco.viewer.launch_passive(m, d, key_callback=bridge.on_key) as v:
            last_write = last_check = 0.0
            wall0, sim0 = time.perf_counter(), d.time
            while v.is_running():
                now = time.perf_counter()
                with v.lock():
                    if CMD.exists():
                        bridge.apply(m, d)
                        wall0, sim0, last_write = now, d.time, 0.0
                    if bridge.reset_requested:
                        bridge.reset_requested = False
                        mujoco.mj_resetDataKeyframe(m, d, 0) if m.nkey else mujoco.mj_resetData(m, d)
                        mujoco.mj_forward(m, d)
                        wall0, sim0 = now, d.time
                    if bridge.paused:
                        wall0, sim0 = now, d.time
                    else:  # real-time pacing
                        while d.time - sim0 < now - wall0:
                            mujoco.mj_step(m, d)
                    if now - last_write > 0.2:
                        last_write = now
                        bridge.write(m, d)
                v.sync()
                if now - last_check > 0.5:
                    last_check = now
                    snap = _snapshot(path.parent)
                    if snap != mtimes:
                        mtimes = snap
                        try:
                            reload_to = mujoco.MjModel.from_xml_path(str(path))
                            bridge.reload_error = None
                            break
                        except ValueError as e:
                            bridge.reload_error = str(e)
                            print(f"reload failed, keeping previous model:\n{e}", file=sys.stderr, flush=True)
                time.sleep(0.005)
        if reload_to is None:
            return  # window closed by the user
        q = d.qpos.copy() if reload_to.nq == m.nq else None
        m, d = reload_to, mujoco.MjData(reload_to)
        if q is not None:
            d.qpos[:] = q
        elif m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        print(f"reloaded {path}", flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
    os._exit(0)  # interpreter teardown can hang after the GLFW window closes
