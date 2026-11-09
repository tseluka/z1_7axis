"""mjt: headless MuJoCo inspection / rendering / simulation CLI.

Every command prints plain text; image commands save PNGs under out/ and print the path.
"""

import csv
import json
import math
import os
import sys
import time
from typing import Annotated

import typer

from tools.mjcore import (
    CLEAN_ENV,
    JNT_TYPE,
    OBJ,
    OUT,
    Image,
    camera_for,
    contacts,
    die,
    fmt,
    label,
    live_state,
    load,
    make_renderer,
    mujoco,
    name,
    np,
    obj_id,
    parse_assign,
    qpos_string,
    render_views,
    save,
    scene_option,
    set_state,
    tile,
)

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)

Model = Annotated[str, typer.Argument(help="Path to MJCF .xml or .mjb")]
Key = Annotated[str | None, typer.Option("--key", "-k", help="Keyframe name to start from")]
Qpos = Annotated[str | None, typer.Option("--qpos", "-q", help="'joint2=0.5,joint3=-1', raw nq numbers, or 'live'")]


def _jname(m, j):
    return name(m, OBJ.mjOBJ_JOINT, j)


def _bname(m, b):
    return name(m, OBJ.mjOBJ_BODY, b)


def _aname(m, a):
    return name(m, OBJ.mjOBJ_ACTUATOR, a)


def _act_joint(m, a) -> int:
    """Joint id an actuator drives, or -1 if its transmission isn't a joint."""
    return int(m.actuator_trnid[a, 0]) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT else -1


def _is_position(m, a) -> bool:
    """Position-servo-like: affine bias with biasprm[1] == -kp."""
    return (m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
            and math.isclose(m.actuator_biasprm[a, 1], -m.actuator_gainprm[a, 0]))


def _actuator_desc(m, a) -> str:
    gt, bt = m.actuator_gaintype[a], m.actuator_biastype[a]
    if _is_position(m, a):
        kv = -m.actuator_biasprm[a, 2]
        return f"position kp={m.actuator_gainprm[a, 0]:g} kv={kv:g}"
    if gt == mujoco.mjtGain.mjGAIN_FIXED and bt == mujoco.mjtBias.mjBIAS_NONE:
        return f"motor gain={m.actuator_gainprm[a, 0]:g}"
    return f"gain={mujoco.mjtGain(gt).name[7:]}{fmt(m.actuator_gainprm[a, :3], 3)} " \
           f"bias={mujoco.mjtBias(bt).name[7:]}{fmt(m.actuator_biasprm[a, :3], 3)}"


def _range(lim, r):
    return f"[{r[0]:.3f}, {r[1]:.3f}]" if lim else "unlimited"


# ---------------------------------------------------------------- inspect

@app.command()
def inspect(model: Model):
    """Compile the model and dump sizes, bodies, joints, actuators, sensors, cameras, sites, keyframes."""
    m, d = load(model)
    mujoco.mj_forward(m, d)
    o = m.opt
    print(f"MODEL {model}  nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} njnt={m.njnt} ngeom={m.ngeom} "
          f"nsite={m.nsite} nsensor={m.nsensor} ncam={m.ncam} nkey={m.nkey}")
    print(f"OPTION timestep={o.timestep:g} integrator={mujoco.mjtIntegrator(o.integrator).name[6:]} "
          f"gravity={fmt(o.gravity, 2)} cone={mujoco.mjtCone(o.cone).name[7:]} total_mass={m.body_subtreemass[0]:.4f}")

    print("\nBODIES  id name <- parent | mass subtree_mass | diaginertia | joints | ngeom(vis/col)")
    for b in range(1, m.nbody):
        js = [_jname(m, j) for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])] if m.body_jntnum[b] else []
        gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        ncol = sum(1 for g in gs if m.geom_contype[g] or m.geom_conaffinity[g])
        print(f"  {b:2d} {_bname(m, b)} <- {_bname(m, m.body_parentid[b])} | {m.body_mass[b]:.4f} "
              f"{m.body_subtreemass[b]:.4f} | {fmt(m.body_inertia[b], 6)} | {','.join(js) or '-'} | "
              f"{len(gs) - ncol}/{ncol}")

    print("\nJOINTS  id name type @body | axis(local) | range | damping armature frictionloss stiffness | qadr dadr")
    for j in range(m.njnt):
        print(f"  {j:2d} {_jname(m, j)} {JNT_TYPE[m.jnt_type[j]]} @{_bname(m, m.jnt_bodyid[j])} | "
              f"{fmt(m.jnt_axis[j], 2)} | {_range(m.jnt_limited[j], m.jnt_range[j])} | "
              f"{m.dof_damping[m.jnt_dofadr[j]]:g} {m.dof_armature[m.jnt_dofadr[j]]:g} "
              f"{m.dof_frictionloss[m.jnt_dofadr[j]]:g} {m.jnt_stiffness[j]:g} | {m.jnt_qposadr[j]} {m.jnt_dofadr[j]}")

    print("\nACTUATORS  id name -> target | type | ctrlrange | forcerange | gear")
    for a in range(m.nu):
        j = _act_joint(m, a)
        tgt = _jname(m, j) if j >= 0 else f"{mujoco.mjtTrn(m.actuator_trntype[a]).name[6:]}#{m.actuator_trnid[a, 0]}"
        print(f"  {a:2d} {_aname(m, a)} -> {tgt} | {_actuator_desc(m, a)} | "
              f"{_range(m.actuator_ctrllimited[a], m.actuator_ctrlrange[a])} | "
              f"{_range(m.actuator_forcelimited[a], m.actuator_forcerange[a])} | {m.actuator_gear[a, 0]:g}")
    acted = {_act_joint(m, a) for a in range(m.nu)}
    unact = [_jname(m, j) for j in range(m.njnt) if j not in acted and m.jnt_type[j] != 0]
    if unact:
        print(f"  (unactuated joints: {', '.join(unact)})")

    if m.nsensor:
        print("\nSENSORS  id name type -> obj | dim")
        for s in range(m.nsensor):
            print(f"  {s:2d} {name(m, OBJ.mjOBJ_SENSOR, s)} {mujoco.mjtSensor(m.sensor_type[s]).name[9:]} "
                  f"-> {mujoco.mjtObj(m.sensor_objtype[s]).name[6:]}#{m.sensor_objid[s]} | {m.sensor_dim[s]}")
    if m.ncam:
        print("\nCAMERAS  id name @body | world pos | fovy")
        for c in range(m.ncam):
            print(f"  {c:2d} {name(m, OBJ.mjOBJ_CAMERA, c)} @{_bname(m, m.cam_bodyid[c])} | {fmt(d.cam_xpos[c])} | "
                  f"{m.cam_fovy[c]:g}")
    if m.nsite:
        print("\nSITES  id name @body | world pos (qpos0)")
        for s in range(m.nsite):
            print(f"  {s:2d} {name(m, OBJ.mjOBJ_SITE, s)} @{_bname(m, m.site_bodyid[s])} | {fmt(d.site_xpos[s])}")
    if m.nkey:
        print("\nKEYFRAMES")
        for k in range(m.nkey):
            print(f"  {k:2d} {name(m, OBJ.mjOBJ_KEY, k)} qpos=[{fmt(m.key_qpos[k])}]"
                  + (f" ctrl=[{fmt(m.key_ctrl[k])}]" if m.nu else ""))


# ---------------------------------------------------------------- tree

@app.command()
def tree(model: Model, key: Key = None, qpos: Qpos = None):
    """Kinematic tree with world-frame body positions and joint anchors/axes at the given pose."""
    m, d = load(model)
    set_state(m, d, key, qpos)
    kids = {b: [] for b in range(m.nbody)}
    for b in range(1, m.nbody):
        kids[m.body_parentid[b]].append(b)

    def walk(b, depth):
        ind = "  " * depth
        print(f"{ind}{_bname(m, b)}  pos={fmt(d.xpos[b])}  local_pos={fmt(m.body_pos[b])}"
              + (f"  mass={m.body_mass[b]:.3f}" if b else ""))
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
            q = d.qpos[m.jnt_qposadr[j]]
            print(f"{ind}  * {_jname(m, j)} [{JNT_TYPE[m.jnt_type[j]]}] q={q:.3f}  anchor={fmt(d.xanchor[j])}  "
                  f"axis_world={fmt(d.xaxis[j], 2)}  range={_range(m.jnt_limited[j], m.jnt_range[j])}")
        for s in range(m.nsite):
            if m.site_bodyid[s] == b:
                print(f"{ind}  @ site {name(m, OBJ.mjOBJ_SITE, s)} pos={fmt(d.site_xpos[s])}")
        for c in kids[b]:
            walk(c, depth + 1)

    walk(0, 0)


# ---------------------------------------------------------------- render

@app.command()
def render(model: Model, key: Key = None, qpos: Qpos = None,
           views: Annotated[str, typer.Option(help="Comma list: front,back,left,right,top,iso,iso2,cam:NAME")] = "iso,front,left,top",
           frames: Annotated[str, typer.Option(help="none|body|geom|site|world|contact")] = "none",
           contacts_: Annotated[bool, typer.Option("--contacts", help="Contact points + forces")] = False,
           joints: Annotated[bool, typer.Option(help="Draw joint axes")] = False,
           collision: Annotated[bool, typer.Option(help="Show collision geoms (group 3) instead of visual")] = False,
           transparent: bool = False, com: Annotated[bool, typer.Option(help="Show centers of mass")] = False,
           labels: Annotated[str, typer.Option(help="none|body|joint|geom|site|actuator|...")] = "none",
           zoom: float = 1.0, width: int = 480, height: int = 360,
           out: Annotated[str, typer.Option(help="Output file (relative names go to out/)")] = "render.png"):
    """Render the model from several angles into one tiled PNG."""
    m, d = load(model)
    set_state(m, d, key, qpos)
    r = make_renderer(m, width, height)
    opt = scene_option(frames, contacts_, joints, collision, transparent, com, labels)
    imgs = render_views(m, d, r, [v.strip() for v in views.split(",")], opt, zoom)
    print(save(tile(imgs, 2 if len(imgs) == 4 else None), out))


# ---------------------------------------------------------------- sim

def _hold_targets(m, d) -> np.ndarray:
    """ctrl that holds the current pose for position actuators (others keep d.ctrl)."""
    ctrl = d.ctrl.copy()
    for a in range(m.nu):
        j = _act_joint(m, a)
        if j >= 0 and _is_position(m, a):
            ctrl[a] = d.qpos[m.jnt_qposadr[j]]
    return ctrl


@app.command()
def sim(model: Model, key: Key = None, qpos: Qpos = None,
        t: Annotated[float, typer.Option(help="Seconds to simulate")] = 2.0,
        hold: Annotated[bool, typer.Option(help="Position actuators hold the starting pose")] = False,
        passive: Annotated[bool, typer.Option(help="Disable all actuators (gravity-only fall test)")] = False,
        ctrl: Annotated[str | None, typer.Option(help="Constant ctrl: 'motor1=0.3,motor2=1'")] = None,
        sine: Annotated[str | None, typer.Option(help="Sinusoid around start: 'motor1=AMP,motor2=AMP'")] = None,
        freq: Annotated[float, typer.Option(help="Sine frequency, Hz")] = 0.5,
        filmstrip: Annotated[int, typer.Option(help="Save N evenly spaced frames as a contact sheet")] = 0,
        view: str = "iso", csv_out: Annotated[str, typer.Option("--csv", help="CSV log path")] = "sim.csv"):
    """Step the simulation, log a CSV, and print a stability / tracking summary.

    Default control: keyframe ctrl if --key has one, else zero. --hold overrides with the start pose.
    """
    m, d = load(model)
    set_state(m, d, key, qpos)
    base = d.ctrl.copy()
    if hold:
        base = _hold_targets(m, d)
    if passive:
        m.actuator_gainprm[:] = 0
        m.actuator_biasprm[:] = 0
    for an, v in parse_assign(ctrl or ""):
        base[obj_id(m, OBJ.mjOBJ_ACTUATOR, an)] = v
    amps = np.zeros(m.nu)
    for an, v in parse_assign(sine or ""):
        amps[obj_id(m, OBJ.mjOBJ_ACTUATOR, an)] = v
    q0 = d.qpos.copy()

    steps = int(round(t / m.opt.timestep))
    log_every = max(1, steps // 1000)
    film_at = set(np.linspace(0, steps, filmstrip, dtype=int).tolist()) if filmstrip else set()
    r = make_renderer(m, 320, 240) if filmstrip else None
    frames, rows = [], []
    max_pen, max_qvel, limit_hits, sat = 0.0, 0.0, {}, np.zeros(m.nu)
    blew_up = None

    hinge = [j for j in range(m.njnt) if m.jnt_type[j] in (2, 3)]
    header = ["t"] + [f"q_{_jname(m, j)}" for j in hinge] + [f"v_{_jname(m, j)}" for j in hinge] \
        + [f"ctrl_{_aname(m, a)}" for a in range(m.nu)] + [f"frc_{_aname(m, a)}" for a in range(m.nu)] + ["ncon"]

    for i in range(steps + 1):
        d.ctrl[:] = base + amps * math.sin(2 * math.pi * freq * d.time)
        if i in film_at:
            r.update_scene(d, camera=camera_for(m, view))
            frames.append(label(Image.fromarray(r.render()), f"t={d.time:.2f}s"))
        if i == steps:
            break
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
            blew_up = d.time
            break
        if d.ncon:
            max_pen = max(max_pen, -min(c.dist for c in d.contact[: d.ncon]))
        max_qvel = max(max_qvel, float(np.abs(d.qvel).max()) if m.nv else 0.0)
        for j in hinge:
            if m.jnt_limited[j]:
                q = d.qpos[m.jnt_qposadr[j]]
                lo, hi = m.jnt_range[j]
                if q < lo - 1e-3 or q > hi + 1e-3:
                    limit_hits[_jname(m, j)] = limit_hits.get(_jname(m, j), 0) + 1
        for a in range(m.nu):
            if m.actuator_forcelimited[a] and abs(d.actuator_force[a]) >= 0.99 * np.abs(m.actuator_forcerange[a]).max():
                sat[a] += 1
        if i % log_every == 0:
            rows.append([d.time] + [d.qpos[m.jnt_qposadr[j]] for j in hinge] + [d.qvel[m.jnt_dofadr[j]] for j in hinge]
                        + list(d.ctrl) + list(d.actuator_force) + [d.ncon])

    p = OUT / csv_out if "/" not in csv_out else csv_out
    OUT.mkdir(exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[f"{x:.6g}" for x in row] for row in rows])

    print(f"SIM {t}s dt={m.opt.timestep:g} steps={i} -> {p}")
    if blew_up is not None:
        print(f"  !! UNSTABLE: NaN/Inf in state at t={blew_up:.4f}s")
    warns = [(mujoco.mjtWarning(k).name[7:], d.warning[k].number) for k in range(len(d.warning)) if d.warning[k].number]
    if warns:
        print("  !! MuJoCo warnings: " + ", ".join(f"{n} x{c}" for n, c in warns))
    print(f"  max|qvel|={max_qvel:.3f}  max_penetration={max_pen * 1000:.2f}mm  final ncon={d.ncon}")
    if limit_hits:
        print("  joint limit violations (steps >1mrad past range): "
              + ", ".join(f"{k} x{v}" for k, v in limit_hits.items()))
    if sat.any():
        print("  force-saturated actuators (fraction of steps): "
              + ", ".join(f"{_aname(m, a)} {sat[a] / max(i, 1):.0%}" for a in range(m.nu) if sat[a]))
    print("  final state vs start / ctrl target:")
    for a in range(m.nu):
        j = _act_joint(m, a)
        if j < 0:
            continue
        q = d.qpos[m.jnt_qposadr[j]]
        tgt = f" target={d.ctrl[a]:+.4f} err={d.ctrl[a] - q:+.4f}" if _is_position(m, a) and not passive else ""
        print(f"    {_jname(m, j)}: start={q0[m.jnt_qposadr[j]]:+.4f} final={q:+.4f}{tgt} "
              f"force={d.actuator_force[a]:+.2f}")
    print(f"  final --qpos '{qpos_string(m, d.qpos)}'")
    if frames:
        print(f"  filmstrip -> {save(tile(frames, min(len(frames), 4)), 'filmstrip.png')}")


# ---------------------------------------------------------------- check

@app.command()
def check(model: Model,
          settle: Annotated[float, typer.Option(help="Seconds of --hold sim per keyframe for the stability test")] = 1.0):
    """Lint the model: mass/inertia, actuation, self-collision at qpos0/keyframes, gravity load, stability."""
    m, d = load(model)
    issues = 0

    def warn(msg):
        nonlocal issues
        issues += 1
        print(f"  WARN {msg}")

    print(f"CHECK {model}: compiled OK (nq={m.nq} nv={m.nv} nu={m.nu})")

    print("[mass/inertia]")
    for b in range(1, m.nbody):
        if m.body_jntnum[b] and m.body_subtreemass[b] < 1e-6:
            warn(f"body '{_bname(m, b)}' has joints but its subtree is massless")
        elif m.body_mass[b] > 0 and m.body_inertia[b].min() < 1e-9:
            warn(f"body '{_bname(m, b)}' has tiny inertia {" ".join(f"{x:.1e}" for x in m.body_inertia[b])} (mass {m.body_mass[b]:.4f})")
    print(f"  total mass {m.body_subtreemass[0]:.3f} kg")

    print("[actuation]")
    acted = {}
    for a in range(m.nu):
        j = _act_joint(m, a)
        if j >= 0:
            acted.setdefault(j, []).append(a)
            if _is_position(m, a) and m.actuator_ctrllimited[a] and m.jnt_limited[j]:
                cr, jr = m.actuator_ctrlrange[a], m.jnt_range[j]
                if cr[0] < jr[0] - 1e-6 or cr[1] > jr[1] + 1e-6:
                    warn(f"{_aname(m, a)} ctrlrange {fmt(cr)} exceeds {_jname(m, j)} range {fmt(jr)}")
    for j in range(m.njnt):
        if m.jnt_type[j] != 0 and j not in acted:
            warn(f"joint '{_jname(m, j)}' has no actuator")
        if len(acted.get(j, [])) > 1:
            warn(f"joint '{_jname(m, j)}' driven by several actuators: {[_aname(m, a) for a in acted[j]]}")

    poses = [("qpos0", None)] + [(name(m, OBJ.mjOBJ_KEY, k), name(m, OBJ.mjOBJ_KEY, k)) for k in range(m.nkey)]
    M = np.zeros((m.nv, m.nv))
    explicit = m.opt.integrator in (mujoco.mjtIntegrator.mjINT_EULER, mujoco.mjtIntegrator.mjINT_RK4)
    for label_, key in poses:
        print(f"[pose {label_}]")
        set_state(m, d, key)
        for j in range(m.njnt):
            if m.jnt_limited[j] and m.jnt_type[j] in (2, 3):
                q, (lo, hi) = d.qpos[m.jnt_qposadr[j]], m.jnt_range[j]
                if not lo - 1e-6 <= q <= hi + 1e-6:
                    warn(f"{_jname(m, j)}={q:.4f} outside range [{lo:.4f}, {hi:.4f}]")
        for c in contacts(m, d):
            if c["dist"] < 0:
                warn(f"penetration {c['dist'] * 1000:.2f}mm: {c['g1']} ({c['b1']}) vs {c['g2']} ({c['b2']})")
        # Gravity load: torque each actuated dof needs to stay still, vs what the actuator can give.
        mujoco.mj_fullM(m, d, M)
        for j, acts in acted.items():
            dof = m.jnt_dofadr[j]
            need = d.qfrc_bias[dof]
            for a in acts:
                gear = m.actuator_gear[a, 0]
                fmax = np.abs(m.actuator_forcerange[a]).max() * abs(gear) if m.actuator_forcelimited[a] else np.inf
                extra = ""
                if _is_position(m, a):
                    kp = m.actuator_gainprm[a, 0] * gear * gear
                    sag = need / kp
                    wn = math.sqrt(kp / (M[dof, dof] + m.dof_armature[dof]))
                    extra = f" sag~{sag:+.4f}rad  wn*dt={wn * m.opt.timestep:.3f}"
                print(f"    {_jname(m, j)}: gravity torque {need:+7.3f}  limit {fmax:6.1f} "
                      f"({abs(need) / fmax:4.0%}){extra}")
                # Implicit integrators absorb the kv damping, so stiff servos are fine there.
                if _is_position(m, a) and wn * m.opt.timestep > 0.5 and explicit:
                    warn(f"{_aname(m, a)} stiff vs timestep (wn*dt={wn * m.opt.timestep:.2f}) under explicit "
                         "integrator; add armature, reduce timestep, or use implicitfast")
                if abs(need) > 0.8 * fmax:
                    warn(f"{_aname(m, a)} needs {abs(need):.1f} of {fmax:.1f} to hold {label_} against gravity")
        # Short hold simulation for stability.
        set_state(m, d, key)
        d.ctrl[:] = _hold_targets(m, d)
        q_start = d.qpos.copy()
        for _ in range(int(settle / m.opt.timestep)):
            mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            warn(f"simulation went NaN within {settle}s holding {label_}")
        else:
            drift = np.abs(d.qpos - q_start).max() if m.nq else 0
            print(f"    hold {settle}s: max joint drift {drift:.4f}  max|qvel| {np.abs(d.qvel).max():.4f}")
            if drift > 0.05:
                warn(f"pose {label_} drifts {drift:.3f} rad/m in {settle}s under position hold")
        for k in range(len(d.warning)):
            if d.warning[k].number:
                warn(f"MuJoCo warning {mujoco.mjtWarning(k).name} x{d.warning[k].number}")

    print(f"DONE: {issues} warning(s)")


# ---------------------------------------------------------------- fk / ik

def _target(m, body: str | None, site: str | None):
    if site:
        return "site", obj_id(m, OBJ.mjOBJ_SITE, site)
    if body:
        return "body", obj_id(m, OBJ.mjOBJ_BODY, body)
    return "body", m.nbody - 1


@app.command()
def fk(model: Model, key: Key = None, qpos: Qpos = None, body: str | None = None, site: str | None = None):
    """World pose of a body or site (default: last body) at the given configuration."""
    m, d = load(model)
    set_state(m, d, key, qpos)
    kind, i = _target(m, body, site)
    if kind == "site":
        pos, R = d.site_xpos[i], d.site_xmat[i].reshape(3, 3)
    else:
        pos, R = d.xpos[i], d.xmat[i].reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    n = name(m, OBJ.mjOBJ_SITE if kind == "site" else OBJ.mjOBJ_BODY, i)
    print(f"{kind} {n}\n  pos  = {fmt(pos, 4)}\n  quat = {fmt(quat, 4)} (wxyz)")
    print("  R    = " + "\n         ".join(fmt(row, 4) for row in R))
    print("  x/y/z axes are the columns of R")


@app.command()
def ik(model: Model, target: Annotated[str, typer.Argument(help="x,y,z world position")],
       key: Key = None, qpos: Qpos = None, body: str | None = None, site: str | None = None,
       iters: int = 500, tol: float = 1e-4, damping: float = 1e-3):
    """Damped-least-squares position IK (joint limits respected). Prints a --qpos string for render/sim."""
    m, d = load(model)
    set_state(m, d, key, qpos)
    kind, i = _target(m, body, site)
    goal = np.array([float(x) for x in target.split(",")])
    jacp = np.zeros((3, m.nv))
    for it in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        cur = d.site_xpos[i] if kind == "site" else d.xpos[i]
        err = goal - cur
        if np.linalg.norm(err) < tol:
            break
        (mujoco.mj_jacSite if kind == "site" else mujoco.mj_jacBody)(m, d, jacp, None, i)
        dq = jacp.T @ np.linalg.solve(jacp @ jacp.T + damping * np.eye(3), err)
        mujoco.mj_integratePos(m, d.qpos, dq, 1.0)
        for j in range(m.njnt):
            if m.jnt_limited[j] and m.jnt_type[j] in (2, 3):
                a = m.jnt_qposadr[j]
                d.qpos[a] = np.clip(d.qpos[a], *m.jnt_range[j])
    mujoco.mj_forward(m, d)
    cur = d.site_xpos[i] if kind == "site" else d.xpos[i]
    e = np.linalg.norm(goal - cur)
    print(f"IK {'converged' if e < tol else 'NOT converged'} after {it + 1} iters: |err|={e * 1000:.2f}mm  "
          f"reached={fmt(cur, 4)}")
    at_lim = [_jname(m, j) for j in range(m.njnt) if m.jnt_limited[j] and m.jnt_type[j] in (2, 3)
              and min(abs(d.qpos[m.jnt_qposadr[j]] - m.jnt_range[j])) < 1e-4]
    if at_lim:
        print(f"  joints at limit: {', '.join(at_lim)}")
    print(f"  --qpos '{qpos_string(m, d.qpos)}'")


# ---------------------------------------------------------------- diff

@app.command()
def diff(a: Model, b: Model):
    """Structural diff of two models: bodies, joints, actuators."""
    ma, _ = load(a)
    mb, _ = load(b)

    def table(m):
        bodies = {_bname(m, i): (f"parent={_bname(m, m.body_parentid[i])} pos={fmt(m.body_pos[i])} "
                                 f"mass={m.body_mass[i]:.4f}") for i in range(1, m.nbody)}
        joints = {_jname(m, j): (f"{JNT_TYPE[m.jnt_type[j]]} @{_bname(m, m.jnt_bodyid[j])} axis={fmt(m.jnt_axis[j], 2)} "
                                 f"range={_range(m.jnt_limited[j], m.jnt_range[j])} "
                                 f"damping={m.dof_damping[m.jnt_dofadr[j]]:g}") for j in range(m.njnt)}
        acts = {_aname(m, x): (f"-> {_jname(m, _act_joint(m, x)) if _act_joint(m, x) >= 0 else '?'} "
                               f"{_actuator_desc(m, x)} ctrl={_range(m.actuator_ctrllimited[x], m.actuator_ctrlrange[x])} "
                               f"force={_range(m.actuator_forcelimited[x], m.actuator_forcerange[x])}") for x in range(m.nu)}
        return {"BODIES": bodies, "JOINTS": joints, "ACTUATORS": acts}

    ta, tb = table(ma), table(mb)
    print(f"--- {a}\n+++ {b}\nsizes: nq {ma.nq}->{mb.nq}  nv {ma.nv}->{mb.nv}  nu {ma.nu}->{mb.nu}  "
          f"nbody {ma.nbody}->{mb.nbody}  mass {ma.body_subtreemass[0]:.3f}->{mb.body_subtreemass[0]:.3f}")
    for sec in ta:
        print(f"\n{sec}")
        for k in list(ta[sec]) + [k for k in tb[sec] if k not in ta[sec]]:
            va, vb = ta[sec].get(k), tb[sec].get(k)
            if va is None:
                print(f"  + {k}: {vb}")
            elif vb is None:
                print(f"  - {k}: {va}")
            elif va != vb:
                print(f"  ~ {k}:\n      {va}\n   -> {vb}")


# ---------------------------------------------------------------- live viewer

@app.command()
def view(model: Model):
    """Open the classic MuJoCo viewer with hot reload and the Claude file bridge.

    Mirrors the live state to out/viewer_state.json (`-q live` renders it) and applies commands from
    out/viewer_cmd.json (`mjt pose`). Space pauses, Backspace resets, saving the XML reloads it.
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.py")
    os.execve(sys.executable, [sys.executable, script, os.path.abspath(model)], CLEAN_ENV)


@app.command()
def pose(key: Key = None, qpos: Annotated[str | None, typer.Option("--qpos", "-q", help="'joint2=0.5,...'")] = None,
         ctrl: Annotated[str | None, typer.Option(help="'motor1=0.3,...'")] = None,
         pause: Annotated[bool | None, typer.Option("--pause/--play")] = None,
         timeout: float = 3.0):
    """Send a pose / ctrl / pause command to the running `mjt view` window and report the result."""
    cmd = {}
    if key:
        cmd["key"] = key
    if qpos:
        cmd["qpos"] = dict(parse_assign(qpos))
    if ctrl:
        cmd["ctrl"] = dict(parse_assign(ctrl))
    if pause is not None:
        cmd["pause"] = pause
    if not cmd:
        die("nothing to send; give --key, --qpos, --ctrl or --pause/--play")
    sent = time.time()
    cmd["id"] = f"{sent:.6f}"  # echoed back in last_cmd, so we never report a stale result
    OUT.mkdir(exist_ok=True)
    (OUT / "viewer_cmd.json").write_text(json.dumps(cmd))
    while time.time() - sent < timeout:
        time.sleep(0.1)
        last = live_state().get("last_cmd") or {}
        if last.get("id") == cmd["id"]:
            print(json.dumps({k: v for k, v in last.items() if k != "id"}))
            return
    die("viewer did not pick up the command (is `mjt view` running?)")


if __name__ == "__main__":
    app()
