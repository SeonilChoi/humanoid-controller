#!/usr/bin/env python3
"""No-load actuator identification for DAMIAO MIT-mode motors (real2sim), numpy + scipy only (runs on the Jetson).

Uses every log of an example_damiao_bam session (bam/*.json and sysid/*.json, including the torque feed-forward
tests that the stock BAM fit cannot use) and fits

    tau_motor = clip(ratio * (kp (q_des - q) + kd (v_des - dq) + tau_ff)(t - delay), torque / speed line)
    armature * ddq = tau_motor - friction,
    friction  = BAM / MuJoCo friction budget: frictionloss = Fc + Fs exp(-(|dq| / vs)^2), damping = b
                (static friction clipped as in bam/simulate.py and MuJoCo dof_frictionloss / dof_damping)

by rolling the model out from the measured state every --segment seconds and matching position and velocity.

    python3 fit_noload.py <session_dir> [<session_dir> ...] --motor dm_j8006 [--out params.json] [--plot]

Output JSON: armature, frictionloss (Fc), stribeck (Fs, vs), damping (b), delay, ratio, effort / velocity limit,
per-log RMS errors, plus the direct estimates (breakaway torques, steady-state friction curve, torque-sensor ratio).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

LIMITS = {  # peak torque [Nm], no-load speed [rad/s] at 24 V (aifitlab DAMIAO wiki manuals): only the model clip
    "dm_j8006": (20.0, 19.9),   # 6:1, 190 rpm
    "dm_j6006": (12.0, 25.1),   # 6:1, 240 rpm
    "dm_j4340": (40.0, 5.9),    # 40:1, 56 rpm (48 V: 112 rpm = 11.7 rad/s)
}

PARAMS = ["armature", "frictionloss", "stribeck", "stribeck_velocity", "damping", "delay", "ratio"]


def load_logs(sessions: list[str], dt: float) -> list[dict]:
    logs = []
    for session in sessions:
        files = sorted(glob.glob(os.path.join(session, "bam", "*.json")) + glob.glob(os.path.join(session, "sysid", "*.json")))
        if not files and session.endswith(".json"):
            files = [session]
        for path in files:
            with open(path) as f:
                raw = json.load(f)
            if raw.get("aborted") or len(raw["entries"]) < 20:
                print(f"  skip {path} (aborted / too short)")
                continue
            entries = raw["entries"]
            t_all = np.array([x["timestamp"] for x in entries])
            # Tests made of several runs (velocity speeds, relay torques, breakaway cycles) have gaps where the
            # motor moved back to the center unlogged: split there, never interpolate across a gap.
            cuts = np.where(np.diff(t_all) > 5 * dt)[0] + 1
            bounds = [0, *cuts.tolist(), len(entries)]
            chunks = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 20]
            for c_i, (a, b) in enumerate(chunks):
                logs.append(make_log(raw, entries[a:b], dt, os.path.relpath(path, session)
                                     + (f"#{c_i + 1}" if len(chunks) > 1 else ""), session))
    return logs


def make_log(raw: dict, e: list, dt: float, name: str, session: str) -> dict:
    t = np.array([x["timestamp"] for x in e])
    # resample on a uniform grid (timestamps jitter with the recorder loop)
    grid = np.arange(t[0], t[-1], dt)
    def col(key, default=0.0):
        return np.interp(grid, t, np.array([float(x.get(key, default)) for x in e]))
    log = {
        "name": name,
        "session": session,
        "trajectory": raw.get("trajectory", "?"),
        "kp_log": raw.get("kp", 0.0),
        "t": grid,
        "q": col("position"),
        "dq": col("speed"),
        "goal": col("goal_position"),
        "goal_velocity": col("goal_velocity"),
        "torque_ff": col("torque_ff"),
        "kp": col("kp"),
        "kd": col("kd"),
        "enable": col("torque_enable") > 0.5,
        "tau_meas": col("control"),
        "tau_demand": col("torque_demand"),
        "meta": {k: v for k, v in raw.items() if k != "entries"},
    }
    return log


# ---------------------------------------------------------------------------------------------------------------
# Direct estimates (no simulation)

def direct_estimates(logs: list[dict], dt: float) -> dict:
    out = {}
    breakaway = [(l["meta"].get("breakaway_positive"), l["meta"].get("breakaway_negative"))
                 for l in logs if l["trajectory"] == "breakaway"]
    if breakaway:
        out["breakaway_positive"], out["breakaway_negative"] = breakaway[-1]

    # torque sensor vs commanded torque (where the command is not saturated and the motor is enabled)
    q = np.concatenate([l["tau_meas"][l["enable"]] for l in logs])
    d = np.concatenate([l["tau_demand"][l["enable"]] for l in logs])
    mask = np.abs(d) > 0.05 * max(1e-6, np.abs(d).max())
    if mask.sum() > 50:
        out["measured_over_demand"] = float(np.dot(q[mask], d[mask]) / np.dot(d[mask], d[mask]))

    # regression on the measured torque: tau = J ddq + Fc sign(dq) + b dq   (moving samples only)
    rows, rhs = [], []
    for l in logs:
        if len(l["q"]) < 31:
            continue
        dq_s = savgol_filter(l["dq"], 21, 3)
        ddq = savgol_filter(l["dq"], 21, 3, deriv=1, delta=dt)
        tau = savgol_filter(l["tau_meas"], 21, 3)
        m = (np.abs(dq_s) > 0.1) & l["enable"]
        rows.append(np.stack([ddq[m], np.sign(dq_s[m]), dq_s[m]], axis=1))
        rhs.append(tau[m])
    if rows:
        A, y = np.concatenate(rows), np.concatenate(rhs)
        if len(y) > 50:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            out["regression_armature"], out["regression_coulomb"], out["regression_viscous"] = map(float, coef)

    # steady-state friction curve from the velocity sweeps: mean measured torque per commanded speed and direction
    curve = []
    for l in logs:
        if l["trajectory"] != "velocity_sweep":
            continue
        gv = l["goal_velocity"]
        for v in sorted(set(np.round(np.abs(gv[np.abs(gv) > 1e-6]), 4))):
            for sign in (1.0, -1.0):
                m = np.isclose(gv, sign * v, rtol=1e-3) & (np.abs(l["dq"] - sign * v) < 0.25 * v + 0.05)
                if m.sum() > 20:
                    curve.append([float(np.mean(l["dq"][m])), float(np.mean(l["tau_meas"][m]))])
    if curve:
        out["friction_curve"] = sorted(curve)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Simulation (vectorized over segments)

def is_open_loop(log: dict) -> bool:
    """kp = 0 (torque feed-forward tests): the rollout has no position feedback and drifts fast."""
    return bool(np.mean(log["kp"] < 1e-6) > 0.5)


def make_segments(logs: list[dict], dt: float, segment: float, which=None) -> dict | None:
    n = max(10, int(round(segment / dt)))
    keys = ["q", "dq", "goal", "goal_velocity", "torque_ff", "kp", "kd", "enable"]
    seg = {k: [] for k in keys}
    owner = []
    for i, l in enumerate(logs):
        if which is not None and not which(l):
            continue
        for s in range(0, len(l["q"]) - n, n):
            for k in keys:
                seg[k].append(l[k][s:s + n])
            owner.append(i)
    if not owner:
        return None
    seg = {k: np.array(v) for k, v in seg.items()}
    seg["owner"] = np.array(owner)
    return seg


def delayed(x: np.ndarray, delay: float, dt: float) -> np.ndarray:
    """Shift every row right by 'delay' seconds (linear interpolation, first sample held)."""
    shift = delay / dt
    k = int(math.floor(shift))
    a = shift - k
    def roll(m):
        if m == 0:
            return x
        if m >= x.shape[1]:
            return np.repeat(x[:, :1], x.shape[1], axis=1)
        return np.concatenate([np.repeat(x[:, :1], m, axis=1), x[:, :-m]], axis=1)
    return (1 - a) * roll(k) + a * roll(k + 1)


def simulate(p: dict, seg: dict, dt: float, limits: tuple[float, float], substeps: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Roll every segment out from its first measured state. The command of sample k is held for dt (after the
    delay); the MIT law and the dynamics run 'substeps' times per sample (the motor computes the law at kHz rates,
    so the kd term acts on the live velocity, not on a 2 ms hold)."""
    t_max, v_max = limits
    goal = delayed(seg["goal"], p["delay"], dt)
    gv = delayed(seg["goal_velocity"], p["delay"], dt)
    ff = delayed(seg["torque_ff"], p["delay"], dt)
    kp = delayed(seg["kp"], p["delay"], dt)
    kd = delayed(seg["kd"], p["delay"], dt)
    en = seg["enable"]
    q = seg["q"][:, 0].copy()
    dq = seg["dq"][:, 0].copy()
    n = seg["q"].shape[1]
    qs = np.empty_like(seg["q"])
    dqs = np.empty_like(seg["dq"])
    inertia = p["armature"]
    h = dt / substeps
    for k in range(n):
        qs[:, k] = q
        dqs[:, k] = dq
        for _ in range(substeps):
            tau = p["ratio"] * (kp[:, k] * (goal[:, k] - q) + kd[:, k] * (gv[:, k] - dq) + ff[:, k])
            upper = np.clip(t_max * (1.0 - dq / v_max), -t_max, t_max)
            lower = np.clip(-t_max * (1.0 + dq / v_max), -t_max, t_max)
            tau = np.clip(tau, lower, upper) * en[:, k]
            frictionloss = p["frictionloss"] + p["stribeck"] * np.exp(-(dq / p["stribeck_velocity"]) ** 2)
            tau_stop = inertia / h * dq + tau
            friction = -np.sign(tau_stop) * np.minimum(np.abs(tau_stop), frictionloss + p["damping"] * np.abs(dq))
            dq = dq + (tau + friction) / inertia * h
            q = q + dq * h
    return qs, dqs


def fit(groups: list[dict], dt: float, limits: tuple[float, float], x0: dict, velocity_weight: float,
        fixed: dict, substeps: int) -> tuple[dict, object]:
    # optimise log(armature) etc. so the parameters stay positive and well scaled
    lo = {"armature": 1e-5, "frictionloss": 0.0, "stribeck": 0.0, "stribeck_velocity": 0.01, "damping": 0.0,
          "delay": 0.0, "ratio": 0.5}
    hi = {"armature": 2.0, "frictionloss": 10.0, "stribeck": 10.0, "stribeck_velocity": 2.0, "damping": 10.0,
          "delay": 0.02, "ratio": 1.5}
    log_params = {"armature", "stribeck_velocity"}

    free = [k for k in PARAMS if k not in fixed]

    def pack(p):
        return np.array([math.log(p[k]) if k in log_params else p[k] for k in free])

    def unpack(x):
        p = dict(fixed)
        p.update({k: (math.exp(v) if k in log_params else v) for k, v in zip(free, x)})
        return p

    # every group (closed-loop / open-loop segments) is normalised so each weighs the same
    scales = []
    for seg in groups:
        scales.append((max(1e-3, float(np.std(seg["q"] - seg["q"][:, :1]))), max(1e-3, float(np.std(seg["dq"]))),
                       1.0 / math.sqrt(seg["q"].size)))

    def residuals(x):
        p = unpack(x)
        r = []
        for seg, (q_scale, v_scale, w) in zip(groups, scales):
            qs, dqs = simulate(p, seg, dt, limits, substeps)
            r.append(w * ((qs - seg["q"]) / q_scale).ravel())
            if velocity_weight > 0:
                r.append(w * velocity_weight * ((dqs - seg["dq"]) / v_scale).ravel())
        return np.concatenate(r)

    lb = np.array([math.log(lo[k]) if k in log_params else lo[k] for k in free])
    ub = np.array([math.log(hi[k]) if k in log_params else hi[k] for k in free])
    x_start = np.clip(pack(x0), lb + 1e-9, ub - 1e-9)
    result = least_squares(residuals, x_start, bounds=(lb, ub), x_scale="jac", diff_step=1e-3, max_nfev=400,
                           verbose=1)
    return unpack(result.x), result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="example_damiao_bam session dirs (or single .json logs)")
    ap.add_argument("--motor", required=True, choices=sorted(LIMITS))
    ap.add_argument("--velocity_limit", type=float, default=None,
                    help="override the no-load speed of the model clip [rad/s] (e.g. 11.7 for a J4340 on 48 V)")
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--segment", type=float, default=0.5, help="rollout length before re-sync to the measured state [s]")
    ap.add_argument("--segment_open_loop", type=float, default=0.1,
                    help="rollout length for the kp = 0 torque tests (no position feedback) [s]")
    ap.add_argument("--velocity_weight", type=float, default=0.3)
    ap.add_argument("--substeps", type=int, default=8, help="simulation steps per log sample (MIT law rate)")
    ap.add_argument("--fit_ratio", action="store_true",
                    help="also fit the command -> torque gain (default fixed 1.0: it trades off against friction)")
    ap.add_argument("--fix", nargs="*", default=[], metavar="NAME=VALUE",
                    help=f"hold parameters fixed, e.g. --fix delay=0.002 stribeck=0 (names: {', '.join(PARAMS)})")
    ap.add_argument("--max_segments", type=int, default=600, help="random subset for speed (0 = all)")
    ap.add_argument("--out", default=None, help="output JSON (default <first session>/fit_noload.json)")
    ap.add_argument("--plot", action="store_true", help="save <out>.png: measured vs simulated per log")
    args = ap.parse_args()

    logs = load_logs(args.sessions, args.dt)
    if not logs:
        sys.exit("No usable logs.")
    print(f"{len(logs)} logs:", ", ".join(l["name"] for l in logs))

    direct = direct_estimates(logs, args.dt)
    print("direct estimates:", json.dumps({k: v for k, v in direct.items() if k != "friction_curve"}, indent=2))

    limits = LIMITS[args.motor]
    if args.velocity_limit:
        limits = (limits[0], args.velocity_limit)
    rng = np.random.default_rng(0)
    groups = []  # (segments, fit subset)
    for which, length, label in ((lambda l: not is_open_loop(l), args.segment, "closed-loop"),
                                 (is_open_loop, args.segment_open_loop, "open-loop (kp = 0)")):
        seg = make_segments(logs, args.dt, length, which)
        if seg is None:
            continue
        if args.max_segments and len(seg["owner"]) > args.max_segments:
            pick = np.sort(rng.choice(len(seg["owner"]), args.max_segments, replace=False))
            sub = {k: v[pick] for k, v in seg.items()}
        else:
            sub = seg
        groups.append((seg, sub))
        print(f"fitting on {len(sub['owner'])} {label} segments of {length} s")
    if not groups:
        sys.exit("No data (logs shorter than one segment?)")

    x0 = {
        "armature": max(1e-4, direct.get("regression_armature", 0.003)),
        "frictionloss": max(0.0, direct.get("regression_coulomb", 0.05)),
        "stribeck": 0.02,
        "stribeck_velocity": 0.1,
        "damping": max(0.0, direct.get("regression_viscous", 0.01)),
        "delay": 0.002,
        "ratio": 1.0,
    }
    fixed = {} if args.fit_ratio else {"ratio": 1.0}
    for item in args.fix:
        name, value = item.split("=")
        if name not in PARAMS:
            sys.exit(f"--fix: unknown parameter {name}")
        fixed[name] = float(value)
    if "stribeck" in fixed and fixed["stribeck"] == 0.0:
        fixed.setdefault("stribeck_velocity", 0.1)
    print("fixed:", fixed)
    p, result = fit([sub for _, sub in groups], args.dt, limits, x0, args.velocity_weight, fixed, args.substeps)

    # per-log RMS on every segment
    per_log = {}
    rollouts = []
    for seg, _ in groups:
        qs, dqs = simulate(p, seg, args.dt, limits, args.substeps)
        rollouts.append((seg, qs, dqs))
        for i, l in enumerate(logs):
            m = seg["owner"] == i
            if m.any():
                per_log[l["name"]] = {
                    "position_rmse": float(np.sqrt(np.mean((qs[m] - seg["q"][m]) ** 2))),
                    "velocity_rmse": float(np.sqrt(np.mean((dqs[m] - seg["dq"][m]) ** 2))),
                    "segment_s": seg["q"].shape[1] * args.dt,
                }

    out = {
        "motor": args.motor,
        "sessions": [os.path.abspath(s) for s in args.sessions],
        "model": "tau = clip(ratio * MIT(t - delay)); armature ddq = tau - friction(frictionloss + stribeck, damping)",
        "params": p,
        "sim": {  # MuJoCo / Isaac Lab joint + actuator settings at the OUTPUT shaft
            "armature": p["armature"],
            "frictionloss": p["frictionloss"],
            "damping": p["damping"],
            "effort_limit": limits[0],
            "velocity_limit": limits[1],
            "delay_s": p["delay"],
            "torque_scale": p["ratio"],
        },
        "segment_s": args.segment,
        "segment_open_loop_s": args.segment_open_loop,
        "dt": args.dt,
        "substeps": args.substeps,
        "fixed": fixed,
        "cost": float(result.cost),
        "per_log": per_log,
        "direct": direct,
    }
    path = args.out or os.path.join(args.sessions[0], "fit_noload.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["params"], indent=2))
    print("per-log RMSE (position rad / velocity rad/s):")
    for name, v in per_log.items():
        print(f"  {name:40s} {v['position_rmse']:.4f}  {v['velocity_rmse']:.3f}")
    print(f"-> {path}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(logs), 2, figsize=(14, 2.4 * len(logs)), squeeze=False)
        for i, l in enumerate(logs):
            found = [(seg, qs, dqs) for seg, qs, dqs in rollouts if (seg["owner"] == i).any()]
            if not found:
                continue
            seg, qs, dqs = found[0]
            n = seg["q"].shape[1]
            idx = np.where(seg["owner"] == i)[0]
            t = np.arange(len(idx) * n) * args.dt
            axes[i, 0].plot(t, seg["q"][idx].ravel(), label="real")
            axes[i, 0].plot(t, qs[idx].ravel(), "--", label="sim")
            axes[i, 0].plot(t, seg["goal"][idx].ravel(), ":", label="goal")
            axes[i, 0].set_ylabel(l["name"], fontsize=7)
            axes[i, 1].plot(t, seg["dq"][idx].ravel(), label="real")
            axes[i, 1].plot(t, dqs[idx].ravel(), "--", label="sim")
        axes[0, 0].set_title("position [rad]")
        axes[0, 1].set_title("velocity [rad/s]")
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.splitext(path)[0] + ".png", dpi=80)
        print(f"-> {os.path.splitext(path)[0]}.png")


if __name__ == "__main__":
    main()
