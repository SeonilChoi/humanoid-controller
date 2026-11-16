# DAMIAO no-load actuator identification (BAM + real2sim)

Motors: DM-J8006-2EC, DM-J6006-2EC, DM-J4340-2EC, one at a time, **nothing attached to the output**.

## 1. Record (Jetson)

```bash
./requirements/udev-rules.sh                 # once: /dev/Damiao (USB2CAN) permissions, replug the adapter
cd build && cmake .. && make example_damiao_bam
./examples/example_damiao_bam ../config/sysid/damiao_bam.yaml dm_j8006     # or dm_j6006 / dm_j4340
```

- Set the CAN ID of each motor to the `id` in `config/hardware/motor/damiao_<motor>.yaml` (1 / 2 / 3), or edit the file.
- Every test asks first (Enter = run, `s` = skip, `q` = quit); `--yes` runs everything. Ctrl+C = motor limp.
- Safety (per motor in `damiao_bam.yaml`): position window, velocity and temperature limits; any trip makes the
  motor limp and stops the session. Logs recorded up to then are kept.

Output `damiao_logs/<motor>/<YYYYmmdd_HHMMSS>/`:

| path | content |
|---|---|
| `bam/<trajectory>_kp<kp>_<n>.json` | BAM raw logs: BAM trajectories, MIT `kp`, `kd` (= `damping`), `v_des = 0`, `tau_ff = 0` |
| `sysid/NN_<test>.json` | chirp, velocity sweep, torque relay, breakaway (same format, uses `goal_velocity` / `torque_ff`) |
| `session.yaml` | motor config + test settings used |

Entry keys: `timestamp, position, speed, goal_position, goal_velocity, torque_ff, kp, kd, torque_enable, control`
(measured torque, Nm), `torque_demand` (kp e + kd de + ff), `temperature, error`. Entry k = state at `timestamp`
and the command sent right after it (acts until entry k+1). Angles are relative to the test center, output shaft.
Top level: `motor, trajectory, kp, damping, mass = arm_mass = length = 0` (no pendulum), `center, loop_period, ...`.

## 2a. Fit on the Jetson (numpy / scipy): `fit_noload.py`

```bash
python3 tools/damiao_sysid/fit_noload.py build/damiao_logs/dm_j8006/<session> --motor dm_j8006 --plot
```

Uses **all** logs (BAM + torque tests). Model: MIT law with delay, armature, BAM/MuJoCo friction budget
(`frictionloss` + Stribeck, viscous `damping`), torque / speed clip. Rollouts restart from the measured state every
0.5 s (0.1 s for the kp = 0 torque tests). Writes `<session>/fit_noload.json` (+ `.png`):

- `sim`: `armature`, `frictionloss`, `damping`, `effort_limit`, `velocity_limit`, `delay_s` for MuJoCo
  (`joint armature / frictionloss / damping`) or Isaac Lab (`armature`, `friction`, `damping` of the joint +
  actuator delay).
- `direct`: model-free checks: breakaway torque, regression `tau_meas = J ddq + Fc sign(dq) + b dq`, steady-state
  friction curve from the velocity sweep, measured / commanded torque ratio.
- `ratio` (command -> torque gain) is fixed to 1 by default because it trades off against friction; `--fit_ratio`
  frees it, `--fix name=value` holds others.

Checked against a simulated motor (J 0.004, Fc 0.15, b 0.02): fit 0.00402 / 0.152 / 0.0195.

## 2b. Fit with BAM (PC, Python 3.12): `damiao_bam.py`

```bash
pip install better-actuator-models optuna wandb       # BAM fit imports optuna / wandb
python damiao_bam.py process --raw <session>/bam --logdir processed/dm_j8006 --dt 0.001
python damiao_bam.py fit  --actuator dm_j8006 --model m1 --logdir processed/dm_j8006 \
                          --output params/dm_j8006/m1.json --trials 2000 --validation_kp 20
python damiao_bam.py plot --actuator dm_j8006 --logdir processed/dm_j8006 --params params/dm_j8006/m1.json --sim
```

`damiao_bam.py` registers `dm_j8006 / dm_j6006 / dm_j4340` (`DamiaoActuator`: MIT law
`ratio * kp * e - kd * dq`, torque / speed clip, armature) and forwards to `bam.process / fit / plot`.
Resample with `--dt 0.001` (2 ms makes the kd term too coarse). No load means: m1 / m2 (constant + viscous
[+ Stribeck]) are identifiable; m3 / m4 only through the motor's own torque; m5 / m6 (external load, directional)
are not — use a pendulum for those.
