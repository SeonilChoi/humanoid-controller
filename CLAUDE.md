# humanoid_controller, branch `arti-h1-olaf`: working notes for Claude

This worktree runs the Olaf-style (arXiv 2512.16705) **walking and standing policies** on the real ARTI-H1 KB robot.
The policies are trained in `/opt/isaac/projects/humanoid_sim` (successor of `arti_h1_olaf`). Read that project's
`CLAUDE.md` for the training history, real2sim and the evaluation criteria. This file covers the controller side.
A Korean version is in `CLAUDE_ko.md`; keep the two in sync.

## 1. Principles

- **The controller must reproduce the training env exactly.** Every observation term, reset, filter and clock here
  mirrors `ArtiH1OlafEnv` (`humanoid_sim/.../tasks/direct/olaf/olaf_env.py`, `olaf_env_cfg.py`). Any change on one side
  needs the same change on the other, in the same piece of work. Most robot failures so far came from a mismatch
  (switch reset, servo start, estimator bias), not from the policy.
- **Change the controller only to match training, or for safety.** Do not tune behaviour here that the policy was not
  trained with (for example a different low-pass for walking: 20 Hz made the sim walk fall 33x more).
- **Every deployment is reversible.** Keep the previous ONNX in `config/robot/ARTI-H1/` and write the fallback in the
  yaml comment.

## 2. Repository and working rules

- **Worktree:** `/opt/isaac/projects/humanoid_controller_olaf` = branch `arti-h1-olaf` of SeonilChoi/humanoid_controller.
  Other worktrees of the same repo: `/opt/isaac/projects/humanoid_controller` (`arti-h1-new-pelvis`) and
  `/opt/isaac/projects/humanoid_controller_d1` (`arti-d1`). Run git commands from this directory only.
- **git stash is shared** with the other worktrees and sessions. Never use bare `git stash` / `git stash pop`; prefer a
  temporary WIP commit.
- **Credentials:** the shell has none. The user pushes, or gives a token in a message; use it only in that push command
  (`https://x-access-token:<token>@github.com/SeonilChoi/humanoid_controller.git arti-h1-olaf:arti-h1-olaf`) and never
  store it. Commit and push only when asked. The user also commits robot logs to this branch from the robot PC, so
  `git fetch` before committing and expect fast-forward conflicts.
- **Build:** only on the robot PC (Jetson Orin, `~/humanoid_controller_olaf`): `mkdir -p build && cd build && cmake .. && make -j2`.
  This dev PC has no yaml-cpp / onnxruntime / system Eigen, so the full build cannot be checked here. For a header-only
  change (for example `olaf_inekf.hpp`), compile a small test program against it with a local Eigen copy, and say
  clearly that the full build was not checked.
- **Real logs:** `arti_h1_olaf_run_<YYYYMMDD_HHMMSS>.csv`, committed by the user from the robot. Commit 6af2da5
  ("log deleted", 11-24) removed the 11-21 / 11-23 logs from the tree; recover them with
  `git show 42ad823:arti_h1_olaf_run_20261123_132540.csv > /tmp/...`.

## 3. Running

```bash
cd ~/humanoid_controller_olaf/build
./ARTI-H1 ../config/robot/ARTI-H1/robot_olaf.yaml             # starts STAND; pad cross = WALK, circle = STAND
./ARTI-H1 ../config/robot/ARTI-H1/robot_olaf.yaml --observe   # estimator / observation check, no policy
```
Other flags: `--imu-calib` / `--no-imu-calib` (default `policy.imu_still_time: 10` s of gyro bias and attitude
convergence), `--blind-imu`, `--model <onnx>`. Ctrl+C stops and flushes the log.

Sticks: WALK left stick = path vx / vy, right stick left / right = turn (vx −0.12..0.15 m/s, vy ±0.068 m/s,
wz ±0.4 rad/s, projected onto the reference keys as in training). STAND left stick = trunk pitch / roll, right stick =
trunk yaw / height (dz −3..+1.5 cm).

## 4. Olaf code map

| Part | Code | Training counterpart |
|---|---|---|
| Observation (walk 80-D, stand 79-D) | `ArtiH1::observation()` (`olaf_obs_` branch), layout in `ARTI_H1.hpp` | `ArtiH1OlafEnv._get_observations` |
| State estimate | `ArtiH1::olaf_estimate`, `robot/robots/olaf_inekf.hpp` (`OlafInEKF`) | `ArtiH1OlafEnv._estimate`, `olaf/inekf.py` |
| Path frame, clock, g | `olaf_advance`, `olaf_reset_path`, `olaf_project_walk` | `_pre_physics_step`, `_step_path`, `_walk_blend` |
| Policy switch | cross / circle handling in `observation()`, `olaf_switch_reset` | `_reset_idx` (reset_stand_pose / switch bank) |
| Runtime chain | `olaf_servo_loop` (600 Hz thread): first-order hold → slew 5.24 rad/s → first-order low-pass | `_apply_action` per physics step |
| Faults | `control()`: tilt > `max_tilt` damps and exits; IMU / motor link / (joystick) faults hold the pose | — |

Observation layout (policy joint order): [0:3] pelvis position in the path frame (x, y, height), [3:9] pelvis
orientation relative to the path frame (first two rotation-matrix columns, row-major), [9:12] pelvis-frame linear
velocity, [12:15] gyro, [15:27] joint pos − default, [27:39] joint vel, [39:51] a(t), [51:63] a(t−1), [63:75] motor
temperatures clipped to [70, 85] °C and scaled to [−1, 1], then walk: [75:77] phase cos / sin, [77:80] g = (vx, vy, wz);
stand: [75:79] g = (dz, roll, pitch, yaw). The heading relative to the path is atan2(obs_5, obs_3).

Key behaviour:
- **Estimator:** starts from the MTi-3 attitude over the first 3 control steps. A sole is in contact when it is within
  `contact_tolerance` (12 mm) of the lowest sole. `ekf.freeze_gyro_bias_z: true` holds the gyro z bias at 0 (11-24): a
  free b_g,z absorbed the true yaw rate, so the estimated heading stood still while the robot turned in place at up to
  +13 deg/s.
- **WALK → STAND:** circle requests the switch; WALK keeps stepping with g = 0 and switches at the next touchdown
  (both soles down), at most `stand_switch_timeout` (1 s) later.
- **Switch reset:** on every policy switch a(t) and a(t−1) become clamp((q − default) / `action_scale`), as in the env
  reset. WALK starts at clock phase `walk_start_phase` (0.92).
- **Servo start:** the first q_des is ramped from the standing pose, not jumped to.
- **Per-mode low-pass:** walking 37.5 Hz (= training), standing 20 Hz (`stand_action_lowpass_hz`, an 11-20 robot fix
  for the stand shake).

## 5. Configuration (`config/robot/ARTI-H1/robot_olaf.yaml`)

| Key | Value | Must match |
|---|---|---|
| `model_file` | `agent_olaf_walk13_480000.onnx` (walk_13) | — |
| `olaf.stand_model_file` | `agent_olaf_stand_stage5_480000.onnx` (stand stage 5). Fallback: `agent_olaf_stand_stage3_480000.onnx` | — |
| `olaf.walk_keys_file` | `olaf_walk_keys.yaml` | `export_onnx.py --keys_out` (reference clip keys) |
| `policy.control_dt` | 0.02 | decimation 12 at 600 Hz |
| `policy.default_pose` | KB stance (hip_pitch 0.33555, knee −0.67109, ankle_pitch −0.33555) | env init joint_pos |
| `olaf.action_scale` | [0.3 ×4, 0.7, 0.7, 0.8, 0.8, 0.6, 0.6, 0.35, 0.35] | `OLAF_ACTION_SCALE` |
| `olaf.walk_period`, `walk_start_phase` | 0.8 s, 0.92 | reference period, `walk_start_phase` |
| `olaf.path_max_distance`, `path_max_yaw`, `stand_path_tau` | 0.10 m, 0.35 rad, 0.5 s | same names in the cfg |
| `olaf.walk_command_range`, `stand_command_range`, `stand_command_rate` | see yaml | same names in the cfg |
| `olaf.actuator_rate`, `action_lowpass_hz`, `action_rate_limit` | 600 Hz, 37.5 Hz, 5.24 rad/s | sim dt, `action_lowpass_hz`, `action_rate_limit` |
| `olaf.thermal_clip` | [70, 85] | `thermal_clip` |
| `olaf.contact_tolerance`, `olaf.ekf.*` | 0.012 m, noise values, `freeze_gyro_bias_z: true` | `est_contact_tolerance`, `ekf_*`, `ekf_freeze_gyro_bias_z` |
| `foots[].offset`, `sensors[].translation` | sole (−0.026, 0, −0.051), IMU (0.04688, 0, 0.34667) | `ARTI_H1_*_FOOT_OFFSET_FROM_FOOT_LINK`, IMU offset |

Motor homing and joint zero (−20/40/−20 deg) are in `motor.yaml` and are shared with the KB / velocity line; do not
change them for Olaf.

## 6. Deploying new policies

1. In the training project: `python scripts/export_onnx.py --checkpoint <run>/checkpoints/agent_N.pt
   --out <this repo>/config/robot/ARTI-H1/agent_olaf_<walkNN|stand_stageN>_<N>.onnx` (walk: add
   `--keys_out <tmp>.yaml` and diff it against `olaf_walk_keys.yaml`). The script checks ONNX against torch (≤ 1e-6).
   The run directory needs `params/env.yaml`; if the server sent only a checkpoint, copy the params of a stage whose
   export fields (joint names, init pose, action scales, joint limits, observation size) are identical.
2. Point `model_file` / `olaf.stand_model_file` at the new files. The comment states what the policy is, its key sim
   numbers, and the previous file (the fallback).
3. Commit the ONNX files and the yaml together; put the sim evaluation numbers in the commit message.
4. Deploy only policies that pass the criteria in `humanoid_sim/CLAUDE.md` section 4, or say clearly which criterion
   they fail (stand stage 5 was deployed on 11-24 with tremor proxy 0.21 > 0.1 at the user's request).

## 7. Robot test checklist

- Pull and rebuild on the robot PC after every change in C++ or the yaml (the yaml is read at start, the C++ needs a build).
- Hard floor (on carpet it fell after 2.9 s). New walking policies: hanger first, then small sticks.
- Watch for: standing tremor (8-10 Hz shaking); in-place rotation with sticks released; behaviour right after a turn
  ends and right after WALK → STAND.
- Faults print `[ArtiH1] t=... s FAULT: ...`; the CSV `fault` column has bits 1 IMU, 2 motor link, 4 joystick,
  8 holding. A fall (tilt > `max_tilt` 0.7 rad) damps the joints and exits. On 11-20 all 8 Unitree motors went stale at
  71 s at once (bus / power); the hold worked.
- Logs: the run writes `arti_h1_olaf_run_<timestamp>.csv` with `t, phase, obs_0..79, q_des_*, action_*,
  pos_/vel_/tau_/temp_<joint>, imu_q*, imu_g*, imu_a*, fault, olaf_walk, est_x..est_vz, path_x/y/psi, contact_l/r,
  g0..g3` (g0..g2 = path vx, vy, wz while walking). Analyse with the real-log metrics in `humanoid_sim/CLAUDE.md`
  section 4 and the replay tools (`scripts/real2sim/walk_replay/`).

## 8. Controller-side fixes so far

| Date | Problem on the robot | Fix |
|---|---|---|
| 11-17 | servo started from the first q_des (jump); hold / servo hand-off; FOH continuity; stand-switch g | review fixes (servo ramps from the standing pose, hold under the servo lock, ...) |
| 11-20 | stand shook at ~9.6 Hz at 37.5 Hz low-pass | per-mode low-pass: stand 20 Hz, walk 37.5 Hz |
| 11-20 | fell right after WALK → STAND: the stand policy got walk's ±1 actions as its history | `olaf_switch_reset` (a(t), a(t−1) from the measured pose, `action_scale`) |
| 11-20 | STAND switch in single support folded the knees | switch at the next touchdown, g = 0 meanwhile, 1 s timeout |
| 11-24 | in-place yaw drift up to +13 deg/s | `ekf.freeze_gyro_bias_z: true` (same in training) |

## 9. Open issues

- Stand stage 5 on the robot: if it shakes, switch `olaf.stand_model_file` back to `agent_olaf_stand_stage3_480000.onnx`.
- A short spike in the estimated yaw over the first ~4 s of walking (seen in sim replay).
- The Unitree bus updates commands at 125 Hz (8 ms master period); the sim does not model it.
- Encoder zero errors: the robot's leg kinematics use encoder angles, the sim's use the true joint angles.
