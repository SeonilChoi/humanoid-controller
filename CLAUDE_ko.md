# humanoid_controller, 브랜치 `arti-h1-olaf`: Claude 작업 지침

이 저장소는 Olaf 방식(arXiv 2512.16705)의 **보행 정책과 서기 정책**을 실물 ARTI-H1 로봇에서 돌린다.
정책은 `/opt/isaac/projects/humanoid_sim`(`arti_h1_olaf`의 후속)에서 학습한다. 학습 내역, real2sim, 평가 기준은
그 프로젝트의 `CLAUDE.md`를 본다. 이 문서는 컨트롤러 쪽만 다룬다. 영어판은 `CLAUDE.md`이고, 두 파일은 같은 내용으로 유지한다.

## 1. 원칙

- **컨트롤러는 학습 env를 그대로 재현해야 한다.** 관측 항목, 리셋, 필터, 클럭은 모두 `ArtiH1OlafEnv`
  (`humanoid_sim/.../tasks/direct/olaf/olaf_env.py`, `olaf_env_cfg.py`)를 따른다. 한쪽을 바꾸면 같은 작업 안에서
  다른 쪽도 바꾼다. 지금까지 실물 문제는 대부분 정책이 아니라 이런 불일치(전환 리셋, 서보 시작, 추정기 바이어스)에서 나왔다.
- **컨트롤러는 학습과 맞추거나 안전을 위해서만 바꾼다.** 정책이 학습하지 않은 동작을 여기서 튜닝하지 않는다
  (예: 보행에 다른 LPF를 쓰면 안 된다. 20 Hz로 바꿨을 때 sim 보행 넘어짐이 33배가 됐다).
- **모든 배포는 되돌릴 수 있어야 한다.** 이전 ONNX를 `config/robot/ARTI-H1/`에 남기고, yaml 주석에 되돌릴 파일을 적는다.

## 2. 저장소와 작업 규칙

- **저장소:** `/opt/isaac/projects/humanoid_controller_olaf`는 SeonilChoi/humanoid_controller의 `arti-h1-olaf` 브랜치를 받은
  독립 저장소다(11-24부터. 그 전에는 `/opt/isaac/projects/humanoid_controller`의 worktree였는데 원본이 지워졌다).
  다른 브랜치(`arti-h1-new-pelvis`, `arti-d1` 등)는 원격에만 있다. `git stash`보다 임시 WIP 커밋을 쓴다.
- **자격 증명:** 셸에 없다. 사용자가 직접 push하거나 메시지로 토큰을 준다. 토큰은 그 push 명령에서만 쓰고
  (`https://x-access-token:<token>@github.com/SeonilChoi/humanoid_controller.git arti-h1-olaf:arti-h1-olaf`) 저장하지 않는다.
  커밋과 push는 요청받았을 때만 한다. 사용자가 로봇 PC에서 이 브랜치로 로그를 커밋하므로, 커밋 전에 `git fetch`하고
  fast-forward 충돌을 예상한다.
- **빌드:** 로봇 PC(Jetson Orin, `~/humanoid_controller_olaf`)에서만 된다: `mkdir -p build && cd build && cmake .. && make -j2`.
  이 개발 PC에는 yaml-cpp, onnxruntime, 시스템 Eigen이 없어 전체 빌드를 확인할 수 없다. 헤더만 바꾼 경우(예: `olaf_inekf.hpp`)는
  로컬 Eigen 복사본으로 작은 테스트 프로그램을 컴파일해 확인하고, 전체 빌드는 확인하지 못했다고 분명히 말한다.
- **실물 로그:** `arti_h1_olaf_run_<YYYYMMDD_HHMMSS>.csv`. 사용자가 로봇에서 커밋한다. 커밋 6af2da5("log deleted", 11-24)에서
  11-21, 11-23 로그가 트리에서 지워졌다. `git show 42ad823:arti_h1_olaf_run_20261123_132540.csv > /tmp/...`로 꺼낸다.

## 3. 실행

```bash
cd ~/humanoid_controller_olaf/build
./ARTI-H1 ../config/robot/ARTI-H1/robot_olaf.yaml             # STAND로 시작; 패드 cross = WALK, circle = STAND
./ARTI-H1 ../config/robot/ARTI-H1/robot_olaf.yaml --observe   # 추정기 / 관측 확인, 정책 없음
```
그 밖의 옵션: `--imu-calib` / `--no-imu-calib`(기본은 `policy.imu_still_time: 10` s 동안 자이로 바이어스와 자세 수렴),
`--blind-imu`, `--model <onnx>`. Ctrl+C로 멈추면 로그가 저장된다.

스틱: WALK 왼쪽 스틱 = 경로 vx / vy, 오른쪽 스틱 좌우 = 회전(vx −0.12..0.15 m/s, vy ±0.068 m/s, wz ±0.4 rad/s,
학습과 같이 참조 키 범위로 투영). STAND 왼쪽 스틱 = 몸통 pitch / roll, 오른쪽 스틱 = 몸통 yaw / 높이(dz −3..+1.5 cm).

## 4. Olaf 코드 구조

| 부분 | 코드 | 학습 쪽 대응 |
|---|---|---|
| 관측 (보행 80차원, 서기 79차원) | `ArtiH1::observation()`의 `olaf_obs_` 분기, 배치는 `ARTI_H1.hpp` | `ArtiH1OlafEnv._get_observations` |
| 상태 추정 | `ArtiH1::olaf_estimate`, `robot/robots/olaf_inekf.hpp` (`OlafInEKF`) | `ArtiH1OlafEnv._estimate`, `olaf/inekf.py` |
| 경로 좌표계, 클럭, g | `olaf_advance`, `olaf_reset_path`, `olaf_project_walk` | `_pre_physics_step`, `_step_path`, `_walk_blend` |
| 정책 전환 | `observation()`의 cross / circle 처리, `olaf_switch_reset` | `_reset_idx` (reset_stand_pose / switch bank) |
| 런타임 체인 | `olaf_servo_loop` (600 Hz 스레드): FOH → 기울기 제한 5.24 rad/s → 1차 LPF | 물리 스텝마다 `_apply_action` |
| 고장 처리 | `control()`: tilt > `max_tilt`이면 감쇠 후 종료, IMU / 모터 링크 / (조이스틱) 고장은 자세 유지 | — |

관측 배치(정책 관절 순서): [0:3] 경로 좌표계 기준 골반 위치(x, y, 높이), [3:9] 경로 좌표계 기준 골반 자세(회전 행렬의
첫 두 열, row-major), [9:12] 골반 좌표계 선속도, [12:15] 자이로, [15:27] 관절각 − 기본 자세, [27:39] 관절 속도,
[39:51] a(t), [51:63] a(t−1), [63:75] 모터 온도([70, 85] °C로 clip, [−1, 1]로 스케일), 이어서 보행: [75:77] 위상 cos / sin,
[77:80] g = (vx, vy, wz). 서기: [75:79] g = (dz, roll, pitch, yaw). 경로 기준 상대 yaw = atan2(obs_5, obs_3).

주요 동작:
- **추정기:** 처음 3 제어 스텝 동안 MTi-3 자세로 시작한다. 가장 낮은 발바닥에서 `contact_tolerance`(12 mm) 안에 있으면 접촉으로 본다.
  `ekf.freeze_gyro_bias_z: true`는 자이로 z 바이어스를 0으로 고정한다(11-24). 바이어스를 자유롭게 두면 실제 yaw 회전을
  흡수해서, 추정 heading은 멈춰 있는데 로봇은 제자리에서 최대 +13°/s로 돌았다.
- **WALK → STAND:** circle이 전환을 요청하면 WALK는 g = 0으로 계속 걷다가 다음 착지(양발 접촉) 때 전환한다.
  최대 `stand_switch_timeout`(1 s)까지 기다린다.
- **전환 리셋:** 정책이 바뀔 때마다 a(t), a(t−1)을 clamp((q − default) / `action_scale`)로 다시 설정한다(env 리셋과 같다).
  WALK는 클럭 위상 `walk_start_phase`(0.92)에서 시작한다.
- **서보 시작:** 첫 q_des로 바로 점프하지 않고 서 있는 자세에서 ramp한다.
- **모드별 LPF:** 보행 37.5 Hz(= 학습), 서기 20 Hz(`stand_action_lowpass_hz`, 11-20 서기 떨림 때문에 실물에서 넣은 설정).

## 4a. 조이스틱: 정책 전환과 조정 (BD-X 참고)

조작 방식은 Olaf의 전작인 Disney BD-X 로봇(Grandia et al., "Design and Control of a Bipedal Robotic Character",
RSS 2024)을 따른다. **모드마다 정책을 따로 두고 버튼으로 고르며, 스틱은 관절 목표가 아니라 각 정책의 제어 입력 g를 정한다.**
패드 처리를 바꿀 때도 이 구조를 유지한다.

- **모드 버튼:** cross = WALK, circle = STAND (누르는 순간에만 반응). 로봇은 STAND로 시작한다. 한 번에 한 정책만 돌고,
  다른 ONNX는 로드된 채로 대기한다.
- **스틱은 모드별 정책 제어 입력이다:**
  - WALK: g = 경로 속도 (vx, vy, wz). 왼쪽 스틱 vx / vy, 오른쪽 스틱 좌우 = wz. 스틱 끝 = 학습 명령 범위.
    명령은 env가 g를 뽑을 때와 똑같이 참조 범위로 투영한다(`olaf_project_walk`, 보행 키 4-NN 블렌드). 그래서 정책은
    학습하지 않은 g를 받지 않는다.
  - STAND: g = 몸통 자세 (dz, roll, pitch, yaw). 왼쪽 스틱 pitch / roll, 오른쪽 스틱 yaw / 높이. 스틱은 목표만 정하고
    g는 초당 `stand_command_rate`만큼 따라간다(env의 속도 제한 추종). 스틱을 갑자기 움직여도 입력이 계단처럼 바뀌지 않는다.
  - 모든 축에 `stick_deadzone` 0.1이 적용된다. 스틱을 놓으면 g = 0이다(WALK: 제자리 걸음, STAND: 기본 자세).
- **BD-X식 전환:** WALK → STAND는 즉시 일어나지 않는다. circle은 요청만 하고, WALK는 g = 0으로 계속 걷다가 다음 착지
  (양발 접촉) 때 전환한다. 최대 `stand_switch_timeout`(1 s)까지 기다린다. sim(`scripts/eval_switch.py`, walk_09 + stand_07)에서
  전환 뒤 넘어짐이 24.6%(즉시 전환)에서 17.6%로 줄었다. STAND → WALK는 서 있는 자세에서 보행 클럭 `walk_start_phase` 0.92로
  시작한다(env의 reset_stand_pose).
- **BD-X와 다른 점:** BD-X는 출력이 매끄러워서 전환 때 출력 이력을 그대로 넘긴다. 우리 보행 정책은 무릎 출력이 포화될
  만큼 매끄럽지 않고, 보행의 ±1 출력을 서기 정책에 넘겼다가 11-20에 넘어졌다. 그래서 전환할 때마다 a(t), a(t−1)을
  측정 자세로 다시 설정한다(`olaf_switch_reset`, `_reset_idx`와 동일). 경로 좌표계도 전환 때 골반 위치로 리셋한다(`olaf_reset_path`).
- **시도했다가 뺀 것 (11-20):** WALK에서 스틱을 놓으면 자동으로 STAND로 바꾸는 기능(`walk_stop_hold`)과 양발 지지 위상
  게이트. 사용자가 버튼으로 명시적으로 전환하는 방식을 택했으니, 묻지 않고 자동 전환을 다시 넣지 않는다.
- 여기를 바꾸면 학습 쪽도 맞춰야 한다. 서기 정책은 보행 정책의 전환 상태(`switch_bank_file`, `reset_walk_pose_fraction`)에서,
  보행 정책은 `walk_start_phase`의 서 있는 자세에서 시작하도록 학습한다.

## 5. 설정 (`config/robot/ARTI-H1/robot_olaf.yaml`)

| 키 | 값 | 맞춰야 할 대상 |
|---|---|---|
| `model_file` | `agent_olaf_walk13_480000.onnx` (walk_13) | — |
| `olaf.stand_model_file` | `agent_olaf_stand_stage5_480000.onnx` (서기 stage5). 되돌릴 파일: `agent_olaf_stand_stage3_480000.onnx` | — |
| `olaf.walk_keys_file` | `olaf_walk_keys.yaml` | `export_onnx.py --keys_out` (참조 클립 키) |
| `policy.control_dt` | 0.02 | 600 Hz에서 decimation 12 |
| `policy.default_pose` | 서 있는 기본 자세 (hip_pitch 0.33555, knee −0.67109, ankle_pitch −0.33555) | env init joint_pos |
| `olaf.action_scale` | [0.3 ×4, 0.7, 0.7, 0.8, 0.8, 0.6, 0.6, 0.35, 0.35] | `OLAF_ACTION_SCALE` |
| `olaf.walk_period`, `walk_start_phase` | 0.8 s, 0.92 | 참조 주기, `walk_start_phase` |
| `olaf.path_max_distance`, `path_max_yaw`, `stand_path_tau` | 0.10 m, 0.35 rad, 0.5 s | cfg의 같은 이름 |
| `olaf.walk_command_range`, `stand_command_range`, `stand_command_rate` | yaml 참고 | cfg의 같은 이름 |
| `olaf.actuator_rate`, `action_lowpass_hz`, `action_rate_limit` | 600 Hz, 37.5 Hz, 5.24 rad/s | sim dt, `action_lowpass_hz`, `action_rate_limit` |
| `olaf.thermal_clip` | [70, 85] | `thermal_clip` |
| `olaf.contact_tolerance`, `olaf.ekf.*` | 0.012 m, 노이즈 값, `freeze_gyro_bias_z: true` | `est_contact_tolerance`, `ekf_*`, `ekf_freeze_gyro_bias_z` |
| `foots[].offset`, `sensors[].translation` | 발바닥 (−0.026, 0, −0.051), IMU (0.04688, 0, 0.34667) | `ARTI_H1_*_FOOT_OFFSET_FROM_FOOT_LINK`, IMU 오프셋 |

모터 호밍과 관절 영점(−20/40/−20°)은 `motor.yaml`에 있고 velocity 정책 라인과 공유한다. Olaf 때문에 바꾸지 않는다.

## 6. 새 정책 배포

1. 학습 프로젝트에서 `python scripts/export_onnx.py --checkpoint <run>/checkpoints/agent_N.pt
   --out <이 저장소>/config/robot/ARTI-H1/agent_olaf_<walkNN|stand_stageN>_<N>.onnx`로 변환한다(보행은 `--keys_out <tmp>.yaml`을
   추가해 `olaf_walk_keys.yaml`과 diff). 스크립트가 ONNX와 torch의 차이(≤ 1e-6)를 확인한다. 실행 폴더에 `params/env.yaml`이
   있어야 한다. 서버에서 체크포인트만 왔으면, export에 쓰는 항목(관절 이름, 초기 자세, 출력 스케일, 관절 한계, 관측 크기)이
   같은 다른 stage의 params를 복사한다.
2. `model_file` / `olaf.stand_model_file`을 새 파일로 바꾼다. 주석에 정책 설명, 주요 sim 수치, 이전 파일(되돌릴 파일)을 적는다.
3. ONNX 파일과 yaml을 함께 커밋하고, 커밋 메시지에 sim 평가 수치를 남긴다.
4. `humanoid_sim/CLAUDE.md` 4절 기준을 통과한 정책만 배포한다. 통과하지 못한 기준이 있으면 분명히 밝힌다
   (서기 stage5는 11-24에 떨림 프록시 0.21 > 0.1인 상태로 사용자 요청에 따라 배포했다).

## 7. 실물 테스트 점검 목록

- C++나 yaml을 바꿀 때마다 로봇 PC에서 pull하고 다시 빌드한다(yaml은 시작할 때 읽고, C++는 빌드가 필요하다).
- 하드 바닥에서 테스트한다(카펫에서는 2.9 s 만에 넘어졌다). 새 보행 정책은 행거에서 먼저, 그다음 작은 스틱 입력으로 한다.
- 확인할 것: 서기 떨림(8~10 Hz 흔들림), 스틱을 놓았을 때 제자리 회전, 회전이 끝난 직후와 WALK → STAND 직후의 동작.
- 고장은 `[ArtiH1] t=... s FAULT: ...`로 출력된다. CSV `fault` 열 비트: 1 IMU, 2 모터 링크, 4 조이스틱, 8 자세 유지 중.
  넘어지면(tilt > `max_tilt` 0.7 rad) 관절을 감쇠하고 종료한다. 11-20에는 71 s에 Unitree 모터 8개가 동시에 stale이 됐다
  (버스 / 전원 문제). 자세 유지는 정상 동작했다.
- 로그: `arti_h1_olaf_run_<timestamp>.csv`에 `t, phase, obs_0..79, q_des_*, action_*, pos_/vel_/tau_/temp_<joint>, imu_q*,
  imu_g*, imu_a*, fault, olaf_walk, est_x..est_vz, path_x/y/psi, contact_l/r, g0..g3`가 기록된다(보행 중 g0..g2 = 경로 vx, vy, wz).
  분석은 `humanoid_sim/CLAUDE.md` 4절의 실물 로그 지표와 리플레이 도구(`scripts/real2sim/walk_replay/`)로 한다.

## 8. 지금까지의 컨트롤러 수정

| 날짜 | 실물 문제 | 수정 |
|---|---|---|
| 11-17 | 서보가 첫 q_des에서 시작(점프), hold / 서보 인계, FOH 연속성, 서기 전환 g | 리뷰 수정(서 있는 자세에서 ramp, 서보 lock 아래에서 hold 등) |
| 11-20 | 37.5 Hz LPF에서 서기가 약 9.6 Hz로 떨림 | 모드별 LPF: 서기 20 Hz, 보행 37.5 Hz |
| 11-20 | WALK → STAND 직후 넘어짐: 서기 정책이 보행의 ±1 출력을 이력으로 받음 | `olaf_switch_reset` (측정 자세로 a(t), a(t−1) 설정, `action_scale`) |
| 11-20 | 단일 지지 중 STAND 전환으로 무릎이 꺾임 | 다음 착지 때 전환, 그동안 g = 0, 타임아웃 1 s |
| 11-24 | 제자리 yaw 회전 최대 +13°/s | `ekf.freeze_gyro_bias_z: true` (학습도 동일) |

## 9. 열린 문제

- 서기 stage5의 실물 결과. 떨리면 `olaf.stand_model_file`을 `agent_olaf_stand_stage3_480000.onnx`로 되돌린다.
- 보행 시작 후 첫 약 4 s 동안 추정 yaw가 잠깐 튄다(sim 리플레이에서 확인).
- Unitree 버스는 125 Hz(마스터 주기 8 ms)로 지령을 갱신하지만 sim은 이를 모델링하지 않는다.
- 엔코더 영점 오차: 로봇의 다리 기구학은 엔코더 각도를 쓰고, sim은 참 관절각을 쓴다.
