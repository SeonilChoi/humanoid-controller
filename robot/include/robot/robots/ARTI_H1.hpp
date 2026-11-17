#ifndef ROBOT_ROBOTS_ARTI_H1_HPP_
#define ROBOT_ROBOTS_ARTI_H1_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "robot/core/robot.hpp"
#include "robot/controllers/amp_controller.hpp"
#include "robot/robots/olaf_inekf.hpp"
#include "sensor/core/sensor_interface.hpp"

namespace robot {

class ArtiH1 : public Robot {
public:
    // model_file: ONNX policy to run instead of robot.yaml model_file (main.cpp --ppo / --model); empty = robot.yaml.
    explicit ArtiH1(const std::string& config_file, const std::string& model_file = "");

    virtual ~ArtiH1();

    void initialize() override;

    void observation() override;

    void control() override;

    // Observation check (main.cpp --observe): print every value of the last observation() in readable units (no
    // policy, the motors keep holding the default pose). Tilt / move the robot by hand and check the signs.
    void print_observation() const;

    // Append the last observation() to the runtime log (if robot.yaml log_file is set); used by --observe.
    void log_observation();

    // Test mode (main.cpp --blind-imu): the policy gets the pelvis tilt latched at the first observation and zero
    // angular velocity instead of the IMU; the real IMU is still logged and still used for the max_tilt fall check.
    void set_blind_imu(bool blind) { blind_imu_ = blind; }
    // IMU stabilisation before the policy starts (main.cpp --imu-calib / --no-imu-calib), overrides policy.imu_still_time:
    // on = gyro bias average over imu_still_time (10 s if the yaml has 0) + wait for the attitude to converge; off = skip.
    void set_imu_calibration(bool on) { imu_still_time_ = on ? (imu_still_time_ > 0.0 ? imu_still_time_ : 10.0) : 0.0; }
    double imu_still_time() const { return imu_still_time_; }
    // control period of the policy loop (robot.yaml policy.control_dt): main.cpp runs observation() / control() at it
    double control_dt() const { return control_dt_; }
    // stop the olaf servo thread (it writes the motor commands); main.cpp calls it before Robot::stop()
    void shutdown_servo();

    // Zero check (main.cpp --observe --limp-right): after initialize() the right-leg motors get kp 0 (policy kd only,
    // so the leg is limp but damped) while the left leg keeps holding the default pose. Hold the pelvis level with
    // both soles pressed flat on the floor, the right foot placed as the mirror image of the left; the right-leg
    // readings then differ from the left-leg ones by the right-leg zero error (motor.yaml has the same
    // homing_offset on both sides). update_limp_stats() averages that difference every control step (~2 s window),
    // print_limp_report() prints it with the homing_offset that would remove it.
    void set_limp_right();

    void update_limp_stats();

    void print_limp_report() const;

private:
    void home(uint8_t index);

    std::vector<pinocchio::JointIndex> joint_ids_;

    std::unordered_map<pinocchio::JointIndex, uint8_t> joint_id_to_motor_index_;

    uint8_t imu_sensor_index_{};

    pinocchio::FrameIndex imu_frame_id_{};

    Eigen::Vector3d imu_translation_{};

    Eigen::Quaterniond frame_q_imu_{
        Eigen::Quaterniond::Identity()
    };

    std::vector<pinocchio::FrameIndex> foot_frame_ids_;

    std::vector<Eigen::Vector3d> foot_toe_offsets_;

    // Policy settings from robot.yaml 'policy' (must match the training env, see arti_h1_env_cfg.py).
    controller::AmpController* amp_controller_{nullptr};

    std::array<double, 12> default_pose_{};   // first reference frame, policy order

    // robot.yaml policy.observation: "amp" (116-D ArtiH1AmpEnv layout, default) or "velocity" (45-D
    // ArtiH1VelocityEnv layout: pelvis gyro 3, gravity in pelvis frame 3, joint pos - default 12, joint vel 12,
    // previous actions 12, command 3). Fixed at construction; observation_size_ follows it.
    bool velocity_obs_{false};

    // robot.yaml policy.observation: "velocity_history" = 81-D (ArtiH1VelocityEnvCfg.obs_action_history): the 45-D
    // layout with the last three actions a(t), a(t-1), a(t-2) and the last sent joint target - default pose before
    // the command.
    bool action_history_obs_{false};

    // robot.yaml policy.target_rate_limit (rad/s, 0 = off; ArtiH1VelocityEnvCfg.target_rate_limit): each control
    // step the joint target moves at most rate * control_dt from the previous target (Open Duck max_motor_velocity).
    double target_rate_limit_{0.0};

    // robot.yaml policy.target_lowpass_alpha (ArtiH1VelocityEnvCfg.target_lowpass_alpha, 1 = off): before the rate limit
    // the joint target becomes previous + alpha * (q_des - previous). 11-11 r12 run: without it the joints chattered
    // at 30 Hz (period-2 limit cycle between the policy and the fast real joints).
    double target_lowpass_alpha_{1.0};

    std::size_t observation_size_{116};

    std::array<double, 12> policy_kp_{};      // motor.yaml kp / kd, driver units, motor index order

    std::array<double, 12> policy_kd_{};

    std::array<double, 3> command_scale_{};   // observation = command * command_scale

    std::array<double, 3> command_max_{};     // command at full stick: vx (m/s), vy (m/s), yaw rate (rad/s)

    double gait_phase_rate_{0.0};             // gait clock, cycles per second

    double control_dt_{1.0 / 60.0};

    double gait_phase_{0.0};                  // 0 = left single support start, as in the reference clips

    double initial_gait_phase_{0.0};          // clock at the policy start (training standstill = reference frame 0)

    double max_tilt_{0.7};

    double imu_still_time_{0.0};              // s: hold the default pose still to average the gyro bias; 0 = off

    double imu_converge_tolerance_{0.5};      // deg: IMU roll / pitch vs accelerometer tilt before the policy starts

    double imu_converge_timeout_{180.0};      // s: start anyway (with a warning) after this long

    std::array<double, 3> imu_gyro_bias_{};   // rad/s, sensor frame; subtracted from the gyro in observation()

    double imu_attitude_gain_{0.0};           // 1/s: own gyro + accelerometer attitude filter; 0 = MTi orientation

    Eigen::Quaterniond attitude_q_{Eigen::Quaterniond::Identity()};   // sensor to world (own filter)

    bool attitude_initialized_{false};

    std::chrono::steady_clock::time_point attitude_time_{};

    std::array<double, 4> last_attitude_{};   // w, x, y, z: the orientation the observation used

    double imu_stale_time_{0.2};              // s: stop when the IMU gyro / accelerometer samples stop changing this long

    std::array<double, 6> imu_last_sample_{};

    std::chrono::steady_clock::time_point imu_sample_time_{};

    bool imu_stale_{false};                   // set in observation(), control() stops on it

    bool blind_imu_{false};

    bool blind_latched_{false};

    Eigen::Matrix3d blind_heading_R_root_{Eigen::Matrix3d::Identity()};

    double true_tilt_cos_{1.0};               // z of the real pelvis z axis (max_tilt check, also when blind)

    Eigen::Quaterniond update_attitude(const sensor_interface::imu_data_t& imu_data, const Eigen::Quaterniond& mti_q);

    // Standing (training cfg gait_rest_phases / stand_command_threshold): under a zero command the clock runs on to
    // the next rest phase (mid double support) and stops there; empty = the clock always runs.
    std::vector<double> gait_rest_phases_;

    double stand_command_threshold_{0.0};

    double stick_deadzone_{0.0};                 // |stick| below it counts as 0, so a released stick means "stand"

    std::array<double, 3> command_{};           // last command (vx m/s, vy m/s, yaw rate rad/s)                    // rad; beyond it the policy stops and the joints are only damped

    std::array<double, 12> previous_actions_{};

    std::array<double, 12> previous_actions2_{};   // a(t-1), velocity_history observation

    std::array<double, 12> previous_actions3_{};   // a(t-2), velocity_history observation

    std::array<double, 12> q_target_{};            // last joint target sent (rad, policy order, after the rate limit)

    // Runtime log (robot.yaml 'log_file', empty = off): one CSV row per control step with the observation, the ONNX
    // outputs, the motor states and the raw IMU, so a real run can be replayed / compared with the simulator.
    std::FILE* log_file_{nullptr};

    std::chrono::steady_clock::time_point log_start_{};

    motor_interface::motor_state_t last_motor_status_[12]{};

    bool limp_right_{false};

    std::array<double, 12> limp_mean_pos_{};   // policy order, EMA of the joint positions while the right leg is limp

    bool limp_stats_init_{false};

    std::array<double, 4> last_imu_orientation_{};   // w, x, y, z (sensor to world)

    std::array<double, 3> last_imu_angular_velocity_{};

    std::array<double, 3> last_imu_linear_acceleration_{};   // sensor frame, m/s^2 (gravity included)

    double last_imu_temperature_{0.0};

    void write_log_row();

    // Fault hold (11-13, user): an IMU stop (no new samples for imu_stale_time), a motor link fault (MotorManager
    // comm_fault) or a lost joystick no longer damps the joints and exits. The policy stops, every joint is held by
    // the policy kp / kd at the last joint target, which then moves to the default stance over fault_settle_time
    // (robot.yaml policy.fault_settle_time, s; 0 = freeze exactly where it was), and the program keeps running and
    // logging (CSV column 'fault') until Ctrl+C. The hold is latched: a fault that clears is reported but the policy
    // does not restart. A fall (tilt > max_tilt) still damps the joints and exits.
    enum FaultBits : int { FAULT_IMU = 1, FAULT_MOTOR_COMM = 2, FAULT_JOY = 4, FAULT_HOLDING = 8 };
    int fault_status() const;
    void report_faults(int status);
    void hold_control();
    int fault_{0};                                   // last fault_status() | FAULT_HOLDING while holding
    bool holding_{false};
    std::array<double, 12> hold_from_{};             // policy order: joint target when the hold started
    std::chrono::steady_clock::time_point hold_start_{};
    double fault_settle_time_{1.0};
    // robot.yaml policy.joy_loss_hold (default false): a lost joystick only logs a FAULT and the policy keeps running
    // on the zero command (stands, walks again once the pad is back); true = hold the pose like an IMU / motor fault.
    // 11-13: the DualSense dropped every ~10 s over Bluetooth and the latched hold made every run unusable.
    bool joy_loss_hold_{false};

    // Damping only (kp 0) on every joint, then throw: used when the robot falls.
    [[noreturn]] void safe_stop(const std::string& reason);

    // ---------------------------------------------------------------------------------------------------------------
    // robot.yaml policy.observation: "olaf" = arti_h1_olaf (arXiv 2512.16705) walking + standing policies, one ONNX
    // each (model_file = walking, olaf.stand_model_file = standing), switched with the pad (cross = walk, circle =
    // stand; the robot starts standing). Observation = ArtiH1OlafEnv._get_observations (policy order):
    //   pelvis position relative to the path frame (x, y, height) 3, pelvis orientation relative to the path frame
    //   (first two rotation-matrix columns, row-major) 6, pelvis linear velocity (pelvis frame) 3, gyro (pelvis frame) 3,
    //   joint pos - default 12, joint vel 12, a(t) 12, a(t-1) 12, motor temperatures (clip [70, 85] C, scaled to
    //   [-1, 1]) 12, then walking: gait phase (cos, sin) 2 + path velocity g (vx, vy, wz) 3 (80-D);
    //   standing: g = (dz, roll, pitch, yaw) of the trunk 4 (79-D).
    // The pelvis pose / velocity / orientation come from olaf_estimate(): the contact-aided invariant EKF (OlafInEKF, the
    // paper's [42]) on the MTi-3 gyro / accelerometer and the leg kinematics; the path frame / phase / g follow the env
    // (olaf_advance()). Runtime (paper Sec. VII-C): the policy runs at 50 Hz (policy.control_dt 0.02); its q_des is
    // upsampled with a first-order hold to olaf.actuator_rate (600 Hz), slew limited and low-pass filtered at
    // olaf.action_lowpass_hz (37.5 Hz) in the servo thread (olaf_servo_loop), which writes the motor commands.
    bool olaf_obs_{false};
    bool olaf_walking_{false};
    std::unique_ptr<controller::AmpController> olaf_stand_controller_;
    double olaf_walk_period_{0.8};
    double olaf_walk_start_phase_{0.92};
    double olaf_path_max_distance_{0.10};
    double olaf_path_max_yaw_{0.35};
    double olaf_stand_path_tau_{0.5};
    std::array<double, 2> olaf_thermal_clip_{70.0, 85.0};
    std::array<std::array<double, 2>, 3> olaf_walk_range_{};     // vx, vy, wz (min, max): full stick
    std::array<std::array<double, 2>, 4> olaf_stand_range_{};    // dz, roll, pitch, yaw (min, max): full stick
    std::array<double, 4> olaf_stand_rate_{};                    // max g change per second (standing)
    std::vector<std::array<double, 3>> olaf_walk_keys_;          // reference clip keys (path velocity)
    int olaf_knn_{4};
    double olaf_contact_tolerance_{0.012};                       // m: a sole this close to the lowest sole is in contact
    // estimator (ArtiH1OlafEnv._estimate): the filter restarts from the MTi-3 attitude on the first olaf_restart_steps
    // estimates (the env: episode_length_buf <= 2), then propagates / corrects every control step
    OlafInEKF olaf_ekf_;
    int olaf_est_steps_{0};
    static constexpr int olaf_restart_steps_ = 3;
    bool olaf_est_init_{false};
    Eigen::Matrix3d olaf_R_{Eigen::Matrix3d::Identity()};       // estimated world_R_pelvis
    Eigen::Vector3d olaf_pos_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d olaf_vel_{Eigen::Vector3d::Zero()};
    std::array<bool, 2> olaf_contact_{{true, true}};
    std::array<Eigen::Vector3d, 2> olaf_feet_world_{};
    std::array<double, 2> olaf_feet_yaw_{};
    double olaf_root_yaw_{0.0};
    std::array<double, 3> olaf_path_{};                          // x, y, psi
    double olaf_phase_{0.0};
    std::array<double, 4> olaf_g_{};
    std::array<double, 4> olaf_g_target_{};
    bool olaf_prev_cross_{false};
    bool olaf_prev_circle_{false};
    bool olaf_stand_switch_{false};
    std::array<double, 3> olaf_project_walk(const std::array<double, 3>& g) const;
    void olaf_estimate(const Eigen::Matrix3d& world_R_root, const Eigen::Vector3d& gyro_root,
                       const Eigen::Vector3d& accel_root, const Eigen::Vector3d& imu_root);
    void olaf_reset_path();
    void olaf_advance();

    // servo thread (runtime chain): first-order hold from olaf_q_des_prev_ to olaf_q_des_ over the control period,
    // slew limit olaf_rate_limit_ (rad/s), low-pass alpha = 1 - exp(-2 pi f_c / actuator_rate); policy order.
    // Mirrors ArtiH1OlafEnv._apply_action (physics step = one servo tick).
    double olaf_actuator_rate_{600.0};
    double olaf_lowpass_hz_{37.5};
    double olaf_rate_limit_{5.24};
    std::mutex olaf_servo_mutex_;
    std::thread olaf_servo_thread_;
    std::atomic<bool> olaf_servo_run_{false};       // thread alive
    std::atomic<bool> olaf_servo_active_{false};    // thread writes the motor commands (false while holding / stopped)
    std::array<double, 12> olaf_q_des_prev_{};
    std::array<double, 12> olaf_q_des_{};
    std::array<double, 12> olaf_q_filt_{};
    int olaf_servo_tick_{0};                        // servo ticks since the last q_des
    void olaf_servo_start();
    void olaf_servo_stop();
    void olaf_servo_loop();
    void olaf_servo_set(const std::array<double, 12>& q_des);
};

} // namespace robot

#endif // ROBOT_ROBOTS_ARTI_H1_HPP_