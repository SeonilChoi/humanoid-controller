#include <algorithm>
#include <array>
#include <cmath>
#include <ctime>
#include <string>
#include <thread>
#include <chrono>
#include <iostream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "motor/core/motor_interface.hpp"
#include "sensor/core/sensor_interface.hpp"
#include "joy/core/joy_interface.hpp"

#include "sensor/core/imu.hpp"
#include "robot/controllers/amp_controller.hpp"

#include "robot/robots/ARTI_H1.hpp"

namespace {

constexpr std::size_t NUM_JOINTS = 12;
// root tan-norm 6, angular velocity 3, joint tan-norm 72, joint velocity 12, toes 6, previous actions 12,
// gait phase sin/cos 2, commands 3 (ArtiH1AmpEnv._get_observations).
constexpr std::size_t OBSERVATION_SIZE = 116;
// ArtiH1VelocityEnv._get_observations: gyro 3, gravity 3, joint pos - default 12, joint vel 12, previous actions 12,
// command 3 (robot.yaml policy.observation: velocity).
constexpr std::size_t VELOCITY_OBSERVATION_SIZE = 45;
// obs_action_history: gyro 3, gravity 3, joint pos - default 12, joint vel 12, actions a(t), a(t-1), a(t-2) 36,
// joint target - default 12, command 3 (robot.yaml policy.observation: velocity_history).
constexpr std::size_t VELOCITY_HISTORY_OBSERVATION_SIZE = 81;
// arti_h1_olaf ArtiH1OlafEnv (robot.yaml policy.observation: olaf): walking 80, standing 79 (see ARTI_H1.hpp).
constexpr std::size_t OLAF_WALK_OBSERVATION_SIZE = 80;
constexpr std::size_t OLAF_STAND_OBSERVATION_SIZE = 79;
constexpr auto STAND_UP_TIME = std::chrono::seconds(3);
constexpr double PI = 3.14159265358979323846;

constexpr auto HOMING_PERIOD = std::chrono::milliseconds(10);
constexpr auto HOMING_SETTLE_TIME = std::chrono::milliseconds(500);
constexpr auto HOMING_TIMEOUT = std::chrono::seconds(15);
constexpr int HOMING_STALL_COUNT = 10;
constexpr double HOMING_MAX_LEAD = 0.15;
constexpr double HOMING_VELOCITY_FILTER = 0.2;
constexpr double HOMING_ZERO_TOLERANCE = 0.03;
constexpr auto HOMING_ZERO_TIMEOUT = std::chrono::seconds(3);

constexpr const char* kPolicyJointNames[NUM_JOINTS] = {
    "left_hip_yaw",
    "right_hip_yaw",
    "left_hip_roll",
    "right_hip_roll",
    "left_hip_pitch",
    "right_hip_pitch",
    "left_knee",
    "right_knee",
    "left_ankle_pitch",
    "right_ankle_pitch",
    "left_ankle_roll",
    "right_ankle_roll",
};

void append_vector(const Eigen::Vector3d& value, std::vector<double>& vector) {
    vector.push_back(value.x());
    vector.push_back(value.y());
    vector.push_back(value.z());
}

template <std::size_t M>
std::array<double, M> read_array(const YAML::Node& node, const char* key) {
    const auto values = node[key].as<std::vector<double>>();
    if (values.size() != M) {
        throw std::runtime_error(std::string("[ArtiH1] robot.yaml policy.") + key + " needs " + std::to_string(M) + " values.");
    }
    std::array<double, M> out{};
    std::copy(values.begin(), values.end(), out.begin());
    return out;
}

void append_rotation(const Eigen::Matrix3d& value, std::vector<double>& vector) {
    append_vector(value.col(0), vector);
    append_vector(value.col(2), vector);
}

} // namespace

robot::ArtiH1::ArtiH1(const std::string& config_file, const std::string& model_file)
    : robot::Robot(config_file) {
    const bool model_override = !model_file.empty() && model_file != model_file_;
    if (!model_file.empty()) model_file_ = model_file;
    std::cout << "[ArtiH1] policy model: " << model_file_ << std::endl;
    if (motors_.size() != NUM_JOINTS) {
        throw std::runtime_error("[ArtiH1::ArtiH1] Invalid number of joints.");
    }

    for (const auto& motor : motors_) {
        const auto joint_id = kinematics_->joint_id(motor.name);
        
        joint_ids_.push_back(joint_id);
        joint_id_to_motor_index_[joint_id] = motor.index;
    }

    for (const auto& sensor : sensors_) {
        if (sensor.name == "imu_body") {
            imu_sensor_index_ = sensor.index;
            imu_frame_id_ = kinematics_->frame_id(sensor.frame);
            imu_translation_ = Eigen::Vector3d(sensor.translation.data());
            frame_q_imu_ = Eigen::Quaterniond(sensor.orientation.data()).normalized();
        }
    }

    for (const auto& foot : foots_) {
        foot_frame_ids_.push_back(kinematics_->frame_id(foot.name));
        foot_toe_offsets_.push_back(Eigen::Vector3d(foot.offset.data()));
    }

    auto amp_controller = std::make_unique<controller::AmpController>(model_file_);
    amp_controller_ = amp_controller.get();
    controller_ = std::move(amp_controller);
    controller_->initialize();

    const YAML::Node policy = YAML::LoadFile(config_file)["policy"];
    if (!policy) throw std::runtime_error("[ArtiH1::ArtiH1] robot.yaml has no 'policy' section.");
    default_pose_ = read_array<NUM_JOINTS>(policy, "default_pose");
    if (policy["observation"]) {
        const auto type = policy["observation"].as<std::string>();
        if (type == "velocity") {
            velocity_obs_ = true;
        } else if (type == "velocity_history") {
            velocity_obs_ = true;
            action_history_obs_ = true;
        } else if (type == "olaf") {
            olaf_obs_ = true;
        } else if (type != "amp") {
            throw std::runtime_error(
                "[ArtiH1::ArtiH1] robot.yaml policy.observation must be 'amp', 'velocity', 'velocity_history' or 'olaf'.");
        }
    }
    observation_size_ = olaf_obs_            ? OLAF_WALK_OBSERVATION_SIZE   // log width; standing pads one column
                        : action_history_obs_ ? VELOCITY_HISTORY_OBSERVATION_SIZE
                        : velocity_obs_       ? VELOCITY_OBSERVATION_SIZE
                                              : OBSERVATION_SIZE;
    std::cout << "[ArtiH1] observation: "
              << (olaf_obs_ ? "olaf (walk 80 / stand 79)"
                  : action_history_obs_ ? "velocity_history (81)" : velocity_obs_ ? "velocity (45)" : "amp (116)")
              << std::endl;
    if (olaf_obs_) {
        const YAML::Node olaf = YAML::LoadFile(config_file)["olaf"];
        if (!olaf) throw std::runtime_error("[ArtiH1::ArtiH1] policy.observation olaf needs an 'olaf' section.");
        olaf_stand_controller_ = std::make_unique<controller::AmpController>(olaf["stand_model_file"].as<std::string>());
        olaf_stand_controller_->initialize();
        olaf_walk_period_ = olaf["walk_period"].as<double>();
        olaf_walk_start_phase_ = olaf["walk_start_phase"].as<double>();
        olaf_path_max_distance_ = olaf["path_max_distance"].as<double>();
        olaf_path_max_yaw_ = olaf["path_max_yaw"].as<double>();
        olaf_stand_path_tau_ = olaf["stand_path_tau"].as<double>();
        const auto clip = olaf["thermal_clip"].as<std::vector<double>>();
        olaf_thermal_clip_ = {clip.at(0), clip.at(1)};
        const auto walk_range = olaf["walk_command_range"].as<std::vector<std::vector<double>>>();
        const auto stand_range = olaf["stand_command_range"].as<std::vector<std::vector<double>>>();
        if (walk_range.size() != 3 || stand_range.size() != 4) {
            throw std::runtime_error("[ArtiH1::ArtiH1] olaf.walk_command_range needs 3, stand_command_range 4 [min, max] pairs.");
        }
        for (std::size_t i = 0; i < 3; ++i) olaf_walk_range_[i] = {walk_range[i].at(0), walk_range[i].at(1)};
        for (std::size_t i = 0; i < 4; ++i) olaf_stand_range_[i] = {stand_range[i].at(0), stand_range[i].at(1)};
        olaf_stand_rate_ = read_array<4>(olaf, "stand_command_rate");
        olaf_knn_ = olaf["reference_knn"].as<int>();
        if (olaf["contact_tolerance"]) olaf_contact_tolerance_ = olaf["contact_tolerance"].as<double>();
        // contact-aided InEKF noise (training ArtiH1OlafEnvCfg.ekf_*)
        OlafInEKF::Params ep;
        if (const YAML::Node e = olaf["ekf"]) {
            const auto get = [&e](const char* key, double& v) { if (e[key]) v = e[key].as<double>(); };
            get("gyro_noise", ep.gyro_noise);
            get("accel_noise", ep.accel_noise);
            get("gyro_bias_noise", ep.gyro_bias_noise);
            get("accel_bias_noise", ep.accel_bias_noise);
            get("contact_noise", ep.contact_noise);
            get("fk_noise", ep.fk_noise);
            get("init_rot", ep.init_rot);
            get("init_vel", ep.init_vel);
            get("init_pos", ep.init_pos);
            get("init_gyro_bias", ep.init_gyro_bias);
            get("init_accel_bias", ep.init_accel_bias);
        }
        olaf_ekf_.configure(ep);
        // runtime chain (training action_first_order_hold / action_lowpass_hz / action_rate_limit, sim 600 Hz)
        if (olaf["actuator_rate"]) olaf_actuator_rate_ = olaf["actuator_rate"].as<double>();
        if (olaf["action_lowpass_hz"]) olaf_lowpass_hz_ = olaf["action_lowpass_hz"].as<double>();
        olaf_stand_lowpass_hz_ = olaf["stand_action_lowpass_hz"] ? olaf["stand_action_lowpass_hz"].as<double>()
                                                                 : olaf_lowpass_hz_;
        if (olaf["walk_stop_hold"]) olaf_walk_stop_hold_ = olaf["walk_stop_hold"].as<double>();
        if (olaf["action_rate_limit"]) olaf_rate_limit_ = olaf["action_rate_limit"].as<double>();
        const auto keys = YAML::LoadFile(olaf["walk_keys_file"].as<std::string>())["walk_keys"]
                              .as<std::vector<std::vector<double>>>();
        for (const auto& k : keys) olaf_walk_keys_.push_back({k.at(0), k.at(1), k.at(2)});
        if (olaf_walk_keys_.empty()) throw std::runtime_error("[ArtiH1::ArtiH1] olaf.walk_keys_file has no walk_keys.");
        std::cout << "[ArtiH1] olaf: walk model " << model_file_ << ", stand model "
                  << olaf["stand_model_file"].as<std::string>() << ", " << olaf_walk_keys_.size()
                  << " walk reference keys. Pad: cross = walk mode (WALK policy only while the stick commands a velocity, else STAND), circle = stand mode (starts standing)." << std::endl;
        std::cout << "[ArtiH1] olaf runtime: first-order hold to " << olaf_actuator_rate_ << " Hz, slew "
                  << olaf_rate_limit_ << " rad/s, low-pass walk " << olaf_lowpass_hz_ << " Hz / stand "
                  << olaf_stand_lowpass_hz_ << " Hz (servo thread)" << std::endl;
    }
    if (policy["target_rate_limit"]) target_rate_limit_ = policy["target_rate_limit"].as<double>();
    if (policy["target_lowpass_alpha"]) target_lowpass_alpha_ = policy["target_lowpass_alpha"].as<double>();
    if (!(target_lowpass_alpha_ > 0.0 && target_lowpass_alpha_ <= 1.0)) {
        throw std::runtime_error("[ArtiH1::ArtiH1] robot.yaml policy.target_lowpass_alpha must be in (0, 1].");
    }
    std::cout << "[ArtiH1] joint target low-pass alpha: " << target_lowpass_alpha_
              << (target_lowpass_alpha_ < 1.0 ? "" : " (off)") << std::endl;
    std::cout << "[ArtiH1] joint target rate limit: "
              << (target_rate_limit_ > 0.0 ? std::to_string(target_rate_limit_) + " rad/s" : std::string("off"))
              << std::endl;
    // Walking PD gains: motor.yaml 'kp' / 'kd' per motor (driver units, the real robot gains).
    for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
        policy_kp_[i] = motor_manager_->gains(static_cast<uint8_t>(i)).kp;
        policy_kd_[i] = motor_manager_->gains(static_cast<uint8_t>(i)).kd;
    }
    command_scale_ = read_array<3>(policy, "command_scale");
    command_max_ = read_array<3>(policy, "command_max");
    gait_phase_rate_ = policy["gait_phase_rate"].as<double>();
    control_dt_ = policy["control_dt"].as<double>();
    if (policy["initial_gait_phase"]) initial_gait_phase_ = policy["initial_gait_phase"].as<double>();
    if (policy["max_tilt"]) max_tilt_ = policy["max_tilt"].as<double>();
    if (policy["imu_still_time"]) imu_still_time_ = policy["imu_still_time"].as<double>();
    if (policy["imu_converge_tolerance"]) imu_converge_tolerance_ = policy["imu_converge_tolerance"].as<double>();
    if (policy["imu_converge_timeout"]) imu_converge_timeout_ = policy["imu_converge_timeout"].as<double>();
    if (policy["imu_attitude_gain"]) imu_attitude_gain_ = policy["imu_attitude_gain"].as<double>();
    if (policy["imu_stale_time"]) imu_stale_time_ = policy["imu_stale_time"].as<double>();
    if (policy["fault_settle_time"]) fault_settle_time_ = policy["fault_settle_time"].as<double>();
    if (policy["joy_loss_hold"]) joy_loss_hold_ = policy["joy_loss_hold"].as<bool>();
    if (policy["gait_rest_phases"]) gait_rest_phases_ = policy["gait_rest_phases"].as<std::vector<double>>();
    if (policy["stand_command_threshold"]) stand_command_threshold_ = policy["stand_command_threshold"].as<double>();
    if (policy["stick_deadzone"]) stick_deadzone_ = policy["stick_deadzone"].as<double>();

    const YAML::Node log_file = YAML::LoadFile(config_file)["log_file"];
    if (log_file && !log_file.as<std::string>().empty()) {
        // <log_file>_YYYYmmdd_HHMMSS.csv, so runs never overwrite each other; a model picked on the command line
        // (--ppo / --model) adds its file name: <log_file>_<model>_YYYYmmdd_HHMMSS.csv
        std::string prefix = log_file.as<std::string>();
        if (model_override) {
            std::string stem = model_file_.substr(model_file_.find_last_of('/') + 1);
            if (stem.size() > 5 && stem.compare(stem.size() - 5, 5, ".onnx") == 0) stem.resize(stem.size() - 5);
            prefix += "_" + stem;
        }
        char stamp[32];
        const std::time_t now = std::time(nullptr);
        std::strftime(stamp, sizeof(stamp), "_%Y%m%d_%H%M%S.csv", std::localtime(&now));
        const std::string path = prefix + stamp;
        log_file_ = std::fopen(path.c_str(), "w");
        if (log_file_ == nullptr) {
            std::cerr << "[ArtiH1] Cannot open log file " << path << ", logging disabled." << std::endl;
        } else {
            std::fprintf(log_file_, "t,phase");
            for (std::size_t i = 0; i < observation_size_; ++i) std::fprintf(log_file_, ",obs_%zu", i);
            for (const char* name : kPolicyJointNames) std::fprintf(log_file_, ",q_des_%s", name);
            for (const char* name : kPolicyJointNames) std::fprintf(log_file_, ",action_%s", name);
            for (const char* name : kPolicyJointNames) {
                std::fprintf(log_file_, ",pos_%s,vel_%s,tau_%s,temp_%s", name, name, name, name);
            }
            std::fprintf(log_file_, ",imu_qw,imu_qx,imu_qy,imu_qz,imu_gx,imu_gy,imu_gz,imu_ax,imu_ay,imu_az,imu_temp,att_qw,att_qx,att_qy,att_qz");
            // motor error code (Unitree: 1 overheat, 2 overcurrent, 3 overvoltage, 4 encoder fault)
            for (const char* name : kPolicyJointNames) std::fprintf(log_file_, ",err_%s", name);
            // fault bits: 1 IMU stopped, 2 motor link, 4 joystick lost, 8 holding (policy stopped, pose held)
            std::fprintf(log_file_, ",fault");
            // olaf: mode (1 walk, 0 stand), estimated pelvis position / velocity (estimator world frame), path frame,
            // estimated foot contacts, control input g (walk: vx vy wz, stand: dz roll pitch yaw)
            if (olaf_obs_) {
                std::fprintf(log_file_, ",olaf_walk,est_x,est_y,est_z,est_vx,est_vy,est_vz,path_x,path_y,path_psi,"
                                        "contact_l,contact_r,g0,g1,g2,g3");
            }
            std::fprintf(log_file_, "\n");
            std::cout << "[ArtiH1] Logging every control step to " << path << std::endl;
        }
    }

    observation_.clear();
    action_.clear();

    for (std::size_t i = 0; i < 8; ++i) {
        motor_command_[i].kp = 0.0;
        motor_command_[i].kd = 0.0;
    }
    for (std::size_t i = 8; i < 12; ++i) {
        motor_command_[i].kp = 0.0;
        motor_command_[i].kd = 0.0;
    }
}

robot::ArtiH1::~ArtiH1() {
    olaf_servo_stop();
    if (log_file_ != nullptr) {
        std::fclose(log_file_);
        log_file_ = nullptr;
    }
}

void robot::ArtiH1::print_observation() const {
    const auto& o = observation_;
    constexpr double DEG = 180.0 / PI;
    if (olaf_obs_) {
        if (o.size() < OLAF_STAND_OBSERVATION_SIZE) return;
        std::printf("\n[observe] olaf %s | pelvis in path frame x %+.3f y %+.3f height %+.4f m | path R row-major "
                    "(%+.2f %+.2f / %+.2f %+.2f / %+.2f %+.2f)\n", olaf_walking_ ? "WALK" : "STAND", o[0], o[1], o[2],
                    o[3], o[4], o[5], o[6], o[7], o[8]);
        std::printf("[observe] pelvis vel (pelvis frame) %+.3f %+.3f %+.3f m/s | gyro %+.3f %+.3f %+.3f rad/s | "
                    "contact L %d R %d | est pos %+.3f %+.3f %+.3f | path %+.3f %+.3f %+.1f deg\n", o[9], o[10], o[11],
                    o[12], o[13], o[14], olaf_contact_[0] ? 1 : 0, olaf_contact_[1] ? 1 : 0, olaf_pos_.x(), olaf_pos_.y(),
                    olaf_pos_.z(), olaf_path_[0], olaf_path_[1], olaf_path_[2] * DEG);
        std::printf("[observe] joint               pos-default(deg)  vel(rad/s)  temp obs\n");
        for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
            std::printf("[observe]   %-18s %+8.2f          %+7.3f     %+.2f\n", kPolicyJointNames[i], o[15 + i] * DEG,
                        o[27 + i], o[63 + i]);
        }
        std::printf("[observe] g (%+.3f %+.3f %+.3f %+.3f)\n", olaf_g_[0], olaf_g_[1], olaf_g_[2], olaf_g_[3]);
        std::fflush(stdout);
        return;
    }
    if (velocity_obs_) {
        if (o.size() != observation_size_) return;
        const std::size_t c = o.size() - 3;   // command index
        // [3:6] = gravity direction in the pelvis frame: (0, 0, -1) when level
        const double roll = std::atan2(o[4], -o[5]) * DEG;    // + = right side down
        const double pitch = std::atan2(-o[3], -o[5]) * DEG;  // + = nose down
        std::printf("\n[observe] pelvis roll %+6.2f deg  pitch %+6.2f deg  | gravity (%+.3f %+.3f %+.3f)  gyro (%+.3f %+.3f %+.3f) rad/s\n",
                    roll, pitch, o[3], o[4], o[5], o[0], o[1], o[2]);
        std::printf("[observe] joint               pos-default(deg)  vel(rad/s)  prev action\n");
        for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
            std::printf("[observe]   %-18s %+8.2f          %+7.3f     %+.3f\n", kPolicyJointNames[i], o[6 + i] * DEG,
                        o[18 + i], o[30 + i]);
        }
        std::printf("[observe] command (%+.4f %+.4f %+.4f)\n", o[c], o[c + 1], o[c + 2]);
        std::fflush(stdout);
        return;
    }
    if (o.size() != OBSERVATION_SIZE) return;
    // [0:6] = heading-frame root x axis (tan) and z axis (norm): body z in the heading frame gives roll / pitch
    const double roll = std::atan2(-o[4], o[5]) * DEG;   // + = right side down (x forward, y left, z up)
    const double pitch = std::atan2(o[3], o[5]) * DEG;   // + = nose down
    std::printf("\n[observe] root roll %+6.2f deg  pitch %+6.2f deg  | tan (%+.3f %+.3f %+.3f) norm (%+.3f %+.3f %+.3f)\n",
                roll, pitch, o[0], o[1], o[2], o[3], o[4], o[5]);
    std::printf("[observe] ang vel heading frame (rad/s)  x %+6.3f  y %+6.3f  z %+6.3f\n", o[6], o[7], o[8]);
    std::printf("[observe] joint               motor(deg)  |angle| from obs tan-norm(deg)  tan (x y z)              norm (x y z)              vel(rad/s)  prev action\n");
    for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
        // joint rotation R: tan = R e_x, norm = R e_z, R e_y = norm x tan; |angle| = acos((trace - 1) / 2) must equal
        // |motor angle| (a mismatch means the observation joint order / mapping is wrong)
        const double* r = &o[9 + 6 * i];
        const double y_y = r[5] * r[0] - r[3] * r[2];   // (norm x tan).y
        const double trace = r[0] + y_y + r[5];
        const double angle = std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0));
        const auto motor_index = joint_id_to_motor_index_.at(kinematics_->joint_id(kPolicyJointNames[i]));
        const double motor = last_motor_status_[motor_index].position;
        std::printf("[observe]   %-18s %+8.2f      %7.2f                      (%+.3f %+.3f %+.3f)  (%+.3f %+.3f %+.3f)  %+7.3f     %+.3f\n",
                    kPolicyJointNames[i], motor * DEG, angle * DEG, r[0], r[1], r[2], r[3], r[4], r[5], o[81 + i],
                    o[99 + i]);
    }
    std::printf("[observe] toes (heading frame, mm)  left (%+6.1f %+6.1f %+6.1f)  right (%+6.1f %+6.1f %+6.1f)\n",
                o[93] * 1000, o[94] * 1000, o[95] * 1000, o[96] * 1000, o[97] * 1000, o[98] * 1000);
    std::printf("[observe] phase sin %+.3f cos %+.3f | command (%+.4f %+.4f %+.4f) | imu q (%+.3f %+.3f %+.3f %+.3f)\n",
                o[111], o[112], o[113], o[114], o[115], last_imu_orientation_[0], last_imu_orientation_[1],
                last_imu_orientation_[2], last_imu_orientation_[3]);
    std::fflush(stdout);
}

void robot::ArtiH1::set_limp_right() {
    for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
        if (std::string(kPolicyJointNames[i]).rfind("right_", 0) != 0) continue;
        const auto motor_index = joint_id_to_motor_index_.at(kinematics_->joint_id(kPolicyJointNames[i]));
        motor_command_[motor_index].kp = 0.0;
        motor_command_[motor_index].kd = policy_kd_[motor_index];
    }
    motor_manager_->write(motor_command_);
    limp_right_ = true;
    limp_stats_init_ = false;
    std::cout << "[ArtiH1] --limp-right: right-leg motors kp 0 (damping only). Hold the pelvis level and press both soles"
                 " flat on the floor, the right foot as the mirror image of the left." << std::endl;
}

void robot::ArtiH1::update_limp_stats() {
    if (!limp_right_) return;
    constexpr double kAlpha = 1.0 / 120.0;   // ~2 s at 60 Hz
    for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
        const auto motor_index = joint_id_to_motor_index_.at(kinematics_->joint_id(kPolicyJointNames[i]));
        const double q = last_motor_status_[motor_index].position;
        limp_mean_pos_[i] = limp_stats_init_ ? limp_mean_pos_[i] + kAlpha * (q - limp_mean_pos_[i]) : q;
    }
    limp_stats_init_ = true;
}

void robot::ArtiH1::print_limp_report() const {
    if (!limp_right_ || !limp_stats_init_) return;
    constexpr double DEG = 180.0 / PI;
    double roll = 0.0, pitch = 0.0;
    if (observation_.size() >= 6) {
        roll = std::atan2(observation_[4], -observation_[5]) * DEG;
        pitch = std::atan2(-observation_[3], -observation_[5]) * DEG;
    }
    std::printf("\n[limp] pelvis roll %+5.2f pitch %+5.2f deg (keep both near 0)   ~2 s averages:\n", roll, pitch);
    std::printf("[limp] joint        left(deg)  right(deg)  right-left(deg)  right motor id  homing_offset now -> suggested\n");
    // policy order is (left, right) pairs: hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll
    for (std::size_t i = 0; i + 1 < NUM_JOINTS; i += 2) {
        const double l = limp_mean_pos_[i], r = limp_mean_pos_[i + 1];
        const auto motor_index = joint_id_to_motor_index_.at(kinematics_->joint_id(kPolicyJointNames[i + 1]));
        const double offset = motor_manager_->homing(motor_index).offset;
        // after homing the reading is raw - (raw_at_stopper - homing_offset): a reading too high by e needs offset - e
        std::printf("[limp]   %-12s %+8.2f   %+8.2f     %+8.2f        %2d          %+.4f -> %+.4f\n",
                    kPolicyJointNames[i + 1] + 6, l * DEG, r * DEG, (r - l) * DEG, static_cast<int>(motor_index) + 1,
                    offset, offset - (r - l));
    }
    std::fflush(stdout);
}

void robot::ArtiH1::log_observation() {
    if (log_file_ != nullptr) write_log_row();
}

void robot::ArtiH1::write_log_row() {
    // row = state the policy saw (observation_, motor / IMU read in observation()) and what it answered
    if (log_start_ == std::chrono::steady_clock::time_point{}) log_start_ = std::chrono::steady_clock::now();
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - log_start_).count();
    std::fprintf(log_file_, "%.4f,%.5f", t, gait_phase_);
    for (const double v : observation_) std::fprintf(log_file_, ",%.7g", v);   // full precision: replayed through ONNX
    for (std::size_t i = observation_.size(); i < observation_size_; ++i) std::fprintf(log_file_, ",nan");
    // --observe mode has no policy output: q_des / action columns are nan there
    const bool policy_ran = action_.size() == NUM_JOINTS;
    for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
        if (policy_ran) std::fprintf(log_file_, ",%.7g", action_[i]); else std::fprintf(log_file_, ",nan");
    }
    for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
        if (policy_ran) std::fprintf(log_file_, ",%.7g", previous_actions_[i]); else std::fprintf(log_file_, ",nan");
    }
    for (const char* name : kPolicyJointNames) {
        const auto& s = last_motor_status_[joint_id_to_motor_index_.at(kinematics_->joint_id(name))];
        std::fprintf(log_file_, ",%.7g,%.7g,%.5g,%.0f", s.position, s.velocity, s.torque, s.temperature);
    }
    for (const double v : last_imu_orientation_) std::fprintf(log_file_, ",%.8g", v);
    for (const double v : last_imu_angular_velocity_) std::fprintf(log_file_, ",%.7g", v);
    for (const double v : last_imu_linear_acceleration_) std::fprintf(log_file_, ",%.7g", v);
    std::fprintf(log_file_, ",%.2f", last_imu_temperature_);
    for (const double v : last_attitude_) std::fprintf(log_file_, ",%.8g", v);
    for (const char* name : kPolicyJointNames) {
        std::fprintf(log_file_, ",%d", last_motor_status_[joint_id_to_motor_index_.at(kinematics_->joint_id(name))].error);
    }
    std::fprintf(log_file_, ",%d", fault_);
    if (olaf_obs_) {
        std::fprintf(log_file_, ",%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%d,%.6g,%.6g,%.6g,%.6g",
                     olaf_walking_ ? 1 : 0, olaf_pos_.x(), olaf_pos_.y(), olaf_pos_.z(), olaf_vel_.x(), olaf_vel_.y(),
                     olaf_vel_.z(), olaf_path_[0], olaf_path_[1], olaf_path_[2], olaf_contact_[0] ? 1 : 0,
                     olaf_contact_[1] ? 1 : 0, olaf_g_[0], olaf_g_[1], olaf_g_[2], olaf_g_[3]);
    }
    std::fprintf(log_file_, "\n");
}

void robot::ArtiH1::initialize() {
    for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
        motor_command_[i] = motor_interface::motor_command_t{};
    }
    motor_manager_->write(motor_command_);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    for (const auto& motor : motors_) {
        home(motor.index);
    }

    // homed motors hold 0.0 with weak gains (homing_hold_kp / homing_hold_kd); now move to the policy's default
    // pose with the policy gains, so the first q_des does not jump.
    motor_interface::motor_state_t motor_status[NUM_JOINTS]{};
    motor_manager_->read(motor_status);
    std::array<double, NUM_JOINTS> start{};
    std::array<double, NUM_JOINTS> goal{};
    for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
        const auto motor_index = joint_id_to_motor_index_.at(kinematics_->joint_id(kPolicyJointNames[i]));
        start[motor_index] = motor_status[motor_index].position;
        goal[motor_index] = default_pose_[i];
    }
    const auto begin = std::chrono::steady_clock::now();
    const double duration = std::chrono::duration<double>(STAND_UP_TIME).count();
    auto next_wakeup = begin;
    while (true) {
        const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        const double alpha = std::min(1.0, t / duration);
        for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
            motor_command_[i].position = start[i] + (goal[i] - start[i]) * alpha;
            motor_command_[i].kp = policy_kp_[i];
            motor_command_[i].kd = policy_kd_[i];
        }
        motor_manager_->write(motor_command_);
        if (alpha >= 1.0) break;
        next_wakeup += HOMING_PERIOD;
        std::this_thread::sleep_until(next_wakeup);
    }

    // 10-28 15:19 observe log (joints still to 0.01 deg): the MTi-3 roll / pitch started 3-4 deg away from the
    // accelerometer tilt and took ~80 s to converge onto it, and the gyro read a constant bias of up to 1 deg/s
    // (training randomizes +-0.57 deg/s). Hold the default pose still: average the gyro bias (subtracted in
    // observation()), then wait until the IMU attitude agrees with the accelerometer before the policy starts.
    if (imu_still_time_ > 0.0) {
        auto& imu = sensor_manager_->sensor<sensor_interface::Imu>(imu_sensor_index_);
        imu.no_rotation_update(static_cast<uint16_t>(std::lround(imu_still_time_)));
        std::cout << "[ArtiH1::initialize] IMU calibration: keep the robot still ..." << std::endl;

        const auto tilt = [](const sensor_interface::imu_data_t& d, double& qr, double& qp, double& ar, double& ap) {
            const Eigen::Quaterniond q(d.orientation[0], d.orientation[1], d.orientation[2], d.orientation[3]);
            const Eigen::Vector3d g = q.normalized().toRotationMatrix().transpose() * Eigen::Vector3d::UnitZ();
            qr = std::atan2(g.y(), g.z()) * 180.0 / PI;
            qp = -std::asin(std::clamp(g.x(), -1.0, 1.0)) * 180.0 / PI;
            const double* a = d.linear_acceleration;
            ar = std::atan2(a[1], a[2]) * 180.0 / PI;
            ap = std::atan2(-a[0], std::hypot(a[1], a[2])) * 180.0 / PI;
        };

        constexpr auto kPeriod = std::chrono::milliseconds(10);
        const auto begin = std::chrono::steady_clock::now();
        auto next = begin;
        std::array<double, 3> gyro_sum{};
        std::size_t gyro_count = 0;
        double agree_since = -1.0;
        double last_print = 0.0;
        while (true) {
            next += kPeriod;
            std::this_thread::sleep_until(next);
            const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            sensor_interface::imu_data_t d{};
            imu.read(d);
            if (t < imu_still_time_) {
                for (std::size_t i = 0; i < 3; ++i) gyro_sum[i] += d.angular_velocity[i];
                ++gyro_count;
                continue;
            }
            double qr, qp, ar, ap;
            tilt(d, qr, qp, ar, ap);
            const double err = std::max(std::abs(qr - ar), std::abs(qp - ap));
            if (err <= imu_converge_tolerance_) {
                if (agree_since < 0.0) agree_since = t;
            } else {
                agree_since = -1.0;
            }
            if (t - last_print >= 5.0) {
                last_print = t;
                std::printf("[ArtiH1::initialize] %5.1f s  IMU roll %+.2f pitch %+.2f | accel roll %+.2f pitch %+.2f deg"
                            "  (temp %.1f C)\n", t, qr, qp, ar, ap, d.temperature);
            }
            if (agree_since >= 0.0 && t - agree_since >= 1.0) break;
            if (t >= imu_converge_timeout_) {
                std::cerr << "[ArtiH1::initialize] WARNING: IMU attitude still " << err
                          << " deg away from the accelerometer after " << t << " s; starting anyway." << std::endl;
                break;
            }
        }
        for (std::size_t i = 0; i < 3; ++i) imu_gyro_bias_[i] = gyro_count > 0 ? gyro_sum[i] / gyro_count : 0.0;
        std::printf("[ArtiH1::initialize] IMU ready: gyro bias (%+.3f %+.3f %+.3f) deg/s removed from the observation\n",
                    imu_gyro_bias_[0] * 180.0 / PI, imu_gyro_bias_[1] * 180.0 / PI, imu_gyro_bias_[2] * 180.0 / PI);
    }
    attitude_initialized_ = false;
    imu_sample_time_ = std::chrono::steady_clock::now();
    gait_phase_ = initial_gait_phase_;
    previous_actions_.fill(0.0);
    previous_actions2_.fill(0.0);
    previous_actions3_.fill(0.0);
    q_target_ = default_pose_;   // the env resets its joint target to the default pose
    // olaf: start standing (nominal standing reference = the default stance), estimator re-anchors on the first step
    olaf_walking_ = false;
    olaf_walk_mode_ = false;
    olaf_zero_cmd_time_ = 0.0;
    olaf_est_init_ = false;
    olaf_est_steps_ = 0;
    olaf_g_.fill(0.0);
    olaf_g_target_.fill(0.0);
    olaf_phase_ = olaf_walk_start_phase_;
}

void robot::ArtiH1::home(uint8_t index) {
    const auto& homing = motor_manager_->homing(index);
    const double dt = std::chrono::duration<double>(HOMING_PERIOD).count();

    motor_interface::motor_state_t motor_status[NUM_JOINTS]{};
    motor_manager_->read(motor_status);

    // only the homing motor is driven, already homed motors hold 0.0, the rest stay passive
    double target = motor_status[index].position;
    motor_command_[index].position = target;
    motor_command_[index].kp = homing.kp;
    motor_command_[index].kd = homing.kd;
    motor_manager_->write(motor_command_);

    double velocity = 0.0;
    int stall_count = 0;
    int debug_count = 0;

    const auto start = std::chrono::steady_clock::now();
    auto next_wakeup = start;

    while (stall_count < HOMING_STALL_COUNT) {
        next_wakeup += HOMING_PERIOD;
        std::this_thread::sleep_until(next_wakeup);
        // homing must not run on stale states (a frozen reading looks like a stall and sets a wrong zero)
        if (!motor_manager_->is_running() || motor_manager_->comm_fault()) {
            throw std::runtime_error("[ArtiH1::home] Motor communication stopped on motor index " + std::to_string(index) + ".");
        }

        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed > HOMING_TIMEOUT) {
            motor_command_[index] = motor_interface::motor_command_t{};
            motor_manager_->write(motor_command_);
            throw std::runtime_error("[ArtiH1::home] Timeout on motor index " + std::to_string(index) + ".");
        }

        motor_manager_->read(motor_status);
        const double position = motor_status[index].position;

        // move the target slowly, but never more than HOMING_MAX_LEAD ahead of the motor,
        // so the force at the stopper stays around kp * HOMING_MAX_LEAD
        target += homing.direction * homing.velocity * dt;
        if (homing.direction * (target - position) > HOMING_MAX_LEAD) {
            target = position + homing.direction * HOMING_MAX_LEAD;
        }
        motor_command_[index].position = target;
        motor_manager_->write(motor_command_);

        // stalled against the stopper: pushing in the homing direction but not moving
        velocity += HOMING_VELOCITY_FILTER * (motor_status[index].velocity - velocity);
        const double torque = motor_status[index].torque * homing.direction;

        if (++debug_count % 10 == 0) {
            std::cout << "[ArtiH1::home] index " << static_cast<int>(index)
                      << " target " << target
                      << " position " << position
                      << " velocity " << velocity
                      << " torque " << motor_status[index].torque << std::endl;
        }

        if (elapsed > HOMING_SETTLE_TIME &&
            torque >= homing.torque_threshold &&
            std::abs(velocity) <= homing.velocity_threshold) {
            ++stall_count;
        } else {
            stall_count = 0;
        }
    }

    // damping only while the offset changes: a position hold would jump when the position frame shifts
    motor_command_[index] = motor_interface::motor_command_t{};
    motor_command_[index].kd = homing.hold_kd;
    motor_manager_->write(motor_command_);

    motor_manager_->set_homing_offset(index);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "[ArtiH1::home] Motor index " << static_cast<int>(index) << " homed." << std::endl;

    // move from the stopper (homing offset) to 0.0
    motor_manager_->read(motor_status);
    target = motor_status[index].position;

    motor_command_[index].position = target;
    motor_command_[index].kp = homing.zero_kp;
    motor_command_[index].kd = homing.zero_kd;
    motor_manager_->write(motor_command_);

    const double step = homing.velocity * dt;
    next_wakeup = std::chrono::steady_clock::now();

    while (target != 0.0) {
        next_wakeup += HOMING_PERIOD;
        std::this_thread::sleep_until(next_wakeup);
        // homing must not run on stale states (a frozen reading looks like a stall and sets a wrong zero)
        if (!motor_manager_->is_running() || motor_manager_->comm_fault()) {
            throw std::runtime_error("[ArtiH1::home] Motor communication stopped on motor index " + std::to_string(index) + ".");
        }

        target = std::abs(target) <= step ? 0.0 : target - std::copysign(step, target);
        motor_command_[index].position = target;
        motor_manager_->write(motor_command_);
    }

    // wait until the motor actually reaches 0.0
    const auto zero_start = std::chrono::steady_clock::now();

    while (true) {
        std::this_thread::sleep_for(HOMING_PERIOD);
        // homing must not run on stale states (a frozen reading looks like a stall and sets a wrong zero)
        if (!motor_manager_->is_running() || motor_manager_->comm_fault()) {
            throw std::runtime_error("[ArtiH1::home] Motor communication stopped on motor index " + std::to_string(index) + ".");
        }

        motor_manager_->read(motor_status);

        if (std::abs(motor_status[index].position) <= HOMING_ZERO_TOLERANCE) break;

        if (std::chrono::steady_clock::now() - zero_start > HOMING_ZERO_TIMEOUT) {
            motor_command_[index] = motor_interface::motor_command_t{};
            motor_manager_->write(motor_command_);
            throw std::runtime_error("[ArtiH1::home] Motor index " + std::to_string(index) +
                                     " did not reach zero, position " + std::to_string(motor_status[index].position) + ".");
        }
    }

    std::cout << "[ArtiH1::home] Motor index " << static_cast<int>(index)
              << " moved to zero, position " << motor_status[index].position << std::endl;

    // keep holding 0.0 with weak gains
    motor_command_[index].position = 0.0;
    motor_command_[index].kp = homing.hold_kp;
    motor_command_[index].kd = homing.hold_kd;
    motor_manager_->write(motor_command_);
}

// 10-28 16:05 walking log: in every still window after walking the MTi-3 roll / pitch was 1.5-3.4 deg away from the
// accelerometer tilt (robot standing still, so the accelerometer is the truth). Own attitude instead: integrate the
// bias-corrected gyro and pull roll / pitch toward the accelerometer with imu_attitude_gain (1/s), trusting the
// accelerometer less when |a| is far from g (impacts). Offline on that log (gain 0.5-1): still-window error <= 0.4 deg.
// Starts from the MTi orientation (checked against the accelerometer in initialize()); yaw only follows the gyro,
// which is fine because the policy only sees the heading frame.
Eigen::Quaterniond robot::ArtiH1::update_attitude(const sensor_interface::imu_data_t& imu_data,
                                                  const Eigen::Quaterniond& mti_q) {
    const auto now = std::chrono::steady_clock::now();
    if (!attitude_initialized_) {
        attitude_q_ = mti_q;
        attitude_time_ = now;
        attitude_initialized_ = true;
        return attitude_q_;
    }
    const double dt = std::clamp(std::chrono::duration<double>(now - attitude_time_).count(), 0.0, 0.1);
    attitude_time_ = now;

    Eigen::Vector3d omega(imu_data.angular_velocity[0] - imu_gyro_bias_[0],
                          imu_data.angular_velocity[1] - imu_gyro_bias_[1],
                          imu_data.angular_velocity[2] - imu_gyro_bias_[2]);
    const Eigen::Vector3d accel(imu_data.linear_acceleration[0], imu_data.linear_acceleration[1],
                                imu_data.linear_acceleration[2]);
    const double accel_norm = accel.norm();
    if (accel_norm > 1.0) {
        // up direction in the sensor frame: estimated (from the attitude) vs measured (accelerometer)
        const Eigen::Vector3d up = attitude_q_.toRotationMatrix().transpose() * Eigen::Vector3d::UnitZ();
        const double trust = std::exp(-std::pow((accel_norm - 9.81) / 0.5, 2));
        omega += imu_attitude_gain_ * trust * (accel / accel_norm).cross(up);
    }
    const Eigen::Quaterniond omega_q(0.0, omega.x(), omega.y(), omega.z());
    attitude_q_.coeffs() += 0.5 * dt * (attitude_q_ * omega_q).coeffs();
    attitude_q_.normalize();
    return attitude_q_;
}

void robot::ArtiH1::observation() {
    observation_.clear();
    observation_.reserve(observation_size_);

    motor_interface::motor_state_t motor_status[NUM_JOINTS]{};
    motor_manager_->read(motor_status);
    std::copy(motor_status, motor_status + NUM_JOINTS, last_motor_status_);

    // update kinematics with current motor positions
    Eigen::VectorXd joint_positions = Eigen::VectorXd::Zero(NUM_JOINTS);

    for (const auto& joint_id : joint_ids_) {
        const int joint_index = kinematics_->joint_index(joint_id);
        joint_positions[joint_index] = motor_status[joint_id_to_motor_index_.at(joint_id)].position;
    }

    kinematics_->update(joint_positions);

    // read IMU data
    sensor_interface::imu_data_t imu_data{};
    sensor_manager_->sensor<sensor_interface::Imu>(imu_sensor_index_).read(imu_data);
    for (std::size_t i = 0; i < 4; ++i) last_imu_orientation_[i] = imu_data.orientation[i];
    for (std::size_t i = 0; i < 3; ++i) last_imu_angular_velocity_[i] = imu_data.angular_velocity[i];
    for (std::size_t i = 0; i < 3; ++i) last_imu_linear_acceleration_[i] = imu_data.linear_acceleration[i];
    last_imu_temperature_ = imu_data.temperature;

    // 10-29 09:44 walking log: every IMU field froze at 71.5 s (no more MTi packets) and the policy walked on the
    // stale attitude for 90 s. Real samples never repeat exactly at 60 Hz, so a sample that stops changing = IMU lost.
    const std::array<double, 6> imu_sample{imu_data.angular_velocity[0], imu_data.angular_velocity[1],
                                           imu_data.angular_velocity[2], imu_data.linear_acceleration[0],
                                           imu_data.linear_acceleration[1], imu_data.linear_acceleration[2]};
    const auto now = std::chrono::steady_clock::now();
    if (imu_sample != imu_last_sample_) {
        imu_last_sample_ = imu_sample;
        imu_sample_time_ = now;
        imu_stale_ = false;
    } else if (imu_stale_time_ > 0.0 &&
               std::chrono::duration<double>(now - imu_sample_time_).count() > imu_stale_time_) {
        imu_stale_ = true;
    }

    Eigen::Quaterniond world_q_imu(
        imu_data.orientation[0],
        imu_data.orientation[1],
        imu_data.orientation[2],
        imu_data.orientation[3]
    );
    if (world_q_imu.norm() < 1.0e-8) {
        throw std::runtime_error("[ArtiH1::observation] Invalid IMU orientation.");
    }
    world_q_imu.normalize();
    if (imu_attitude_gain_ > 0.0) world_q_imu = update_attitude(imu_data, world_q_imu);
    last_attitude_ = {world_q_imu.w(), world_q_imu.x(), world_q_imu.y(), world_q_imu.z()};

    const Eigen::Matrix3d world_R_imu = world_q_imu.toRotationMatrix();

    const Eigen::Matrix3d root_R_frame = kinematics_->frame_rotation(imu_frame_id_);

    const Eigen::Matrix3d frame_R_imu = frame_q_imu_.toRotationMatrix();

    const Eigen::Matrix3d root_R_imu = root_R_frame * frame_R_imu;

    const Eigen::Matrix3d world_R_root = world_R_imu * root_R_imu.transpose();

    const double heading = std::atan2(world_R_root(1, 0), world_R_root(0, 0));

    const Eigen::Matrix3d heading_R_world = Eigen::AngleAxisd(
        -heading,
        Eigen::Vector3d::UnitZ()
    ).toRotationMatrix();

    const Eigen::Matrix3d true_heading_R_root = heading_R_world * world_R_root;
    true_tilt_cos_ = true_heading_R_root(2, 2);
    if (blind_imu_ && !blind_latched_) {
        blind_heading_R_root_ = true_heading_R_root;
        blind_latched_ = true;
    }
    const Eigen::Matrix3d& heading_R_root = blind_imu_ ? blind_heading_R_root_ : true_heading_R_root;

    const Eigen::Vector3d angular_velocity_imu(
        imu_data.angular_velocity[0] - imu_gyro_bias_[0],
        imu_data.angular_velocity[1] - imu_gyro_bias_[1],
        imu_data.angular_velocity[2] - imu_gyro_bias_[2]
    );

    const Eigen::Vector3d angular_velocity_world = world_R_imu * angular_velocity_imu;

    const Eigen::Vector3d angular_velocity_heading =
        blind_imu_ ? Eigen::Vector3d(Eigen::Vector3d::Zero()) : Eigen::Vector3d(heading_R_world * angular_velocity_world);

    if (olaf_obs_) {
        // InEKF inputs in the pelvis axes at the IMU (the sim IMU: pelvis orientation, ARTI_H1_IMU_OFFSET_FROM_PELVIS):
        // gyro (bias calibrated), accelerometer (specific force), IMU position in the pelvis frame
        const Eigen::Vector3d accel_imu(imu_data.linear_acceleration[0], imu_data.linear_acceleration[1],
                                        imu_data.linear_acceleration[2]);
        const Eigen::Vector3d imu_root = kinematics_->frame_position(imu_frame_id_) + root_R_frame * imu_translation_;
        olaf_estimate(world_R_root, root_R_imu * angular_velocity_imu, root_R_imu * accel_imu, imu_root);
        joy_interface::joy_data_t joy_data{};
        joy_handler_->read(joy_data);
        // pad mode (paper: the Animation Engine switches policies): cross = walk mode, circle = stand mode.
        // Walk mode runs the WALK policy only while the left / right stick asks for a velocity; with the sticks at zero
        // (for walk_stop_hold s) it holds still with the STAND policy at the neutral pose instead of stepping in place.
        if (joy_data.cross && !olaf_prev_cross_ && !olaf_walk_mode_) {
            olaf_walk_mode_ = true;
            olaf_zero_cmd_time_ = 0.0;
            std::cout << "[ArtiH1] olaf: WALK mode (WALK policy while the stick commands a velocity)" << std::endl;
        } else if (joy_data.circle && !olaf_prev_circle_ && olaf_walk_mode_) {
            olaf_walk_mode_ = false;
            std::cout << "[ArtiH1] olaf: STAND mode" << std::endl;
        }
        olaf_prev_cross_ = joy_data.cross;
        olaf_prev_circle_ = joy_data.circle;
        const auto dead = [this](double v) { return std::abs(v) < stick_deadzone_ ? 0.0 : v; };
        // stick in [-1, 1] -> [min, max] of the training range (0 stays 0)
        const auto span = [](double v, const std::array<double, 2>& r) { return v >= 0.0 ? v * r[1] : -v * r[0]; };
        const std::array<double, 3> g_walk{span(dead(joy_data.stick_ly), olaf_walk_range_[0]),
                                           span(-dead(joy_data.stick_lx), olaf_walk_range_[1]),
                                           span(-dead(joy_data.stick_rx), olaf_walk_range_[2])};
        const bool moving = g_walk[0] != 0.0 || g_walk[1] != 0.0 || g_walk[2] != 0.0;
        olaf_zero_cmd_time_ = moving ? 0.0 : olaf_zero_cmd_time_ + control_dt_;
        const bool want_walk = olaf_walk_mode_ && (moving || (olaf_walking_ && olaf_zero_cmd_time_ < olaf_walk_stop_hold_));
        if (want_walk && !olaf_walking_) {
            olaf_walking_ = true;
            olaf_phase_ = olaf_walk_start_phase_;   // training: walking episodes from the stance start at this phase
            olaf_reset_path();
            std::cout << "[ArtiH1] olaf: WALK policy" << std::endl;
        } else if (!want_walk && olaf_walking_) {
            olaf_walking_ = false;
            olaf_stand_switch_ = true;              // g = the current stick target below (env: g = g_target at reset)
            olaf_reset_path();                      // training: standing episodes from mid-stride start at the torso
            std::cout << "[ArtiH1] olaf: STAND policy" << std::endl;
        }
        if (olaf_walking_) {
            const auto p = olaf_project_walk(g_walk);   // onto the reference coverage, as the env samples g
            olaf_g_ = {p[0], p[1], p[2], 0.0};
            command_ = p;
        } else {
            if (olaf_walk_mode_) {
                olaf_g_target_.fill(0.0);   // walk mode standing still: neutral pose (the sticks command velocity)
            } else {
                // left stick: pitch (push = nose down, +pitch) / roll (right = lean right, +roll), right stick: yaw
                // (right = turn right, -yaw) / height (up = higher, +dz); same Euler ZYX convention as the IK grid
                olaf_g_target_ = {span(dead(joy_data.stick_ry), olaf_stand_range_[0]),
                                  span(dead(joy_data.stick_lx), olaf_stand_range_[1]),
                                  span(dead(joy_data.stick_ly), olaf_stand_range_[2]),
                                  span(-dead(joy_data.stick_rx), olaf_stand_range_[3])};
            }
            if (olaf_stand_switch_) olaf_g_ = olaf_g_target_;
            olaf_stand_switch_ = false;
            command_ = {0.0, 0.0, 0.0};
        }
        // pelvis pose / velocity relative to the path frame
        const double psi = olaf_path_[2];
        const double c = std::cos(psi), s = std::sin(psi);
        const double dx = olaf_pos_.x() - olaf_path_[0], dy = olaf_pos_.y() - olaf_path_[1];
        observation_.push_back(c * dx + s * dy);                       // [0:3] position in the path frame, height
        observation_.push_back(-s * dx + c * dy);
        observation_.push_back(olaf_pos_.z());
        const Eigen::Matrix3d path_R_root = Eigen::AngleAxisd(-psi, Eigen::Vector3d::UnitZ()).toRotationMatrix() * olaf_R_;
        for (int r = 0; r < 3; ++r) {                                   // [3:9] first two columns, row-major
            observation_.push_back(path_R_root(r, 0));
            observation_.push_back(path_R_root(r, 1));
        }
        append_vector(olaf_R_.transpose() * olaf_vel_, observation_);        // [9:12] pelvis frame velocity
        append_vector(root_R_imu * angular_velocity_imu, observation_);      // [12:15] gyro, pelvis frame
        for (std::size_t i = 0; i < NUM_JOINTS; ++i) {                 // [15:27] joint pos - default pose
            const auto joint_id = kinematics_->joint_id(kPolicyJointNames[i]);
            observation_.push_back(motor_status[joint_id_to_motor_index_.at(joint_id)].position - default_pose_[i]);
        }
        for (const char* name : kPolicyJointNames) {                   // [27:39] joint velocities
            observation_.push_back(motor_status[joint_id_to_motor_index_.at(kinematics_->joint_id(name))].velocity);
        }
        for (const double action : previous_actions_) observation_.push_back(action);    // [39:51] a(t)
        for (const double action : previous_actions2_) observation_.push_back(action);   // [51:63] a(t-1)
        const double t_mid = 0.5 * (olaf_thermal_clip_[0] + olaf_thermal_clip_[1]);
        const double t_half = 0.5 * (olaf_thermal_clip_[1] - olaf_thermal_clip_[0]);
        for (const char* name : kPolicyJointNames) {                   // [63:75] motor temperatures
            const double temp = motor_status[joint_id_to_motor_index_.at(kinematics_->joint_id(name))].temperature;
            observation_.push_back((std::clamp(temp, olaf_thermal_clip_[0], olaf_thermal_clip_[1]) - t_mid) / t_half);
        }
        if (olaf_walking_) {
            observation_.push_back(std::cos(2.0 * PI * olaf_phase_));   // [75:77] gait phase
            observation_.push_back(std::sin(2.0 * PI * olaf_phase_));
            for (std::size_t i = 0; i < 3; ++i) observation_.push_back(olaf_g_[i]);   // [77:80] path velocity
        } else {
            for (std::size_t i = 0; i < 4; ++i) observation_.push_back(olaf_g_[i]);   // [75:79] torso dz roll pitch yaw
        }
        gait_phase_ = olaf_walking_ ? olaf_phase_ : 0.0;   // logged in the 'phase' column
        return;
    }

    if (velocity_obs_) {
        // ArtiH1VelocityEnv: the sim IMU sits on the pelvis with identity orientation, so its gyro and gravity are in
        // the pelvis frame (root_R_imu maps the real sensor axes onto it).
        const Eigen::Vector3d gyro_root =
            blind_imu_ ? Eigen::Vector3d(Eigen::Vector3d::Zero()) : Eigen::Vector3d(root_R_imu * angular_velocity_imu);
        // gravity direction in the pelvis frame; heading_R_root has the same tilt as world_R_root (heading is about z)
        const Eigen::Vector3d gravity_root = heading_R_root.transpose() * Eigen::Vector3d(0.0, 0.0, -1.0);
        append_vector(gyro_root, observation_);                       // [0:3]
        append_vector(gravity_root, observation_);                    // [3:6]
        for (std::size_t i = 0; i < NUM_JOINTS; ++i) {                // [6:18] joint pos - default pose, policy order
            const auto joint_id = kinematics_->joint_id(kPolicyJointNames[i]);
            observation_.push_back(motor_status[joint_id_to_motor_index_.at(joint_id)].position - default_pose_[i]);
        }
        for (const char* name : kPolicyJointNames) {                  // [18:30] joint velocities
            const auto joint_id = kinematics_->joint_id(name);
            observation_.push_back(motor_status[joint_id_to_motor_index_.at(joint_id)].velocity);
        }
        for (const double action : previous_actions_) observation_.push_back(action);   // [30:42] a(t)
        if (action_history_obs_) {
            for (const double action : previous_actions2_) observation_.push_back(action);   // [42:54] a(t-1)
            for (const double action : previous_actions3_) observation_.push_back(action);   // [54:66] a(t-2)
            for (std::size_t i = 0; i < NUM_JOINTS; ++i) {                                // [66:78] target - default
                observation_.push_back(q_target_[i] - default_pose_[i]);
            }
        }
        joy_interface::joy_data_t joy_data{};
        joy_handler_->read(joy_data);
        const auto dead = [this](double v) { return std::abs(v) < stick_deadzone_ ? 0.0 : v; };
        command_ = {dead(joy_data.stick_ly) * command_max_[0], -dead(joy_data.stick_lx) * command_max_[1],
                    -dead(joy_data.stick_rx) * command_max_[2]};
        for (std::size_t i = 0; i < 3; ++i) observation_.push_back(command_[i] * command_scale_[i]);   // last 3
        return;
    }

    // heading rotation [0:6]
    append_rotation(heading_R_root, observation_);

    // body angular velocity in heading frame [6:9]
    append_vector(angular_velocity_heading, observation_);

    // joint rotations [9:81], policy order
    for (const char* name : kPolicyJointNames) {
        const auto joint_id = kinematics_->joint_id(name);
        append_rotation(kinematics_->joint_local_rotation(joint_id), observation_);
    }

    // joint velocities [81:93], policy order
    for (const char* name : kPolicyJointNames) {
        const auto joint_id = kinematics_->joint_id(name);
        observation_.push_back(motor_status[joint_id_to_motor_index_.at(joint_id)].velocity);
    }

    // foot positions [93:99]
    for (std::size_t i = 0; i < foot_frame_ids_.size(); ++i) {
        const auto foot_frame_id = foot_frame_ids_[i];
        const auto foot_toe_offset = foot_toe_offsets_[i];

        const Eigen::Vector3d toe_position = 
            kinematics_->frame_position(foot_frame_id) +
            kinematics_->frame_rotation(foot_frame_id) * foot_toe_offset;

        append_vector(heading_R_root * toe_position, observation_);
    }

    // previous actions [99:111], policy order (raw ONNX "actions", already clamped to [-1, 1])
    for (const double action : previous_actions_) {
        observation_.push_back(action);
    }

    // gait clock [111:113]
    observation_.push_back(std::sin(2.0 * PI * gait_phase_));
    observation_.push_back(std::cos(2.0 * PI * gait_phase_));

    // read joy data
    joy_interface::joy_data_t joy_data{};
    joy_handler_->read(joy_data);

    // command [113:116] = command * command_scale, full stick = the training command range
    const auto dead = [this](double v) { return std::abs(v) < stick_deadzone_ ? 0.0 : v; };
    command_ = {dead(joy_data.stick_ly) * command_max_[0], -dead(joy_data.stick_lx) * command_max_[1],
                -dead(joy_data.stick_rx) * command_max_[2]};
    for (std::size_t i = 0; i < 3; ++i) observation_.push_back(command_[i] * command_scale_[i]);
}

void robot::ArtiH1::safe_stop(const std::string& reason) {
    olaf_servo_stop();
    for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
        motor_command_[i] = motor_interface::motor_command_t{};
        motor_command_[i].kd = policy_kd_[i];
    }
    if (motor_manager_->is_running()) {
        motor_manager_->write(motor_command_);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));   // let every master send the damping command
    }
    if (log_file_ != nullptr) std::fflush(log_file_);
    throw std::runtime_error("[ArtiH1::control] " + reason);
}

int robot::ArtiH1::fault_status() const {
    int status = 0;
    if (imu_stale_) status |= FAULT_IMU;
    if (motor_manager_->comm_fault()) status |= FAULT_MOTOR_COMM;
    if (!joy_handler_->connected()) status |= FAULT_JOY;
    return status;
}

void robot::ArtiH1::report_faults(int status) {
    const int previous = fault_ & ~FAULT_HOLDING;
    if (status == previous) return;
    const double t = log_start_ == std::chrono::steady_clock::time_point{} ? 0.0
                     : std::chrono::duration<double>(std::chrono::steady_clock::now() - log_start_).count();
    const auto say = [&](int bit, const char* what) {
        if ((status & bit) && !(previous & bit)) std::cerr << "[ArtiH1] t=" << t << " s FAULT: " << what << std::endl;
        if (!(status & bit) && (previous & bit)) std::cerr << "[ArtiH1] t=" << t << " s cleared: " << what
                                                           << (holding_ ? " (still holding, restart to walk again)" : "")
                                                           << std::endl;
    };
    say(FAULT_IMU, "IMU samples stopped changing (MTi-3 disconnected?)");
    say(FAULT_MOTOR_COMM, "motor link (see the [MotorManager] / [UnitreeMaster] messages)");
    say(FAULT_JOY, joy_loss_hold_ ? "joystick lost" : "joystick lost (zero command: the policy stands until the pad is back)");
    if (log_file_ != nullptr) std::fflush(log_file_);
}

void robot::ArtiH1::hold_control() {
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - hold_start_).count();
    const double alpha = fault_settle_time_ > 0.0 ? std::min(1.0, t / fault_settle_time_) : 0.0;
    for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
        q_target_[i] = hold_from_[i] + alpha * (default_pose_[i] - hold_from_[i]);
        const auto motor_index = joint_id_to_motor_index_.at(kinematics_->joint_id(kPolicyJointNames[i]));
        motor_command_[motor_index].position = q_target_[i];
        motor_command_[motor_index].kp = policy_kp_[motor_index];
        motor_command_[motor_index].kd = policy_kd_[motor_index];
    }
    motor_manager_->write(motor_command_);
    if (log_file_ != nullptr) write_log_row();
}

void robot::ArtiH1::control() {
    // real cos(tilt) of the pelvis (= observation_[5] unless --blind-imu); a fallen robot is only damped
    if (!observation_.empty() && true_tilt_cos_ < std::cos(max_tilt_)) {
        safe_stop("Pelvis tilt " + std::to_string(std::acos(std::clamp(true_tilt_cos_, -1.0, 1.0)) * 180.0 / PI) +
                  " deg > max_tilt: fallen, joints damped.");
    }
    // IMU / motor link / joystick fault: stop the policy and hold the pose, keep running and logging
    const int status = fault_status();
    report_faults(status);
    const int hold_mask = FAULT_IMU | FAULT_MOTOR_COMM | (joy_loss_hold_ ? FAULT_JOY : 0);
    if ((status & hold_mask) != 0 && !holding_) {
        holding_ = true;
        hold_from_ = q_target_;
        if (olaf_obs_ && olaf_servo_run_) {
            // the hold writes the motor commands from here on; under the servo lock, so a servo write cannot land after
            // it. Before the servo has started q_target_ is still the standing pose (olaf_q_filt_ is not set yet).
            std::lock_guard<std::mutex> lock(olaf_servo_mutex_);
            olaf_servo_active_ = false;
            hold_from_ = olaf_q_filt_;
        }
        hold_start_ = std::chrono::steady_clock::now();
        std::cerr << "[ArtiH1] policy stopped, holding the pose (settle to the default stance over "
                  << fault_settle_time_ << " s). Ctrl+C to exit." << std::endl;
    }
    fault_ = status | (holding_ ? FAULT_HOLDING : 0);
    if (holding_) {
        hold_control();
        return;
    }
    if (!observation_.empty()) {
        // action_ = q_des (rad, policy order): default pose + scaled action + nominal gait at the next clock
        // phase, clamped to the training joint limits (hip_roll +-15 deg).
        controller::AmpController* active = amp_controller_;
        if (olaf_obs_ && !olaf_walking_) active = olaf_stand_controller_.get();
        active->update(observation_, action_);
        if (olaf_obs_) {
            // q_des goes to the servo thread (first-order hold, slew limit, low-pass at the actuator rate). On the first
            // step the servo starts from q_target_ = the default pose the robot stands in (the env resets q_des, its
            // history and the filter to the reset pose and ramps from there), so the first q_des is ramped, not jumped.
            const auto& actions = active->policy_actions();
            if (actions.size() != NUM_JOINTS) {
                throw std::runtime_error("[ArtiH1::control] The ONNX model has no 'actions' output (needs export_onnx.py).");
            }
            if (!olaf_servo_run_) olaf_servo_start();
            std::copy(action_.begin(), action_.end(), q_target_.begin());
            olaf_servo_set(q_target_);
            previous_actions3_ = previous_actions2_;
            previous_actions2_ = previous_actions_;
            std::copy(actions.begin(), actions.end(), previous_actions_.begin());
            if (log_file_ != nullptr) write_log_row();
            olaf_advance();   // ArtiH1OlafEnv._pre_physics_step: phase, path frame, standing g follow
            return;
        }
        // ArtiH1VelocityEnv._pre_physics_step: low-pass and slew limit against the previous target, then remember it.
        for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
            if (target_lowpass_alpha_ < 1.0) action_[i] = q_target_[i] + target_lowpass_alpha_ * (action_[i] - q_target_[i]);
            if (target_rate_limit_ > 0.0) {
                const double dq = target_rate_limit_ * control_dt_;
                action_[i] = std::clamp(action_[i], q_target_[i] - dq, q_target_[i] + dq);
            }
            q_target_[i] = action_[i];
        }
        for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
            const auto joint_id = kinematics_->joint_id(kPolicyJointNames[i]);
            const auto motor_index = joint_id_to_motor_index_.at(joint_id);
            motor_command_[motor_index].position = action_[i];
            motor_command_[motor_index].kp = policy_kp_[motor_index];
            motor_command_[motor_index].kd = policy_kd_[motor_index];
        }

        const auto& actions = active->policy_actions();
        if (actions.size() != NUM_JOINTS) {
            throw std::runtime_error("[ArtiH1::control] The ONNX model has no 'actions' output (needs pt_to_onnx.py export).");
        }
        previous_actions3_ = previous_actions2_;
        previous_actions2_ = previous_actions_;
        std::copy(actions.begin(), actions.end(), previous_actions_.begin());

        if (log_file_ != nullptr) write_log_row();

        // the env advances the clock once per control step (ArtiH1AmpEnv._pre_physics_step)
        // the env advances the clock the same way (ArtiH1AmpEnv._advance_gait_phase)
        const double step = control_dt_ * gait_phase_rate_;
        double next = std::fmod(gait_phase_ + step, 1.0);
        const bool standing = std::max({std::abs(command_[0]), std::abs(command_[1]), std::abs(command_[2])})
                              <= stand_command_threshold_;
        if (standing) {
            for (const double rest : gait_rest_phases_) {
                const double ahead = std::fmod(rest - gait_phase_ + 1.0, 1.0);
                if (ahead < step || ahead > 1.0 - 1e-6) next = rest;
            }
        }
        gait_phase_ = next;
    }
    motor_manager_->write(motor_command_);
}

// ---------------------------------------------------------------------------------------------------------------------
// arti_h1_olaf (arXiv 2512.16705) support: state estimate, path frame, control input g. Mirrors ArtiH1OlafEnv.

namespace {
double wrap_angle(double a) { return std::atan2(std::sin(a), std::cos(a)); }
double yaw_of(const Eigen::Matrix3d& R) { return std::atan2(R(1, 0), R(0, 0)); }
}

std::array<double, 3> robot::ArtiH1::olaf_project_walk(const std::array<double, 3>& g) const {
    // ArtiH1OlafEnv._walk_blend: inverse-distance blend of the reference_knn nearest clip keys, distances scaled by the
    // command span; the commanded path velocity becomes the blend of those keys
    std::array<double, 3> scale{};
    for (std::size_t i = 0; i < 3; ++i) scale[i] = 1.0 / (olaf_walk_range_[i][1] - olaf_walk_range_[i][0]);
    std::vector<std::pair<double, std::size_t>> d;
    d.reserve(olaf_walk_keys_.size());
    for (std::size_t k = 0; k < olaf_walk_keys_.size(); ++k) {
        double s = 0.0;
        for (std::size_t i = 0; i < 3; ++i) s += std::pow((g[i] - olaf_walk_keys_[k][i]) * scale[i], 2);
        d.emplace_back(std::sqrt(s), k);
    }
    const std::size_t n = std::min<std::size_t>(static_cast<std::size_t>(olaf_knn_), d.size());
    std::partial_sort(d.begin(), d.begin() + n, d.end());
    std::array<double, 3> out{};
    double wsum = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        const double w = 1.0 / (d[j].first + 1e-3);
        wsum += w;
        for (std::size_t i = 0; i < 3; ++i) out[i] += w * olaf_walk_keys_[d[j].second][i];
    }
    for (auto& v : out) v /= wsum;
    return out;
}

void robot::ArtiH1::olaf_estimate(const Eigen::Matrix3d& world_R_root, const Eigen::Vector3d& gyro_root,
                                  const Eigen::Vector3d& accel_root, const Eigen::Vector3d& imu_root) {
    // ArtiH1OlafEnv._estimate: contact-aided InEKF with the body origin at the IMU, pelvis axes. The soles s_i are the
    // leg kinematics relative to the IMU. The policy only sees the pelvis relative to the path frame, which is
    // integrated in the same estimator frame, so the unobservable absolute position / yaw drift does not matter.
    std::array<Eigen::Vector3d, 2> s{};
    for (std::size_t i = 0; i < 2; ++i) {
        s[i] = kinematics_->frame_position(foot_frame_ids_[i]) +
               kinematics_->frame_rotation(foot_frame_ids_[i]) * foot_toe_offsets_[i] - imu_root;
    }
    // one filter step, then (first olaf_restart_steps_ estimates) the restart from the MTi-3 attitude, as the env
    const bool restart = !olaf_est_init_ || olaf_est_steps_ < olaf_restart_steps_;
    if (olaf_est_init_) {
        olaf_ekf_.propagate(gyro_root, accel_root, control_dt_);
        std::array<Eigen::Vector3d, 2> rel{olaf_ekf_.R() * s[0], olaf_ekf_.R() * s[1]};
        const double zmin = std::min(rel[0].z(), rel[1].z());
        std::array<bool, 2> contact{};
        for (std::size_t i = 0; i < 2; ++i) contact[i] = rel[i].z() - zmin < olaf_contact_tolerance_;
        olaf_ekf_.correct(s, contact);
    }
    if (restart) olaf_ekf_.reset(world_R_root, s);
    ++olaf_est_steps_;
    olaf_est_init_ = true;
    olaf_R_ = olaf_ekf_.R();
    const Eigen::Vector3d w = gyro_root - olaf_ekf_.gyro_bias();
    olaf_pos_ = olaf_ekf_.p() - olaf_R_ * imu_root;
    olaf_vel_ = olaf_ekf_.v() - olaf_R_ * w.cross(imu_root);
    olaf_contact_ = olaf_ekf_.contact();
    for (std::size_t i = 0; i < 2; ++i) {
        olaf_feet_world_[i] = olaf_ekf_.p() + olaf_R_ * s[i];
        olaf_feet_yaw_[i] = yaw_of(olaf_R_ * kinematics_->frame_rotation(foot_frame_ids_[i]));
    }
    olaf_root_yaw_ = yaw_of(olaf_R_);
    if (restart) olaf_reset_path();
}

void robot::ArtiH1::olaf_reset_path() {
    // the env starts the path frame at the spawned torso (position, heading)
    olaf_path_ = {olaf_pos_.x(), olaf_pos_.y(), olaf_root_yaw_};
    if (!olaf_walking_) olaf_g_ = olaf_g_target_;
}

void robot::ArtiH1::olaf_advance() {
    const double dt = control_dt_;
    if (olaf_walking_) {
        olaf_phase_ = std::fmod(olaf_phase_ + dt / olaf_walk_period_, 1.0);
        const double psi = olaf_path_[2];
        olaf_path_[0] += (std::cos(psi) * olaf_g_[0] - std::sin(psi) * olaf_g_[1]) * dt;
        olaf_path_[1] += (std::sin(psi) * olaf_g_[0] + std::cos(psi) * olaf_g_[1]) * dt;
        olaf_path_[2] = psi + olaf_g_[2] * dt;
    } else {
        for (std::size_t i = 0; i < 4; ++i) {
            const double step = olaf_stand_rate_[i] * dt;
            olaf_g_[i] += std::clamp(olaf_g_target_[i] - olaf_g_[i], -step, step);
        }
        // converge toward the feet midpoint and the mean sole heading (right foot link faces -x: yaw + pi)
        const double yl = olaf_feet_yaw_[0], yr = olaf_feet_yaw_[1] + PI;
        const double psi_f = std::atan2(std::sin(yl) + std::sin(yr), std::cos(yl) + std::cos(yr));
        const double a = 1.0 - std::exp(-dt / olaf_stand_path_tau_);
        olaf_path_[0] += a * (0.5 * (olaf_feet_world_[0].x() + olaf_feet_world_[1].x()) - olaf_path_[0]);
        olaf_path_[1] += a * (0.5 * (olaf_feet_world_[0].y() + olaf_feet_world_[1].y()) - olaf_path_[1]);
        olaf_path_[2] += a * wrap_angle(psi_f - olaf_path_[2]);
    }
    // keep the path frame within a bounded distance of the torso
    const double ex = olaf_path_[0] - olaf_pos_.x(), ey = olaf_path_[1] - olaf_pos_.y();
    const double dist = std::max(std::hypot(ex, ey), 1e-6);
    const double k = std::min(1.0, olaf_path_max_distance_ / dist);
    olaf_path_[0] = olaf_pos_.x() + ex * k;
    olaf_path_[1] = olaf_pos_.y() + ey * k;
    olaf_path_[2] = olaf_root_yaw_ + std::clamp(wrap_angle(olaf_path_[2] - olaf_root_yaw_), -olaf_path_max_yaw_, olaf_path_max_yaw_);
}

// ---------------------------------------------------------------------------------------------------------------------
// olaf runtime chain (paper Sec. VII-C): ArtiH1OlafEnv._apply_action at the actuator rate

void robot::ArtiH1::olaf_servo_set(const std::array<double, 12>& q_des) {
    std::lock_guard<std::mutex> lock(olaf_servo_mutex_);
    // the new ramp starts where the current one is (= the previous q_des after the 12 ticks of a control period, as in
    // the env); if the 50 Hz loop and the servo drift apart (11 or 13 ticks) the hold input stays continuous
    const int n = std::max(1, static_cast<int>(std::lround(control_dt_ * olaf_actuator_rate_)));
    const double k = static_cast<double>(std::min(olaf_servo_tick_, n)) / n;
    for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
        olaf_q_des_prev_[i] = olaf_q_des_prev_[i] + k * (olaf_q_des_[i] - olaf_q_des_prev_[i]);
    }
    olaf_q_des_ = q_des;
    olaf_servo_walking_ = olaf_walking_;
    olaf_servo_tick_ = 0;
}

void robot::ArtiH1::olaf_servo_start() {
    {
        // the env resets q_des / its history / the filter to the reset joint target (the default pose here)
        std::lock_guard<std::mutex> lock(olaf_servo_mutex_);
        olaf_q_des_prev_ = q_target_;
        olaf_q_des_ = q_target_;
        olaf_q_filt_ = q_target_;
        olaf_servo_tick_ = 0;
    }
    olaf_servo_active_ = true;
    olaf_servo_run_ = true;
    olaf_servo_thread_ = std::thread(&ArtiH1::olaf_servo_loop, this);
}

void robot::ArtiH1::shutdown_servo() { olaf_servo_stop(); }

void robot::ArtiH1::olaf_servo_stop() {
    olaf_servo_active_ = false;
    olaf_servo_run_ = false;
    if (olaf_servo_thread_.joinable()) olaf_servo_thread_.join();
}

void robot::ArtiH1::olaf_servo_loop() {
    const auto period = std::chrono::nanoseconds(static_cast<long long>(std::llround(1e9 / olaf_actuator_rate_)));
    const int n = std::max(1, static_cast<int>(std::lround(control_dt_ * olaf_actuator_rate_)));
    const auto lp_alpha = [this](double hz) { return hz > 0.0 ? 1.0 - std::exp(-2.0 * PI * hz / olaf_actuator_rate_) : 1.0; };
    const double alpha_walk = lp_alpha(olaf_lowpass_hz_), alpha_stand = lp_alpha(olaf_stand_lowpass_hz_);
    const double slew = olaf_rate_limit_ > 0.0 ? olaf_rate_limit_ / olaf_actuator_rate_ : 1e9;
    motor_interface::motor_command_t cmd[motor_interface::MAX_MOTORS]{};
    std::array<double, NUM_JOINTS> q{};
    auto next = std::chrono::steady_clock::now();
    while (olaf_servo_run_) {
        next += period;
        std::this_thread::sleep_until(next);
        const auto now = std::chrono::steady_clock::now();
        if (next < now - period) next = now;   // overrun: do not try to catch up
        if (!olaf_servo_active_) continue;
        {
            std::lock_guard<std::mutex> lock(olaf_servo_mutex_);
            olaf_servo_tick_ = std::min(olaf_servo_tick_ + 1, n);
            const double k = static_cast<double>(olaf_servo_tick_) / n;
            const double alpha = olaf_servo_walking_ ? alpha_walk : alpha_stand;   // per policy (stand may be slower)
            for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
                double u = olaf_q_des_prev_[i] + k * (olaf_q_des_[i] - olaf_q_des_prev_[i]);
                u = std::clamp(u, olaf_q_filt_[i] - slew, olaf_q_filt_[i] + slew);
                olaf_q_filt_[i] += alpha * (u - olaf_q_filt_[i]);
            }
            q = olaf_q_filt_;
        }
        for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
            const auto motor_index = joint_id_to_motor_index_.at(kinematics_->joint_id(kPolicyJointNames[i]));
            cmd[motor_index].position = q[i];
            cmd[motor_index].kp = policy_kp_[motor_index];
            cmd[motor_index].kd = policy_kd_[motor_index];
        }
        std::lock_guard<std::mutex> lock(olaf_servo_mutex_);   // the hold hand-off clears the flag under this lock
        if (olaf_servo_active_) motor_manager_->write(cmd);
    }
}
