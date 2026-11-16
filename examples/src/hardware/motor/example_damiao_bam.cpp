// No-load BAM / system-identification recorder for one DAMIAO motor (DM-J8006 / J6006 / J4340, MIT mode).
//
//   ./examples/example_damiao_bam <bam_config.yaml> <motor_name> [output_root] [--yes]
//   ./examples/example_damiao_bam ../config/sysid/damiao_bam.yaml dm_j8006
//
// Records
//   - the BAM trajectories (Rhoban BAM bam/trajectory.py, ported) for every kp in the config -> <session>/bam/*.json,
//     raw BAM log format (mass = arm_mass = length = 0: no pendulum, the motor's own inertia only);
//   - extra no-load tests (chirp, velocity sweep, torque relay, breakaway) -> <session>/sysid/*.json, same format plus
//     the torque feed-forward, for tools/damiao_sysid/fit_noload.py.
// See config/sysid/damiao_bam.yaml for the tests and the log format.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "motor/motor_manager.hpp"
#include "motor/core/motor_interface.hpp"

namespace {

volatile std::sig_atomic_t running = 1;
void on_sigint(int) { running = 0; }

constexpr double PI = 3.14159265358979323846;

using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

// One MIT command, relative to the test center.
struct Sample {
    double goal{0.0};           // rad
    double goal_velocity{0.0};  // rad/s
    double torque_ff{0.0};      // Nm
    double kp{0.0};
    double kd{0.0};
    bool enable{true};          // false = limp (kp = kd = tau = 0), BAM torque_enable
};

struct Entry {
    double t;
    double position;
    double speed;
    Sample cmd;
    double control;        // measured torque (Nm)
    double torque_demand;  // kp * (goal - q) + kd * (goal_velocity - dq) + ff
    double temperature;
    int error;
};

struct Safety {
    double abort_window{4.0};
    double velocity{25.0};
    double temperature{70.0};
};

struct Settings {
    std::string motor_name;
    fs::path motor_config;
    int loop_period_us{2000};
    YAML::Node center;
    double move_speed{1.0};
    double move_kp{10.0};
    double move_kd{0.5};
    int repeat{1};
    double time_scale{1.0};
    double amplitude{1.0};
    Safety safety;
    std::vector<std::string> bam_trajectories;
    std::vector<double> bam_kp;
    double bam_kd{0.3};
    YAML::Node sysid;
    YAML::Node motor_section;
};

// Motor section value, else defaults, else fallback.
template <typename T>
T pick(const YAML::Node& motor, const YAML::Node& defaults, const std::string& key, const T& fallback) {
    if (motor && motor[key]) return motor[key].as<T>();
    if (defaults && defaults[key]) return defaults[key].as<T>();
    return fallback;
}

Settings load_settings(const fs::path& config_path, const std::string& motor_name) {
    const YAML::Node root = YAML::LoadFile(config_path.string());
    const YAML::Node defaults = root["defaults"];
    const YAML::Node motor = root["motors"][motor_name];
    if (!motor) {
        std::ostringstream names;
        for (const auto& m : root["motors"]) names << " " << m.first.as<std::string>();
        throw std::runtime_error("Motor '" + motor_name + "' not in " + config_path.string() + " (have:" + names.str() + ").");
    }

    Settings s;
    s.motor_name = motor_name;
    s.motor_section = motor;
    // Relative motor_config paths are relative to the bam config file.
    s.motor_config = fs::path(motor["motor_config"].as<std::string>());
    if (s.motor_config.is_relative()) s.motor_config = config_path.parent_path() / s.motor_config;

    s.loop_period_us = pick<int>(motor, defaults, "loop_period_us", 2000);
    s.center = (motor["center"] ? motor["center"] : (defaults ? defaults["center"] : YAML::Node()));
    s.move_speed = pick<double>(motor, defaults, "move_speed", 1.0);
    s.move_kp = pick<double>(motor, defaults, "move_kp", 10.0);
    s.move_kd = pick<double>(motor, defaults, "move_kd", 0.5);
    s.repeat = pick<int>(motor, defaults, "repeat", 1);
    s.time_scale = pick<double>(motor, defaults, "time_scale", 1.0);
    s.amplitude = pick<double>(motor, defaults, "amplitude", 1.0);
    s.bam_trajectories = pick<std::vector<std::string>>(motor, defaults, "bam_trajectories", {});

    const YAML::Node ds = defaults ? defaults["safety"] : YAML::Node();
    const YAML::Node ms = motor["safety"];
    s.safety.abort_window = pick<double>(ms, ds, "abort_window", 4.0);
    s.safety.velocity = pick<double>(ms, ds, "velocity", 25.0);
    s.safety.temperature = pick<double>(ms, ds, "temperature", 70.0);

    if (motor["bam"]) {
        s.bam_kp = motor["bam"]["kp"].as<std::vector<double>>();
        s.bam_kd = motor["bam"]["kd"].as<double>();
    }
    s.sysid = motor["sysid"];

    if (s.loop_period_us < 500) throw std::runtime_error("loop_period_us must be >= 500.");
    if (s.time_scale <= 0.0) throw std::runtime_error("time_scale must be > 0.");
    return s;
}

// ---------------------------------------------------------------------------------------------------------------
// BAM trajectories (bam/trajectory.py): angle(t), torque_enable(t), 6 s each.

double cubic_interpolate(const std::vector<std::array<double, 3>>& keyframes, double t) {
    if (t < keyframes.front()[0]) return keyframes.front()[1];
    if (t > keyframes.back()[0]) return keyframes.back()[1];
    for (std::size_t i = 0; i + 1 < keyframes.size(); ++i) {
        const auto& [t0, x0, v0] = keyframes[i];
        const auto& [t1, x1, v1] = keyframes[i + 1];
        if (t0 <= t && t <= t1) {
            // Hermite form of the same cubic (value and slope at both ends).
            const double h = t1 - t0;
            const double s = (t - t0) / h;
            const double h00 = 2 * s * s * s - 3 * s * s + 1;
            const double h10 = s * s * s - 2 * s * s + s;
            const double h01 = -2 * s * s * s + 3 * s * s;
            const double h11 = s * s * s - s * s;
            return h00 * x0 + h10 * h * v0 + h01 * x1 + h11 * h * v1;
        }
    }
    return keyframes.back()[1];
}

bool bam_trajectory(const std::string& name, double t, double& angle, bool& enable) {
    enable = true;
    if (name == "lift_and_drop") {
        angle = cubic_interpolate({{0.0, 0.0, 0.0}, {2.0, -PI / 2, 0.0}}, t);
        enable = t < 2.0;
    } else if (name == "sin_time_square") {
        angle = std::sin(t * t);
    } else if (name == "up_and_down") {
        angle = cubic_interpolate({{0.0, 0.0, 0.0}, {3.0, PI / 2, 0.0}, {6.0, 0.8 * PI / 2, 0.0}}, t);
    } else if (name == "sin_sin") {
        angle = std::sin(t) * PI / 2 + std::sin(5.0 * t) * 0.5 * std::sin(t * 2.0);
    } else if (name == "half_sine") {
        angle = std::sin(t / 2) * PI / 2;
    } else if (name == "steps") {
        static const double stages[][2] = {{0, 0}, {1, 0.75}, {2.5, 1.55}, {4, 0.75}, {5, 0}, {6, 0}};
        angle = 0.0;
        for (int i = 0; i < 5; ++i) {
            if (stages[i][0] <= t && t <= stages[i + 1][0]) {
                angle = stages[i][1];
                break;
            }
        }
    } else if (name == "nothing") {
        angle = 0.0;
        enable = false;
    } else {
        return false;
    }
    return true;
}

constexpr double BAM_DURATION = 6.0;

// ---------------------------------------------------------------------------------------------------------------

std::string timestamp_string() {
    const std::time_t now = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&now));
    return buf;
}

std::string number_string(double v) {
    std::ostringstream o;
    o << v;
    std::string s = o.str();
    std::replace(s.begin(), s.end(), '.', 'p');
    std::replace(s.begin(), s.end(), '-', 'm');
    return s;
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (const char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

// Raw BAM log (+ extra keys). 'meta' holds already-formatted JSON values.
void write_log(const fs::path& path, const std::vector<std::pair<std::string, std::string>>& meta,
               const std::vector<Entry>& entries) {
    std::FILE* f = std::fopen(path.string().c_str(), "w");
    if (f == nullptr) throw std::runtime_error("Cannot write " + path.string());
    std::fprintf(f, "{\n");
    for (const auto& [key, value] : meta) std::fprintf(f, "  \"%s\": %s,\n", key.c_str(), value.c_str());
    std::fprintf(f, "  \"entries\": [\n");
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const Entry& e = entries[i];
        std::fprintf(f,
                     "    {\"timestamp\": %.6f, \"position\": %.6f, \"speed\": %.5f, \"goal_position\": %.6f, "
                     "\"goal_velocity\": %.5f, \"torque_ff\": %.5f, \"kp\": %.4f, \"kd\": %.4f, \"torque_enable\": %s, "
                     "\"control\": %.5f, \"torque_demand\": %.5f, \"temperature\": %.1f, \"error\": %d}%s\n",
                     e.t, e.position, e.speed, e.cmd.goal, e.cmd.goal_velocity, e.cmd.torque_ff, e.cmd.kp, e.cmd.kd,
                     e.cmd.enable ? "true" : "false", e.control, e.torque_demand, e.temperature, e.error,
                     i + 1 < entries.size() ? "," : "");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
}

std::string jstr(const std::string& s) { return "\"" + json_escape(s) + "\""; }
std::string jnum(double v) {
    std::ostringstream o;
    o.precision(10);
    o << v;
    return o.str();
}

class Recorder {
public:
    Recorder(motor_manager::MotorManager& manager, const Settings& settings)
    : manager_(manager), s_(settings) {}

    void set_center(double center) { center_ = center; }
    double center() const { return center_; }
    bool aborted() const { return aborted_; }

    motor_interface::motor_state_t read() {
        motor_interface::motor_state_t status[motor_interface::MAX_MOTORS]{};
        manager_.read(status);
        return status[0];
    }

    void limp() {
        motor_interface::motor_command_t cmd[motor_interface::MAX_MOTORS]{};
        manager_.write(cmd);
    }

    // Runs profile(t, state) for 'duration' s (profile may end it early by returning false).
    // Logs to 'entries' if not null. False when aborted or interrupted.
    bool run(double duration, const std::function<bool(double, const motor_interface::motor_state_t&, Sample&)>& profile,
             std::vector<Entry>* entries) {
        // Each tick: read the state -> safety check -> command from that state -> send -> log (state, command).
        // Entry k = state at t_k and the command that acts from t_k to t_k+1 (the order a simulator steps in).
        const auto period = std::chrono::microseconds(s_.loop_period_us);
        const auto start = Clock::now();
        auto next = start;

        while (true) {
            const motor_interface::motor_state_t state = read();
            const double t = std::chrono::duration<double>(Clock::now() - start).count();
            if (t > duration) return true;
            if (!running) {
                std::cout << "[bam] Ctrl+C: motor limp." << std::endl;
                limp();
                aborted_ = true;
                return false;
            }
            const std::string why = check(state);
            if (!why.empty()) {
                limp();
                aborted_ = true;
                std::cerr << "[bam] ABORT: " << why << " -> motor limp." << std::endl;
                return false;
            }

            Sample cmd;
            if (!profile(t, state, cmd)) return true;
            write(cmd);

            if (entries != nullptr) {
                const double q = state.position - center_;
                Entry e{};
                e.t = t;
                e.position = q;
                e.speed = state.velocity;
                e.cmd = cmd;
                e.control = state.torque;
                e.torque_demand = cmd.enable ? cmd.kp * (cmd.goal - q) + cmd.kd * (cmd.goal_velocity - state.velocity) + cmd.torque_ff
                                             : 0.0;
                e.temperature = state.temperature;
                e.error = state.error;
                entries->push_back(e);
            }

            next += period;
            const auto now = Clock::now();
            if (next < now) {
                next = now;
            } else {
                std::this_thread::sleep_until(next);
            }
        }
    }

    // Cosine move from the current position to center + goal, then 0.5 s hold (not logged).
    bool move_to(double goal) {
        const double from = read().position - center_;
        const double duration = std::max(0.5, std::abs(goal - from) / s_.move_speed * PI / 2);
        return run(duration, [&](double t, const motor_interface::motor_state_t&, Sample& c) {
            const double a = 0.5 - 0.5 * std::cos(PI * std::min(1.0, t / duration));
            c.goal = from + (goal - from) * a;
            c.kp = s_.move_kp;
            c.kd = s_.move_kd;
            return true;
        }, nullptr) && run(0.5, [&](double, const motor_interface::motor_state_t&, Sample& c) {
            c.goal = goal;
            c.kp = s_.move_kp;
            c.kd = s_.move_kd;
            return true;
        }, nullptr);
    }

private:
    void write(const Sample& c) {
        motor_interface::motor_command_t cmd[motor_interface::MAX_MOTORS]{};
        if (c.enable) {
            cmd[0].position = center_ + c.goal;
            cmd[0].velocity = c.goal_velocity;
            cmd[0].torque = c.torque_ff;
            cmd[0].kp = c.kp;
            cmd[0].kd = c.kd;
        }
        manager_.write(cmd);
    }

    std::string check(const motor_interface::motor_state_t& s) const {
        std::ostringstream why;
        if (!manager_.is_running()) {
            why << "motor communication stopped";
        } else if (s.error != 0) {
            why << "motor fault 0x" << std::hex << s.error << std::dec
                << " (8 over-voltage, 9 under-voltage, A over-current, B MOS temp, C coil temp, D comm lost, E overload)";
        } else if (std::abs(s.position - center_) > s_.safety.abort_window) {
            why << "position " << s.position - center_ << " rad from the center (abort_window " << s_.safety.abort_window << ")";
        } else if (std::abs(s.velocity) > s_.safety.velocity) {
            why << "velocity " << s.velocity << " rad/s (limit " << s_.safety.velocity << ")";
        } else if (s.temperature > s_.safety.temperature) {
            why << "temperature " << s.temperature << " C";
        }
        return why.str();
    }

    motor_manager::MotorManager& manager_;
    const Settings& s_;
    double center_{0.0};
    bool aborted_{false};
};

bool ask(const std::string& message, bool yes) {
    if (yes) {
        std::cout << message << std::endl;
        return true;
    }
    std::cout << message << "  [Enter = run, s + Enter = skip, q + Enter = quit] " << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) return false;
    if (line == "q") {
        running = 0;
        return false;
    }
    return line != "s";
}

void print_summary(const std::vector<Entry>& entries) {
    if (entries.empty()) return;
    double sq = 0.0, max_err = 0.0, max_v = 0.0, max_tau = 0.0;
    for (const auto& e : entries) {
        const double err = e.cmd.goal - e.position;
        sq += err * err;
        max_err = std::max(max_err, std::abs(err));
        max_v = std::max(max_v, std::abs(e.speed));
        max_tau = std::max(max_tau, std::abs(e.control));
    }
    std::printf("[bam]   %zu samples, %.2f s, tracking rms %.4f / max %.4f rad, |v| max %.2f rad/s, |tau| max %.3f Nm\n",
                entries.size(), entries.back().t, std::sqrt(sq / entries.size()), max_err, max_v, max_tau);
}

// Meta common to every log.
std::vector<std::pair<std::string, std::string>> base_meta(const Settings& s, const std::string& trajectory, double kp,
                                                           double kd, double center) {
    return {
        {"motor", jstr(s.motor_name)},
        {"trajectory", jstr(trajectory)},
        {"kp", jnum(kp)},
        {"damping", jnum(kd)},
        {"mass", "0.0"},
        {"arm_mass", "0.0"},
        {"length", "0.0"},
        {"load", jstr("none")},
        {"center", jnum(center)},
        {"loop_period", jnum(s.loop_period_us * 1e-6)},
        {"time_scale", jnum(s.time_scale)},
        {"amplitude", jnum(s.amplitude)},
        {"date", jstr(timestamp_string())},
    };
}

}  // namespace

int main(int argc, char* argv[]) {
    bool yes = false;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--yes" || a == "-y") {
            yes = true;
        } else {
            args.push_back(a);
        }
    }
    if (args.size() < 2) {
        std::cerr << "Usage: " << argv[0] << " <bam_config.yaml> <motor_name> [output_root=./damiao_logs] [--yes]" << std::endl;
        std::cerr << "  e.g. " << argv[0] << " ../config/sysid/damiao_bam.yaml dm_j8006" << std::endl;
        return 1;
    }
    std::signal(SIGINT, on_sigint);

    const fs::path config_path = fs::absolute(args[0]);
    const Settings settings = load_settings(config_path, args[1]);
    const fs::path session = fs::path(args.size() > 2 ? args[2] : "./damiao_logs") / settings.motor_name / timestamp_string();
    fs::create_directories(session / "bam");
    fs::create_directories(session / "sysid");

    {
        std::ofstream info((session / "session.yaml").string());
        info << "# example_damiao_bam session\nmotor: " << settings.motor_name << "\nbam_config: " << config_path.string()
             << "\nmotor_config: " << fs::absolute(settings.motor_config).string() << "\nsection:\n";
        std::stringstream section;
        section << settings.motor_section;
        std::string line;
        while (std::getline(section, line)) info << "  " << line << "\n";
    }

    std::cout << "[bam] motor " << settings.motor_name << ", motor config " << settings.motor_config.string()
              << "\n[bam] logs -> " << session.string() << std::endl;
    if (!ask("[bam] NO LOAD on the output, free to turn +-" + std::to_string(settings.safety.abort_window) +
             " rad, hands clear?", yes)) {
        return 0;
    }

    motor_manager::MotorManager manager(settings.motor_config.string());
    if (manager.number_of_motors() != 1) {
        std::cerr << "[bam] " << settings.motor_config << " must hold exactly one motor (has "
                  << manager.number_of_motors() << ")." << std::endl;
        return 1;
    }

    Recorder rec(manager, settings);
    rec.limp();
    manager.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (!manager.is_running()) {
        std::cerr << "[bam] Motor communication stopped." << std::endl;
        return 1;
    }

    const double start_position = rec.read().position;
    double center = start_position;
    if (settings.center && settings.center.as<std::string>() != "current") center = settings.center.as<double>();
    rec.set_center(center);
    std::cout << "[bam] position " << start_position << " rad, center " << center << " rad" << std::endl;

    int exit_code = 0;
    try {
        // ---- BAM trajectories ----------------------------------------------------------------------------------
        for (const double kp : settings.bam_kp) {
            for (const auto& name : settings.bam_trajectories) {
                for (int n = 1; n <= settings.repeat && running && !rec.aborted(); ++n) {
                    double a0;
                    bool e0;
                    if (!bam_trajectory(name, 0.0, a0, e0)) throw std::runtime_error("Unknown BAM trajectory " + name);

                    std::ostringstream title;
                    title << "[bam] BAM " << name << " kp " << kp << " kd " << settings.bam_kd << " (" << n << "/"
                          << settings.repeat << ", " << BAM_DURATION / settings.time_scale << " s)";
                    if (!ask(title.str(), yes)) {
                        if (!running) break;
                        continue;
                    }
                    if (!rec.move_to(settings.amplitude * a0)) break;

                    std::vector<Entry> entries;
                    const double duration = BAM_DURATION / settings.time_scale;
                    const bool ok = rec.run(duration, [&](double t, const motor_interface::motor_state_t&, Sample& c) {
                        double angle;
                        bool enable;
                        bam_trajectory(name, std::min(t * settings.time_scale, BAM_DURATION), angle, enable);
                        c.goal = settings.amplitude * angle;
                        c.kp = kp;
                        c.kd = settings.bam_kd;
                        c.enable = enable;
                        return true;
                    }, &entries);

                    auto meta = base_meta(settings, name, kp, settings.bam_kd, center);
                    if (!ok) meta.emplace_back("aborted", "true");
                    const fs::path file = session / (ok ? "bam" : "sysid") /
                                          (name + "_kp" + number_string(kp) + "_" + std::to_string(n) + (ok ? "" : "_aborted") + ".json");
                    write_log(file, meta, entries);
                    print_summary(entries);
                    std::cout << "[bam]   -> " << file.string() << std::endl;
                    if (!ok) break;
                }
                if (!running || rec.aborted()) break;
            }
            if (!running || rec.aborted()) break;
        }

        // ---- extra sysid tests ---------------------------------------------------------------------------------
        int index = 0;
        for (const auto& test : settings.sysid) {
            if (!running || rec.aborted()) break;
            const std::string type = test["type"].as<std::string>();
            ++index;
            std::ostringstream params;
            params << test;
            std::string flat = params.str();
            std::replace(flat.begin(), flat.end(), '\n', ' ');
            if (!ask("[bam] sysid " + type + " " + flat, yes)) {
                if (!running) break;
                continue;
            }
            if (!rec.move_to(0.0)) break;

            std::vector<Entry> entries;
            double kp = test["kp"].as<double>(0.0);
            double kd = test["kd"].as<double>(0.0);
            bool ok = true;
            std::vector<std::pair<std::string, std::string>> extra;

            if (type == "chirp") {
                const double amplitude = test["amplitude"].as<double>();
                const double f0 = test["f0"].as<double>();
                const double f1 = test["f1"].as<double>();
                const double duration = test["duration"].as<double>();
                const double k = f1 / f0;
                ok = rec.run(duration, [&](double t, const motor_interface::motor_state_t&, Sample& c) {
                    const double phase = 2.0 * PI * f0 * duration / std::log(k) * (std::pow(k, t / duration) - 1.0);
                    const double fade = std::min({1.0, t, duration - t});
                    const double window = fade <= 0.0 ? 0.0 : 0.5 - 0.5 * std::cos(PI * std::min(1.0, fade));
                    c.goal = amplitude * window * std::sin(phase);
                    c.kp = kp;
                    c.kd = kd;
                    return true;
                }, &entries);
                extra = {{"f0", jnum(f0)}, {"f1", jnum(f1)}, {"chirp_amplitude", jnum(amplitude)}};
            } else if (type == "velocity_sweep") {
                const double window = test["window"].as<double>();
                const auto speeds = test["speeds"].as<std::vector<double>>();
                const int cycles = test["cycles"].as<int>(2);
                // Start at -window, then triangle -window -> +window -> -window at each speed.
                if (!rec.move_to(-window)) break;
                double t_offset = 0.0;
                for (const double v : speeds) {
                    if (!ok) break;
                    const double leg = 2.0 * window / v;
                    const double duration = cycles * 2.0 * leg;
                    const std::size_t first = entries.size();
                    ok = rec.run(duration, [&](double t, const motor_interface::motor_state_t&, Sample& c) {
                        const double phase = std::fmod(t, 2.0 * leg);
                        if (phase < leg) {
                            c.goal = -window + v * phase;
                            c.goal_velocity = v;
                        } else {
                            c.goal = window - v * (phase - leg);
                            c.goal_velocity = -v;
                        }
                        c.kp = kp;
                        c.kd = kd;
                        return true;
                    }, &entries);
                    // continuous timestamps across speeds
                    for (std::size_t i = first; i < entries.size(); ++i) entries[i].t += t_offset;
                    t_offset = entries.empty() ? 0.0 : entries.back().t + settings.loop_period_us * 1e-6;
                }
                std::ostringstream sp;
                sp << "[";
                for (std::size_t i = 0; i < speeds.size(); ++i) sp << (i ? ", " : "") << speeds[i];
                sp << "]";
                extra = {{"window", jnum(window)}, {"speeds", sp.str()}, {"cycles", jnum(cycles)}};
            } else if (type == "torque_relay") {
                const double window = test["window"].as<double>();
                const auto torques = test["torques"].as<std::vector<double>>();
                const double duration = test["duration"].as<double>();
                const double v_cap = test["velocity_cap"].as<double>(settings.safety.velocity * 0.6);
                kp = 0.0;
                double t_offset = 0.0;
                for (const double tau : torques) {
                    if (!ok) break;
                    if (!rec.move_to(0.0)) {
                        ok = false;
                        break;
                    }
                    double sign = 1.0;
                    const std::size_t first = entries.size();
                    ok = rec.run(duration, [&](double, const motor_interface::motor_state_t& s, Sample& c) {
                        const double q = s.position - rec.center();
                        if (q > window) sign = -1.0;
                        if (q < -window) sign = 1.0;
                        c.goal = q;  // kp = 0: goal only for the log
                        c.kp = 0.0;
                        c.kd = kd;
                        c.torque_ff = std::abs(s.velocity) > v_cap ? 0.0 : sign * tau;
                        return true;
                    }, &entries);
                    for (std::size_t i = first; i < entries.size(); ++i) entries[i].t += t_offset;
                    t_offset = entries.empty() ? 0.0 : entries.back().t + 0.5;
                }
                extra = {{"window", jnum(window)}, {"velocity_cap", jnum(v_cap)}};
            } else if (type == "breakaway") {
                const double rate = test["rate"].as<double>();
                const double max_torque = test["max_torque"].as<double>();
                const double move = test["move"].as<double>(0.02);
                const double velocity_threshold = test["velocity"].as<double>(0.15);
                const int cycles = test["cycles"].as<int>(4);
                kp = 0.0;
                kd = 0.0;
                std::vector<double> breakaway[2];
                double t_offset = 0.0;
                for (int c_i = 0; c_i < 2 * cycles && ok; ++c_i) {
                    const double sign = (c_i % 2 == 0) ? 1.0 : -1.0;
                    if (!rec.move_to(0.0)) {
                        ok = false;
                        break;
                    }
                    const double q0 = rec.read().position - rec.center();
                    double tau_at_break = 0.0;
                    double t_break = -1.0;
                    const std::size_t first = entries.size();
                    ok = rec.run(max_torque / rate + 1.0, [&](double t, const motor_interface::motor_state_t& s, Sample& c) {
                        const double q = s.position - rec.center();
                        if (t_break < 0.0 && (std::abs(q - q0) > move || std::abs(s.velocity) > velocity_threshold)) {
                            t_break = t;
                            tau_at_break = std::min(rate * t, max_torque);
                        }
                        if (t_break >= 0.0 && t > t_break + 0.5) return false;
                        c.goal = q;
                        c.kp = 0.0;
                        c.kd = 0.0;
                        c.torque_ff = t_break >= 0.0 ? 0.0 : sign * std::min(rate * t, max_torque);
                        return true;
                    }, &entries);
                    for (std::size_t i = first; i < entries.size(); ++i) entries[i].t += t_offset;
                    t_offset = entries.empty() ? 0.0 : entries.back().t + 0.5;
                    if (t_break >= 0.0) {
                        breakaway[c_i % 2].push_back(tau_at_break);
                        std::printf("[bam]   %s breakaway %.3f Nm\n", sign > 0 ? "+" : "-", tau_at_break);
                    } else {
                        std::printf("[bam]   %s no motion up to %.3f Nm\n", sign > 0 ? "+" : "-", max_torque);
                    }
                }
                auto mean = [](const std::vector<double>& v) {
                    double sum = 0.0;
                    for (const double x : v) sum += x;
                    return v.empty() ? 0.0 : sum / v.size();
                };
                std::printf("[bam]   breakaway mean: + %.3f Nm, - %.3f Nm\n", mean(breakaway[0]), mean(breakaway[1]));
                extra = {{"breakaway_positive", jnum(mean(breakaway[0]))}, {"breakaway_negative", jnum(mean(breakaway[1]))},
                         {"rate", jnum(rate)}};
            } else {
                throw std::runtime_error("Unknown sysid test type: " + type);
            }

            auto meta = base_meta(settings, type, kp, kd, center);
            for (auto& e : extra) meta.push_back(e);
            if (!ok) meta.emplace_back("aborted", "true");
            char name[96];
            std::snprintf(name, sizeof(name), "%02d_%s%s.json", index, type.c_str(), ok ? "" : "_aborted");
            write_log(session / "sysid" / name, meta, entries);
            print_summary(entries);
            std::cout << "[bam]   -> " << (session / "sysid" / name).string() << std::endl;
        }

        if (!rec.aborted() && running) rec.move_to(0.0);
    } catch (const std::exception& e) {
        std::cerr << "[bam] " << e.what() << std::endl;
        exit_code = 1;
    }

    rec.limp();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    manager.stop();
    std::cout << "[bam] done, logs in " << session.string() << std::endl;
    return exit_code;
}
