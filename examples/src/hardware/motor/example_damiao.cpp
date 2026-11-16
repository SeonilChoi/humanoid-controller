// DAMIAO DM-J8006-2EC example (MIT mode, CANable / SocketCAN or DAMIAO USB2CAN serial, see damiao.yaml 'device').
//   ./example_damiao ../config/hardware/motor/damiao.yaml 0.5 [ramp_seconds]
//
// Moves every motor from its current position to <goal_position> (rad, output side) with a smooth ramp using
// the kp / kd from the config, then holds and prints the state. Ctrl+C disables the motors.
#include <cmath>
#include <thread>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <iostream>

#include "motor/motor_manager.hpp"
#include "motor/core/motor_interface.hpp"

namespace {
volatile std::sig_atomic_t running = 1;
void on_sigint(int) { running = 0; }
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <config_file> <goal_position> [ramp_seconds]" << std::endl;
        return 1;
    }

    std::signal(SIGINT, on_sigint);

    const double goal_position = std::stod(argv[2]);
    const double ramp_seconds = (argc > 3) ? std::stod(argv[3]) : 3.0;

    motor_manager::MotorManager motor_manager(argv[1]);
    const int n_motors = motor_manager.number_of_motors();

    motor_interface::motor_command_t command[motor_interface::MAX_MOTORS]{};
    motor_interface::motor_state_t status[motor_interface::MAX_MOTORS]{};

    // Limp (kp = kd = 0) until the first feedback arrives.
    motor_manager.write(command);
    motor_manager.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (!motor_manager.is_running()) {
        std::cerr << "Motor thread stopped (no feedback?)." << std::endl;
        return 1;
    }

    motor_manager.read(status);
    double start_position[motor_interface::MAX_MOTORS]{};
    for (int i = 0; i < n_motors; ++i) {
        start_position[i] = status[i].position;
        std::printf("Motor %d start position %.4f rad -> goal %.4f rad\n", i, start_position[i], goal_position);
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto last_print = t0;

    while (running && motor_manager.is_running()) {
        const auto now = std::chrono::steady_clock::now();
        const double t = std::chrono::duration<double>(now - t0).count();
        // Cosine ramp: zero velocity at both ends.
        const double s = (ramp_seconds > 0.0 && t < ramp_seconds) ? 0.5 - 0.5 * std::cos(M_PI * t / ramp_seconds) : 1.0;
        const double ds = (ramp_seconds > 0.0 && t < ramp_seconds) ? 0.5 * M_PI / ramp_seconds * std::sin(M_PI * t / ramp_seconds) : 0.0;

        for (int i = 0; i < n_motors; ++i) {
            const auto& gains = motor_manager.gains(static_cast<uint8_t>(i));
            command[i].position = start_position[i] + (goal_position - start_position[i]) * s;
            command[i].velocity = (goal_position - start_position[i]) * ds;
            command[i].torque = 0.0;
            command[i].kp = gains.kp;
            command[i].kd = gains.kd;
        }
        motor_manager.write(command);

        if (now - last_print >= std::chrono::milliseconds(100)) {
            last_print = now;
            motor_manager.read(status);
            for (int i = 0; i < n_motors; ++i) {
                std::printf("Motor %d | cmd %+.4f | pos %+.4f rad | vel %+.4f rad/s | tau %+.3f Nm | temp %.0f C | err 0x%X\n",
                            i, command[i].position, status[i].position, status[i].velocity, status[i].torque,
                            status[i].temperature, status[i].error);
            }
            std::printf("\n");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    motor_manager.stop();

    return 0;
}
