#ifndef MOTOR_DRIVER_DAMIAO_DRIVER_HPP_
#define MOTOR_DRIVER_DAMIAO_DRIVER_HPP_

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <algorithm>

#include "motor/core/motor_driver.hpp"

// DAMIAO DM-J8006-2EC (and other DM-J series) over CAN, MIT mode.
// Reference: https://wiki.aifitlab.com/damiao-docs/dm-j8006-2ec-motor-instruction-manual
//
//   Command (MIT)     : CAN ID = motor ID, 8 bytes
//                       D0-1 p_des[16] | D2 v_des[11:4] | D3 v_des[3:0] kp[11:8] | D4 kp[7:0]
//                       D5 kd[11:4] | D6 kd[3:0] t_ff[11:8] | D7 t_ff[7:0]           (all big endian, linear map)
//   Special command   : CAN ID = motor ID, FF FF FF FF FF FF FF {FC enable, FD disable, FE set zero, FB clear error}
//   Feedback          : CAN ID = master ID, 8 bytes
//                       D0 [7:4] status, [3:0] ID low nibble | D1-2 pos[16] | D3 vel[11:4] | D4 vel[3:0] tau[11:8]
//                       D5 tau[7:0] | D6 MOS temp | D7 rotor temp
//   Register access   : CAN ID = 0x7FF, D0-1 motor ID (LE), D2 {0x33 read, 0x55 write, 0xAA save}, D3 RID, D4-7 value (LE)
//
// All MIT quantities are on the OUTPUT shaft (the 6:1 gearbox is inside the motor), so gear_ratio is not applied.
namespace damiao {

constexpr std::size_t TX_PACKET_SIZE = 8;
constexpr std::size_t RX_PACKET_SIZE = 8;

constexpr uint32_t REGISTER_CAN_ID = 0x7FF;

constexpr uint8_t CMD_ENABLE      = 0xFC;
constexpr uint8_t CMD_DISABLE     = 0xFD;
constexpr uint8_t CMD_SAVE_ZERO   = 0xFE;
constexpr uint8_t CMD_CLEAR_ERROR = 0xFB;

constexpr uint8_t REG_READ  = 0x33;
constexpr uint8_t REG_WRITE = 0x55;
constexpr uint8_t REG_SAVE  = 0xAA;

// Register IDs (RID).
constexpr uint8_t RID_MST_ID    = 0x07;  // feedback (master) CAN ID
constexpr uint8_t RID_ESC_ID    = 0x08;  // motor CAN ID
constexpr uint8_t RID_TIMEOUT   = 0x09;  // CAN timeout (x 50 us), 0 = disabled
constexpr uint8_t RID_CTRL_MODE = 0x0A;  // 1 MIT, 2 position-velocity, 3 velocity, 4 force-position
constexpr uint8_t RID_PMAX      = 0x15;  // float, rad
constexpr uint8_t RID_VMAX      = 0x16;  // float, rad/s
constexpr uint8_t RID_TMAX      = 0x17;  // float, Nm
constexpr uint8_t RID_CAN_BR    = 0x23;  // 0 125K, 1 200K, 2 250K, 3 500K, 4 1M

constexpr uint32_t CTRL_MODE_MIT = 1;

// Factory mapping ranges of the DM-J8006 (used only if the PMAX / VMAX / TMAX registers cannot be read).
constexpr double DEFAULT_PMAX = 12.5;
constexpr double DEFAULT_VMAX = 45.0;
constexpr double DEFAULT_TMAX = 20.0;

constexpr double KP_MIN = 0.0;
constexpr double KP_MAX = 500.0;

constexpr double KD_MIN = 0.0;
constexpr double KD_MAX = 5.0;

// Status nibble of the feedback frame (D0[7:4]).
constexpr uint8_t STATUS_DISABLED       = 0x0;
constexpr uint8_t STATUS_ENABLED        = 0x1;
constexpr uint8_t STATUS_OVER_VOLTAGE   = 0x8;
constexpr uint8_t STATUS_UNDER_VOLTAGE  = 0x9;
constexpr uint8_t STATUS_OVER_CURRENT   = 0xA;
constexpr uint8_t STATUS_MOS_OVER_TEMP  = 0xB;
constexpr uint8_t STATUS_COIL_OVER_TEMP = 0xC;
constexpr uint8_t STATUS_COMM_LOST      = 0xD;
constexpr uint8_t STATUS_OVERLOAD       = 0xE;

inline const char* status_string(uint8_t status) {
    switch (status) {
        case STATUS_DISABLED:       return "disabled";
        case STATUS_ENABLED:        return "enabled";
        case STATUS_OVER_VOLTAGE:   return "over-voltage";
        case STATUS_UNDER_VOLTAGE:  return "under-voltage";
        case STATUS_OVER_CURRENT:   return "over-current";
        case STATUS_MOS_OVER_TEMP:  return "MOS over-temperature";
        case STATUS_COIL_OVER_TEMP: return "coil over-temperature";
        case STATUS_COMM_LOST:      return "communication lost";
        case STATUS_OVERLOAD:       return "overload";
        default:                    return "unknown";
    }
}

class DamiaoDriver : public motor_interface::MotorDriver {
public:
    DamiaoDriver(uint8_t id, double gear_ratio, int8_t axis, uint32_t pulse_per_revolution, double min, double max)
    : motor_interface::MotorDriver(id, gear_ratio, axis, pulse_per_revolution, min, max) {}

    virtual ~DamiaoDriver() = default;

    uint8_t id() const { return id_; }

    // Mapping ranges (PMAX / VMAX / TMAX registers); must match the motor or every value is scaled wrong.
    void set_limits(double pmax, double vmax, double tmax) {
        if (pmax <= 0.0 || vmax <= 0.0 || tmax <= 0.0) throw std::runtime_error("[DamiaoDriver::set_limits] Invalid limits.");
        pmax_ = pmax;
        vmax_ = vmax;
        tmax_ = tmax;
    }

    double pmax() const { return pmax_; }
    double vmax() const { return vmax_; }
    double tmax() const { return tmax_; }

    // Raw status nibble of the last decoded feedback frame (STATUS_*).
    uint8_t last_status() const { return last_status_; }

    // Feedback frames carry only the low nibble of the motor ID.
    bool matches(const uint8_t* buffer, std::size_t size) const {
        return buffer != nullptr && size == RX_PACKET_SIZE && (buffer[0] & 0x0F) == (id_ & 0x0F);
    }

    std::size_t encode(const motor_interface::motor_command_t& command, uint8_t* buffer, std::size_t size) override {
        if (buffer == nullptr) throw std::runtime_error("[DamiaoDriver::encode] Null buffer.");

        if (size < TX_PACKET_SIZE) throw std::runtime_error("[DamiaoDriver::encode] Invalid buffer size.");

        const uint16_t position_raw = position(command.position);
        const uint16_t velocity_raw = float_to_uint(command.velocity * axis_, -vmax_, vmax_, 12);
        const uint16_t torque_raw = float_to_uint(command.torque * axis_, -tmax_, tmax_, 12);
        const uint16_t kp_raw = float_to_uint(command.kp, KP_MIN, KP_MAX, 12);
        const uint16_t kd_raw = float_to_uint(command.kd, KD_MIN, KD_MAX, 12);

        write_u16_big_endian(buffer, position_raw);
        buffer[2] = static_cast<uint8_t>(velocity_raw >> 4);
        buffer[3] = static_cast<uint8_t>(((velocity_raw & 0x0F) << 4) | ((kp_raw >> 8) & 0x0F));
        buffer[4] = static_cast<uint8_t>(kp_raw);
        buffer[5] = static_cast<uint8_t>(kd_raw >> 4);
        buffer[6] = static_cast<uint8_t>(((kd_raw & 0x0F) << 4) | ((torque_raw >> 8) & 0x0F));
        buffer[7] = static_cast<uint8_t>(torque_raw);

        return TX_PACKET_SIZE;
    }

    void decode(const uint8_t* buffer, std::size_t size, motor_interface::motor_state_t& status) override {
        if (buffer == nullptr) throw std::runtime_error("[DamiaoDriver::decode] Null buffer.");

        if (size != RX_PACKET_SIZE) throw std::runtime_error("[DamiaoDriver::decode] Invalid buffer size.");

        if (!matches(buffer, size)) throw std::runtime_error("[DamiaoDriver::decode] Motor ID mismatch.");

        const uint8_t state = static_cast<uint8_t>(buffer[0] >> 4);
        const uint16_t position_raw = read_u16_big_endian(buffer + 1);
        const uint16_t velocity_raw = static_cast<uint16_t>((static_cast<uint16_t>(buffer[3]) << 4) | (buffer[4] >> 4));
        const uint16_t torque_raw = static_cast<uint16_t>(((static_cast<uint16_t>(buffer[4]) & 0x0F) << 8) | buffer[5]);
        const uint8_t mos_temperature = buffer[6];
        const uint8_t rotor_temperature = buffer[7];

        last_status_ = state;

        status.position = position(position_raw);
        status.velocity = uint_to_float(velocity_raw, -vmax_, vmax_, 12) * axis_;
        status.torque = uint_to_float(torque_raw, -tmax_, tmax_, 12) * axis_;
        status.temperature = static_cast<double>(std::max(mos_temperature, rotor_temperature));
        // 0 = no fault (enabled / disabled), otherwise the DAMIAO fault code 0x8..0xE.
        status.error = (state >= STATUS_OVER_VOLTAGE) ? state : 0;
    }

    std::size_t enable(uint8_t* buffer, std::size_t size) override {
        return special(CMD_ENABLE, buffer, size);
    }

    std::size_t disable(uint8_t* buffer, std::size_t size) override {
        return special(CMD_DISABLE, buffer, size);
    }

    // Sets the current position as the motor's zero (stored in the motor). Send while disabled.
    std::size_t save_zero(uint8_t* buffer, std::size_t size) {
        return special(CMD_SAVE_ZERO, buffer, size);
    }

    std::size_t clear_error(uint8_t* buffer, std::size_t size) {
        return special(CMD_CLEAR_ERROR, buffer, size);
    }

    // Register frames go to REGISTER_CAN_ID (0x7FF).
    std::size_t read_register(uint8_t rid, uint8_t* buffer, std::size_t size) const {
        return register_frame(REG_READ, rid, 0U, buffer, size);
    }

    std::size_t write_register(uint8_t rid, uint32_t value, uint8_t* buffer, std::size_t size) const {
        return register_frame(REG_WRITE, rid, value, buffer, size);
    }

    std::size_t write_register(uint8_t rid, float value, uint8_t* buffer, std::size_t size) const {
        uint32_t raw;
        std::memcpy(&raw, &value, sizeof(raw));
        return register_frame(REG_WRITE, rid, raw, buffer, size);
    }

    // Saves all registers to flash (motor must be disabled, takes up to 30 ms, ~10k flash cycles).
    std::size_t save_registers(uint8_t* buffer, std::size_t size) const {
        if (size < TX_PACKET_SIZE) throw std::runtime_error("[DamiaoDriver::save_registers] Invalid buffer size.");
        write_u16_little_endian(buffer, id_);
        buffer[2] = REG_SAVE;
        buffer[3] = 0x01;
        buffer[4] = buffer[5] = buffer[6] = buffer[7] = 0x00;
        return TX_PACKET_SIZE;
    }

    // True if the frame is the reply to a read / write of `rid` on this motor.
    bool is_register_reply(const uint8_t* buffer, std::size_t size, uint8_t rid) const {
        return buffer != nullptr && size == RX_PACKET_SIZE &&
               read_u16_little_endian(buffer) == id_ &&
               (buffer[2] == REG_READ || buffer[2] == REG_WRITE) &&
               buffer[3] == rid;
    }

    static uint32_t register_u32(const uint8_t* buffer) { return read_u32_little_endian(buffer + 4); }

    static float register_float(const uint8_t* buffer) {
        const uint32_t raw = read_u32_little_endian(buffer + 4);
        float value;
        std::memcpy(&value, &raw, sizeof(value));
        return value;
    }

private:
    std::size_t special(uint8_t command, uint8_t* buffer, std::size_t size) const {
        if (buffer == nullptr) throw std::runtime_error("[DamiaoDriver::special] Null buffer.");
        if (size < TX_PACKET_SIZE) throw std::runtime_error("[DamiaoDriver::special] Invalid buffer size.");

        for (std::size_t i = 0; i < TX_PACKET_SIZE - 1; ++i) buffer[i] = 0xFF;
        buffer[TX_PACKET_SIZE - 1] = command;

        return TX_PACKET_SIZE;
    }

    std::size_t register_frame(uint8_t op, uint8_t rid, uint32_t value, uint8_t* buffer, std::size_t size) const {
        if (buffer == nullptr) throw std::runtime_error("[DamiaoDriver::register_frame] Null buffer.");
        if (size < TX_PACKET_SIZE) throw std::runtime_error("[DamiaoDriver::register_frame] Invalid buffer size.");

        write_u16_little_endian(buffer, id_);
        buffer[2] = op;
        buffer[3] = rid;
        write_u32_little_endian(buffer + 4, value);

        return TX_PACKET_SIZE;
    }

    static uint16_t float_to_uint(double value, double min, double max, uint8_t bits) {
        value = clamp(value, min, max);

        const double span = max - min;
        const uint32_t max_raw = (1U << bits) - 1U;

        return static_cast<uint16_t>(std::lround((value - min) * static_cast<double>(max_raw) / span));
    }

    static double uint_to_float(uint16_t value, double min, double max, uint8_t bits) {
        const double span = max - min;
        const uint32_t max_raw = (1U << bits) - 1U;

        return static_cast<double>(value) * span / static_cast<double>(max_raw) + min;
    }

    // Same zero convention as CubeMars: motor_position * axis = joint_position + zero_offset
    // (zero_offset_ == -10.0 means "not set yet": raw motor position, no joint limit clamp).
    uint16_t position(const double& value) const {
        double position;
        if (zero_offset_ == -10.0) {
            position = value * axis_;
        } else {
            position = (clamp(value, min_, max_) + zero_offset_) * axis_;
        }
        return float_to_uint(position, -pmax_, pmax_, 16);
    }

    double position(const uint16_t& value) const {
        const double position = uint_to_float(value, -pmax_, pmax_, 16);
        if (zero_offset_ == -10.0) {
            return position * axis_;
        } else {
            return position * axis_ - zero_offset_;
        }
    }

    double pmax_{DEFAULT_PMAX};
    double vmax_{DEFAULT_VMAX};
    double tmax_{DEFAULT_TMAX};

    uint8_t last_status_{STATUS_DISABLED};
};

} // namespace damiao

#endif // MOTOR_DRIVER_DAMIAO_DRIVER_HPP_
