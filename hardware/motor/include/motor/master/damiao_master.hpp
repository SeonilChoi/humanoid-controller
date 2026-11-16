#ifndef MOTOR_MASTER_DAMIAO_MASTER_HPP_
#define MOTOR_MASTER_DAMIAO_MASTER_HPP_

#include <cerrno>
#include <chrono>
#include <string>
#include <thread>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#include <poll.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>

#include "motor/core/motor_master.hpp"
#include "motor/driver/damiao_driver.hpp"

// DAMIAO motors over CAN. Two transports, chosen by the 'device' in the config:
//   - SocketCAN interface name (e.g. "can1"): any SocketCAN adapter, e.g. the CANable (slcand) used for CubeMars.
//   - Device path (e.g. "/dev/ttyACM0"): the DAMIAO USB2CAN adapter (USB CDC serial, 921600 8N1).
//     It is not SLCAN: CAN frames are wrapped in its own serial packets:
//
//   Host -> adapter (30 bytes, one CAN frame):
//     [0]  0x55  [1] 0xAA  header
//     [2]  0x1E  packet length (30)
//     [3]  0x03  command: send CAN frame
//     [4..7]     send count (u32 LE, 1)
//     [8..11]    retransmit interval (u32 LE, 10)
//     [12]       ID type (0 standard, 1 extended)
//     [13..16]   CAN ID (u32 LE)
//     [17]       frame type (0 data, 1 remote)
//     [18]       DLC (8)
//     [19..20]   reserved
//     [21..28]   CAN data
//     [29]       CRC (unused, 0)
//
//   Adapter -> host (16 bytes, one received CAN frame):
//     [0]  0xAA  header
//     [1]  command (0x11 = CAN frame received)
//     [2]  flags / DLC
//     [3..6]     CAN ID (u32 LE)
//     [7..14]    CAN data
//     [15] 0x55  tail
namespace damiao {

constexpr speed_t SERIAL_BAUDRATE = B921600;

constexpr std::size_t SERIAL_TX_SIZE = 30;
constexpr std::size_t SERIAL_RX_SIZE = 16;
constexpr uint8_t SERIAL_RX_HEADER = 0xAA;
constexpr uint8_t SERIAL_RX_TAIL = 0x55;
constexpr uint8_t SERIAL_RX_CAN_FRAME = 0x11;

// Reply window for one update(), as a fraction of the control period (USB adapters add up to ~1 ms round trip).
constexpr double UPDATE_TIMEOUT_RATIO = 0.8;
// Reply window for register / enable commands during initialize().
constexpr auto SETUP_TIMEOUT = std::chrono::milliseconds(50);
// Consecutive cycles without feedback from one motor before update() throws (stops the control thread).
constexpr uint32_t MAX_MISSED_FEEDBACK = 100;

struct can_frame_t {
    uint32_t id{};
    uint8_t dlc{};
    uint8_t data[8]{};
};

class DamiaoMaster : public motor_interface::MotorMaster {
public:
    DamiaoMaster(uint32_t period, const std::string& device)
    : motor_interface::MotorMaster(period),
      device_(device),
      update_timeout_(static_cast<int64_t>(period * UPDATE_TIMEOUT_RATIO)) {}

    virtual ~DamiaoMaster() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    void add_motor(uint8_t id, double gear_ratio, int8_t axis, uint32_t pulse_per_revolution, double min, double max) override {
        if (drivers_.count(id)) throw std::runtime_error("[DamiaoMaster::add_motor] Duplicate motor ID " + std::to_string(id) + ".");
        // Feedback frames identify the motor by the low nibble of its CAN ID only.
        for (const auto& [other, driver] : drivers_) {
            if ((other & 0x0F) == (id & 0x0F)) {
                throw std::runtime_error("[DamiaoMaster::add_motor] Motor IDs " + std::to_string(other) + " and " +
                                         std::to_string(id) + " share the low nibble; feedback cannot tell them apart.");
            }
        }

        auto driver = std::make_unique<DamiaoDriver>(id, gear_ratio, axis, pulse_per_revolution, min, max);
        damiao_drivers_[id] = driver.get();
        drivers_[id] = std::move(driver);
        ids_[n_ids_++] = id;
        missed_[id] = 0;
    }

    void initialize() override {
        if (fd_ < 0) open_device();

        for (uint8_t i = 0; i < n_ids_; ++i) {
            const uint8_t id = ids_[i];
            DamiaoDriver& driver = *damiao_drivers_.at(id);

            // Start from a known state: disabled, MIT mode, faults cleared.
            send_special(driver, &DamiaoDriver::disable);
            drain(SETUP_TIMEOUT);

            uint32_t mode = 0;
            if (!read_register_u32(driver, RID_CTRL_MODE, mode)) {
                throw std::runtime_error("[DamiaoMaster::initialize] Motor " + std::to_string(id) +
                                         " does not answer on " + device_ + " (check CAN ID, CAN bitrate, power).");
            }
            if (mode != CTRL_MODE_MIT) {
                if (!write_register_u32(driver, RID_CTRL_MODE, CTRL_MODE_MIT)) {
                    throw std::runtime_error("[DamiaoMaster::initialize] Failed to set MIT mode on motor " + std::to_string(id) + ".");
                }
                std::cerr << "[DamiaoMaster::initialize] Motor " << static_cast<int>(id)
                          << ": control mode " << mode << " -> MIT (not saved to flash)." << std::endl;
            }

            // MIT mapping ranges must match the motor's PMAX / VMAX / TMAX.
            float pmax = 0.0f, vmax = 0.0f, tmax = 0.0f;
            if (read_register_float(driver, RID_PMAX, pmax) &&
                read_register_float(driver, RID_VMAX, vmax) &&
                read_register_float(driver, RID_TMAX, tmax) &&
                pmax > 0.0f && vmax > 0.0f && tmax > 0.0f) {
                driver.set_limits(pmax, vmax, tmax);
            } else {
                std::cerr << "[DamiaoMaster::initialize] Motor " << static_cast<int>(id)
                          << ": failed to read PMAX/VMAX/TMAX, using defaults " << DEFAULT_PMAX << "/"
                          << DEFAULT_VMAX << "/" << DEFAULT_TMAX << "." << std::endl;
            }

            send_special(driver, &DamiaoDriver::clear_error);
            drain(SETUP_TIMEOUT);

            send_special(driver, &DamiaoDriver::enable);

            // The motor answers the enable command with a feedback frame.
            can_frame_t rx{};
            const auto deadline = std::chrono::steady_clock::now() + SETUP_TIMEOUT;
            bool enabled = false;
            while (receive_frame(rx, deadline)) {
                if (is_feedback(rx) && driver.matches(rx.data, rx.dlc)) {
                    motor_interface::motor_state_t state{};
                    driver.decode(rx.data, rx.dlc, state);
                    enabled = true;
                    break;
                }
            }
            if (!enabled) {
                throw std::runtime_error("[DamiaoMaster::initialize] No feedback after enabling motor " + std::to_string(id) + ".");
            }
            if (driver.last_status() != STATUS_ENABLED) {
                std::cerr << "[DamiaoMaster::initialize] Motor " << static_cast<int>(id) << " status after enable: "
                          << status_string(driver.last_status()) << std::endl;
            }

            std::cout << "[DamiaoMaster::initialize] Motor " << static_cast<int>(id) << " enabled on " << device_
                      << " (PMAX " << driver.pmax() << ", VMAX " << driver.vmax() << ", TMAX " << driver.tmax() << ")."
                      << std::endl;
        }
    }

    void update(const motor_interface::motor_command_t* commands, motor_interface::motor_state_t* status) override {
        // Send every command first, then collect the replies: one USB round trip per cycle instead of one per motor.
        for (uint8_t i = 0; i < n_ids_; ++i) {
            const uint8_t id = ids_[i];
            DamiaoDriver& driver = *damiao_drivers_.at(id);

            can_frame_t tx{};
            tx.id = id;
            tx.dlc = static_cast<uint8_t>(driver.encode(commands[i], tx.data, sizeof(tx.data)));
            send_frame(tx);

            received_[i] = false;
        }

        uint8_t remaining = n_ids_;
        const auto deadline = std::chrono::steady_clock::now() + update_timeout_;
        can_frame_t rx{};

        while (remaining > 0 && receive_frame(rx, deadline)) {
            if (!is_feedback(rx)) continue;

            for (uint8_t i = 0; i < n_ids_; ++i) {
                if (received_[i]) continue;

                DamiaoDriver& driver = *damiao_drivers_.at(ids_[i]);
                if (!driver.matches(rx.data, rx.dlc)) continue;

                driver.decode(rx.data, rx.dlc, status[i]);
                received_[i] = true;
                --remaining;
                break;
            }
        }

        // A motor without a reply keeps its previous state for this cycle.
        for (uint8_t i = 0; i < n_ids_; ++i) {
            const uint8_t id = ids_[i];
            if (received_[i]) {
                missed_[id] = 0;
                continue;
            }
            if (++missed_[id] >= MAX_MISSED_FEEDBACK) {
                throw std::runtime_error("[DamiaoMaster::update] No feedback from motor " + std::to_string(id) + " for " +
                                         std::to_string(MAX_MISSED_FEEDBACK) + " cycles.");
            }
        }
    }

    void shutdown() override {
        if (fd_ < 0) return;

        for (uint8_t i = 0; i < n_ids_; ++i) {
            try {
                send_special(*damiao_drivers_.at(ids_[i]), &DamiaoDriver::disable);
            } catch (const std::exception& e) {
                std::cerr << "[DamiaoMaster::shutdown] " << e.what() << std::endl;
            }
        }
        drain(SETUP_TIMEOUT);
    }

    // Stores the current position as the motor's zero in flash. Call before initialize() or after shutdown() (motor disabled).
    void save_zero(uint8_t id) {
        if (fd_ < 0) open_device();

        DamiaoDriver& driver = *damiao_drivers_.at(id);
        send_special(driver, &DamiaoDriver::disable);
        drain(SETUP_TIMEOUT);
        send_special(driver, &DamiaoDriver::save_zero);
        drain(SETUP_TIMEOUT);
    }

    uint32_t feedback_id() const { return feedback_id_; }

private:
    using SpecialFn = std::size_t (DamiaoDriver::*)(uint8_t*, std::size_t);

    void open_device() {
        serial_ = !device_.empty() && device_[0] == '/';
        if (serial_) {
            open_serial();
        } else {
            open_socketcan();
        }
    }

    void open_socketcan() {
        fd_ = ::socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (fd_ < 0) throw std::runtime_error("[DamiaoMaster::initialize] Failed to create CAN socket.");

        ifreq ifr{};
        std::strncpy(ifr.ifr_name, device_.c_str(), IFNAMSIZ - 1);
        if (::ioctl(fd_, SIOCGIFINDEX, &ifr) < 0) {
            throw std::runtime_error("[DamiaoMaster::initialize] CAN interface " + device_ +
                                     " not found (bring it up first, e.g. requirements/udev-rules.sh for the CANable).");
        }

        sockaddr_can addr{};
        addr.can_family = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;
        if (::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            throw std::runtime_error("[DamiaoMaster::initialize] Failed to bind CAN socket to " + device_ + ".");
        }
    }

    void open_serial() {
        fd_ = ::open(device_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0) {
            throw std::runtime_error("[DamiaoMaster::initialize] Failed to open " + device_ + ": " + std::strerror(errno) +
                                     " (permission? run requirements/udev-rules.sh).");
        }

        termios tio{};
        if (::tcgetattr(fd_, &tio) != 0) throw std::runtime_error("[DamiaoMaster::initialize] tcgetattr failed.");

        ::cfmakeraw(&tio);
        tio.c_cflag |= CLOCAL | CREAD;
        tio.c_cflag &= ~(PARENB | CSTOPB | CSIZE | CRTSCTS);
        tio.c_cflag |= CS8;
        tio.c_cc[VMIN] = 0;
        tio.c_cc[VTIME] = 0;

        if (::cfsetispeed(&tio, SERIAL_BAUDRATE) != 0 || ::cfsetospeed(&tio, SERIAL_BAUDRATE) != 0) {
            throw std::runtime_error("[DamiaoMaster::initialize] Failed to set baudrate.");
        }
        if (::tcsetattr(fd_, TCSANOW, &tio) != 0) {
            throw std::runtime_error("[DamiaoMaster::initialize] Failed to configure " + device_ + ".");
        }
        ::tcflush(fd_, TCIOFLUSH);
        rx_size_ = 0;
    }

    // Feedback / register replies: 8-byte frames that are not one of our own command IDs.
    bool is_feedback(const can_frame_t& frame) const {
        if (frame.dlc != RX_PACKET_SIZE) return false;
        if (feedback_id_ != UINT32_MAX) return frame.id == feedback_id_;
        return frame.id != REGISTER_CAN_ID && damiao_drivers_.count(static_cast<uint8_t>(frame.id)) == 0;
    }

    void send_special(DamiaoDriver& driver, SpecialFn fn) {
        can_frame_t tx{};
        tx.id = driver.id();
        tx.dlc = static_cast<uint8_t>((driver.*fn)(tx.data, sizeof(tx.data)));
        send_frame(tx);
    }

    bool register_transaction(DamiaoDriver& driver, uint8_t rid, const can_frame_t& tx, uint8_t* reply) {
        for (int attempt = 0; attempt < 3; ++attempt) {
            send_frame(tx);

            can_frame_t rx{};
            const auto deadline = std::chrono::steady_clock::now() + SETUP_TIMEOUT;
            while (receive_frame(rx, deadline)) {
                if (!driver.is_register_reply(rx.data, rx.dlc, rid)) continue;

                // The reply comes from the motor's master (feedback) ID: only accept feedback from it from now on.
                if (feedback_id_ == UINT32_MAX && rx.id != REGISTER_CAN_ID && damiao_drivers_.count(static_cast<uint8_t>(rx.id)) == 0) {
                    feedback_id_ = rx.id;
                }

                std::memcpy(reply, rx.data, RX_PACKET_SIZE);
                return true;
            }
        }
        return false;
    }

    bool read_register_raw(DamiaoDriver& driver, uint8_t rid, uint8_t* reply) {
        can_frame_t tx{};
        tx.id = REGISTER_CAN_ID;
        tx.dlc = static_cast<uint8_t>(driver.read_register(rid, tx.data, sizeof(tx.data)));
        return register_transaction(driver, rid, tx, reply);
    }

    bool read_register_u32(DamiaoDriver& driver, uint8_t rid, uint32_t& value) {
        uint8_t reply[RX_PACKET_SIZE]{};
        if (!read_register_raw(driver, rid, reply)) return false;
        value = DamiaoDriver::register_u32(reply);
        return true;
    }

    bool read_register_float(DamiaoDriver& driver, uint8_t rid, float& value) {
        uint8_t reply[RX_PACKET_SIZE]{};
        if (!read_register_raw(driver, rid, reply)) return false;
        value = DamiaoDriver::register_float(reply);
        return true;
    }

    bool write_register_u32(DamiaoDriver& driver, uint8_t rid, uint32_t value) {
        can_frame_t tx{};
        tx.id = REGISTER_CAN_ID;
        tx.dlc = static_cast<uint8_t>(driver.write_register(rid, value, tx.data, sizeof(tx.data)));
        uint8_t reply[RX_PACKET_SIZE]{};
        return register_transaction(driver, rid, tx, reply) && DamiaoDriver::register_u32(reply) == value;
    }

    void send_frame(const can_frame_t& frame) {
        if (serial_) {
            send_serial(frame);
        } else {
            send_socketcan(frame);
        }
    }

    void send_socketcan(const can_frame_t& frame) {
        can_frame tx{};
        tx.can_id = frame.id & CAN_SFF_MASK;
        tx.can_dlc = frame.dlc;
        std::memcpy(tx.data, frame.data, frame.dlc);

        for (int attempt = 0; attempt < 10; ++attempt) {
            const ssize_t written = ::write(fd_, &tx, sizeof(tx));
            if (written == sizeof(tx)) return;
            // TX queue full (ENOBUFS) or interrupted: retry briefly instead of killing the control loop.
            if (written < 0 && (errno == ENOBUFS || errno == EAGAIN || errno == EINTR)) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
            }
            break;
        }
        throw std::runtime_error("[DamiaoMaster::send_frame] Failed to send CAN frame on " + device_ + ": " + std::strerror(errno));
    }

    void send_serial(const can_frame_t& frame) {
        uint8_t packet[SERIAL_TX_SIZE]{};
        packet[0] = 0x55;
        packet[1] = 0xAA;
        packet[2] = static_cast<uint8_t>(SERIAL_TX_SIZE);
        packet[3] = 0x03;                    // send CAN frame
        packet[4] = 0x01;                    // send count = 1
        packet[8] = 0x0A;                    // retransmit interval
        packet[12] = 0x00;                   // standard ID
        packet[13] = static_cast<uint8_t>(frame.id);
        packet[14] = static_cast<uint8_t>(frame.id >> 8);
        packet[15] = static_cast<uint8_t>(frame.id >> 16);
        packet[16] = static_cast<uint8_t>(frame.id >> 24);
        packet[17] = 0x00;                   // data frame
        packet[18] = frame.dlc;
        std::memcpy(packet + 21, frame.data, frame.dlc);

        std::size_t written = 0;
        int retries = 0;
        while (written < SERIAL_TX_SIZE) {
            const ssize_t n = ::write(fd_, packet + written, SERIAL_TX_SIZE - written);
            if (n > 0) {
                written += static_cast<std::size_t>(n);
                continue;
            }
            if (n < 0 && (errno == EAGAIN || errno == EINTR) && ++retries < 100) {
                pollfd pfd{fd_, POLLOUT, 0};
                ::poll(&pfd, 1, 1);
                continue;
            }
            throw std::runtime_error("[DamiaoMaster::send_frame] Failed to write to " + device_ + ": " + std::strerror(errno));
        }
    }

    // Pops one received CAN frame from the serial stream before `deadline`; false on timeout.
    bool receive_frame(can_frame_t& frame, std::chrono::steady_clock::time_point deadline) {
        while (true) {
            if (serial_ && pop_frame(frame)) return true;

            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) return false;

            const auto remaining = std::chrono::duration_cast<std::chrono::microseconds>(deadline - now).count();
            timespec ts{};
            ts.tv_sec = static_cast<time_t>(remaining / 1000000);
            ts.tv_nsec = static_cast<long>((remaining % 1000000) * 1000);

            pollfd pfd{fd_, POLLIN, 0};
            const int ready = ::ppoll(&pfd, 1, &ts, nullptr);
            if (ready < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("[DamiaoMaster::receive_frame] poll failed.");
            }
            if (ready == 0) return false;
            if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
                throw std::runtime_error("[DamiaoMaster::receive_frame] " + device_ + " disconnected.");
            }

            if (!serial_) {
                can_frame rx{};
                const ssize_t n = ::read(fd_, &rx, sizeof(rx));
                if (n < 0) {
                    if (errno == EINTR || errno == EAGAIN) continue;
                    throw std::runtime_error("[DamiaoMaster::receive_frame] read failed: " + std::string(std::strerror(errno)));
                }
                if (n != sizeof(rx) || (rx.can_id & (CAN_EFF_FLAG | CAN_RTR_FLAG | CAN_ERR_FLAG))) continue;
                frame.id = rx.can_id & CAN_SFF_MASK;
                frame.dlc = rx.can_dlc;
                std::memcpy(frame.data, rx.data, sizeof(frame.data));
                return true;
            }

            if (rx_size_ == sizeof(rx_buffer_)) {
                // Garbage without a valid packet: keep the tail that may hold a partial packet.
                std::memmove(rx_buffer_, rx_buffer_ + rx_size_ - (SERIAL_RX_SIZE - 1), SERIAL_RX_SIZE - 1);
                rx_size_ = SERIAL_RX_SIZE - 1;
            }

            const ssize_t n = ::read(fd_, rx_buffer_ + rx_size_, sizeof(rx_buffer_) - rx_size_);
            if (n < 0) {
                if (errno == EINTR || errno == EAGAIN) continue;
                throw std::runtime_error("[DamiaoMaster::receive_frame] read failed: " + std::string(std::strerror(errno)));
            }
            rx_size_ += static_cast<std::size_t>(n);
        }
    }

    // Extracts the first complete adapter packet carrying a CAN frame from rx_buffer_.
    bool pop_frame(can_frame_t& frame) {
        std::size_t i = 0;
        bool found = false;

        while (i + SERIAL_RX_SIZE <= rx_size_) {
            const uint8_t* p = rx_buffer_ + i;
            if (p[0] != SERIAL_RX_HEADER || p[SERIAL_RX_SIZE - 1] != SERIAL_RX_TAIL) {
                ++i;
                continue;
            }

            if (p[1] == SERIAL_RX_CAN_FRAME) {
                frame.id = static_cast<uint32_t>(p[3]) |
                           (static_cast<uint32_t>(p[4]) << 8) |
                           (static_cast<uint32_t>(p[5]) << 16) |
                           (static_cast<uint32_t>(p[6]) << 24);
                frame.dlc = static_cast<uint8_t>(RX_PACKET_SIZE);
                std::memcpy(frame.data, p + 7, RX_PACKET_SIZE);
                found = true;
            }
            i += SERIAL_RX_SIZE;  // other adapter replies (e.g. send acknowledgements) are skipped
            if (found) break;
        }

        // Bytes before i are consumed or cannot start a packet; bytes from i on may be a partial packet.
        const std::size_t consumed = i;
        if (consumed > 0) {
            std::memmove(rx_buffer_, rx_buffer_ + consumed, rx_size_ - consumed);
            rx_size_ -= consumed;
        }
        return found;
    }

    void drain(std::chrono::microseconds timeout) {
        can_frame_t rx{};
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (receive_frame(rx, deadline)) {}
    }

    std::string device_;

    int fd_{-1};

    bool serial_{false};

    const std::chrono::microseconds update_timeout_;

    uint32_t feedback_id_{UINT32_MAX};

    uint8_t rx_buffer_[1024]{};

    std::size_t rx_size_{0};

    std::unordered_map<uint8_t, DamiaoDriver*> damiao_drivers_;

    std::unordered_map<uint8_t, uint32_t> missed_;

    bool received_[motor_interface::MAX_MOTORS]{};
};

} // namespace damiao

#endif // MOTOR_MASTER_DAMIAO_MASTER_HPP_
