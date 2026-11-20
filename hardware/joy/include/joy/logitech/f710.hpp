#ifndef JOY_LOGITECH_F710_HPP_
#define JOY_LOGITECH_F710_HPP_

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <cstdint>

#include "joy/core/joy.hpp"
#include "joy/core/joy_interface.hpp"

struct libusb_context;
struct libusb_device_handle;

namespace logitech {

// Logitech F710 wireless gamepad, switch on "X" (XInput mode, USB 046d:c21f).
// The Jetson kernel has no xpad driver, so the receiver gets no /dev/input/js*: the XInput reports are read
// directly from the interrupt IN endpoint with libusb. `device` is "vid:pid" in hex (e.g. "046d:c21f").
// Button mapping onto the PlayStation names: A=cross, B=circle, X=square, Y=triangle, LB/RB=l1/r1,
// LT/RT=l2/r2, BACK=select, START=start, LOGITECH=center.
class F710 : public joy_interface::Joy {
public:
    explicit F710(const std::string& device, uint32_t period, const std::string& axis_layout);

    virtual ~F710();

    void initialize() override;

    void update() override;

    void shutdown() override;

    void read(joy_interface::joy_data_t& data) override;

    bool connected() const override { return connected_.load(); }

private:
    bool open_device();

    void close_device();

    void disconnect(const char* reason);

    void try_reconnect();

    void parse(const uint8_t* report, int length);

    uint16_t vendor_id_{0x046d};

    uint16_t product_id_{0xc21f};

    libusb_context* context_{nullptr};

    libusb_device_handle* handle_{nullptr};

    joy_interface::joy_data_t data_;

    std::mutex mutex_;

    bool initialized_{false};

    std::atomic<bool> connected_{false};

    // receiver unplugged: the stick state is cleared (zero command) and update() retries every RECONNECT_PERIOD
    std::chrono::steady_clock::time_point next_open_{};
};

} // namespace logitech

#endif // JOY_LOGITECH_F710_HPP_
