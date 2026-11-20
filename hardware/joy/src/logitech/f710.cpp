#include <libusb-1.0/libusb.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "joy/logitech/f710.hpp"

namespace {
constexpr auto RECONNECT_PERIOD = std::chrono::seconds(1);

constexpr unsigned char EP_IN = 0x81;
constexpr unsigned char EP_OUT = 0x02;
constexpr int INTERFACE = 0;
constexpr int REPORT_SIZE = 32;
constexpr unsigned int READ_TIMEOUT_MS = 1;

float stick(int16_t raw, float threshold = 0.1f) {
    const float v = std::clamp(static_cast<float>(raw) / 32767.0f, -1.0f, 1.0f);
    return (v > -threshold && v < threshold) ? 0.0f : v;
}

int16_t le16(const uint8_t* p) {
    return static_cast<int16_t>(static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8));
}
} // namespace

logitech::F710::F710(const std::string& device, uint32_t period, const std::string& axis_layout)
: joy_interface::Joy(device, period, axis_layout) {
    const auto colon = device.find(':');
    if (colon == std::string::npos) {
        throw std::runtime_error("[F710] device must be \"vid:pid\" in hex (e.g. 046d:c21f), got: " + device);
    }
    vendor_id_ = static_cast<uint16_t>(std::stoul(device.substr(0, colon), nullptr, 16));
    product_id_ = static_cast<uint16_t>(std::stoul(device.substr(colon + 1), nullptr, 16));
}

logitech::F710::~F710() {
    shutdown();
}

bool logitech::F710::open_device() {
    handle_ = libusb_open_device_with_vid_pid(context_, vendor_id_, product_id_);
    if (!handle_) return false;

    if (libusb_kernel_driver_active(handle_, INTERFACE) == 1) {
        libusb_detach_kernel_driver(handle_, INTERFACE);
    }
    if (libusb_claim_interface(handle_, INTERFACE) != LIBUSB_SUCCESS) {
        libusb_close(handle_);
        handle_ = nullptr;
        return false;
    }

    // LED ring: player 1 (also tells the receiver a host is listening)
    unsigned char led[3] = {0x01, 0x03, 0x06};
    int transferred = 0;
    libusb_interrupt_transfer(handle_, EP_OUT, led, sizeof(led), &transferred, 100);

    return true;
}

void logitech::F710::close_device() {
    if (!handle_) return;
    libusb_release_interface(handle_, INTERFACE);
    libusb_close(handle_);
    handle_ = nullptr;
}

void logitech::F710::initialize() {
    if (initialized_) return;

    if (libusb_init(&context_) != LIBUSB_SUCCESS) {
        context_ = nullptr;
        throw std::runtime_error("[F710::initialize] libusb_init failed");
    }

    if (!open_device()) {
        libusb_exit(context_);
        context_ = nullptr;
        throw std::runtime_error("[F710::initialize] Failed to open USB device " + device_ +
                                 " (receiver plugged in, switch on X, udev permission?)");
    }

    initialized_ = true;
    connected_.store(true);
}

void logitech::F710::disconnect(const char* reason) {
    close_device();
    connected_.store(false);
    {
        // a dropped pad sends no release events: clear everything so the robot gets the zero command (stand)
        std::lock_guard<std::mutex> lock(mutex_);
        data_ = joy_interface::joy_data_t{};
    }
    next_open_ = std::chrono::steady_clock::now() + RECONNECT_PERIOD;
    std::cerr << "[F710] " << device_ << " lost (" << reason << "): command set to zero, reconnecting..." << std::endl;
}

void logitech::F710::try_reconnect() {
    const auto now = std::chrono::steady_clock::now();
    if (now < next_open_) return;
    next_open_ = now + RECONNECT_PERIOD;
    if (open_device()) {
        connected_.store(true);
        std::cerr << "[F710] " << device_ << " reconnected." << std::endl;
    }
}

void logitech::F710::update() {
    if (!initialized_) return;
    if (!handle_) {
        try_reconnect();
        if (!handle_) return;
    }

    // drain every report queued since the last cycle (XInput only reports on change)
    unsigned char report[REPORT_SIZE];
    while (true) {
        int transferred = 0;
        const int rc = libusb_interrupt_transfer(handle_, EP_IN, report, sizeof(report), &transferred, READ_TIMEOUT_MS);
        if (rc == LIBUSB_ERROR_TIMEOUT) break;
        if (rc == LIBUSB_ERROR_INTERRUPTED) continue;
        if (rc != LIBUSB_SUCCESS) {
            disconnect(libusb_error_name(rc));
            return;
        }
        parse(report, transferred);
    }
}

void logitech::F710::parse(const uint8_t* r, int length) {
    // XInput input report: type 0x00, length 0x14
    if (length < 14 || r[0] != 0x00 || r[1] < 14) return;

    std::lock_guard<std::mutex> lock(mutex_);

    const uint8_t b0 = r[2];
    const uint8_t b1 = r[3];

    const bool up = b0 & 0x01, down = b0 & 0x02, left = b0 & 0x04, right = b0 & 0x08;
    data_.dpad_x = static_cast<float>(right) - static_cast<float>(left);
    data_.dpad_y = static_cast<float>(up) - static_cast<float>(down);   // up = +1, as the DualSense
    data_.start = b0 & 0x10;
    data_.select = b0 & 0x20;

    data_.l1 = b1 & 0x01;
    data_.r1 = b1 & 0x02;
    data_.center = b1 & 0x04;
    data_.cross = b1 & 0x10;      // A
    data_.circle = b1 & 0x20;     // B
    data_.square = b1 & 0x40;     // X
    data_.triangle = b1 & 0x80;   // Y

    // triggers 0..255 -> [-1, 1] (released = -1, the joydev convention of the DualSense)
    data_.l2_analog = static_cast<float>(r[4]) / 127.5f - 1.0f;
    data_.r2_analog = static_cast<float>(r[5]) / 127.5f - 1.0f;
    data_.l2 = r[4] > 30;
    data_.r2 = r[5] > 30;

    // XInput sticks: x right +, y up + (same sign as the DualSense values after its y flip)
    data_.stick_lx = stick(le16(r + 6));
    data_.stick_ly = stick(le16(r + 8));
    data_.stick_rx = stick(le16(r + 10));
    data_.stick_ry = stick(le16(r + 12));
}

void logitech::F710::shutdown() {
    close_device();
    if (context_) {
        libusb_exit(context_);
        context_ = nullptr;
    }
    initialized_ = false;
}

void logitech::F710::read(joy_interface::joy_data_t& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    data = data_;
}
