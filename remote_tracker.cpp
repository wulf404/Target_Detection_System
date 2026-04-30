#include "remote_tracker.h"

#include "can_work.h"
#include "tracking_state.h"
#include "turret_command.h"

#include <chrono>
#include <iostream>

extern can_work* g_can;

static constexpr int MIN_PERIOD_MS = 5;

RemoteTracker::RemoteTracker(QObject* parent) : QObject(parent) {}

void RemoteTracker::setEnabled(bool en)
{
    m_enabled = en;
}

void RemoteTracker::onPixels(int x, int y, bool valid)
{
    if (!m_enabled) return;
    if (!g_can) return;

    refreshMainCamState();

    if (g_main_cam_has_target.load(std::memory_order_relaxed)) {
        return;
    }

    if (!valid) {
        return;
    }

    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
    if (dt < MIN_PERIOD_MS) return;
    last = now;

    const double az_cmd = static_cast<double>(x) / 100.0;
    const double el_cmd = static_cast<double>(y) / 100.0;

    turret_control::sendPosition(g_can, {az_cmd, el_cmd});

    std::cout << "[UART TRACK] angles_x100=(" << x << "," << y << ") -> "
              << "AZ=" << az_cmd << "  EL=" << el_cmd
              << std::endl;
}
