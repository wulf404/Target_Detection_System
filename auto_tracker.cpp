#include "auto_tracker.h"
#include "app_config.h"
#include "can_work.h"
#include "tower_state.h"
#include "turret_command.h"

#include <cmath>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>

extern std::atomic<can_work*> g_can;

static constexpr double CAMERA_FOV_H_DEG = app_config::kCameraFovHDeg;
static constexpr double CAMERA_FOV_V_DEG = app_config::kCameraFovVDeg;

double g_AZ_K = 1.0;
double g_EL_K = 1.0;

double g_P_AZ = 1.0;
double g_P_EL = 1.0;

int g_DEADZONE_X_PX = 100;
int g_DEADZONE_Y_PX = 100;

bool g_enableDeadzone = true;
bool g_enableP = false;

namespace {
using clock_type = std::chrono::steady_clock;

struct TrackerRuntimeState
{
    clock_type::time_point last_send_t{};
    clock_type::time_point hold_start_t{};
    double last_target_az = 0.0;
    double last_target_el = 0.0;
    double az_rel_f = 0.0;
    double el_rel_f = 0.0;
    bool have_last_target = false;
};

std::mutex g_tracker_mutex;
TrackerRuntimeState g_tracker_state;

void resetTrackerState()
{
    g_tracker_state = TrackerRuntimeState{};
}
} // namespace

void AutoTracker::processPixelCenter(const cv::Point& center)
{
    processPixelCenter(center, cv::Size(app_config::kCameraRequestedWidth,
                                        app_config::kCameraRequestedWidth * 9 / 16));
}

void AutoTracker::processPixelCenter(const cv::Point& center, const cv::Size& frameSize)
{
    std::lock_guard<std::mutex> lock(g_tracker_mutex);

    can_work* can = g_can.load(std::memory_order_acquire);
    if (!can) return;
    if (frameSize.width <= 0 || frameSize.height <= 0) return;

    static constexpr double REACH_EPS_DEG = 0.10;
    static constexpr int SEND_PERIOD_MS = 20;
    static constexpr int MAX_HOLD_MS = 100;
    static constexpr double MIN_DELTA_DEG = 0.02;
    static constexpr double LPF_ALPHA = 0.35;

    using clock = clock_type;

    const double frame_width = static_cast<double>(frameSize.width);
    const double frame_height = static_cast<double>(frameSize.height);
    const double cx = frame_width / 2.0;
    const double cy = frame_height / 2.0;

    const double dx = static_cast<double>(center.x) - cx;
    const double dy = static_cast<double>(center.y) - cy;

    if (g_enableDeadzone &&
        std::abs(dx) < g_DEADZONE_X_PX &&
        std::abs(dy) < g_DEADZONE_Y_PX)
    {
        return;
    }

    const double nx = dx / frame_width;
    const double ny = dy / frame_height;

    const double az_err_deg = nx * CAMERA_FOV_H_DEG;
    const double el_err_deg = ny * CAMERA_FOV_V_DEG;

    double az_rel = az_err_deg * g_AZ_K;
    double el_rel = -el_err_deg * g_EL_K;

    g_tracker_state.az_rel_f = (1.0 - LPF_ALPHA) * g_tracker_state.az_rel_f + LPF_ALPHA * az_rel;
    g_tracker_state.el_rel_f = (1.0 - LPF_ALPHA) * g_tracker_state.el_rel_f + LPF_ALPHA * el_rel;

    double az_cmd_rel = g_tracker_state.az_rel_f;
    double el_cmd_rel = g_tracker_state.el_rel_f;
    if (g_enableP) {
        az_cmd_rel *= g_P_AZ;
        el_cmd_rel *= g_P_EL;
    }

    const TurretState turret = getTurretState();

    const auto target = turret_control::normalize({
        turret.az_deg + az_cmd_rel,
        turret.el_deg + el_cmd_rel
    });

    const auto now = clock::now();
    if (g_tracker_state.last_send_t == clock::time_point{}) {
        g_tracker_state.last_send_t = now - std::chrono::milliseconds(SEND_PERIOD_MS);
        g_tracker_state.hold_start_t = now;
    }

    if (turret.fresh && g_tracker_state.have_last_target) {
        const double eaz = std::abs(turret.az_deg - g_tracker_state.last_target_az);
        const double eel = std::abs(turret.el_deg - g_tracker_state.last_target_el);
        const bool reached = (eaz < REACH_EPS_DEG) && (eel < REACH_EPS_DEG);

        const auto hold_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_tracker_state.hold_start_t).count();
        if (!reached && hold_ms < MAX_HOLD_MS) {
            return;
        }
    }

    const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_tracker_state.last_send_t).count();
    if (dt_ms < SEND_PERIOD_MS) return;

    if (g_tracker_state.have_last_target) {
        if (std::abs(target.az - g_tracker_state.last_target_az) < MIN_DELTA_DEG &&
            std::abs(target.el - g_tracker_state.last_target_el) < MIN_DELTA_DEG)
        {
            return;
        }
    }

    g_tracker_state.last_send_t = now;
    g_tracker_state.hold_start_t = now;
    g_tracker_state.last_target_az = target.az;
    g_tracker_state.last_target_el = target.el;
    g_tracker_state.have_last_target = true;

    turret_control::sendPosition(can, target);

    std::cout << "[AUTO TRACK] pixel=(" << center.x << "," << center.y << ") -> "
              << "AZ=" << target.az << "  EL=" << target.el
              << "  (relAZ=" << az_cmd_rel << " relEL=" << el_cmd_rel << ")"
              << std::endl;
}

void AutoTracker::reset()
{
    std::lock_guard<std::mutex> lock(g_tracker_mutex);
    resetTrackerState();
}

AutoTracker::OverlayConfig AutoTracker::overlayConfig()
{
    return {
        g_DEADZONE_X_PX,
        g_DEADZONE_Y_PX,
        g_enableDeadzone
    };
}
