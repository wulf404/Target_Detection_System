#include "auto_tracker.h"
#include "app_config.h"
#include "can_work.h"
#include "tower_state.h"
#include "turret_command.h"

#include <algorithm>
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
    double center_x_f = 0.0;
    double center_y_f = 0.0;
    int active_deadzone_x = 0;
    int active_deadzone_y = 0;
    int active_deadzone_outer_x = 0;
    int active_deadzone_outer_y = 0;
    bool have_last_target = false;
    bool have_filtered_center = false;
    bool deadzone_hold_active = false;
};

std::mutex g_tracker_mutex;
TrackerRuntimeState g_tracker_state;

void resetTrackerState()
{
    g_tracker_state = TrackerRuntimeState{};
}

struct DeadzoneConfig
{
    int innerX = 0;
    int innerY = 0;
    int outerX = 0;
    int outerY = 0;
};

int outerDeadzoneFromInner(int inner)
{
    return std::max(inner + 1, static_cast<int>(std::round(inner * 1.55)));
}

DeadzoneConfig deadzoneForTarget(const cv::Rect& targetBox)
{
    static constexpr double INNER_BOX_RATIO = 0.35;
    static constexpr int MIN_DYNAMIC_DEADZONE_PX = 16;

    DeadzoneConfig config;

    if (targetBox.width > 0 && targetBox.height > 0) {
        config.innerX = std::clamp(
            static_cast<int>(std::round(targetBox.width * INNER_BOX_RATIO)),
            MIN_DYNAMIC_DEADZONE_PX,
            std::max(MIN_DYNAMIC_DEADZONE_PX, g_DEADZONE_X_PX)
        );
        config.innerY = std::clamp(
            static_cast<int>(std::round(targetBox.height * INNER_BOX_RATIO)),
            MIN_DYNAMIC_DEADZONE_PX,
            std::max(MIN_DYNAMIC_DEADZONE_PX, g_DEADZONE_Y_PX)
        );
    } else {
        config.innerX = g_DEADZONE_X_PX;
        config.innerY = g_DEADZONE_Y_PX;
    }

    config.outerX = outerDeadzoneFromInner(config.innerX);
    config.outerY = outerDeadzoneFromInner(config.innerY);
    return config;
}
} // namespace

void AutoTracker::processPixelCenter(const cv::Point& center)
{
    processPixelCenter(center, cv::Size(app_config::kCameraRequestedWidth,
                                        app_config::kCameraRequestedHeight));
}

void AutoTracker::processPixelCenter(const cv::Point& center, const cv::Size& frameSize)
{
    processTarget(center, cv::Rect(), frameSize);
}

void AutoTracker::processTarget(const cv::Point& center, const cv::Rect& targetBox, const cv::Size& frameSize)
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
    static constexpr double CENTER_LPF_ALPHA = 0.25;

    using clock = clock_type;

    const double frame_width = static_cast<double>(frameSize.width);
    const double frame_height = static_cast<double>(frameSize.height);
    const double cx = frame_width / 2.0;
    const double cy = frame_height / 2.0;
    const DeadzoneConfig deadzone = deadzoneForTarget(targetBox);
    g_tracker_state.active_deadzone_x = deadzone.innerX;
    g_tracker_state.active_deadzone_y = deadzone.innerY;
    g_tracker_state.active_deadzone_outer_x = deadzone.outerX;
    g_tracker_state.active_deadzone_outer_y = deadzone.outerY;

    if (!g_tracker_state.have_filtered_center) {
        g_tracker_state.center_x_f = static_cast<double>(center.x);
        g_tracker_state.center_y_f = static_cast<double>(center.y);
        g_tracker_state.have_filtered_center = true;
    } else {
        g_tracker_state.center_x_f =
            (1.0 - CENTER_LPF_ALPHA) * g_tracker_state.center_x_f +
            CENTER_LPF_ALPHA * static_cast<double>(center.x);
        g_tracker_state.center_y_f =
            (1.0 - CENTER_LPF_ALPHA) * g_tracker_state.center_y_f +
            CENTER_LPF_ALPHA * static_cast<double>(center.y);
    }

    const double dx = g_tracker_state.center_x_f - cx;
    const double dy = g_tracker_state.center_y_f - cy;

    if (g_enableDeadzone) {
        const double abs_dx = std::abs(dx);
        const double abs_dy = std::abs(dy);
        const bool insideInner =
            abs_dx <= static_cast<double>(deadzone.innerX) &&
            abs_dy <= static_cast<double>(deadzone.innerY);
        const bool insideOuter =
            abs_dx <= static_cast<double>(deadzone.outerX) &&
            abs_dy <= static_cast<double>(deadzone.outerY);

        if (insideInner) {
            g_tracker_state.deadzone_hold_active = true;
            g_tracker_state.az_rel_f = 0.0;
            g_tracker_state.el_rel_f = 0.0;
            return;
        }

        if (g_tracker_state.deadzone_hold_active && insideOuter) {
            g_tracker_state.az_rel_f = 0.0;
            g_tracker_state.el_rel_f = 0.0;
            return;
        }

        g_tracker_state.deadzone_hold_active = false;
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

    std::cout << "[AUTO TRACK] pixel=(" << center.x << "," << center.y << ")"
              << " filtered=(" << std::lround(g_tracker_state.center_x_f)
              << "," << std::lround(g_tracker_state.center_y_f) << ") -> "
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
    std::lock_guard<std::mutex> lock(g_tracker_mutex);
    return {
        g_tracker_state.active_deadzone_x > 0
            ? g_tracker_state.active_deadzone_x
            : g_DEADZONE_X_PX,
        g_tracker_state.active_deadzone_y > 0
            ? g_tracker_state.active_deadzone_y
            : g_DEADZONE_Y_PX,
        g_tracker_state.active_deadzone_outer_x > 0
            ? g_tracker_state.active_deadzone_outer_x
            : outerDeadzoneFromInner(g_DEADZONE_X_PX),
        g_tracker_state.active_deadzone_outer_y > 0
            ? g_tracker_state.active_deadzone_outer_y
            : outerDeadzoneFromInner(g_DEADZONE_Y_PX),
        g_enableDeadzone,
        g_tracker_state.deadzone_hold_active
    };
}
