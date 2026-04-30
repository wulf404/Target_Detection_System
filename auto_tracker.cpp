#include "auto_tracker.h"
#include "can_work.h"
#include "tower_state.h"
#include "turret_command.h"

#include <cmath>
#include <chrono>
#include <iostream>

extern can_work* g_can;

static constexpr double CAMERA_FOV_H_DEG = 25.0;
static constexpr double CAMERA_FOV_V_DEG = 14.5;

double g_AZ_K = 1.0;
double g_EL_K = 1.0;

double g_P_AZ = 1.0;
double g_P_EL = 1.0;

int g_DEADZONE_X_PX = 100;
int g_DEADZONE_Y_PX = 100;

bool g_enableDeadzone = true;
bool g_enableP = false;

void AutoTracker::processPixelCenter(const cv::Point& center)
{
    processPixelCenter(center, cv::Size(1920, 1080));
}

void AutoTracker::processPixelCenter(const cv::Point& center, const cv::Size& frameSize)
{
    if (!g_can) return;
    if (frameSize.width <= 0 || frameSize.height <= 0) return;

    static constexpr double REACH_EPS_DEG = 0.10;
    static constexpr int SEND_PERIOD_MS = 20;
    static constexpr int MAX_HOLD_MS = 100;
    static constexpr double MIN_DELTA_DEG = 0.02;
    static constexpr double LPF_ALPHA = 0.35;

    using clock = std::chrono::steady_clock;
    static auto last_send_t = clock::now();
    static auto hold_start_t = clock::now();

    static double last_target_az = 0.0;
    static double last_target_el = 0.0;
    static bool have_last_target = false;

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

    static double az_rel_f = 0.0;
    static double el_rel_f = 0.0;
    az_rel_f = (1.0 - LPF_ALPHA) * az_rel_f + LPF_ALPHA * az_rel;
    el_rel_f = (1.0 - LPF_ALPHA) * el_rel_f + LPF_ALPHA * el_rel;

    if (g_enableP) {
        az_rel_f *= g_P_AZ;
        el_rel_f *= g_P_EL;
    }

    const TurretState turret = getTurretState();
    const auto target = turret_control::normalize({
        turret.az_deg + az_rel_f,
        turret.el_deg + el_rel_f
    });

    const auto now = clock::now();

    if (have_last_target) {
        const double eaz = std::abs(turret.az_deg - last_target_az);
        const double eel = std::abs(turret.el_deg - last_target_el);
        const bool reached = (eaz < REACH_EPS_DEG) && (eel < REACH_EPS_DEG);

        const auto hold_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - hold_start_t).count();
        if (!reached && hold_ms < MAX_HOLD_MS) {
            return;
        }
    }

    const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_send_t).count();
    if (dt_ms < SEND_PERIOD_MS) return;

    if (have_last_target) {
        if (std::abs(target.az - last_target_az) < MIN_DELTA_DEG &&
            std::abs(target.el - last_target_el) < MIN_DELTA_DEG)
        {
            return;
        }
    }

    last_send_t = now;
    hold_start_t = now;
    last_target_az = target.az;
    last_target_el = target.el;
    have_last_target = true;

    turret_control::sendPosition(g_can, target);

    std::cout << "[AUTO TRACK] pixel=(" << center.x << "," << center.y << ") -> "
              << "AZ=" << target.az << "  EL=" << target.el
              << "  (relAZ=" << az_rel_f << " relEL=" << el_rel_f << ")"
              << std::endl;
}
