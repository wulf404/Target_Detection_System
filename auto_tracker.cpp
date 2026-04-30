#include "auto_tracker.h"
#include "can_work.h"
#include "tower_state.h"
#include "tracking_state.h"

#include <cmath>
#include <iostream>
#include <chrono>
#include <algorithm>

extern can_work* g_can;

// ПАРАМЕТРЫ КАМЕРЫ
static constexpr double CAMERA_WIDTH        = 1920.0;
static constexpr double CAMERA_HEIGHT       = 1080.0;
static constexpr double CAMERA_FOV_H_DEG    = 25.0;
static constexpr double CAMERA_FOV_V_DEG    = 14.5;

// ОГРАНИЧЕНИЯ БАШНИ
static constexpr double TURRET_LIMIT_AZ_DEG = 170.0;
static constexpr double TURRET_LIMIT_EL_DEG = 80.0;
static constexpr double TURRET_STEP_DEG     = 0.01;

// ОТЛАДОЧНЫЕ КОЭФФИЦИЕНТЫ
double g_AZ_K = 1.0;
double g_EL_K = 1.0;

double g_P_AZ = 1.0;
double g_P_EL = 1.0;

// Мёртвая зона в пикселях
int g_DEADZONE_X_PX = 100;
int g_DEADZONE_Y_PX = 100;

bool g_enableDeadzone = true;
bool g_enableP        = false;

static double roundToStep(double v_deg)
{
    return std::round(v_deg / TURRET_STEP_DEG) * TURRET_STEP_DEG;
}

static double clampDeg(double v, double lim)
{
    return std::max(-lim, std::min(lim, v));
}

void AutoTracker::processPixelCenter(const cv::Point& center)
{
    if (!g_can) return;

    // Раз main-cam дала центр цели, значит цель видна прямо сейчас.
    updateMainCamSeen(true);

    static constexpr double REACH_EPS_DEG   = 0.10;
    static constexpr int    SEND_PERIOD_MS  = 20;
    static constexpr int    MAX_HOLD_MS     = 100;
    static constexpr double MIN_DELTA_DEG   = 0.02;
    static constexpr double LPF_ALPHA       = 0.35;

    using clock = std::chrono::steady_clock;
    static auto last_send_t   = clock::now();
    static auto hold_start_t  = clock::now();

    static double last_target_az = 0.0;
    static double last_target_el = 0.0;
    static bool   have_last_target = false;

    const double cx = CAMERA_WIDTH  / 2.0;
    const double cy = CAMERA_HEIGHT / 2.0;

    const double dx = static_cast<double>(center.x) - cx;
    const double dy = static_cast<double>(center.y) - cy;

    if (g_enableDeadzone &&
        std::abs(dx) < g_DEADZONE_X_PX &&
        std::abs(dy) < g_DEADZONE_Y_PX)
    {
        return;
    }

    const double nx = dx / CAMERA_WIDTH;
    const double ny = dy / CAMERA_HEIGHT;

    double az_err_deg = nx * CAMERA_FOV_H_DEG;
    double el_err_deg = ny * CAMERA_FOV_V_DEG;

    double az_rel =  az_err_deg * g_AZ_K;
    double el_rel = -el_err_deg * g_EL_K;

    static double az_rel_f = 0.0;
    static double el_rel_f = 0.0;
    az_rel_f = (1.0 - LPF_ALPHA) * az_rel_f + LPF_ALPHA * az_rel;
    el_rel_f = (1.0 - LPF_ALPHA) * el_rel_f + LPF_ALPHA * el_rel;

    if (g_enableP) {
        az_rel_f *= g_P_AZ;
        el_rel_f *= g_P_EL;
    }

    double target_az = g_turret_az_deg + az_rel_f;
    double target_el = g_turret_el_deg + el_rel_f;

    target_az = roundToStep(clampDeg(target_az, TURRET_LIMIT_AZ_DEG));
    target_el = roundToStep(clampDeg(target_el, TURRET_LIMIT_EL_DEG));

    const auto now = clock::now();

    if (have_last_target) {
        const double eaz = std::abs(g_turret_az_deg - last_target_az);
        const double eel = std::abs(g_turret_el_deg - last_target_el);
        const bool reached = (eaz < REACH_EPS_DEG) && (eel < REACH_EPS_DEG);

        const auto hold_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - hold_start_t).count();
        if (!reached && hold_ms < MAX_HOLD_MS) {
            return;
        }
    }

    const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_send_t).count();
    if (dt_ms < SEND_PERIOD_MS) return;

    if (have_last_target) {
        if (std::abs(target_az - last_target_az) < MIN_DELTA_DEG &&
            std::abs(target_el - last_target_el) < MIN_DELTA_DEG)
        {
            return;
        }
    }

    last_send_t = now;
    hold_start_t = now;
    last_target_az = target_az;
    last_target_el = target_el;
    have_last_target = true;

    const int Data1 = static_cast<int>(std::round(target_az * 100.0));
    const int Data2 = static_cast<int>(std::round(target_el * 100.0));

    struct can_frame frame{};
    frame.can_id  = 0x05;
    frame.can_dlc = 5;

    frame.data[4] = static_cast<uint8_t>(Data2 & 0xFF);
    frame.data[3] = static_cast<uint8_t>((Data2 >> 8) & 0xFF);
    frame.data[2] = static_cast<uint8_t>(Data1 & 0xFF);
    frame.data[1] = static_cast<uint8_t>((Data1 >> 8) & 0xFF);
    frame.data[0] = 192;

    g_can->writeCan(frame);

    std::cout << "[AUTO TRACK] pixel=(" << center.x << "," << center.y << ") -> "
              << "AZ=" << target_az << "  EL=" << target_el
              << "  (relAZ=" << az_rel_f << " relEL=" << el_rel_f << ")"
              << std::endl;
}
