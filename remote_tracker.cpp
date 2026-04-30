#include "remote_tracker.h"
#include <iostream>
#include "tracking_state.h"
#include "tower_state.h"
#include "can_work.h"

#include <cmath>
#include <chrono>
#include <algorithm>

extern can_work* g_can;

// Пределы башни
static constexpr double LIMIT_AZ = 170.0;
static constexpr double LIMIT_EL = 80.0;
static constexpr double STEP     = 0.01;

// Ограничение частоты отправки: 5 ms = до 200 Гц
static constexpr int MIN_PERIOD_MS = 5;

static double roundStep(double v) { return std::round(v / STEP) * STEP; }
static double clamp(double v, double lim) { return std::max(-lim, std::min(lim, v)); }

static void sendTurretCmd(double az_deg, double el_deg)
{
    az_deg = roundStep(clamp(az_deg, LIMIT_AZ));
    el_deg = roundStep(clamp(el_deg, LIMIT_EL));

    int Data1 = static_cast<int>(std::round(az_deg * 100.0));
    int Data2 = static_cast<int>(std::round(el_deg * 100.0));

    can_frame fr{};
    fr.can_id  = 0x05;
    fr.can_dlc = 5;

    fr.data[4] = static_cast<uint8_t>(Data2 & 0xFF);
    fr.data[3] = static_cast<uint8_t>((Data2 >> 8) & 0xFF);
    fr.data[2] = static_cast<uint8_t>(Data1 & 0xFF);
    fr.data[1] = static_cast<uint8_t>((Data1 >> 8) & 0xFF);
    fr.data[0] = 192;

    g_can->writeCan(fr);
}

RemoteTracker::RemoteTracker(QObject* parent) : QObject(parent) {}

void RemoteTracker::setEnabled(bool en)
{
    m_enabled = en;
}

void RemoteTracker::onPixels(int x, int y, bool valid)
{
    if (!m_enabled) return;
    if (!g_can) return;

    // UART не должен подменять собой состояние main-cam.
    // Он только спрашивает: не истёк ли timeout с последней детекции камеры?
    refreshMainCamState();

    // Пока основная камера ещё активна — UART-команды игнорируем.
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

    // x и y — углы * 100
    const double az_err = static_cast<double>(x) / 100.0;
    const double el_err = static_cast<double>(y) / 100.0;

    // Прибавка к текущему положению башни
    const double az_cmd = az_err;
    const double el_cmd = el_err;

    sendTurretCmd(az_cmd, el_cmd);

    std::cout << "[UART TRACK] angles_x100=(" << x << "," << y << ") -> "
              << "angles_deg=(" << az_err << "," << el_err << ") -> "
              << "AZ=" << az_cmd << "  EL=" << el_cmd
              << std::endl;
}
