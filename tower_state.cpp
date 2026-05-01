#include "tower_state.h"

#include <atomic>
#include <chrono>

namespace {
std::atomic<double> g_turret_az_deg{0.0};
std::atomic<double> g_turret_el_deg{0.0};
std::atomic<uint64_t> g_turret_last_update_ms{0};

uint64_t nowMs()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}
} // namespace

TurretState getTurretState()
{
    const uint64_t now = nowMs();
    const uint64_t last = g_turret_last_update_ms.load(std::memory_order_relaxed);
    return {
        g_turret_az_deg.load(std::memory_order_relaxed),
        g_turret_el_deg.load(std::memory_order_relaxed),
        last,
        last != 0 && ((now - last) <= TURRET_STATE_TIMEOUT_MS)
    };
}

void setTurretState(double az_deg, double el_deg)
{
    g_turret_az_deg.store(az_deg, std::memory_order_relaxed);
    g_turret_el_deg.store(el_deg, std::memory_order_relaxed);
    g_turret_last_update_ms.store(nowMs(), std::memory_order_relaxed);
}

bool turretStateFresh(uint64_t timeout_ms)
{
    const uint64_t last = g_turret_last_update_ms.load(std::memory_order_relaxed);
    return last != 0 && ((nowMs() - last) <= timeout_ms);
}
