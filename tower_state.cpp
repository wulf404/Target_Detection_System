#include "tower_state.h"

#include <atomic>

namespace {
std::atomic<double> g_turret_az_deg{0.0};
std::atomic<double> g_turret_el_deg{0.0};
} // namespace

TurretState getTurretState()
{
    return {
        g_turret_az_deg.load(std::memory_order_relaxed),
        g_turret_el_deg.load(std::memory_order_relaxed)
    };
}

void setTurretState(double az_deg, double el_deg)
{
    g_turret_az_deg.store(az_deg, std::memory_order_relaxed);
    g_turret_el_deg.store(el_deg, std::memory_order_relaxed);
}
