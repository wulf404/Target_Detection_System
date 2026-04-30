#include "tracking_state.h"
#include <chrono>
#include <iostream>

std::atomic<bool>     g_main_cam_has_target{false};
std::atomic<int>      g_target_distance_mm{-1};
std::atomic<uint64_t> g_main_cam_last_seen_ms{0};

static uint64_t nowMs()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static void recalcMainCamState(bool print_debug)
{
    const uint64_t now  = nowMs();
    const uint64_t last = g_main_cam_last_seen_ms.load(std::memory_order_relaxed);
    const bool active   = (last != 0) && ((now - last) <= MAIN_CAM_LOST_TIMEOUT_MS);

    const bool prev = g_main_cam_has_target.exchange(active, std::memory_order_relaxed);

    if (print_debug) {
        std::cout << "[TRACK] now=" << now
                  << " last_seen=" << last
                  << " diff_ms=" << (last ? (now - last) : 0)
                  << " active=" << (active ? 1 : 0)
                  << std::endl;
    }

    if (prev != active) {
        std::cout << "[FLAG] main_cam_has_target=" << (active ? 1 : 0)
                  << " (timeout=" << MAIN_CAM_LOST_TIMEOUT_MS << "ms)"
                  << std::endl;
    }
}

void updateMainCamSeen(bool seen)
{
    if (seen) {
        g_main_cam_last_seen_ms.store(nowMs(), std::memory_order_relaxed);
    }

    recalcMainCamState(false);
}

void refreshMainCamState()
{
    recalcMainCamState(false);
}
