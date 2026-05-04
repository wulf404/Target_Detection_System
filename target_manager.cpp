#include "target_manager.h"

#include "auto_tracker.h"
#include "can_work.h"
#include "tracking_state.h"
#include "turret_command.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <mutex>

extern std::atomic<can_work*> g_can;

namespace {

constexpr uint64_t EXTERNAL_TARGET_TIMEOUT_MS = 500;
constexpr uint64_t EXTERNAL_SEND_PERIOD_MS = 20;
constexpr int EXTERNAL_AZ_LIMIT_CENTIDEG = 17000;
constexpr int EXTERNAL_EL_LIMIT_CENTIDEG = 8000;

struct ExternalTarget
{
    int az_centideg = 0;
    int el_centideg = 0;
    uint64_t last_seen_ms = 0;
    uint64_t last_packet_ms = 0;
    bool no_target = false;
};

std::mutex g_mutex;
TargetManager::Source g_active_source = TargetManager::Source::None;
ExternalTarget g_external;

uint64_t nowMs()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

const char* sourceName(TargetManager::Source source)
{
    switch (source) {
    case TargetManager::Source::Camera: return "camera";
    case TargetManager::Source::External: return "external";
    case TargetManager::Source::None:
    default:
        return "none";
    }
}

bool cameraFresh(uint64_t now)
{
    const uint64_t last = g_main_cam_last_seen_ms.load(std::memory_order_relaxed);
    return last != 0 && (now - last) <= MAIN_CAM_LOST_TIMEOUT_MS;
}

bool externalFresh(uint64_t now)
{
    return g_external.last_seen_ms != 0 &&
           (now - g_external.last_seen_ms) <= EXTERNAL_TARGET_TIMEOUT_MS;
}

bool externalLinkFresh(uint64_t now)
{
    return g_external.last_packet_ms != 0 &&
           (now - g_external.last_packet_ms) <= EXTERNAL_TARGET_TIMEOUT_MS;
}

void setActiveSource(TargetManager::Source next, const char* reason)
{
    if (g_active_source == next) {
        return;
    }

    const TargetManager::Source prev = g_active_source;
    g_active_source = next;
    if (prev == TargetManager::Source::Camera && next != TargetManager::Source::Camera) {
        AutoTracker::reset();
    }

    std::cout << "[TARGET] source=" << sourceName(next)
              << " reason=" << reason
              << std::endl;
}

bool sendExternalIfAllowed(uint64_t now)
{
    static uint64_t last_send_ms = 0;

    if (!externalFresh(now)) {
        return false;
    }
    can_work* can = g_can.load(std::memory_order_acquire);
    if (!can) {
        return false;
    }
    if (last_send_ms != 0 && (now - last_send_ms) < EXTERNAL_SEND_PERIOD_MS) {
        return false;
    }

    last_send_ms = now;
    const double az = static_cast<double>(g_external.az_centideg) / 100.0;
    const double el = static_cast<double>(g_external.el_centideg) / 100.0;

    turret_control::sendPosition(can, {az, el});

    std::cout << "[TARGET] external angles_x100=("
              << g_external.az_centideg << ","
              << g_external.el_centideg << ") -> AZ="
              << az << " EL=" << el
              << std::endl;

    return true;
}

void refreshSourceAfterNonCameraEvent(uint64_t now, const char* reason)
{
    if (cameraFresh(now)) {
        setActiveSource(TargetManager::Source::Camera, reason);
        return;
    }

    if (externalFresh(now)) {
        setActiveSource(TargetManager::Source::External, reason);
        sendExternalIfAllowed(now);
        return;
    }

    setActiveSource(TargetManager::Source::None, reason);
}

} // namespace

void TargetManager::submitCameraTarget(const cv::Point& center, const cv::Size& frameSize)
{
    updateMainCamSeen(true);

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        setActiveSource(Source::Camera, "camera target");
    }

    AutoTracker::processPixelCenter(center, frameSize);
}

void TargetManager::submitCameraMiss()
{
    updateMainCamSeen(false);

    std::lock_guard<std::mutex> lock(g_mutex);
    refreshSourceAfterNonCameraEvent(nowMs(), "camera miss");
}

void TargetManager::submitExternalAnglesCentideg(int az_centideg, int el_centideg, bool valid)
{
    std::lock_guard<std::mutex> lock(g_mutex);

    const uint64_t now = nowMs();
    g_external.last_packet_ms = now;
    const bool in_range =
        std::abs(az_centideg) <= EXTERNAL_AZ_LIMIT_CENTIDEG &&
        std::abs(el_centideg) <= EXTERNAL_EL_LIMIT_CENTIDEG;

    if (valid && in_range) {
        g_external.az_centideg = az_centideg;
        g_external.el_centideg = el_centideg;
        g_external.last_seen_ms = now;
        g_external.no_target = false;
    } else {
        g_external.last_seen_ms = 0;
        g_external.no_target = !valid;
        if (valid && !in_range) {
            std::cout << "[TARGET] external target rejected: out of range az_x100="
                      << az_centideg << " el_x100=" << el_centideg << std::endl;
        }
    }

    refreshMainCamState();
    refreshSourceAfterNonCameraEvent(now, valid ? "external target" : "external invalid");
}

void TargetManager::refresh()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    refreshMainCamState();
    refreshSourceAfterNonCameraEvent(nowMs(), "watchdog");
}

TargetManager::Snapshot TargetManager::snapshot()
{
    std::lock_guard<std::mutex> lock(g_mutex);

    const uint64_t now = nowMs();

    Snapshot snap;
    snap.cameraFresh = cameraFresh(now);
    snap.externalFresh = externalFresh(now);
    snap.externalLinkFresh = externalLinkFresh(now);
    snap.externalNoTarget = g_external.no_target && snap.externalLinkFresh && !snap.externalFresh;
    snap.externalLastSeenMs = g_external.last_seen_ms;
    snap.externalLastPacketMs = g_external.last_packet_ms;
    if (snap.cameraFresh) {
        snap.activeSource = Source::Camera;
    } else if (snap.externalFresh) {
        snap.activeSource = Source::External;
    } else {
        snap.activeSource = Source::None;
    }
    snap.externalAzCentideg = g_external.az_centideg;
    snap.externalElCentideg = g_external.el_centideg;
    return snap;
}
