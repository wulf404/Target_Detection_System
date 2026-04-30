#include "target_manager.h"

#include "auto_tracker.h"
#include "can_work.h"
#include "tracking_state.h"
#include "turret_command.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>

extern can_work* g_can;

namespace {

constexpr uint64_t EXTERNAL_TARGET_TIMEOUT_MS = 500;
constexpr uint64_t EXTERNAL_SEND_PERIOD_MS = 5;

struct ExternalTarget
{
    int az_centideg = 0;
    int el_centideg = 0;
    uint64_t last_seen_ms = 0;
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

void setActiveSource(TargetManager::Source next, const char* reason)
{
    if (g_active_source == next) {
        return;
    }

    g_active_source = next;
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
    if (!g_can) {
        return false;
    }
    if (last_send_ms != 0 && (now - last_send_ms) < EXTERNAL_SEND_PERIOD_MS) {
        return false;
    }

    last_send_ms = now;
    const double az = static_cast<double>(g_external.az_centideg) / 100.0;
    const double el = static_cast<double>(g_external.el_centideg) / 100.0;

    turret_control::sendPosition(g_can, {az, el});

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

    if (valid) {
        g_external.az_centideg = az_centideg;
        g_external.el_centideg = el_centideg;
        g_external.last_seen_ms = now;
    } else {
        g_external.last_seen_ms = 0;
    }

    refreshMainCamState();
    refreshSourceAfterNonCameraEvent(now, valid ? "external target" : "external invalid");
}

TargetManager::Snapshot TargetManager::snapshot()
{
    std::lock_guard<std::mutex> lock(g_mutex);

    const uint64_t now = nowMs();

    Snapshot snap;
    snap.cameraFresh = cameraFresh(now);
    snap.externalFresh = externalFresh(now);
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
