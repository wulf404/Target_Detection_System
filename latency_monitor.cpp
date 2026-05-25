#include "latency_monitor.h"
#include "app_config.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace latency_monitor {
namespace {

using Clock = std::chrono::steady_clock;

struct Sample
{
    std::uint64_t frameNumber = 0;
    double captureMs = 0.0;
    double preprocessMs = 0.0;
    double inferenceMs = 0.0;
    double postprocessMs = 0.0;
    double controlMs = 0.0;
    double sendMs = 0.0;
    Clock::time_point controlStartedAt{};
    Clock::time_point sendQueuedAt{};
    bool controlReady = false;
    bool sendQueued = false;
    bool sendReady = false;
    bool commandSent = false;
};

std::mutex g_mutex;
std::unordered_map<Token, Sample> g_samples;
Token g_nextToken = 1;
std::uint64_t g_cameraFrames = 0;
thread_local Token g_currentCameraToken = 0;

void printAndEraseIfComplete(Token token)
{
    const auto it = g_samples.find(token);
    if (it == g_samples.end()) {
        return;
    }

    const Sample& sample = it->second;
    if (!sample.controlReady || !sample.sendReady) {
        return;
    }

    const double totalMs =
        sample.captureMs +
        sample.preprocessMs +
        sample.inferenceMs +
        sample.postprocessMs +
        sample.controlMs +
        sample.sendMs;

    std::ostringstream line;
    line << std::fixed << std::setprecision(3)
         << "[LATENCY CAM FRAME #" << sample.frameNumber << "] "
         << "T=" << totalMs << "ms"
         << " | T_cap=" << sample.captureMs << "ms"
         << " | T_prep=" << sample.preprocessMs << "ms"
         << " | T_inf=" << sample.inferenceMs << "ms"
         << " | T_post=" << sample.postprocessMs << "ms"
         << " | T_ctrll=" << sample.controlMs << "ms"
         << " | T_send=" << sample.sendMs << "ms"
         << " | send=" << (sample.commandSent ? "can" : "none");
    std::cerr << line.str() << std::endl;

    g_samples.erase(it);
}

} // namespace

Token beginCameraFrame(double captureMs,
                       double preprocessMs,
                       double inferenceMs,
                       double postprocessMs)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_cameraFrames;
    constexpr std::uint64_t kPrintEveryN =
        app_config::kLatencyLogEveryNFrames > 0 ? app_config::kLatencyLogEveryNFrames : 1;
    if ((g_cameraFrames % kPrintEveryN) != 0) {
        g_currentCameraToken = 0;
        return 0;
    }

    const Token token = g_nextToken++;

    Sample sample;
    sample.frameNumber = g_cameraFrames;
    sample.captureMs = captureMs;
    sample.preprocessMs = preprocessMs;
    sample.inferenceMs = inferenceMs;
    sample.postprocessMs = postprocessMs;
    sample.controlStartedAt = Clock::now();
    g_samples[token] = sample;
    g_currentCameraToken = token;
    return token;
}

Token currentCameraToken()
{
    return g_currentCameraToken;
}

void markControlCompleteAndSendQueued(Token token)
{
    if (token == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_samples.find(token);
    if (it == g_samples.end()) {
        return;
    }

    const Clock::time_point now = Clock::now();
    it->second.controlMs = std::chrono::duration<double, std::milli>(
        now - it->second.controlStartedAt).count();
    it->second.controlReady = true;
    it->second.sendQueuedAt = now;
    it->second.sendQueued = true;
}

void finishSend(Token token)
{
    if (token == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_samples.find(token);
    if (it == g_samples.end() || !it->second.sendQueued) {
        return;
    }

    it->second.sendMs = std::chrono::duration<double, std::milli>(
        Clock::now() - it->second.sendQueuedAt).count();
    it->second.commandSent = true;
    it->second.sendReady = true;
    printAndEraseIfComplete(token);
}

void finishFrameWithoutSend(Token token)
{
    if (token == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_samples.find(token);
    if (it == g_samples.end() || it->second.sendQueued) {
        return;
    }

    it->second.controlMs = std::chrono::duration<double, std::milli>(
        Clock::now() - it->second.controlStartedAt).count();
    it->second.controlReady = true;
    it->second.sendReady = true;
    printAndEraseIfComplete(token);
}

void cancel(Token token)
{
    if (token == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    g_samples.erase(token);
}

void clearCurrentCameraToken()
{
    g_currentCameraToken = 0;
}

} // namespace latency_monitor
