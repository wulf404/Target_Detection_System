#pragma once

#include <cstdint>

namespace latency_monitor {

using Token = std::uint64_t;

Token beginCameraFrame(double captureMs,
                       double preprocessMs,
                       double inferenceMs,
                       double postprocessMs);
Token currentCameraToken();
void markControlCompleteAndSendQueued(Token token);
void finishSend(Token token);
void finishFrameWithoutSend(Token token);
void cancel(Token token);
void clearCurrentCameraToken();

} // namespace latency_monitor
