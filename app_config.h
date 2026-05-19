#pragma once

#include <cstdint>

namespace app_config {

// Main input camera width. Change this value to switch 4K/FullHD/etc.
constexpr int kCameraRequestedWidth = 3840;
// Keep empty for automatic /dev/video0..9 probing.
constexpr const char* kCameraDevicePathOverride = "";
constexpr const char* kCameraPreferredNameContains = "HDMI USB Camera";
constexpr const char* kCameraPreferredVid = "32e4";
constexpr const char* kCameraPreferredPid = "3415";

constexpr const char* kYoloOnnxPath = "/home/nick/qt/yolo_quadro_weights/quadron_1280.onnx";
constexpr const char* kYoloEnginePath = "/home/nick/qt/yolo_quadro_weights/quadron_1280_fp16.engine";
constexpr const char* kYoloWeightsPath = kYoloEnginePath;
constexpr const char* kYoloClassesPath = "/home/nick/qt/yolo_quadro_weights/quadro_3000.names";

constexpr double kCameraFovHDeg = 25.0;
constexpr double kCameraFovVDeg = 14.5;

// External coordinate protocol uses centidegrees.
// yaw=-0.01 and pitch=-0.01 are sent as -1, -1: link is alive, but there is no target.
constexpr std::int16_t kExternalNoTargetAzCentideg = -1;
constexpr std::int16_t kExternalNoTargetElCentideg = -1;

constexpr bool kRangefinderSendDistanceToCan = false;

} // namespace app_config
