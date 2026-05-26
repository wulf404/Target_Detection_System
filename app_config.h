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

constexpr const char* kYoloWeightsPath = "/home/nick/qt/yolo_quadro_weights/quadron_1280.onnx";
constexpr const char* kYoloClassesPath = "/home/nick/qt/yolo_quadro_weights/quadro_3000.names";

constexpr bool kUseDeepStream = true;
// DeepStream owns a target-built engine file. Keep it local to the Jetson:
// when it is absent, nvinfer builds it from this ONNX on that device.
constexpr const char* kDeepStreamOnnxPath = "/home/nick/qt/yolo_quadro_weights/quadron_1280.onnx";
constexpr const char* kDeepStreamEnginePath = "/home/nick/qt/yolo_quadro_weights/quadron_1280_fp16_ds_jp621_orin_nx.engine";
constexpr const char* kDeepStreamInferConfigPath = "/tmp/target_detection_system_nvinfer.txt";
constexpr int kDeepStreamNetworkInputWidth = 1280;
constexpr int kDeepStreamNetworkInputHeight = 1280;
constexpr int kDeepStreamDisplayWidth = 1280;
constexpr int kDeepStreamDisplayHeight = 720;
// HDMI USB Camera advertises H264 3840x2160@60; it is lighter and more stable
// for the USB/decoder path than MJPEG at this data rate.
constexpr bool kDeepStreamPreferH264Capture = true;
// Keep direct-resize semantics, but do the 4K -> network-size conversion in
// nvstreammux instead of feeding a full 4K surface into nvinfer.
constexpr bool kDeepStreamScaleInMuxToNetworkInput = true;
// Invalid nvinfer configuration and CUDA failures are not repaired by
// reconnecting the camera or switching its capture format.
constexpr bool kDeepStreamStopOnNvinferError = true;
constexpr int kDeepStreamNormalStartupProbeTimeoutMs = 2400;
constexpr int kDeepStreamEngineBuildStartupTimeoutMs = 600000;
constexpr float kDeepStreamConfThreshold = 0.3f;
constexpr float kDeepStreamNmsThreshold = 0.5f;

constexpr double kCameraFovHDeg = 25.0;
constexpr double kCameraFovVDeg = 14.5;

// External coordinate protocol uses centidegrees.
// yaw=-0.01 and pitch=-0.01 are sent as -1, -1: link is alive, but there is no target.
constexpr std::int16_t kExternalNoTargetAzCentideg = -1;
constexpr std::int16_t kExternalNoTargetElCentideg = -1;

constexpr bool kRangefinderSendDistanceToCan = false;

} // namespace app_config
