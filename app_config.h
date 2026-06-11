#pragma once

#include <cstdint>

namespace app_config {

// Main input camera resolution. Change these values when the main camera mode changes.
constexpr int kCameraRequestedWidth = 3840;
constexpr int kCameraRequestedHeight = 2160;
// Keep empty for automatic /dev/video0..9 probing.
constexpr const char* kCameraDevicePathOverride = "";
constexpr const char* kCameraPreferredNameContains = "HDMI USB Camera";
constexpr const char* kCameraPreferredVid = "32e4";
constexpr const char* kCameraPreferredPid = "3415";

constexpr const char* kYoloOnnxPath = "/home/nick/qt/yolo_quadro_weights/quadron_1280.onnx";
constexpr const char* kYoloEnginePath = "/home/nick/qt/yolo_quadro_weights/quadron_1280_fp16.engine";
constexpr const char* kYoloWeightsPath = kYoloEnginePath;
constexpr const char* kYoloClassesPath = "/home/nick/qt/yolo_quadro_weights/quadro_3000.names";

// Latency console log period: 1 = every inference frame, 5 = every fifth frame.
constexpr std::uint64_t kLatencyLogEveryNFrames = 1;

// Dynamic YOLO pipeline over a 4K input frame:
// SEARCH = full frame, TRACK = ROI around the last target, LOST = expanding ROI.
constexpr bool kYoloDynamicRoiEnabled = true;
constexpr bool kYoloDynamicRoiDrawOverlay = true;
constexpr double kYoloRoiBoxScale = 4.0;
constexpr double kYoloRoiLostExpansion = 1.35;
constexpr double kYoloRoiMinWidthRatio = 0.25;
constexpr double kYoloRoiMinHeightRatio = 0.25;
constexpr int kYoloRoiFullScanPeriodFrames = 45;
constexpr int kYoloRoiMaxLostFramesBeforeSearch = 8;

constexpr double kCameraFovHDeg = 25.0;
constexpr double kCameraFovVDeg = 14.5;

// Stereo rangefinder connected to Jetson UART.
constexpr bool kStereoRangefinderEnabled = true;
constexpr const char* kStereoRangefinderPort = "/dev/ttyTHS1";
constexpr int kStereoRangefinderBaudRate = 9600;
constexpr int kStereoRangefinderPollPeriodMs = 100;
constexpr int kStereoRangefinderReconnectDelayMs = 1000;
constexpr int kStereoRangefinderBoxRefreshMs = 500;
constexpr int kStereoRangefinderBoxRefreshMinMovePx = 20;
constexpr double kStereoRangefinderBoxRefreshMoveRatio = 0.12;
constexpr double kStereoRangefinderDistanceToMm = 1000.0;
// BBox coordinates from YOLO are scaled from source frame to the stereo frame.
// Set source to the main camera coordinate system and frame to the resolution
// expected by the stereo rangefinder. 0 means "use the actual camera frame".
constexpr int kStereoRangefinderSourceFrameWidth = kCameraRequestedWidth;
constexpr int kStereoRangefinderSourceFrameHeight = kCameraRequestedHeight;
constexpr int kStereoRangefinderFrameWidth = 1920;
constexpr int kStereoRangefinderFrameHeight = 1080;
constexpr bool kStereoRangefinderUseRightStream = false;
constexpr bool kStereoRangefinderSendStopProgramOnClose = false;

// External coordinate protocol uses centidegrees.
// yaw=-0.01 and pitch=-0.01 are sent as -1, -1: link is alive, but there is no target.
constexpr std::int16_t kExternalNoTargetAzCentideg = -1;
constexpr std::int16_t kExternalNoTargetElCentideg = -1;

constexpr bool kRangefinderSendDistanceToCan = false;

} // namespace app_config
