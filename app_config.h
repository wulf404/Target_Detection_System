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
// SEARCH = full frame, TRACK = virtual padded square input-sized ROI around target,
// LOST = expanding ROI from that square up to full frame.
constexpr bool kYoloDynamicRoiEnabled = true;
constexpr bool kYoloDynamicRoiDrawOverlay = true;
constexpr double kYoloRoiBoxScale = 4.0; // Used when target is lost.
constexpr double kYoloRoiLostExpansion = 1.35;
constexpr double kYoloRoiMinWidthRatio = 0.25;
constexpr double kYoloRoiMinHeightRatio = 0.25;
constexpr int kYoloRoiFullScanPeriodFrames = 45;
constexpr int kYoloRoiMaxLostFramesBeforeSearch = 8;

// Candidate scoring: higher confidence/area helps, distance from predicted target hurts.
constexpr double kYoloCandidateConfidenceScore = 1000.0; // Weight of NN confidence.
constexpr double kYoloCandidateAreaScore = 0.7;          // Small bonus for larger bbox.
constexpr double kYoloCandidateDistancePenalty = 0.65;   // Penalty per px from prediction.
constexpr double kYoloCandidateStickyRadiusPx = 260.0;   // Radius where old target gets bonus.
constexpr double kYoloCandidateStickyBonus = 180.0;      // Bonus inside sticky radius.

// Motion filter: alpha corrects position, beta corrects velocity.
constexpr double kYoloMotionAlpha = 0.68;                // Position correction strength.
constexpr double kYoloMotionBeta = 0.22;                 // Velocity correction strength.
constexpr double kYoloMotionMaxPredictMs = 220.0;        // Prediction horizon clamp.
constexpr double kYoloMotionMaxVelocityPxPerSec = 7000.0;// Velocity clamp.
constexpr double kYoloTrackBoxLpfAlpha = 0.45;           // BBox size smoothing.

// Track quality/acquire: target is submitted to guidance only after stable acquire.
constexpr int kYoloTrackAcquireFrames = 2;               // Hits before guidance accepts target.
constexpr int kYoloTrackMemoryFrames = 15;               // Misses before track is forgotten.
constexpr int kYoloTrackSuspiciousResetFrames = 2;       // Repeated jumps start new acquire.
constexpr double kYoloTrackAcquireMinQuality = 0.28;     // Min quality to leave ACQUIRE.
constexpr double kYoloTrackReleaseQuality = 0.12;        // Quality below this drops track.
constexpr double kYoloTrackQualityHitGain = 0.20;        // Quality gain on accepted hit.
constexpr double kYoloTrackQualityMissDecay = 0.16;      // Quality loss on miss.
constexpr double kYoloTrackQualitySuspiciousDecay = 0.24;// Quality loss on rejected jump.

// Jump guard: rejects sudden bbox jumps unless they repeat and start a new acquire.
constexpr double kYoloTrackMaxJumpBasePx = 170.0;        // Base allowed center jump.
constexpr double kYoloTrackMaxJumpBoxDiagRatio = 1.20;   // Extra gate from bbox diagonal.
constexpr double kYoloTrackVelocityGateScale = 1.20;     // Extra gate from predicted speed.
constexpr double kYoloTrackLowQualityGateRelax = 0.80;   // Wider gate when quality is low.
constexpr double kYoloTrackMaxBoxSizeChangeRatio = 2.60; // Max one-frame bbox size ratio.

// TRACK ROI: high quality keeps 1x input side, lower quality widens the square.
constexpr double kYoloTrackRoiMinInputScale = 1.0;       // ROI side at good quality.
constexpr double kYoloTrackRoiLowQualityInputScale = 1.35;// ROI side at low quality.
constexpr double kYoloTrackRoiQualityForMinScale = 0.75; // Quality that reaches min ROI.
constexpr double kYoloTrackRoiTargetBoxScale = 2.2;      // ROI cannot be smaller than bbox*scale.

constexpr double kCameraFovHDeg = 25.0;
constexpr double kCameraFovVDeg = 14.5;

// Turret motion compensation in auto-guidance.
// LeadMs is expected camera+inference+command delay; gains set sign and strength.
constexpr bool kAutoTrackTurretMotionCompEnabled = true;
constexpr double kAutoTrackTurretMotionCompLeadMs = 60.0;          // Fixed delay estimate.
constexpr double kAutoTrackTurretMotionCompTelemetryAgeScale = 0.5;// Adds part of telemetry age.
constexpr double kAutoTrackTurretMotionCompMaxLeadMs = 140.0;      // Lead clamp.
constexpr double kAutoTrackTurretMotionCompAzGain = -1.0;          // AZ compensation sign/gain.
constexpr double kAutoTrackTurretMotionCompElGain = -1.0;          // EL compensation sign/gain.
constexpr double kAutoTrackTurretMotionCompVelLpfAlpha = 0.35;     // Turret velocity smoothing.
constexpr double kAutoTrackTurretMotionCompMaxDeg = 1.2;           // Max angular correction.

// Stereo rangefinder connected to Jetson UART.
constexpr bool kStereoRangefinderEnabled = true;
constexpr const char* kStereoRangefinderPort = "/dev/ttyTHS1";
constexpr int kStereoRangefinderBaudRate = 9600;
constexpr int kStereoRangefinderPollPeriodMs = 100;
constexpr int kStereoRangefinderReconnectDelayMs = 1000;
constexpr int kStereoRangefinderBoxRefreshMs = 200;
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

// Pulse rangefinder connected through USB-UART. "auto" uses serial_port_resolver.
constexpr bool kPulseRangefinderEnabled = true;
constexpr const char* kPulseRangefinderPort = "auto";

// External coordinate protocol uses centidegrees.
// yaw=-0.01 and pitch=-0.01 are sent as -1, -1: link is alive, but there is no target.
constexpr std::int16_t kExternalNoTargetAzCentideg = -1;
constexpr std::int16_t kExternalNoTargetElCentideg = -1;

constexpr bool kRangefinderSendDistanceToCan = false;

} // namespace app_config
