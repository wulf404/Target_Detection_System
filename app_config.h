#pragma once

#include <cstdint>

namespace app_config {

// Основное разрешение камеры входного сигнала. Изменяйте эти значения при смене режима основной камеры.
constexpr int kCameraRequestedWidth = 3840;
constexpr int kCameraRequestedHeight = 2160;
// Оставьте пустым для автоматического поиска /dev/video0..9.
constexpr const char* kCameraDevicePathOverride = "";
constexpr const char* kCameraPreferredNameContains = "HDMI USB Camera";
constexpr const char* kCameraPreferredVid = "32e4";
constexpr const char* kCameraPreferredPid = "3415";

constexpr const char* kYoloOnnxPath = "/home/nick/qt/yolo_quadro_weights/quadron_1280.onnx";
constexpr const char* kYoloEnginePath = "/home/nick/qt/yolo_quadro_weights/quadron_1280_fp16.engine";
constexpr const char* kYoloWeightsPath = kYoloEnginePath;
constexpr const char* kYoloClassesPath = "/home/nick/qt/yolo_quadro_weights/quadro_3000.names";

// Период логирования задержки: 1 = каждый кадр инференса, 5 = каждый пятый кадр.
constexpr std::uint64_t kLatencyLogEveryNFrames = 30;

// Динамический YOLO пайплайн для входного 4K кадра:
// SEARCH = полный кадр, TRACK = виртуальный ROI фиксированного размера вокруг цели,
// LOST = расширяющийся ROI от этого квадрата до полного кадра.
constexpr bool kYoloDynamicRoiEnabled = true;
constexpr bool kYoloDynamicRoiDrawOverlay = true;
constexpr double kYoloRoiBoxScale = 4.0; // Используется когда цель потеряна.
constexpr double kYoloRoiLostExpansion = 1.35;
constexpr double kYoloRoiMinWidthRatio = 0.25;
constexpr double kYoloRoiMinHeightRatio = 0.25;
constexpr int kYoloRoiFullScanPeriodFrames = 45;
constexpr int kYoloRoiMaxLostFramesBeforeSearch = 8;

// Оценка кандидатов: высокая уверенность/размер помогают, расстояние от предсказания — штраф.
constexpr double kYoloCandidateConfidenceScore = 1000.0; // Вес уверенности нейросети.
constexpr double kYoloCandidateAreaScore = 0.7;          // Небольшой бонус за размер bbox.
constexpr double kYoloCandidateDistancePenalty = 0.65;   // Штраф за пиксель от предсказания.
constexpr double kYoloCandidateStickyRadiusPx = 260.0;   // Радиус, где старая цель получает бонус.
constexpr double kYoloCandidateStickyBonus = 180.0;      // Бонус внутри "липкого" радиуса.

// Фильтр движения: alpha корректирует позицию, beta корректирует скорость.
constexpr double kYoloMotionAlpha = 0.68;                // Сила коррекции позиции.
constexpr double kYoloMotionBeta = 0.22;                 // Сила коррекции скорости.
constexpr double kYoloMotionMaxPredictMs = 220.0;        // Ограничение горизонта предсказания.
constexpr double kYoloMotionMaxVelocityPxPerSec = 7000.0;// Ограничение скорости.
constexpr double kYoloTrackBoxLpfAlpha = 0.45;           // Сглаживание размера bbox.

// Качество трека/захвата: цель передаётся в наведение только после стабильного захвата.
constexpr int kYoloTrackAcquireFrames = 2;               // Кол-во попаданий для перехода в захват.
constexpr int kYoloTrackMemoryFrames = 15;               // Кол-во пропусков до удаления трека.
constexpr int kYoloTrackSuspiciousResetFrames = 2;       // Повторные скачки начинают новый захват.
constexpr double kYoloTrackAcquireMinQuality = 0.28;     // Минимальное качество для выхода из ACQUIRE.
constexpr double kYoloTrackReleaseQuality = 0.12;        // Качество ниже этого сбрасывает трек.
constexpr double kYoloTrackQualityHitGain = 0.20;        // Рост качества при попадании.
constexpr double kYoloTrackQualityMissDecay = 0.16;      // Падение качества при пропуске.
constexpr double kYoloTrackQualitySuspiciousDecay = 0.24;// Падение качества при подозрительном скачке.

// Predictive coast: коротко ведем подтвержденную цель по модели движения, если YOLO пропустила кадр.
constexpr bool kYoloTrackCoastEnabled = true;            // Включить прогноз на коротких пропусках.
constexpr int kYoloTrackCoastMaxFrames = 6;              // Макс. кадров прогноза без новой детекции.
constexpr double kYoloTrackCoastMaxMs = 220.0;           // Макс. время прогноза от последней детекции.
constexpr double kYoloTrackCoastMinQuality = 0.14;       // Минимальное качество, ниже которого прогноз запрещен.
constexpr double kYoloTrackCoastQualityDecay = 0.05;     // Падение качества на кадр во время прогноза.

// Защита от скачков: отклоняет резкие прыжки bbox, если они не повторяются.
constexpr double kYoloTrackMaxJumpBasePx = 170.0;        // Базовый допустимый скачок центра.
constexpr double kYoloTrackMaxJumpBoxDiagRatio = 1.20;   // Доп. порог от диагонали bbox.
constexpr double kYoloTrackVelocityGateScale = 1.20;     // Доп. порог от скорости.
constexpr double kYoloTrackLowQualityGateRelax = 0.80;   // Более мягкий порог при низком качестве.
constexpr double kYoloTrackMaxBoxSizeChangeRatio = 2.60; // Макс. изменение размера bbox за кадр.

// ROI трекинга: высокое качество держит 1x вход, низкое качество расширяет квадрат.
constexpr double kYoloTrackRoiMinInputScale = 1.0;       // Размер ROI при хорошем качестве.
constexpr double kYoloTrackRoiLowQualityInputScale = 1.35;// Размер ROI при низком качестве.
constexpr double kYoloTrackRoiQualityForMinScale = 0.75; // Качество для минимального ROI.
constexpr double kYoloTrackRoiTargetBoxScale = 2.2;      // ROI не может быть меньше bbox*scale.

constexpr double kCameraFovHDeg = 25.0;
constexpr double kCameraFovVDeg = 14.5;

// Компенсация движения турели в автотрекинге.
// LeadMs — ожидаемая задержка камеры+инференса+команд; знаки задают направление компенсации.
constexpr bool kAutoTrackTurretMotionCompEnabled = true;
constexpr double kAutoTrackTurretMotionCompLeadMs = 60.0;          // Оценка фиксированной задержки.
constexpr double kAutoTrackTurretMotionCompTelemetryAgeScale = 0.5;// Добавляет часть возраста телеметрии.
constexpr double kAutoTrackTurretMotionCompMaxLeadMs = 140.0;      // Ограничение задержки.
constexpr double kAutoTrackTurretMotionCompAzGain = -1.0;          // Коэффициент компенсации по азимуту.
constexpr double kAutoTrackTurretMotionCompElGain = -1.0;          // Коэффициент компенсации по углу места.
constexpr double kAutoTrackTurretMotionCompVelLpfAlpha = 0.35;     // Сглаживание скорости турели.
constexpr double kAutoTrackTurretMotionCompMaxDeg = 1.2;           // Максимальная угловая коррекция.

// Стереодальномер, подключённый к UART Jetson.
constexpr bool kStereoRangefinderEnabled = true;
constexpr const char* kStereoRangefinderPort = "/dev/ttyTHS1";
constexpr int kStereoRangefinderBaudRate = 9600;
constexpr int kStereoRangefinderPollPeriodMs = 100;
constexpr int kStereoRangefinderReconnectDelayMs = 1000;
constexpr int kStereoRangefinderBoxRefreshMs = 200;
constexpr int kStereoRangefinderBoxRefreshMinMovePx = 20;
constexpr double kStereoRangefinderBoxRefreshMoveRatio = 0.12;
constexpr double kStereoRangefinderDistanceToMm = 1000.0;
// Координаты bbox из YOLO масштабируются из исходного кадра в стерео-кадр.
// Установите исходный размер как разрешение основной камеры и кадр как разрешение,
// ожидаемое дальномером. 0 означает "использовать фактический кадр камеры".
constexpr int kStereoRangefinderSourceFrameWidth = kCameraRequestedWidth;
constexpr int kStereoRangefinderSourceFrameHeight = kCameraRequestedHeight;
constexpr int kStereoRangefinderFrameWidth = 1920;
constexpr int kStereoRangefinderFrameHeight = 1080;
constexpr bool kStereoRangefinderUseRightStream = false;
constexpr bool kStereoRangefinderSendStopProgramOnClose = false;

// Пульсовый дальномер через USB-UART. "auto" использует serial_port_resolver.
constexpr bool kPulseRangefinderEnabled = true;
constexpr const char* kPulseRangefinderPort = "auto";

// Внешний координатный протокол использует сантиградусы.
// yaw=-0.01 и pitch=-0.01 отправляются как -1, -1: связь есть, но цели нет.
constexpr std::int16_t kExternalNoTargetAzCentideg = -1;
constexpr std::int16_t kExternalNoTargetElCentideg = -1;

constexpr bool kRangefinderSendDistanceToCan = false;

}// namespace app_config
