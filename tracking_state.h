#pragma once
#include <atomic>
#include <cstdint>

// True, если основная камера на башне (YOLO) "считается" видит цель
extern std::atomic<bool> g_main_cam_has_target;

// Дистанция до цели (мм), если есть дальномер. -1 если неизвестно
extern std::atomic<int> g_target_distance_mm;
extern std::atomic<uint64_t> g_target_distance_last_seen_ms;

// Последний момент (ms), когда main-cam реально видела цель
extern std::atomic<uint64_t> g_main_cam_last_seen_ms;

// Через сколько ms без детекции считаем, что main-cam потеряла цель
constexpr uint64_t MAIN_CAM_LOST_TIMEOUT_MS = 400;
constexpr uint64_t RANGEFINDER_LOST_TIMEOUT_MS = 1000;

// Вызывать при детекции с main-cam.
// seen=true  -> в этом событии цель видна, обновляем last_seen.
// seen=false -> просто пересчитываем состояние по timeout, не трогая last_seen.
void updateMainCamSeen(bool seen);

// Просто пересчитать флаг активности main-cam по последнему времени last_seen.
// Удобно вызывать из UART-ветки или из реального таймера.
void refreshMainCamState();
void updateTargetDistance(int distance_mm);
bool targetDistanceFresh();
