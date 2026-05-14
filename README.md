# Target Detection System

Qt/C++ система обнаружения и сопровождения цели для башенной платформы.
Основной источник цели - USB-камера с YOLO. Если камера теряет цель, система
переходит на внешний UART-источник координат. Управление башней отправляется в
SocketCAN кадром `0x05`.

## Потоки данных

```text
USB camera -> YOLO -> TargetManager -> AutoTracker -> CAN 0x05
External UART -> RemoteTracker -> TargetManager -> CAN 0x05
Rangefinder UART -> UI/status distance only
CAN telemetry -> can_work -> TurretState / UI/status
```

Дальномер сейчас намеренно не отправляет дистанцию в CAN. Данные продолжают
читаться, показываться в консоли и использоваться в панели состояния. Чтобы
вернуть отправку `0x09`, включите `app_config::kRangefinderSendDistanceToCan`.

## Что Менять Руками

Основные ручные параметры собраны в `app_config.h`:

| Параметр | Назначение |
| --- | --- |
| `kCameraRequestedWidth` | Желаемая входная ширина камеры. Сейчас `3840`, то есть 4K. Менять нужно здесь, а не в `camera.h`. |
| `kYoloWeightsPath` | ONNX-веса YOLO. |
| `kYoloClassesPath` | Файл классов YOLO. |
| `kCameraFovHDeg`, `kCameraFovVDeg` | Горизонтальный и вертикальный угол камеры. |

В обычной работе этого достаточно. Служебные параметры тоже лежат в `app_config.h`,
но менять их нужно только при изменении протокола или диагностике:

| Параметр | Назначение |
| --- | --- |
| `kCameraDevicePathOverride` | Принудительный путь камеры. По умолчанию пустой, поэтому камера ищется автоматически через `/sys/class/video4linux`. |
| `kCameraPreferredNameContains`, `kCameraPreferredVid`, `kCameraPreferredPid` | Предпочитаемая USB-камера. Сейчас `HDMI USB Camera`, `32e4:3415`. |
| `kUseDeepStream` | Включает DeepStream/TensorRT pipeline вместо OpenCV DNN ONNX. |
| `kDeepStreamEnginePath` | TensorRT engine для `nvinfer`. Сейчас `/home/nick/qt/yolo_quadro_weights/quadron_1280_fp16.engine`. |
| `kDeepStreamNetworkInputWidth`, `kDeepStreamNetworkInputHeight` | Входной размер engine. Сейчас `1280x1280`. |
| `kExternalNoTargetAzCentideg`, `kExternalNoTargetElCentideg` | Маркер "связь есть, цели нет". Сейчас это `-1, -1`, то есть yaw/pitch `-0.01`. |
| `kRangefinderSendDistanceToCan` | Отправлять ли дальномер в CAN `0x09`. Сейчас `false`. |

Остальное подтягивается автоматически: CAN-интерфейс, USB-UART роли
дальномера/внешнего источника, путь USB-камеры, свежесть данных, активный источник цели.

## Выбор Источника Цели

`TargetManager` задает приоритет:

1. Если YOLO видит свежую цель, используется камера.
2. Если камера потеряла цель, но внешний UART прислал свежие валидные углы,
   используется внешний источник.
3. Если внешний UART прислал `-1, -1`, это считается живой связью без цели:
   статус показывает `LINK OK, NO TARGET`, но CAN-команды по этой метке не шлются.
4. Если свежей цели нет ни от камеры, ни от UART, новые команды сопровождения не
   отправляются.

## Панель Состояния

В главном окне есть рамка `Состояние системы`. Она показывает:

- общее состояние;
- CAN;
- поток камеры;
- активный источник сопровождения;
- свежесть YOLO-цели;
- внешний UART;
- дальномер;
- дистанцию;
- свежесть телеметрии башни.

Общее состояние:

- красное `DEVICE MISSING`, если не найдено хотя бы одно нужное устройство;
- желтое `STABLE, WAIT DATA`, если устройства найдены, но от чего-то нет свежей обратной связи;
- зеленое `OK`, когда устройства найдены и свежие данные идут от всех узлов.

Наличие цели при этом не обязательно: система может быть полностью исправна, но
цель отсутствует.

## Визуализация На Кадре

На видеокадре дополнительно рисуются:

- центр кадра;
- внутренняя и внешняя рамки мертвой зоны из `AutoTracker`;
- пунктирная линия от центра кадра до центра выбранного YOLO-бокса.

Это помогает проверить, нет ли смещения центра на 4K/FullHD режимах.

Для защиты от дрожания YOLO-бокса центр цели сглаживается, а мертвая зона работает
с гистерезисом: внутри малой зоны команда удерживается, и сопровождение снова
активируется только после выхода за внешнюю рамку. Размер зон динамический:
для маленького bbox зона уменьшается, чтобы она не была больше цели в несколько
раз, а для крупного bbox допускается более широкая окрестность центра.

## USB-UART

Жестких `ttyUSB0/ttyUSB1` нет. Роли определяются через `/sys/class/tty`:

- дальномер: FTDI/FT232BM;
- внешний источник координат: CP210x.

При отключении устройство закрывается, затем порты сканируются заново и новый
порт открывается автоматически.

## Камера

Камера открывается через GStreamer: в DeepStream-режиме это pipeline с
`nvv4l2decoder/nvstreammux/nvinfer`, в fallback-режиме используется OpenCV
VideoCapture поверх GStreamer. При старте и после пропадания потока
`Camera` сканирует `/sys/class/video4linux`, читает имя, VID/PID и предпочитает
`HDMI USB Camera` с VID/PID `32e4:3415`. Если найдено несколько `/dev/video*`,
сначала пробуются наиболее похожие устройства, а не просто первый номер.

В статус камеры выводятся requested/actual параметры, например:
`requested 3840x2160@60 actual 3840x2160@60`. Если поток пропал после
выдергивания USB-камеры, устройство закрывается и попытки открыть его снова
продолжаются до восстановления, без перезапуска приложения.

## DeepStream

Основной режим инференса теперь задается `app_config::kUseDeepStream`. Когда он
включен, используется GStreamer/DeepStream pipeline:

```text
v4l2src -> jpegparse -> nvv4l2decoder -> nvstreammux -> nvinfer -> appsink
```

`nvinfer` получает готовый TensorRT engine `quadron_1280_fp16.engine`.
Конфиг `nvinfer` создается автоматически в `/tmp/target_detection_system_nvinfer.txt`.
YOLO tensor output читается из `NvDsInferTensorMeta`, затем постобработка, NMS,
выбор цели, динамическая мертвая зона и CAN-команды идут через прежние
`TargetManager`, `AutoTracker`, `CAN/UART` модули.

## Сборка

Проектный файл и бинарная цель qmake называются `Target_Detection_System`.

```bash
qmake Target_Detection_System.pro
make -j$(nproc)
```

Основные зависимости:

- Linux + SocketCAN;
- Qt 5 Widgets + SerialPort;
- OpenCV с CUDA/DNN;
- JetPack 6.2.1, CUDA 12.6, TensorRT 10.3, cuDNN 9.0;
- DeepStream SDK 7.1.0;
- GStreamer + NVIDIA DeepStream plugins;
- `aarch64-linux-gnu-g++` на Jetson Orin NX.

## Основные Модули

| Файл | Назначение |
| --- | --- |
| `app_config.h` | Ручные настройки камеры, YOLO, FOV и внешнего UART-маркера. |
| `camera.*` | Захват и reconnect USB-камеры. |
| `deepstream_yolo.*` | DeepStream/TensorRT inference через `nvinfer` и парсинг tensor meta. |
| `my_yolo.*` | Инференс, выбор цели и отрисовка боксов/оверлея. |
| `target_manager.*` | Приоритет источников цели и watchdog свежести данных. |
| `auto_tracker.*` | Пересчет пиксельного смещения в команду башни. |
| `uart_receiver.*` | Внешние координаты и heartbeat `-1, -1`. |
| `rangefinder_uart.*` | Дальномер и его статус. |
| `serial_port_resolver.*` | Автоопределение USB-UART устройств. |
| `can_work.*` | SocketCAN RX/TX и разбор телеметрии. |
| `tower_state.*` | Последнее положение башни и свежесть телеметрии. |
