#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "app_config.h"
#include <cstdint>
#include <atomic>
#include <filesystem>
#include <QString>
#include "uart_receiver.h"
#include <QThread>
#include <iostream>
#include <QFile>
#include "remote_tracker.h"
#include "target_manager.h"
#include "tracking_state.h"
#include "tower_state.h"
#include "turret_command.h"

// ===== NEW for separate video window =====
#include <opencv2/imgproc.hpp>
#include <QGridLayout>
#include <QGroupBox>
#include <QSizePolicy>
#include <QVBoxLayout>
#include <QMutexLocker>

#include <algorithm>
#include <chrono>
#include <cmath>

#define autostart
std::atomic<can_work*> g_can{nullptr};

namespace {

enum class StatusLevel
{
    Ok,
    Warn,
    Bad,
    Neutral
};

uint64_t monotonicMs()
{
    using namespace std::chrono;
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

QString statusStyle(StatusLevel level)
{
    QString bg = "#46515f";
    QString fg = "#ffffff";
    switch (level) {
    case StatusLevel::Ok:
        bg = "#1f7a3a";
        break;
    case StatusLevel::Warn:
        bg = "#8a6d1a";
        break;
    case StatusLevel::Bad:
        bg = "#8a2d2d";
        break;
    case StatusLevel::Neutral:
    default:
        break;
    }

    return QString("QLabel { background:%1; color:%2; padding:3px 6px; "
                   "border-radius:3px; font-weight:600; }")
        .arg(bg)
        .arg(fg);
}

void setStatus(QLabel* label, const QString& text, StatusLevel level)
{
    if (!label) {
        return;
    }

    label->setText(text);
    label->setStyleSheet(statusStyle(level));
}

QLabel* addStatusRow(QGridLayout* layout, int row, const QString& name)
{
    auto* title = new QLabel(name);
    title->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    title->setMinimumWidth(92);

    auto* value = new QLabel("WAIT");
    value->setTextInteractionFlags(Qt::TextSelectableByMouse);
    value->setWordWrap(true);
    value->setMinimumWidth(210);
    value->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);

    layout->addWidget(title, row, 0);
    layout->addWidget(value, row, 1);
    return value;
}

QString ageText(uint64_t now, uint64_t last)
{
    if (last == 0) {
        return "never";
    }
    return QString("%1 ms ago").arg(now - last);
}

QString targetSourceText(TargetManager::Source source)
{
    switch (source) {
    case TargetManager::Source::Camera:
        return "CAMERA";
    case TargetManager::Source::External:
        return "EXTERNAL";
    case TargetManager::Source::None:
    default:
        return "NONE";
    }
}

QString shortDeviceText(const QString& text)
{
    if (text.size() <= 128) {
        return text;
    }
    return text.left(125) + "...";
}

} // namespace

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // =========================================================
    // NEW: отдельное окно для видео (UI НЕ ТРОГАЕМ!)
    // =========================================================
    videoWindow = new QWidget(nullptr, Qt::Window);
    setWindowTitle("Target_Detection_System");
    videoWindow->setWindowTitle("Target_Detection_System Video");
    videoWindow->resize(1100, 700);
    videoWindow->move(50, 50);

    view = new QLabel(videoWindow);
    view->setAlignment(Qt::AlignCenter);
    view->setMinimumSize(640, 360);
    view->setScaledContents(true);

    auto *vLayout = new QVBoxLayout(videoWindow);
    vLayout->setContentsMargins(8, 8, 8, 8);
    vLayout->addWidget(view);

    videoWindow->show();
    // =========================================================

    // === Инициализация вспомогательных классов ===
    balancer = new Balancer();

    // NEW: кадры детекции -> в отдельное окно
    connect(balancer, &Balancer::ui_frame,
            this, &MainWindow::onNewFrame,
            Qt::QueuedConnection);
    connect(balancer, &Balancer::cameraDeviceStateChanged,
            this,
            [this](bool connected, const QString& description) {
                cameraDeviceConnected = connected;
                cameraDeviceDescription = description;
                updateSystemStatusPanel();
            },
            Qt::QueuedConnection);

    connect(ui->start, &QPushButton::pressed, balancer, &Balancer::startAll);
    connect(ui->stop,  &QPushButton::pressed, balancer, &Balancer::stopAll);
    connect(this, &MainWindow::closeWindow,   balancer, &Balancer::closeAll);
    connect(this, &MainWindow::closeWindow,   balancer, &Balancer::deleteLater);

    // NEW: при закрытии основного окна — закрыть и videoWindow
    connect(this, &MainWindow::closeWindow, this, [this]() {
        if (videoWindow) videoWindow->close();
    });

    // === Настройка UI ===
    ui->gbTest->hide();
    ui->textBrowser->clear();
    setupSystemStatusPanel();

    ui->lData1->setText("Data1");
    ui->lData1->setEnabled(false);
    ui->leData1->setEnabled(false);
    ui->lData2->setText("Data2");
    ui->lData2->setEnabled(false);
    ui->leData2->setEnabled(false);
    ui->lData3->setText("Data3");
    ui->lData3->setEnabled(false);
    ui->leData3->setEnabled(false);

    ui->cbData1->setEnabled(false);
    ui->cbData2->setEnabled(false);
    ui->cbData3->setEnabled(false);
    ui->cbData4->setEnabled(false);
    ui->cbData5->setEnabled(false);

    ui->pbSend->setEnabled(false);

    // === Таймеры для циклических кадров ===
    timerX10 = new QTimer(this);
    connect(timerX10, &QTimer::timeout, this, &MainWindow::sendX10Frame);
    timerX05 = new QTimer(this);
    connect(timerX05, &QTimer::timeout, this, &MainWindow::sendX05Frame);
    timerX0F = new QTimer(this);
    connect(timerX0F, &QTimer::timeout, this, &MainWindow::sendX0FFrame);

    // === Заполняем список CAN-интерфейсов ===
    namespace fs = std::filesystem;
    std::vector<QString> can_devices;

    for (const auto &entry : fs::directory_iterator("/sys/class/net")) {
        QString interface = QString::fromStdString(entry.path().filename().string());
        if (interface.startsWith("can", Qt::CaseInsensitive))
            can_devices.push_back(interface);
    }

    ui->cbCan->clear();
    for (const auto &can_dev : can_devices)
        ui->cbCan->addItem(can_dev);

    if (app_config::kStereoRangefinderEnabled) {
        stereoRangefinder = new StereoRangefinder(app_config::kStereoRangefinderPort, this);
        connect(stereoRangefinder, &StereoRangefinder::distanceReady,
                this, &MainWindow::showStereoDistance);
        connect(stereoRangefinder, &StereoRangefinder::errorText,
                this, &MainWindow::showStatus);
        connect(stereoRangefinder, &StereoRangefinder::deviceStateChanged, this,
                [this](bool connected, const QString& description) {
                    rangefinderConnected = connected;
                    rangefinderDescription = description;
                    updateSystemStatusPanel();
                });
        stereoRangefinder->start();
    } else {
        rangefinderConnected = true;
        rangefinderDescription = "disabled";
    }

    // УМНЫЙ СТАРТ CAN
    QTimer *canWaiter = new QTimer(this);
    connect(canWaiter, &QTimer::timeout, this, [this, canWaiter]() {
        if (QFile::exists("/sys/class/net/can0")) {
            canWaiter->stop();
            canWaiter->deleteLater();
            this->initCan();
            MY_CONSOLE_A("[CAN] Interface detected, started immediately");
        }
    });
    canWaiter->start(200);

    // === UART PIXELS ===
    uartReceiver = new UartReceiver();         // без parent, т.к. уйдёт в другой поток
    uartThread = new QThread(this);            // parent = this, чтобы поток не утёк

    uartReceiver->moveToThread(uartThread);
    connect(uartReceiver, &QObject::destroyed, this, [this]() { uartReceiver = nullptr; });
    connect(uartThread, &QObject::destroyed, this, [this]() { uartThread = nullptr; });

    // старт чтения в потоке
    connect(uartThread, &QThread::started, uartReceiver, &UartReceiver::start);

    // при закрытии окна — корректно остановить
    connect(this, &MainWindow::destroyed, uartReceiver, &UartReceiver::stop, Qt::DirectConnection);
    connect(this, &MainWindow::destroyed, uartThread, &QThread::quit);

    // удалить объекты после завершения потока
    connect(uartThread, &QThread::finished, uartReceiver, &QObject::deleteLater);

    // ошибки в консоль
    connect(uartReceiver, &UartReceiver::error, this, [this](const QString& msg){
        MY_CONSOLE_A("[UART ERROR] " + msg);
    });
    connect(uartReceiver, &UartReceiver::status, this, [this](const QString& msg){
        MY_CONSOLE_A(msg);
    });
    connect(uartReceiver, &UartReceiver::deviceStateChanged, this,
            [this](bool connected, const QString& description) {
                externalDeviceConnected = connected;
                externalDeviceDescription = description;
                updateSystemStatusPanel();
            });

    uartThread->start();

    RemoteTracker* remote = new RemoteTracker(this);
    connect(uartReceiver, &UartReceiver::pixelsReceived,
            remote, &RemoteTracker::onPixels,
            Qt::QueuedConnection);

    healthTimer = new QTimer(this);
    connect(healthTimer, &QTimer::timeout, this, &MainWindow::refreshTrackingHealth);
    healthTimer->start(100);

#ifdef autostart
    ui->start->pressed();
#endif
}

// ===== NEW: эти 2 функции тоже должны быть в файле (можешь вставить ниже по тексту) =====
QImage MainWindow::matToQImageRGB(const cv::Mat &bgr)
{
    if (bgr.empty()) return QImage();
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows, (int)rgb.step, QImage::Format_RGB888);
    return img.copy();
}

void MainWindow::onNewFrame(const cv::Mat &frame)
{
    lastVideoFrameMs = monotonicMs();
    if (!view) return;
    QMutexLocker locker(&draw_mutex);
    QImage img = matToQImageRGB(frame);
    if (img.isNull()) return;
    view->setPixmap(QPixmap::fromImage(img));
}

void MainWindow::setupSystemStatusPanel()
{
    if (systemStatusGroup) {
        return;
    }

    systemStatusGroup = new QGroupBox("Состояние системы", this);
    systemStatusGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);

    auto* layout = new QGridLayout(systemStatusGroup);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setHorizontalSpacing(8);
    layout->setVerticalSpacing(5);

    int row = 0;
    statusOverall = addStatusRow(layout, row++, "Общее");
    statusCan = addStatusRow(layout, row++, "CAN");
    statusCamera = addStatusRow(layout, row++, "Камера");
    statusTargetSource = addStatusRow(layout, row++, "Источник");
    statusYoloTarget = addStatusRow(layout, row++, "YOLO");
    statusExternal = addStatusRow(layout, row++, "Внешн.");
    statusRangefinder = addStatusRow(layout, row++, "Дальномер");
    statusDistance = addStatusRow(layout, row++, "Дистанция");
    statusTurret = addStatusRow(layout, row++, "Башня");

    if (ui && ui->gridLayout) {
        ui->gridLayout->addWidget(systemStatusGroup, 3, 0, 1, 2);
    }

    updateSystemStatusPanel();
}

void MainWindow::updateSystemStatusPanel()
{
    if (!systemStatusGroup) {
        return;
    }

    constexpr uint64_t CAMERA_FRAME_TIMEOUT_MS = 800;

    const uint64_t now = monotonicMs();
    const auto target = TargetManager::snapshot();
    const TurretState turret = getTurretState();
    const bool canOk = g_can.load(std::memory_order_acquire) != nullptr;
    const bool cameraFramesFresh =
        lastVideoFrameMs != 0 && (now - lastVideoFrameMs) <= CAMERA_FRAME_TIMEOUT_MS;
    const bool distanceFresh = targetDistanceFresh();
    const bool distanceRequired = target.cameraFresh && target.cameraBoxValid;
    const bool distanceFeedbackOk = !distanceRequired || distanceFresh;

    const bool allDevicesDetected =
        canOk &&
        cameraDeviceConnected &&
        externalDeviceConnected &&
        rangefinderConnected;
    const bool allFeedbackFresh =
        cameraFramesFresh &&
        turret.fresh &&
        target.externalLinkFresh &&
        distanceFeedbackOk;

    StatusLevel overallLevel = StatusLevel::Bad;
    QString overallText = "DEVICE MISSING";
    if (allDevicesDetected && allFeedbackFresh) {
        overallLevel = StatusLevel::Ok;
        overallText = "OK";
    } else if (allDevicesDetected) {
        overallLevel = StatusLevel::Warn;
        overallText = "STABLE, WAIT DATA";
    }
    setStatus(statusOverall, overallText, overallLevel);

    const QString canName = ui->cbCan->currentText().isEmpty()
        ? QString("can?")
        : ui->cbCan->currentText();
    const QString canText = canOk
        ? QString("OK %1 %2").arg(canName).arg(canReaderWriter && canReaderWriter->isRunning() ? "running" : "idle")
        : QString("WAIT %1").arg(canName);
    setStatus(statusCan, canText, canOk ? StatusLevel::Ok : StatusLevel::Bad);

    if (!cameraDeviceConnected) {
        setStatus(statusCamera,
                  "WAIT " + shortDeviceText(cameraDeviceDescription),
                  StatusLevel::Bad);
    } else if (cameraFramesFresh) {
        setStatus(statusCamera,
                  "OK frame " + ageText(now, lastVideoFrameMs) + " " +
                      shortDeviceText(cameraDeviceDescription),
                  StatusLevel::Ok);
    } else if (lastVideoFrameMs == 0) {
        setStatus(statusCamera,
                  "WAIT no frames " + shortDeviceText(cameraDeviceDescription),
                  StatusLevel::Warn);
    } else {
        setStatus(statusCamera,
                  "STALE frame " + ageText(now, lastVideoFrameMs) + " " +
                      shortDeviceText(cameraDeviceDescription),
                  StatusLevel::Warn);
    }

    const StatusLevel sourceLevel = target.activeSource == TargetManager::Source::None
        ? StatusLevel::Warn
        : StatusLevel::Ok;
    setStatus(statusTargetSource, targetSourceText(target.activeSource), sourceLevel);

    if (target.cameraFresh) {
        setStatus(statusYoloTarget,
                  "OK target " + ageText(now, g_main_cam_last_seen_ms.load(std::memory_order_relaxed)),
                  StatusLevel::Ok);
    } else {
        setStatus(statusYoloTarget,
                  "LOST " + ageText(now, g_main_cam_last_seen_ms.load(std::memory_order_relaxed)),
                  StatusLevel::Warn);
    }

    if (externalDeviceConnected && target.externalFresh) {
        setStatus(statusExternal,
                  QString("OK AZ=%1 EL=%2 %3")
                      .arg(target.externalAzCentideg / 100.0, 0, 'f', 2)
                      .arg(target.externalElCentideg / 100.0, 0, 'f', 2)
                      .arg(shortDeviceText(externalDeviceDescription)),
                  StatusLevel::Ok);
    } else if (externalDeviceConnected && target.externalLinkFresh && target.externalNoTarget) {
        setStatus(statusExternal,
                  "LINK OK, NO TARGET " + ageText(now, target.externalLastPacketMs) + " " +
                      shortDeviceText(externalDeviceDescription),
                  StatusLevel::Ok);
    } else if (externalDeviceConnected && target.externalLinkFresh) {
        setStatus(statusExternal,
                  "LINK OK, INVALID TARGET " + ageText(now, target.externalLastPacketMs) + " " +
                      shortDeviceText(externalDeviceDescription),
                  StatusLevel::Warn);
    } else if (externalDeviceConnected) {
        setStatus(statusExternal,
                  "STALE link " + ageText(now, target.externalLastPacketMs) + " " +
                      shortDeviceText(externalDeviceDescription),
                  StatusLevel::Warn);
    } else {
        setStatus(statusExternal,
                  "WAIT " + shortDeviceText(externalDeviceDescription),
                  StatusLevel::Bad);
    }

    if (rangefinderConnected && distanceRequired && distanceFresh) {
        setStatus(statusRangefinder,
                  "OK " + shortDeviceText(rangefinderDescription),
                  StatusLevel::Ok);
    } else if (rangefinderConnected && !distanceRequired) {
        setStatus(statusRangefinder,
                  "READY " + shortDeviceText(rangefinderDescription),
                  StatusLevel::Ok);
    } else if (rangefinderConnected) {
        setStatus(statusRangefinder,
                  "STALE " + shortDeviceText(rangefinderDescription),
                  StatusLevel::Warn);
    } else {
        setStatus(statusRangefinder,
                  "WAIT " + shortDeviceText(rangefinderDescription),
                  StatusLevel::Bad);
    }

    const int distance = g_target_distance_mm.load(std::memory_order_relaxed);
    if (distanceRequired && distanceFresh && distance >= 0) {
        setStatus(statusDistance,
                  QString("OK %1 mm %2")
                      .arg(distance)
                      .arg(ageText(now, g_target_distance_last_seen_ms.load(std::memory_order_relaxed))),
                  StatusLevel::Ok);
    } else if (!distanceRequired) {
        setStatus(statusDistance,
                  "NO CAMERA TARGET",
                  StatusLevel::Neutral);
    } else {
        setStatus(statusDistance,
                  "WAIT " + ageText(now, g_target_distance_last_seen_ms.load(std::memory_order_relaxed)),
                  StatusLevel::Warn);
    }

    if (turret.fresh) {
        setStatus(statusTurret,
                  QString("OK AZ=%1 EL=%2 %3")
                      .arg(turret.az_deg, 0, 'f', 2)
                      .arg(turret.el_deg, 0, 'f', 2)
                      .arg(ageText(now, turret.last_update_ms)),
                  StatusLevel::Ok);
    } else {
        setStatus(statusTurret,
                  "WAIT " + ageText(now, turret.last_update_ms),
                  StatusLevel::Warn);
    }
}


void MainWindow::initCan()
{
    if (canReaderWriter) {
        return;
    }
    const QString can_name = (ui->cbCan->count() > 0)
                           ? ui->cbCan->itemText(0)
                           : QStringLiteral("can0");
    QFile canPath("/sys/class/net/" + can_name);
    if (!canPath.exists()) {
        MY_CONSOLE_A("[CAN] Interface not ready, retry in 500ms...");
        QTimer::singleShot(500, this, [this]() { this->initCan(); });
        return;
    }
    canReaderWriter = new can_work(can_name.toLocal8Bit().constData());

    connect(this, &MainWindow::setReadState,  canReaderWriter, &can_work::setStopFlag);
    connect(this, &MainWindow::putFrame,      canReaderWriter, &can_work::writeCan);
    connect(this, &MainWindow::startCanRecv,  canReaderWriter, &can_work::startLoop);

    connect(canReaderWriter, &can_work::frameReady,     this, &MainWindow::showFrame);
    connect(canReaderWriter, &can_work::insertConsole,  this, &MainWindow::consoleInsertText);
    connect(canReaderWriter, &can_work::appendConsole,  this, &MainWindow::consoleAppendText);

    ui->gbSend->setEnabled(true);
    ui->pbSend->setEnabled(true);
    ui->gbReceive->setEnabled(true);

    isRecieve = true;
    emit startCanRecv(true);

    if (!threadPool) threadPool = new QThreadPool(this);
    canReaderWriter->setAutoDelete(false);
    if (!canReaderWriter->isRunning()) {
        threadPool->start(canReaderWriter);
    }

    MY_CONSOLE_A(QString(">>> CAN %1 auto-started").arg(can_name));
    g_can.store(canReaderWriter, std::memory_order_release);

}

void MainWindow::refreshTrackingHealth()
{
    TargetManager::refresh();
    updateStereoRangefinder();
    updateSystemStatusPanel();
}

void MainWindow::updateStereoRangefinder()
{
    if (!stereoRangefinder) {
        return;
    }

    const auto target = TargetManager::snapshot();
    const bool hasCameraBox =
        target.cameraFresh &&
        target.cameraBoxValid &&
        target.cameraBoxW > 0 &&
        target.cameraBoxH > 0;

    if (!hasCameraBox) {
        if (stereoRangefinderTargetActive) {
            stereoRangefinder->clearTarget();
            stereoRangefinderTargetActive = false;
            stereoRangefinderLastTargetSeq = 0;
        }
        return;
    }

    int boxX = target.cameraBoxX;
    int boxY = target.cameraBoxY;
    int boxW = target.cameraBoxW;
    int boxH = target.cameraBoxH;

    const int sourceFrameW = app_config::kStereoRangefinderSourceFrameWidth > 0
        ? app_config::kStereoRangefinderSourceFrameWidth
        : target.cameraFrameW;
    const int sourceFrameH = app_config::kStereoRangefinderSourceFrameHeight > 0
        ? app_config::kStereoRangefinderSourceFrameHeight
        : target.cameraFrameH;
    const int stereoFrameW = app_config::kStereoRangefinderFrameWidth;
    const int stereoFrameH = app_config::kStereoRangefinderFrameHeight;

    if (stereoFrameW > 0 &&
        stereoFrameH > 0 &&
        sourceFrameW > 0 &&
        sourceFrameH > 0)
    {
        const double sx = static_cast<double>(stereoFrameW) /
                          static_cast<double>(sourceFrameW);
        const double sy = static_cast<double>(stereoFrameH) /
                          static_cast<double>(sourceFrameH);
        boxX = static_cast<int>(std::lround(boxX * sx));
        boxY = static_cast<int>(std::lround(boxY * sy));
        boxW = std::max(1, static_cast<int>(std::lround(boxW * sx)));
        boxH = std::max(1, static_cast<int>(std::lround(boxH * sy)));

        boxX = std::clamp(boxX, 0, std::max(0, stereoFrameW - 1));
        boxY = std::clamp(boxY, 0, std::max(0, stereoFrameH - 1));
        boxW = std::min(boxW, stereoFrameW - boxX);
        boxH = std::min(boxH, stereoFrameH - boxY);
    }

    stereoRangefinder->updateTargetBox(boxX, boxY, boxW, boxH);
    stereoRangefinderTargetActive = true;
    stereoRangefinderLastTargetSeq = target.cameraBoxSequence;
}

void MainWindow::shutdownServices()
{
    if (healthTimer) {
        healthTimer->stop();
    }
    if (timerX10) timerX10->stop();
    if (timerX05) timerX05->stop();
    if (timerX0F) timerX0F->stop();

    if (stereoRangefinder) {
        stereoRangefinder->clearTarget();
        stereoRangefinder->stop();
    }

    if (canReaderWriter) {
        emit startCanRecv(false);
        canReaderWriter->setStopFlag(true);
    }
    if (threadPool) {
        threadPool->waitForDone(1000);
    }
    if (canReaderWriter) {
        canReaderWriter->closeCan();
        g_can.store(nullptr, std::memory_order_release);
        if (!canReaderWriter->isRunning()) {
            delete canReaderWriter;
            canReaderWriter = nullptr;
        }
    }

    if (uartReceiver) {
        uartReceiver->stop();
    }
    if (uartThread && uartThread->isRunning()) {
        uartThread->quit();
        uartThread->wait(1000);
    }
}



MainWindow::~MainWindow()
{
    shutdownServices();
    delete ui;
}


void MainWindow::on_start_pressed()
{
    ui->start->setEnabled(0);
}

void MainWindow::on_stop_pressed()
{

}

//Функция для показа сообщения в текстовом окне
void MainWindow::showFrame(const can_frame &frame)
{
    QString can_id  = QString("%1").arg(frame.can_id, 0, 16);
    QString can_dlc = QString::number(frame.can_dlc);

    MY_CONSOLE_I("ID: ");
    MY_CONSOLE_I(can_id);
    MY_CONSOLE_I(" [");
    MY_CONSOLE_I(can_dlc);
    MY_CONSOLE_I("] ");

    // --- Печатаем всегда 8 байт ---
    for (int i = 0; i < 8; ++i) {
        uint8_t b = (i < frame.can_dlc) ? frame.data[i] : 0;
        MY_CONSOLE_I(QString("%1 ").arg(b, 2, 16, QLatin1Char('0')));
    }
    MY_CONSOLE_I("\n");
}
//Слоты для вывода текста
void MainWindow::consoleAppendText(QString str)
{
    MY_CONSOLE_A(str);
}
void MainWindow::consoleInsertText(QString str)
{
    MY_CONSOLE_I(str);
}

//Автоматическая прокрутка текста
void MainWindow::on_textBrowser_textChanged()
{
    ui->textBrowser->moveCursor(QTextCursor::End);
}

//Кнопка "Очистить"
void MainWindow::on_pbClear_clicked()
{
    ui->textBrowser->clear();
}



//Dalnomer (knopky ON u OFF)
void MainWindow::on_pbTestSend_clicked()
{
    if (stereoRangefinder) {
        stereoRangefinder->start();
        stereoRangefinder->requestDistance();
        MY_CONSOLE_A(">>> SRF started");
    }
}

void MainWindow::on_pbTestReceive_clicked()
{
    if (stereoRangefinder) {
        stereoRangefinder->clearTarget();
        MY_CONSOLE_A(">>> SRF tracking stopped");
    }
}

void MainWindow::showDistance(int mm)
{
    updateTargetDistance(mm);
    int dm = mm / 100;

    QString msg = QString("Distance = %1 mm").arg(mm);
    MY_CONSOLE_A(msg);

    if constexpr (app_config::kRangefinderSendDistanceToCan) {

    // 2. Формируем CAN кадр
    struct can_frame frame{};
    frame.can_id  = 0x09;  // ID = 9
    frame.can_dlc = 4;     // 2 байта данных

    // кладем в данные миллиметры (ограничим 2 байтами, т.е. максимум 65535 мм)

    frame.data[3] = (uint8_t)(dm & 0xFF);
    frame.data[2] = (uint8_t)((dm >> 8) & 0xFF);
    frame.data[1] = 0 & 0xFF; //zag
    frame.data[0] = (0 >> 8) & 0xFF;//zaglushka




    // 3. Отправляем в CAN
    if (canReaderWriter) {
        canReaderWriter->writeCan(frame);
    }
    }
}

void MainWindow::showStereoDistance(double rawDistance, int mm)
{
    showDistance(mm);
    MY_CONSOLE_A(QString("Stereo distance raw = %1").arg(rawDistance, 0, 'f', 3));
}

void MainWindow::showStatus(const QString& msg)
{
    MY_CONSOLE_A(">>> " + msg);
}





//цикл отправка X10
void MainWindow::sendX10Frame()
{
    struct can_frame cFrame{};
    int temp = 0;

    cFrame.can_id = 0x10;
    cFrame.can_dlc = 1;

    if(ui->cbData4->isChecked()) temp += 1;
    if(ui->cbData3->isChecked()) temp += 2;
    if(ui->cbData2->isChecked()) temp += 4;
    if(ui->cbData1->isChecked()) temp += 8;
    cFrame.data[0] = static_cast<uchar>(temp << 4);

    emit putFrame(cFrame);
    emit showFrame(cFrame);

}
//цикл отправка X05
void MainWindow::sendX05Frame()
{
    struct can_frame cFrame;
    int temp = 0;

    if(ui->cbData3->isChecked()) temp += 1;
    if(ui->cbData2->isChecked()) temp += 2;
    if(ui->cbData1->isChecked()) temp += 4;

    QString s1 = ui->leData1->text().trimmed(); s1.replace(',','.');
    QString s2 = ui->leData2->text().trimmed(); s2.replace(',','.');

    int Data1 = qRound(s1.toDouble() * 100.0);
    int Data2 = qRound(s2.toDouble() * 100.0);

    if (std::abs(Data1) > 18000 || std::abs(Data2) > 4000) {
        MY_CONSOLE_A("Неправильные координаты!");
        return;
    }

    cFrame = turret_control::makePositionFrameCentideg(
        static_cast<int16_t>(Data1),
        static_cast<int16_t>(Data2),
        static_cast<uint8_t>(temp << 5)
    );


    emit putFrame(cFrame);
    emit showFrame(cFrame);

}
//цикл отправка X0F
void MainWindow::sendX0FFrame()
{
    struct can_frame cFrame;
    int temp = 0;

    cFrame.can_id = 0xF;
    cFrame.can_dlc = 5;

    int Data1 = ui->leData1->text().toInt();
    int Data2 = ui->leData2->text().toInt();
    int Data3 = ui->leData3->text().toInt();


    if (Data1 < 0)
    {
        MY_CONSOLE_A("Неправильные данные!");
        return;
    }

    if (Data2 < 0)
    {
        MY_CONSOLE_A("Неправильные данные!");
        return;
    }

    if (Data3 < 0)
    {
        MY_CONSOLE_A("Неправильные данные!");
        return;
    }

    cFrame.data[4] = 0;
    cFrame.data[3] = static_cast<uchar>(Data3);
    cFrame.data[2] = 0;
    cFrame.data[1] = static_cast<uchar>(Data2);
    cFrame.data[0] = static_cast<uchar>(Data1);

    emit putFrame(cFrame);
    emit showFrame(cFrame);

}


//Запуск интерфейса
void MainWindow::on_pbSet_clicked()
{
    initCan();
}

//Кнопка "Отправить"
void MainWindow::on_pbSend_clicked()
{
    int index = ui->cmbID->currentIndex();

    int Data1 = ui->leData1->text().toInt();
    int Data2 = ui->leData2->text().toInt();
    int Data3 = ui->leData3->text().toInt();
    int check1 = 0;
    int check2 = 0;
    int check3 = 0;
    int check4 = 0;
    int check5 = 0;

    ui->cbData1->isChecked() ? check1 = 1 : check1 = 0;
    ui->cbData2->isChecked() ? check2 = 1 : check2 = 0;
    ui->cbData3->isChecked() ? check3 = 1 : check3 = 0;
    ui->cbData4->isChecked() ? check4 = 1 : check4 = 0;
    ui->cbData5->isChecked() ? check5 = 1 : check5 = 0;



    struct can_frame cFrame;
    bool isSend = true;
    int temp = 0;

    switch (index)
    {
    case 1://0x10
        cFrame.can_id = 0x10;
        cFrame.can_dlc = 0x1;


        check4 ? temp += 1 : temp += 0;
        check3 ? temp += 2 : temp += 0;
        check2 ? temp += 4 : temp += 0;
        check1 ? temp += 8 : temp += 0;
        cFrame.data[0] = temp << 4;


        if (ui->cbData5->isChecked()) {
            if (!timerX10->isActive()){
                timerX10->start(500);
                std::cerr << "Цикл начался" << std::endl;
            }
            isSend = false;
            break;
        }

        if (timerX10->isActive()) {
            timerX10->stop();
            std::cerr << "Цикл закончился" << std::endl;

        }


        break;


    case 2: //0x5
        if (std::abs(Data1) > 180)
        {
            MY_CONSOLE_A("Неправильные координаты! Диапазон [-180 .. 180]");
            isSend = false;
            break;
        }

        if (std::abs(Data2) > 40)
        {
            MY_CONSOLE_A("Неправильный угол! Диапазон [-40 .. 40]");
            isSend = false;
            break;
        }



        Data1 = qRound(QString(ui->leData1->text()).replace(',','.').toDouble() * 100.0);
        Data2 = qRound(QString(ui->leData2->text()).replace(',','.').toDouble() * 100.0);


        //Test
        std::cerr << "Ввод" << std::endl;
        std::cerr << "Азимут" << Data1 << std::endl;
        std::cerr << "Угол" << Data2 << std::endl;

        check3 ? temp += 1 : temp += 0;
        check2 ? temp += 2 : temp += 0;
        check1 ? temp += 4 : temp += 0;
        cFrame = turret_control::makePositionFrameCentideg(
            static_cast<int16_t>(Data1),
            static_cast<int16_t>(Data2),
            static_cast<uint8_t>(temp << 5)
        );

        //Test
        std::cerr << "Передача" << std::endl;
        std::cerr << "Азимут" << Data1  << "   MSB:" << int(cFrame.data[1]) << "   LSB:" << int(cFrame.data[2]) << std::endl;
        std::cerr << "Азимут" << Data2  << "   MSB:" << int(cFrame.data[3]) << "    LSB:" << int(cFrame.data[4]) << std::endl;

        if (ui->cbData5->isChecked()) {
            if (!timerX05->isActive()){
                timerX05->start(20);
                std::cerr << "Цикл начался" << std::endl;
            }
            isSend = false;
            break;
        }

        if (timerX05->isActive()) {
            timerX05->stop();
            std::cerr << "Цикл закончился" << std::endl;

        }

        break;




    case 3: //0x9
        cFrame.can_id = 0x9;
        cFrame.can_dlc = 4;

        //cFrame.data[3] = (Data2 - ((Data2 >> 8) << 8));//Data2 % 10000;
        //cFrame.data[2] = Data2 >> 8;
        //cFrame.data[1] = (Data1 - ((Data1 >> 8) << 8));//Data1 % 10000;
        //cFrame.data[0] = Data1 >> 8;

        if (Data1 < 0)
        {
            MY_CONSOLE_A("Неправильная рад. скорость! Диапазон [0 .. 1000]");
            isSend = false;
            break;
        }

        if (Data2 < 0 || Data2 > 2000)
        {
            MY_CONSOLE_A("Неправильная дальность цели! Диапазон [0 .. 2000]");
            isSend = false;
            break;
        }



        Data1 = qRound(QString(ui->leData1->text()).replace(',','.').toDouble() * 10.0);
        Data2 = qRound(QString(ui->leData2->text()).replace(',','.').toDouble() * 10.0);

        std::cerr << "Ввод" << std::endl;
        std::cerr << " Рад скорость: " << Data1 << std::endl;
        std::cerr << " дальность: " << Data2 << std::endl;

        cFrame.data[3] = Data2  & 0xFF;
        cFrame.data[2] = (Data2 >> 8) & 0xFF;
        cFrame.data[1] = Data1 & 0xFF;
        cFrame.data[0] = (Data1 >> 8) & 0xFF;

        //Test
        std::cerr << "Передача" << std::endl;
        std::cerr << "Рад скорость:" << Data1  << "   MSB:" << int(cFrame.data[1]) << "   LSB:" << int(cFrame.data[2]) << std::endl;
        std::cerr << "дальность" << Data2 << "   MSB:" << int(cFrame.data[3]) << "    LSB:" << int(cFrame.data[4]) << std::endl;





        break;



    case 4: //0xF
        cFrame.can_id = 0xF;
        cFrame.can_dlc = 5;

        if (Data1 < 0)
        {
            MY_CONSOLE_A("Неправильные данные!");
            isSend = false;
            break;
        }

        if (Data2 < 0)
        {
            MY_CONSOLE_A("Неправильные данные!");
            isSend = false;
            break;
        }

        if (Data3 < 0)
        {
            MY_CONSOLE_A("Неправильные данные!");
            isSend = false;
            break;
        }

        cFrame.data[4] = 0;
        cFrame.data[3] = Data3;
        cFrame.data[2] = 0;
        cFrame.data[1] = Data2;
        cFrame.data[0] = Data1;

        if (ui->cbData5->isChecked()) {
            if (!timerX0F->isActive()){
                timerX0F->start(500);
                std::cerr << "Цикл начался" << std::endl;
            }
            isSend = false;
            break;
        }

        if (timerX0F->isActive()) {
            timerX0F->stop();
            std::cerr << "Цикл закончился" << std::endl;

        }

        break;


    case 5: //0x1A
        cFrame.can_id = 0x1A;
        cFrame.can_dlc = 3;

        if (Data1 < 0)
        {
            MY_CONSOLE_A("Неправильный код устройства!");
            isSend = false;
            break;
        }

        if (Data2 > 9999 || Data2 < 0)
        {
            MY_CONSOLE_A("Неправильное время наработки! Max - 9999, min - 0");
            isSend = false;
            break;
        }

        cFrame.data[2] = (Data2 - ((Data2 >> 8) << 8));
        cFrame.data[1] = Data2 >> 8;
        cFrame.data[0] = Data1;
        break;
    default:
        MY_CONSOLE_A("Невозможно отправить сообщение: не выбран тип кадра\n");
        isSend = false;
        break;
    }

//    std::cerr << isSend;
    const struct can_frame ccFrame {cFrame};
    if (isSend)
    {
        MY_CONSOLE_A("Сообщение для отправки :");
        putFrame(ccFrame);
    }
}

//Кнопка "Получить"
void MainWindow::on_pbReceive_clicked()
{
    if (!canReaderWriter) {
        MY_CONSOLE_A("[CAN] Receiver is not initialized");
        return;
    }

    isRecieve = !isRecieve;
    emit startCanRecv(isRecieve);
    if (isRecieve)
    {
        ui->pbReceive->setText("Остановить прием");
    }
    else
    {
        ui->pbReceive->setText("Принять данные");
    }

    if (isRecieve && !canReaderWriter->isRunning()) {
        if (!threadPool) threadPool = new QThreadPool(this);
        canReaderWriter->setAutoDelete(false);
        threadPool->start(canReaderWriter);
    }
}

// ========= Показать тестовый блок ==========
void MainWindow::on_cbShowTest_stateChanged(int arg1)
{
    if(arg1)
        ui->gbTest->show();
    else
        ui->gbTest->hide();
}

// Тестовая дебажная кнопка
void MainWindow::on_pbTest_clicked()
{}


void MainWindow::on_cmbID_currentIndexChanged(int index)
{
    switch (index) {
    case 1: //0x10
        ui->lData1->setText("Data1");
        ui->lData1->setEnabled(false);
        ui->leData1->setEnabled(false);

        ui->lData2->setText("Data2");
        ui->lData2->setEnabled(false);
        ui->leData2->setEnabled(false);

        ui->lData3->setText("Data3");
        ui->lData3->setEnabled(false);
        ui->leData3->setEnabled(false);

        ui->cbData1->setEnabled(true);
        ui->cbData1->setText("Неисправность СЧ СО");
        ui->cbData2->setEnabled(true);
        ui->cbData2->setText("Загрязнение приемников");
        ui->cbData3->setEnabled(true);
        ui->cbData3->setText("Наличие цели");
        ui->cbData4->setEnabled(true);
        ui->cbData4->setText("Технологический режим СО");

        ui->cbData5->setEnabled(true);
        ui->cbData5->setText("Циклическая отправка (500 мс)");
        break;
    case 2: //0x5
        ui->lData1->setText("Коорд. по азимуту");
        ui->lData1->setEnabled(true);
        ui->leData1->setEnabled(true);

        ui->lData2->setText("Коорд. по углу");
        ui->lData2->setEnabled(true);
        ui->leData2->setEnabled(true);

        ui->lData3->setText("Data3");
        ui->lData3->setEnabled(false);
        ui->leData3->setEnabled(false);

        ui->cbData1->setEnabled(true);
        ui->cbData1->setText("Атака");
        ui->cbData2->setEnabled(true);
        ui->cbData2->setText("Рубеж 30м");
        ui->cbData3->setEnabled(true);
        ui->cbData3->setText("Запрос положения УПУ");
        ui->cbData4->setEnabled(false);
        ui->cbData4->setText("Data4");

        ui->cbData5->setEnabled(true);
        ui->cbData5->setText("Циклическая отправка (20 мс)");
        break;
    case 3: //0x9
        ui->lData1->setText("Радиальная скор.");
        ui->lData1->setEnabled(true);
        ui->leData1->setEnabled(true);

        ui->lData2->setText("Дальность (м)");
        ui->lData2->setEnabled(true);
        ui->leData2->setEnabled(true);

        ui->lData3->setText("Data3");
        ui->lData3->setEnabled(false);
        ui->leData3->setEnabled(false);

        ui->cbData1->setEnabled(false);
        ui->cbData1->setText("Data1");
        ui->cbData2->setEnabled(false);
        ui->cbData2->setText("Data2");
        ui->cbData3->setEnabled(false);
        ui->cbData3->setText("Data3");
        ui->cbData4->setEnabled(false);
        ui->cbData4->setText("Data4");

        ui->cbData5->setEnabled(false);
        ui->cbData5->setText("Циклическая отправка (500 мс)");
        break;
    case 4: //0xF
        ui->lData1->setText("Неисправности ...");
        ui->lData1->setEnabled(true);
        ui->leData1->setEnabled(true);

        ui->lData2->setText("Неиспр. приемн.");
        ui->lData2->setEnabled(true);
        ui->leData2->setEnabled(true);

        ui->lData3->setText("Загрязн. приемн.");
        ui->lData3->setEnabled(true);
        ui->leData3->setEnabled(true);

        ui->cbData1->setEnabled(false);
        ui->cbData1->setText("Data1");
        ui->cbData2->setEnabled(false);
        ui->cbData2->setText("Data2");
        ui->cbData3->setEnabled(false);
        ui->cbData3->setText("Data3");
        ui->cbData4->setEnabled(false);
        ui->cbData4->setText("Data4");

        ui->cbData5->setEnabled(true);
        ui->cbData5->setText("Циклическая отправка (500 мс)");
        break;
    case 5: //0x1A
        ui->lData1->setText("Код устройства");
        ui->lData1->setEnabled(true);
        ui->leData1->setEnabled(true);

        ui->lData2->setText("Время наработки");
        ui->lData2->setEnabled(true);
        ui->leData2->setEnabled(true);

        ui->lData3->setText("Data3");
        ui->lData3->setEnabled(false);
        ui->leData3->setEnabled(false);

        ui->cbData1->setEnabled(false);
        ui->cbData1->setText("Data1");
        ui->cbData2->setEnabled(false);
        ui->cbData2->setText("Data2");
        ui->cbData3->setEnabled(false);
        ui->cbData3->setText("Data3");
        ui->cbData4->setEnabled(false);
        ui->cbData4->setText("Data4");

        ui->cbData5->setEnabled(false);
        ui->cbData5->setText("Циклическая отправка (500 мс)");
        break;
    default:
        ui->lData1->setText("Data1");
        ui->lData1->setEnabled(false);
        ui->leData1->setEnabled(false);
        ui->lData2->setText("Data2");
        ui->lData2->setEnabled(false);
        ui->leData2->setEnabled(false);
//        ui->pbSend->setEnabled(false);

        ui->cbData1->setEnabled(false);
        ui->cbData1->setText("Data1");
        ui->cbData2->setEnabled(false);
        ui->cbData2->setText("Data2");
        ui->cbData3->setEnabled(false);
        ui->cbData3->setText("Data3");
        ui->cbData4->setEnabled(false);
        ui->cbData4->setText("Data4");
        ui->cbData5->setEnabled(false);
        ui->cbData5->setText("Data5");
        break;
    }
}
