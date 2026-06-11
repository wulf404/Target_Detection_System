#include "stereo_rangefinder.h"

#include "app_config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace {

constexpr std::uint8_t kStartByte = 0xFF;
constexpr std::uint8_t kCmdSetInitialBox = 0x01;
constexpr std::uint8_t kCmdRequestDistance = 0x02;
constexpr std::uint8_t kCmdStopTracking = 0x03;
constexpr std::uint8_t kCmdStopProgram = 0x04;
constexpr std::uint8_t kCmdSwitchVideoStream = 0x05;
constexpr int kPacketSize = 11;
constexpr int kDataSize = 8;

std::uint64_t monotonicMs()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

std::uint8_t byteAt(const QByteArray& bytes, int index)
{
    return static_cast<std::uint8_t>(static_cast<unsigned char>(bytes.at(index)));
}

void putU16BE(QByteArray& data, int offset, int value)
{
    const auto clamped = static_cast<std::uint16_t>(
        std::clamp(value, 0, static_cast<int>(std::numeric_limits<std::uint16_t>::max()))
    );
    data[offset] = static_cast<char>((clamped >> 8) & 0xFF);
    data[offset + 1] = static_cast<char>(clamped & 0xFF);
}

double decodeDoubleBE(const QByteArray& data)
{
    std::uint64_t bits = 0;
    for (int i = 0; i < kDataSize; ++i) {
        bits = (bits << 8) | byteAt(data, i);
    }

    double value = 0.0;
    static_assert(sizeof(value) == sizeof(bits), "double must be 64-bit IEEE 754");
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

} // namespace

StereoRangefinder::StereoRangefinder(const QString& portName_, QObject* parent)
    : QObject(parent),
      sp(new QSerialPort(this)),
      pollTimer(new QTimer(this)),
      portName(portName_)
{
    pollTimer->setInterval(app_config::kStereoRangefinderPollPeriodMs);
    connect(pollTimer, &QTimer::timeout, this, &StereoRangefinder::onPollTimer);

    connect(sp, &QSerialPort::readyRead, this, &StereoRangefinder::onReadyRead);
    connect(sp, &QSerialPort::errorOccurred, this, &StereoRangefinder::onError);
}

StereoRangefinder::~StereoRangefinder()
{
    stop();
}

QString StereoRangefinder::description() const
{
    return QString("%1 %2 8N1")
        .arg(portName)
        .arg(app_config::kStereoRangefinderBaudRate);
}

bool StereoRangefinder::start()
{
    if (!sp) {
        return false;
    }
    stopRequested = false;
    if (sp->isOpen()) {
        return true;
    }
    if (retryScheduled) {
        return false;
    }

    sp->setPortName(portName);
    sp->setBaudRate(app_config::kStereoRangefinderBaudRate);
    sp->setDataBits(QSerialPort::Data8);
    sp->setParity(QSerialPort::NoParity);
    sp->setStopBits(QSerialPort::OneStop);
    sp->setFlowControl(QSerialPort::NoFlowControl);

    if (!sp->open(QIODevice::ReadWrite)) {
        emit deviceStateChanged(false, description());
        emit errorText("[SRF] Can't open " + portName + ": " + sp->errorString());
        scheduleRetry(app_config::kStereoRangefinderReconnectDelayMs);
        return false;
    }

    rxBuffer.clear();
    crcFailCount = 0;
    retryScheduled = false;
    sp->clear(QSerialPort::AllDirections);

    emit deviceStateChanged(true, description());
    emit errorText("[SRF] Opened stereo rangefinder: " + description());
    std::cout << "[SRF] Port opened: " << description().toStdString() << std::endl;

    switchVideoStream(app_config::kStereoRangefinderUseRightStream);
    pollTimer->start();
    return true;
}

void StereoRangefinder::stop()
{
    stopRequested = true;
    retryScheduled = false;

    if (pollTimer) {
        pollTimer->stop();
    }

    if (sp && sp->isOpen()) {
        if (trackingActive) {
            stopTracking();
        }
        if (app_config::kStereoRangefinderSendStopProgramOnClose) {
            stopProgram();
        }
        sp->flush();
        emit deviceStateChanged(false, description());
        emit errorText("[SRF] Port closed: " + portName);
        sp->close();
    }
}

bool StereoRangefinder::isConnected() const
{
    return sp && sp->isOpen();
}

void StereoRangefinder::scheduleRetry(int delayMs)
{
    if (retryScheduled || stopRequested) {
        return;
    }

    retryScheduled = true;
    QTimer::singleShot(delayMs, this, [this]() {
        if (stopRequested) {
            retryScheduled = false;
            return;
        }
        retryScheduled = false;
        this->start();
    });
}

bool StereoRangefinder::sendCommand(std::uint8_t cmd, const QByteArray& data)
{
    if (!sp || !sp->isOpen()) {
        return false;
    }

    QByteArray payload(kDataSize, char(0));
    for (int i = 0; i < std::min(data.size(), kDataSize); ++i) {
        payload[i] = data[i];
    }

    QByteArray packet;
    packet.reserve(kPacketSize);
    packet.append(static_cast<char>(kStartByte));
    packet.append(static_cast<char>(cmd));

    std::uint8_t crc = cmd;
    for (int i = 0; i < kDataSize; ++i) {
        const std::uint8_t b = byteAt(payload, i);
        packet.append(static_cast<char>(b));
        crc = static_cast<std::uint8_t>(crc + b);
    }
    packet.append(static_cast<char>(crc));

    const qint64 written = sp->write(packet);
    if (written != packet.size()) {
        emit errorText(QString("[SRF] Short write cmd=0x%1 written=%2/%3")
                           .arg(static_cast<int>(cmd), 2, 16, QLatin1Char('0'))
                           .arg(written)
                           .arg(packet.size()));
        return false;
    }

    sp->flush();
    return true;
}

bool StereoRangefinder::sendTargetBox(const Box& box)
{
    QByteArray data(kDataSize, char(0));
    putU16BE(data, 0, box.x);
    putU16BE(data, 2, box.y);
    putU16BE(data, 4, box.w);
    putU16BE(data, 6, box.h);

    if (!sendCommand(kCmdSetInitialBox, data)) {
        return false;
    }

    std::cout << "[SRF] Sent target box x=" << box.x
              << " y=" << box.y
              << " w=" << box.w
              << " h=" << box.h
              << std::endl;
    return true;
}

bool StereoRangefinder::shouldRefreshBox(const Box& box, std::uint64_t nowMs) const
{
    if (!trackingActive || !haveLastSentBox) {
        return true;
    }
    if (nowMs - lastBoxSentMs < static_cast<std::uint64_t>(app_config::kStereoRangefinderBoxRefreshMs)) {
        return false;
    }

    const int lastCx = lastSentBox.x + lastSentBox.w / 2;
    const int lastCy = lastSentBox.y + lastSentBox.h / 2;
    const int cx = box.x + box.w / 2;
    const int cy = box.y + box.h / 2;
    const int moveThresholdX = std::max(
        app_config::kStereoRangefinderBoxRefreshMinMovePx,
        static_cast<int>(std::round(lastSentBox.w * app_config::kStereoRangefinderBoxRefreshMoveRatio))
    );
    const int moveThresholdY = std::max(
        app_config::kStereoRangefinderBoxRefreshMinMovePx,
        static_cast<int>(std::round(lastSentBox.h * app_config::kStereoRangefinderBoxRefreshMoveRatio))
    );

    return std::abs(cx - lastCx) >= moveThresholdX ||
           std::abs(cy - lastCy) >= moveThresholdY ||
           std::abs(box.w - lastSentBox.w) >= moveThresholdX ||
           std::abs(box.h - lastSentBox.h) >= moveThresholdY;
}

void StereoRangefinder::rememberBox(const Box& box, std::uint64_t nowMs)
{
    lastSentBox = box;
    lastBoxSentMs = nowMs;
    haveLastSentBox = true;
    trackingActive = true;
}

void StereoRangefinder::updateTargetBox(int x, int y, int width, int height)
{
    if (width <= 0 || height <= 0) {
        clearTarget();
        return;
    }

    if (!isConnected()) {
        start();
        return;
    }

    const Box box{x, y, width, height};
    const std::uint64_t now = monotonicMs();
    if (!shouldRefreshBox(box, now)) {
        return;
    }

    if (sendTargetBox(box)) {
        rememberBox(box, now);
        requestDistance();
    }
}

void StereoRangefinder::clearTarget()
{
    if (trackingActive) {
        stopTracking();
    }
    haveLastSentBox = false;
    lastBoxSentMs = 0;
}

void StereoRangefinder::requestDistance()
{
    QByteArray data(kDataSize, char(0));
    sendCommand(kCmdRequestDistance, data);
}

void StereoRangefinder::stopTracking()
{
    QByteArray data(kDataSize, char(0));
    if (sendCommand(kCmdStopTracking, data)) {
        std::cout << "[SRF] Sent stop tracking" << std::endl;
    }
    trackingActive = false;
}

void StereoRangefinder::stopProgram()
{
    QByteArray data(kDataSize, char(0));
    if (sendCommand(kCmdStopProgram, data)) {
        std::cout << "[SRF] Sent stop program" << std::endl;
    }
}

void StereoRangefinder::switchVideoStream(bool rightStream)
{
    QByteArray data(kDataSize, char(0));
    data[0] = static_cast<char>(rightStream ? 0x01 : 0x00);
    if (sendCommand(kCmdSwitchVideoStream, data)) {
        std::cout << "[SRF] Switched to "
                  << (rightStream ? "right" : "left")
                  << " video stream" << std::endl;
    }
}

void StereoRangefinder::onPollTimer()
{
    if (!isConnected()) {
        start();
        return;
    }
    if (!trackingActive) {
        return;
    }

    requestDistance();
}

void StereoRangefinder::onReadyRead()
{
    if (!sp || !sp->isOpen()) {
        return;
    }

    rxBuffer += sp->readAll();
    if (rxBuffer.size() > 512) {
        rxBuffer.clear();
        emit errorText("[SRF] RX buffer overflow, clearing");
        return;
    }

    while (rxBuffer.size() >= kPacketSize) {
        if (byteAt(rxBuffer, 0) != kStartByte) {
            rxBuffer.remove(0, 1);
            continue;
        }

        const QByteArray packet = rxBuffer.left(kPacketSize);
        rxBuffer.remove(0, kPacketSize);

        std::uint8_t crc = 0;
        for (int i = 1; i <= 9; ++i) {
            crc = static_cast<std::uint8_t>(crc + byteAt(packet, i));
        }

        if (crc != byteAt(packet, 10)) {
            ++crcFailCount;
            if (crcFailCount == 1 || crcFailCount % 10 == 0) {
                emit errorText(QString("[SRF] CRC mismatch count=%1").arg(crcFailCount));
            }
            continue;
        }

        crcFailCount = 0;
        handlePacket(packet);
    }
}

void StereoRangefinder::handlePacket(const QByteArray& packet)
{
    const std::uint8_t cmd = byteAt(packet, 1);
    const QByteArray data = packet.mid(2, kDataSize);

    if (cmd != kCmdRequestDistance) {
        return;
    }

    const double rawDistance = decodeDoubleBE(data);
    if (!std::isfinite(rawDistance) || rawDistance < 0.0) {
        emit errorText(QString("[SRF] Invalid distance value: %1").arg(rawDistance));
        return;
    }

    const double mmDouble = rawDistance * app_config::kStereoRangefinderDistanceToMm;
    if (mmDouble < 0.0 ||
        mmDouble > static_cast<double>(std::numeric_limits<int>::max()))
    {
        emit errorText(QString("[SRF] Distance out of int range: %1").arg(rawDistance));
        return;
    }

    const int mm = static_cast<int>(std::llround(mmDouble));
    emit distanceReady(rawDistance, mm);
}

void StereoRangefinder::onError(QSerialPort::SerialPortError error)
{
    if (error == QSerialPort::NoError) {
        return;
    }

    const QString text = sp ? sp->errorString() : QString("unknown serial error");
    emit errorText("[SRF] Serial error on " + portName + ": " + text);

    if (!sp || !sp->isOpen()) {
        emit deviceStateChanged(false, description());
        scheduleRetry(app_config::kStereoRangefinderReconnectDelayMs);
        return;
    }

    if (error == QSerialPort::ResourceError ||
        error == QSerialPort::DeviceNotFoundError ||
        error == QSerialPort::PermissionError)
    {
        sp->close();
        trackingActive = false;
        haveLastSentBox = false;
        emit deviceStateChanged(false, description());
        scheduleRetry(app_config::kStereoRangefinderReconnectDelayMs);
    }
}
