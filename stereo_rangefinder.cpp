#include "stereo_rangefinder.h"

#include "app_config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <QString>

namespace {

constexpr std::uint8_t kStartByte = 0xFF;
constexpr std::uint8_t kCmdSetInitialBox = 0x01;
constexpr std::uint8_t kCmdRequestDistance = 0x02;
constexpr std::uint8_t kCmdStopTracking = 0x03;
constexpr std::uint8_t kCmdStopProgram = 0x04;
constexpr std::uint8_t kCmdSwitchVideoStream = 0x05;
constexpr int kPacketSize = 11;
constexpr int kDataSize = 8;

constexpr std::uint64_t kDistanceRequestDelayAfterBoxMs = 150;

std::uint64_t monotonicMs()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

std::uint8_t byteAt(const QByteArray& bytes, int index)
{
    return static_cast<std::uint8_t>(static_cast<unsigned char>(bytes.at(index)));
}

QString hexBytes(const QByteArray& bytes)
{
    return QString(bytes.toHex(' ')).toUpper();
}

void logSrf(const QString& text)
{
    (void)text;
}

void logSrf(const char* text)
{
    (void)text;
}

void logSrfImportant(const QString& text)
{
    std::cout << text.toStdString() << std::endl;
}

void logSrfImportant(const char* text)
{
    std::cout << text << std::endl;
}

const char* cmdName(std::uint8_t cmd)
{
    switch (cmd) {
    case kCmdSetInitialBox:
        return "SET_INITIAL_BOX";
    case kCmdRequestDistance:
        return "REQUEST_DISTANCE";
    case kCmdStopTracking:
        return "STOP_TRACKING";
    case kCmdStopProgram:
        return "STOP_PROGRAM";
    case kCmdSwitchVideoStream:
        return "SWITCH_VIDEO_STREAM";
    default:
        return "UNKNOWN";
    }
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

    connect(pollTimer, &QTimer::timeout,
            this, &StereoRangefinder::onPollTimer);

    connect(sp, &QSerialPort::readyRead,
            this, &StereoRangefinder::onReadyRead);

    connect(sp, &QSerialPort::errorOccurred,
            this, &StereoRangefinder::onError);
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
    logSrf("[SRF] start() called");

    if (!sp) {
        const QString text = "[SRF] start() failed: QSerialPort is null";
        logSrfImportant(text);
        emit errorText(text);
        return false;
    }

    stopRequested = false;

    if (sp->isOpen()) {
        logSrf("[SRF] start(): port already open");
        return true;
    }

    if (retryScheduled) {
        logSrf("[SRF] start(): retry already scheduled, skip");
        return false;
    }

    sp->setPortName(portName);
    sp->setBaudRate(app_config::kStereoRangefinderBaudRate);
    sp->setDataBits(QSerialPort::Data8);
    sp->setParity(QSerialPort::NoParity);
    sp->setStopBits(QSerialPort::OneStop);
    sp->setFlowControl(QSerialPort::NoFlowControl);

    logSrf("[SRF] Opening stereo rangefinder: " + description());

    if (!sp->open(QIODevice::ReadWrite)) {
        emit deviceStateChanged(false, description());

        const QString text = "[SRF] Can't open " + portName + ": " + sp->errorString();
        logSrfImportant(text);
        emit errorText(text);

        scheduleRetry(app_config::kStereoRangefinderReconnectDelayMs);
        return false;
    }

    rxBuffer.clear();
    crcFailCount = 0;
    retryScheduled = false;

    sp->clear(QSerialPort::AllDirections);

    emit deviceStateChanged(true, description());

    logSrf("[SRF] Opened stereo rangefinder: " + description());

    logSrf(QString("[SRF] Initial stream switch, rightStream=%1")
               .arg(app_config::kStereoRangefinderUseRightStream ? "true" : "false"));

    switchVideoStream(app_config::kStereoRangefinderUseRightStream);

    logSrf(QString("[SRF] Starting poll timer, period=%1 ms")
               .arg(app_config::kStereoRangefinderPollPeriodMs));

    pollTimer->start();

    return true;
}

void StereoRangefinder::stop()
{
    logSrf("[SRF] stop() called");

    stopRequested = true;
    retryScheduled = false;

    if (pollTimer) {
        pollTimer->stop();
        logSrf("[SRF] pollTimer stopped");
    }

    if (sp && sp->isOpen()) {
        if (trackingActive) {
            logSrf("[SRF] stop(): trackingActive=true -> stopTracking()");
            stopTracking();
        }

        if (app_config::kStereoRangefinderSendStopProgramOnClose) {
            logSrf("[SRF] stop(): sendStopProgramOnClose=true -> stopProgram()");
            stopProgram();
        }

        sp->flush();

        emit deviceStateChanged(false, description());

        logSrf("[SRF] Port closed: " + portName);

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

    logSrf(QString("[SRF] Reconnect scheduled after %1 ms").arg(delayMs));

    QTimer::singleShot(delayMs, this, [this]() {
        if (stopRequested) {
            retryScheduled = false;
            logSrf("[SRF] Retry cancelled: stopRequested=true");
            return;
        }

        retryScheduled = false;

        logSrf("[SRF] Retry start()");
        this->start();
    });
}

bool StereoRangefinder::sendCommand(std::uint8_t cmd, const QByteArray& data)
{
    if (!sp || !sp->isOpen()) {
        const QString text = QString("[SRF] TX failed: port is not open, CMD=0x%1 (%2)")
                                 .arg(static_cast<int>(cmd), 2, 16, QLatin1Char('0')).toUpper()
                                 .arg(QString::fromLatin1(cmdName(cmd)));
        logSrfImportant(text);
        emit errorText(text);
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

    logSrf(QString("[SRF] TX CMD=0x%1 (%2) PACKET=%3")
               .arg(static_cast<int>(cmd), 2, 16, QLatin1Char('0')).toUpper()
               .arg(QString::fromLatin1(cmdName(cmd)))
               .arg(hexBytes(packet)));

    const qint64 written = sp->write(packet);

    if (written != packet.size()) {
        const QString text = QString("[SRF] Short write CMD=0x%1 (%2), written=%3/%4")
                                 .arg(static_cast<int>(cmd), 2, 16, QLatin1Char('0')).toUpper()
                                 .arg(QString::fromLatin1(cmdName(cmd)))
                                 .arg(written)
                                 .arg(packet.size());
        logSrfImportant(text);
        emit errorText(text);
        return false;
    }

    const bool flushed = sp->flush();

    logSrf(QString("[SRF] TX DONE CMD=0x%1 (%2), WRITTEN=%3, FLUSH=%4")
               .arg(static_cast<int>(cmd), 2, 16, QLatin1Char('0')).toUpper()
               .arg(QString::fromLatin1(cmdName(cmd)))
               .arg(written)
               .arg(flushed ? "true" : "false"));

    return true;
}

bool StereoRangefinder::sendTargetBox(const Box& box)
{
    logSrf(QString("[SRF] sendTargetBox() x=%1 y=%2 w=%3 h=%4")
               .arg(box.x)
               .arg(box.y)
               .arg(box.w)
               .arg(box.h));

    QByteArray data(kDataSize, char(0));

    putU16BE(data, 0, box.x);
    putU16BE(data, 2, box.y);
    putU16BE(data, 4, box.w);
    putU16BE(data, 6, box.h);

    logSrf("[SRF] Target box DATA=" + hexBytes(data));

    if (!sendCommand(kCmdSetInitialBox, data)) {
        logSrf("[SRF] sendTargetBox() failed");
        return false;
    }

    logSrf(QString("[SRF] Sent target box x=%1 y=%2 w=%3 h=%4")
               .arg(box.x)
               .arg(box.y)
               .arg(box.w)
               .arg(box.h));

    return true;
}

bool StereoRangefinder::shouldRefreshBox(const Box& box, std::uint64_t nowMs) const
{
    (void)box;

    if (!trackingActive || !haveLastSentBox) {
        return true;
    }

    const std::uint64_t dt = nowMs - lastBoxSentMs;

    /*
     * Спокойный режим:
     * CMD 01 отправляется строго по таймеру.
     *
     * Это убирает порядок команд вида:
     * CMD 01 -> сразу CMD 02 -> снова CMD 01.
     *
     * Для проверки дальномера лучше сначала стабилизировать обмен.
     */
    return dt >= static_cast<std::uint64_t>(app_config::kStereoRangefinderBoxRefreshMs);
}

void StereoRangefinder::rememberBox(const Box& box, std::uint64_t nowMs)
{
    lastSentBox = box;
    lastBoxSentMs = nowMs;
    haveLastSentBox = true;
    trackingActive = true;

    logSrf(QString("[SRF] rememberBox(): trackingActive=true, x=%1 y=%2 w=%3 h=%4 time=%5")
               .arg(box.x)
               .arg(box.y)
               .arg(box.w)
               .arg(box.h)
               .arg(nowMs));
}

void StereoRangefinder::updateTargetBox(int x, int y, int width, int height)
{
    logSrf(QString("[SRF] updateTargetBox() x=%1 y=%2 w=%3 h=%4 connected=%5 trackingActive=%6 haveLastSentBox=%7")
               .arg(x)
               .arg(y)
               .arg(width)
               .arg(height)
               .arg(isConnected() ? "true" : "false")
               .arg(trackingActive ? "true" : "false")
               .arg(haveLastSentBox ? "true" : "false"));

    if (width <= 0 || height <= 0) {
        logSrf("[SRF] updateTargetBox(): invalid box size -> clearTarget()");
        clearTarget();
        return;
    }

    if (!isConnected()) {
        logSrf("[SRF] updateTargetBox(): not connected -> start()");
        start();
        return;
    }

    const Box box{x, y, width, height};
    const std::uint64_t now = monotonicMs();

    const bool refresh = shouldRefreshBox(box, now);

    if (!refresh) {
        const std::uint64_t dt = haveLastSentBox ? now - lastBoxSentMs : 0;

        logSrf(QString("[SRF] updateTargetBox(): skipped, shouldRefreshBox=false, dt=%1 ms, refreshPeriod=%2 ms")
                   .arg(dt)
                   .arg(app_config::kStereoRangefinderBoxRefreshMs));
        return;
    }

    const std::uint64_t dt = haveLastSentBox ? now - lastBoxSentMs : 0;

    logSrf(QString("[SRF] updateTargetBox(): shouldRefreshBox=true, dt=%1 ms -> sendTargetBox()")
               .arg(dt));

    if (sendTargetBox(box)) {
        rememberBox(box, now);

        /*
         * ВАЖНО:
         * Сразу после CMD 01 больше не отправляем CMD 02.
         *
         * Раньше было:
         * CMD 01
         * CMD 02 сразу после него
         *
         * Дальномер мог не успевать начать сопровождение
         * и возвращал 0. Теперь запрос дальности делает только pollTimer,
         * причём с задержкой после последнего CMD 01.
         */
        logSrf("[SRF] updateTargetBox(): box sent, distance request will be done by pollTimer");
    } else {
        logSrf("[SRF] updateTargetBox(): sendTargetBox() returned false");
    }
}

void StereoRangefinder::clearTarget()
{
    logSrf(QString("[SRF] clearTarget() called, trackingActive=%1")
               .arg(trackingActive ? "true" : "false"));

    if (trackingActive) {
        logSrf("[SRF] clearTarget(): trackingActive=true -> stopTracking()");
        stopTracking();
    }

    haveLastSentBox = false;
    lastBoxSentMs = 0;

    logSrf("[SRF] clearTarget(): haveLastSentBox=false, lastBoxSentMs=0");
}

void StereoRangefinder::requestDistance()
{
    logSrf(QString("[SRF] requestDistance() called, trackingActive=%1 connected=%2")
               .arg(trackingActive ? "true" : "false")
               .arg(isConnected() ? "true" : "false"));

    QByteArray data(kDataSize, char(0));

    if (!sendCommand(kCmdRequestDistance, data)) {
        logSrf("[SRF] requestDistance(): sendCommand() failed");
    }
}

void StereoRangefinder::stopTracking()
{
    logSrf(QString("[SRF] stopTracking() called, connected=%1")
               .arg(isConnected() ? "true" : "false"));

    QByteArray data(kDataSize, char(0));

    if (sendCommand(kCmdStopTracking, data)) {
        logSrf("[SRF] stopTracking(): CMD 03 sent");
    } else {
        logSrf("[SRF] stopTracking(): sendCommand() failed");
    }

    trackingActive = false;

    logSrf("[SRF] stopTracking(): trackingActive=false");
}

void StereoRangefinder::stopProgram()
{
    logSrf(QString("[SRF] stopProgram() called, connected=%1")
               .arg(isConnected() ? "true" : "false"));

    QByteArray data(kDataSize, char(0));

    if (sendCommand(kCmdStopProgram, data)) {
        logSrf("[SRF] stopProgram(): CMD 04 sent");
    } else {
        logSrf("[SRF] stopProgram(): sendCommand() failed");
    }
}

void StereoRangefinder::switchVideoStream(bool rightStream)
{
    logSrf(QString("[SRF] switchVideoStream() rightStream=%1")
               .arg(rightStream ? "true" : "false"));

    QByteArray data(kDataSize, char(0));
    data[0] = static_cast<char>(rightStream ? 0x01 : 0x00);

    if (sendCommand(kCmdSwitchVideoStream, data)) {
        logSrf(QString("[SRF] switchVideoStream(): CMD 05 sent, stream=%1")
                   .arg(rightStream ? "right" : "left"));
    } else {
        logSrf("[SRF] switchVideoStream(): sendCommand() failed");
    }
}

void StereoRangefinder::onPollTimer()
{
    if (!isConnected()) {
        logSrf("[SRF] onPollTimer(): not connected -> start()");
        start();
        return;
    }

    if (!trackingActive) {
        logSrf("[SRF] onPollTimer(): trackingActive=false, skip requestDistance()");
        return;
    }

    const std::uint64_t now = monotonicMs();

    /*
     * После отправки CMD 01 ждём 150 мс.
     * Это даёт дальномеру время принять рамку и начать сопровождение.
     */
    if (haveLastSentBox) {
        const std::uint64_t dtAfterBox = now - lastBoxSentMs;

        if (dtAfterBox < kDistanceRequestDelayAfterBoxMs) {
            logSrf(QString("[SRF] onPollTimer(): wait after CMD 01, dt=%1 ms, delay=%2 ms")
                       .arg(dtAfterBox)
                       .arg(kDistanceRequestDelayAfterBoxMs));
            return;
        }
    }

    logSrf("[SRF] onPollTimer(): trackingActive=true -> requestDistance()");
    requestDistance();
}

void StereoRangefinder::onReadyRead()
{
    if (!sp || !sp->isOpen()) {
        logSrf("[SRF] onReadyRead(): port is not open");
        return;
    }

    const QByteArray chunk = sp->readAll();

    logSrf(QString("[SRF] RX raw chunk size=%1 DATA=%2")
               .arg(chunk.size())
               .arg(hexBytes(chunk)));

    rxBuffer += chunk;

    logSrf(QString("[SRF] RX buffer size=%1 DATA=%2")
               .arg(rxBuffer.size())
               .arg(hexBytes(rxBuffer)));

    if (rxBuffer.size() > 512) {
        rxBuffer.clear();
        const QString text = "[SRF] RX buffer overflow, clearing";
        logSrfImportant(text);
        emit errorText(text);
        return;
    }

    while (rxBuffer.size() >= kPacketSize) {
        if (byteAt(rxBuffer, 0) != kStartByte) {
            logSrf(QString("[SRF] RX sync: drop byte 0x%1 before START")
                       .arg(byteAt(rxBuffer, 0), 2, 16, QLatin1Char('0')).toUpper());

            rxBuffer.remove(0, 1);
            continue;
        }

        const QByteArray packet = rxBuffer.left(kPacketSize);
        rxBuffer.remove(0, kPacketSize);

        logSrf("[SRF] RX packet candidate: " + hexBytes(packet));

        std::uint8_t crc = 0;

        for (int i = 1; i <= 9; ++i) {
            crc = static_cast<std::uint8_t>(crc + byteAt(packet, i));
        }

        const std::uint8_t receivedCrc = byteAt(packet, 10);

        if (crc != receivedCrc) {
            ++crcFailCount;

            if (crcFailCount == 1 || crcFailCount % 10 == 0) {
                const QString text =
                    QString("[SRF] CRC mismatch count=%1 expected=0x%2 received=0x%3")
                        .arg(crcFailCount)
                        .arg(crc, 2, 16, QLatin1Char('0')).toUpper()
                        .arg(receivedCrc, 2, 16, QLatin1Char('0')).toUpper();
                logSrfImportant(text);
                emit errorText(text);
            }

            continue;
        }

        crcFailCount = 0;

        logSrf("[SRF] RX packet OK: " + hexBytes(packet));

        handlePacket(packet);
    }
}

void StereoRangefinder::handlePacket(const QByteArray& packet)
{
    const std::uint8_t cmd = byteAt(packet, 1);
    const QByteArray data = packet.mid(2, kDataSize);

    logSrf(QString("[SRF] RX CMD=0x%1 (%2) DATA=%3")
               .arg(static_cast<int>(cmd), 2, 16, QLatin1Char('0')).toUpper()
               .arg(QString::fromLatin1(cmdName(cmd)))
               .arg(hexBytes(data)));

    switch (cmd) {
    case kCmdSetInitialBox: {
        const int x = (byteAt(data, 0) << 8) | byteAt(data, 1);
        const int y = (byteAt(data, 2) << 8) | byteAt(data, 3);
        const int w = (byteAt(data, 4) << 8) | byteAt(data, 5);
        const int h = (byteAt(data, 6) << 8) | byteAt(data, 7);

        logSrf(QString("[SRF] CMD 01 response: x=%1 y=%2 w=%3 h=%4")
                   .arg(x)
                   .arg(y)
                   .arg(w)
                   .arg(h));

        return;
    }

    case kCmdRequestDistance:
        break;

    case kCmdStopTracking:
        logSrf("[SRF] CMD 03 response: stop tracking ACK");
        return;

    case kCmdStopProgram:
        logSrf("[SRF] CMD 04 response: stop program ACK");
        return;

    case kCmdSwitchVideoStream:
        logSrf("[SRF] CMD 05 response: switch stream ACK");
        return;

    default:
        {
            const QString text = QString("[SRF] Unknown CMD=0x%1, ignored")
                                     .arg(static_cast<int>(cmd), 2, 16, QLatin1Char('0')).toUpper();
            logSrfImportant(text);
            emit errorText(text);
        }
        return;
    }

    const double rawDistance = decodeDoubleBE(data);

    logSrf(QString("[SRF] Decoded raw distance=%1")
               .arg(rawDistance, 0, 'f', 6));

    if (!std::isfinite(rawDistance) || rawDistance < 0.0) {
        const QString text = QString("[SRF] Invalid distance value: %1").arg(rawDistance);
        logSrfImportant(text);
        emit errorText(text);
        return;
    }

    const double mmDouble = rawDistance * app_config::kStereoRangefinderDistanceToMm;

    if (mmDouble < 0.0 ||
        mmDouble > static_cast<double>(std::numeric_limits<int>::max()))
    {
        const QString text = QString("[SRF] Distance out of int range: %1").arg(rawDistance);
        logSrfImportant(text);
        emit errorText(text);
        return;
    }

    const int mm = static_cast<int>(std::llround(mmDouble));

    logSrfImportant(QString("[SRF] Distance = %1 mm (%2 m)")
                        .arg(mm)
                        .arg(rawDistance, 0, 'f', 3));

    emit distanceReady(rawDistance, mm);
}

void StereoRangefinder::onError(QSerialPort::SerialPortError error)
{
    if (error == QSerialPort::NoError) {
        return;
    }

    const QString errorTextValue = sp ? sp->errorString() : QString("unknown serial error");
    const QString text = QString("[SRF] Serial error on %1: code=%2 text=%3")
                             .arg(portName)
                             .arg(static_cast<int>(error))
                             .arg(errorTextValue);
    logSrfImportant(text);
    emit errorText(text);

    if (!sp || !sp->isOpen()) {
        emit deviceStateChanged(false, description());
        scheduleRetry(app_config::kStereoRangefinderReconnectDelayMs);
        return;
    }

    if (error == QSerialPort::ResourceError ||
        error == QSerialPort::DeviceNotFoundError ||
        error == QSerialPort::PermissionError)
    {
        logSrf("[SRF] Critical serial error -> close port and schedule retry");

        sp->close();

        trackingActive = false;
        haveLastSentBox = false;

        emit deviceStateChanged(false, description());

        scheduleRetry(app_config::kStereoRangefinderReconnectDelayMs);
    }
}
