#pragma once

#include <QObject>
#include <QByteArray>
#include <QSerialPort>
#include <QString>
#include <QTimer>

#include <cstdint>

class StereoRangefinder : public QObject
{
    Q_OBJECT
public:
    explicit StereoRangefinder(const QString& portName, QObject* parent = nullptr);
    ~StereoRangefinder() override;

    bool start();
    void stop();
    bool isConnected() const;

public slots:
    void updateTargetBox(int x, int y, int width, int height);
    void clearTarget();
    void requestDistance();
    void stopTracking();
    void stopProgram();
    void switchVideoStream(bool rightStream);

signals:
    void distanceReady(double distance, int distanceMm);
    void errorText(const QString& text);
    void deviceStateChanged(bool connected, const QString& description);

private slots:
    void onReadyRead();
    void onError(QSerialPort::SerialPortError error);
    void onPollTimer();

private:
    struct Box
    {
        int x = 0;
        int y = 0;
        int w = 0;
        int h = 0;
    };

    void scheduleRetry(int delayMs);
    bool sendTargetBox(const Box& box);
    bool sendCommand(std::uint8_t cmd, const QByteArray& data);
    bool shouldRefreshBox(const Box& box, std::uint64_t nowMs) const;
    void rememberBox(const Box& box, std::uint64_t nowMs);
    void handlePacket(const QByteArray& packet);
    QString description() const;

    QSerialPort* sp = nullptr;
    QTimer* pollTimer = nullptr;
    QByteArray rxBuffer;
    QString portName;
    Box lastSentBox;
    std::uint64_t lastBoxSentMs = 0;
    int crcFailCount = 0;
    bool retryScheduled = false;
    bool stopRequested = false;
    bool haveLastSentBox = false;
    bool trackingActive = false;
};
