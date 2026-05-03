#pragma once
#include <QObject>
#include <QSerialPort>
#include <QString>
#include <QTimer>

class RangefinderUart : public QObject {
    Q_OBJECT
public:
    explicit RangefinderUart(const QString& port, QObject* parent = nullptr);
    ~RangefinderUart();

    bool start();   // открыть порт
    void stop();    // закрыть порт

    void startContinuous();   // отправить команду на непрерывные измерения
    void stopMeasurement();   // отправить команду на остановку

signals:
    void distanceReady(int mm); // готовое расстояние
    void errorText(const QString& text);
    void deviceStateChanged(bool connected, const QString& description);
    void usbDevicesChanged(const QString& summary);

private slots:
    void onReadyRead();
    void onError(QSerialPort::SerialPortError error);


private:
    void scheduleRetry(int delay_ms);
    void reportUsbPorts(const QString& summary);

    QSerialPort* sp = nullptr;
    QByteArray buffer;
    QString lastUsbSummary;
    QString lastSelectedPort;
    bool retryScheduled = false;
};
