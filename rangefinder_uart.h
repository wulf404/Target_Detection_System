#pragma once
#include <QObject>
#include <QSerialPort>
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

private slots:
    void onReadyRead();
    void onError(QSerialPort::SerialPortError error);


private:
    QSerialPort* sp = nullptr;
    QByteArray buffer;
};
