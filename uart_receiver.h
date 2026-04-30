#ifndef UART_RECEIVER_H
#define UART_RECEIVER_H

#include <QObject>
#include <QString>

#include <atomic>

class UartReceiver : public QObject
{
    Q_OBJECT
public:
    explicit UartReceiver(QObject* parent = nullptr);

public slots:
    void start();   // запуск чтения
    void stop();    // остановка

signals:
    void error(QString msg);
    void pixelsReceived(int x, int y, bool valid);

private:
    bool configurePort(int fd, const QString& devicePath);
    bool readFromOpenPort(int fd, const QString& devicePath);
    bool waitBeforeReconnect(int delay_ms);

    std::atomic<bool> m_stop{false};
};

#endif // UART_RECEIVER_H
