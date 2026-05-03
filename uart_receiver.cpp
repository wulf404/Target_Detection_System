#include "uart_receiver.h"
#include "serial_port_resolver.h"

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <termios.h>
#include <thread>
#include <unistd.h>

namespace {
constexpr int RECONNECT_DELAY_MS = 1000;
constexpr int POLL_TIMEOUT_MS = 50;
}

UartReceiver::UartReceiver(QObject* parent)
    : QObject(parent)
{
}

void UartReceiver::stop()
{
    m_stop.store(true, std::memory_order_relaxed);
}

bool UartReceiver::waitBeforeReconnect(int delay_ms)
{
    constexpr int STEP_MS = 50;
    int waited_ms = 0;

    while (!m_stop.load(std::memory_order_relaxed) && waited_ms < delay_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(STEP_MS));
        waited_ms += STEP_MS;
    }

    return !m_stop.load(std::memory_order_relaxed);
}

bool UartReceiver::configurePort(int fd, const QString& devicePath)
{
    termios tty{};
    if (tcgetattr(fd, &tty) != 0) {
        emit error(QString("tcgetattr %1 failed: %2").arg(devicePath).arg(strerror(errno)));
        return false;
    }

    cfsetispeed(&tty, B921600);
    cfsetospeed(&tty, B921600);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag |= CLOCAL | CREAD;
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        emit error(QString("tcsetattr %1 failed: %2").arg(devicePath).arg(strerror(errno)));
        return false;
    }

    return true;
}

bool UartReceiver::readFromOpenPort(int fd, const QString& devicePath)
{
    std::uint8_t buf[4]{};
    std::size_t received = 0;

    while (!m_stop.load(std::memory_order_relaxed))
    {
        pollfd pfd{};
        pfd.fd = fd;
        pfd.events = POLLIN;

        const int ready = poll(&pfd, 1, POLL_TIMEOUT_MS);
        if (ready == 0) {
            continue;
        }
        if (ready < 0) {
            if (errno == EINTR) continue;
            emit error(QString("poll %1 failed: %2").arg(devicePath).arg(strerror(errno)));
            return true;
        }

        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
            emit error(QString("%1 disconnected").arg(devicePath));
            return true;
        }
        if (!(pfd.revents & POLLIN)) {
            continue;
        }

        const ssize_t n = read(fd, buf + received, sizeof(buf) - received);
        if (n > 0)
        {
            received += static_cast<std::size_t>(n);
            if (received == sizeof(buf))
            {
                received = 0;

                const auto raw_x = static_cast<std::uint16_t>((buf[0] << 8) | buf[1]);
                const auto raw_y = static_cast<std::uint16_t>((buf[2] << 8) | buf[3]);
                const auto x = static_cast<std::int16_t>(raw_x);
                const auto y = static_cast<std::int16_t>(raw_y);

                if (x == -1 && y == -1) {
                    emit pixelsReceived(0, 0, false);
                    continue;
                }

                emit pixelsReceived(x, y, true);
            }
        }
        else if (n == 0) {
            emit error(QString("%1 returned EOF").arg(devicePath));
            return true;
        }
        else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
        {
            emit error(QString("read %1 failed: %2").arg(devicePath).arg(strerror(errno)));
            return true;
        }
    }

    return false;
}

void UartReceiver::start()
{
    m_stop.store(false, std::memory_order_relaxed);

    std::cerr << "[UART] receiver loop entered\n";
    bool reported_wait = false;
    QString last_usb_summary;
    QString last_selected_port;
    QString last_open_error;

    while (!m_stop.load(std::memory_order_relaxed))
    {
        const QVector<serial_ports::PortInfo> ports = serial_ports::listUsbSerialPorts();
        const QString usb_summary = serial_ports::describeList(ports);
        if (usb_summary != last_usb_summary) {
            last_usb_summary = usb_summary;
            emit usbDevicesChanged(usb_summary);
            emit status("[USB] ttyUSB devices: " + usb_summary);
        }

        const auto port = serial_ports::findPort(serial_ports::Role::CoordinateSource, ports);
        if (!port) {
            if (!reported_wait) {
                emit deviceStateChanged(false, "cp210x not found");
                emit status(QString("[UART] Waiting for cp210x coordinate source USB serial port"));
                reported_wait = true;
            }
            waitBeforeReconnect(RECONNECT_DELAY_MS);
            continue;
        }

        reported_wait = false;

        const QString selectedPort = serial_ports::describe(*port);
        if (selectedPort != last_selected_port) {
            last_selected_port = selectedPort;
            emit status("[UART] Selected coordinate source: " + selectedPort);
        }

        const QByteArray devicePathBytes = port->devicePath.toLocal8Bit();
        const int fd = open(devicePathBytes.constData(), O_RDONLY | O_NOCTTY | O_NONBLOCK);
        if (fd < 0) {
            emit deviceStateChanged(false, selectedPort);
            const QString open_error = QString("Waiting for %1: %2")
                                           .arg(port->devicePath)
                                           .arg(strerror(errno));
            if (open_error != last_open_error) {
                last_open_error = open_error;
                emit error(open_error);
            }
            waitBeforeReconnect(RECONNECT_DELAY_MS);
            continue;
        }

        last_open_error.clear();

        if (!configurePort(fd, port->devicePath)) {
            close(fd);
            waitBeforeReconnect(RECONNECT_DELAY_MS);
            continue;
        }

        std::cout << "[UART] "
                  << serial_ports::describe(*port).toStdString()
                  << " opened" << std::endl;
        emit deviceStateChanged(true, selectedPort);
        emit status("[UART] Opened coordinate source: " + selectedPort);
        const bool should_reconnect = readFromOpenPort(fd, port->devicePath);
        close(fd);

        if (m_stop.load(std::memory_order_relaxed)) {
            break;
        }

        if (should_reconnect) {
            emit deviceStateChanged(false, selectedPort);
            emit status("[UART] Coordinate source disconnected, rescanning USB");
            emit pixelsReceived(0, 0, false);
            waitBeforeReconnect(RECONNECT_DELAY_MS);
        }
    }

    std::cout << "[UART] receiver loop stopped" << std::endl;
}
