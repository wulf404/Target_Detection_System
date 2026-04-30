#include "uart_receiver.h"
#include <iostream>
#include <iomanip>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <cstdint>

UartReceiver::UartReceiver(QObject* parent)
    : QObject(parent)
{
}

void UartReceiver::stop()
{
    m_stop = true;
}

void UartReceiver::start()
{


    std::cerr << "[UART] start() entered\n";
    const char* device = "/dev/ttyUSB1";
    int fd = open(device, O_RDONLY | O_NOCTTY);
    if (fd < 0) {
        emit error(QString("Ошибка открытия %1: %2").arg(device).arg(strerror(errno)));
        return;
    }

    struct termios tty{};
    tcgetattr(fd, &tty);

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

    tcsetattr(fd, TCSANOW, &tty);

    std::cout << "[UART] Приём координат запущен" << std::endl;

    uint8_t buf[4];
    size_t received = 0;

    while (!m_stop)
    {
        ssize_t n = read(fd, buf + received, 4 - received);
        if (n > 0)
        {
            received += n;
            if (received == 4)
            {
                received = 0;

                int16_t x = (int16_t)((buf[0] << 8) | buf[1]);
                int16_t y = (int16_t)((buf[2] << 8) | buf[3]);

                if (x == -1 && y == -1) {
                    emit pixelsReceived(0, 0, false);
                    continue;
                }

                emit pixelsReceived(x, y, true);
                //std::cout << "[UART] x=" << std::setw(6) << x
                //          << "  y=" << std::setw(6) << y << std::endl;
            }
        }
        else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
        {
            emit error(QString("Ошибка чтения: %1").arg(strerror(errno)));
            break;
        }
    }

    close(fd);
    std::cout << "[UART] Приём остановлен" << std::endl;
}
