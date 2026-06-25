#include "rangefinder_uart.h"
#include "serial_port_resolver.h"
#include <iostream>
#include <QThread>
#include <QTimer>

RangefinderUart::RangefinderUart(const QString& port, QObject* parent)
    : QObject(parent), sp(new QSerialPort(this))
{
    if (port != "auto") {
        const QString dev = port.startsWith("/dev/") ? port : ("/dev/" + port);
        sp->setPortName(dev);
    }
    sp->setBaudRate(QSerialPort::Baud115200);
    sp->setDataBits(QSerialPort::Data8);
    sp->setParity(QSerialPort::NoParity);
    sp->setStopBits(QSerialPort::OneStop);
    sp->setFlowControl(QSerialPort::NoFlowControl);

    connect(sp, &QSerialPort::readyRead, this, &RangefinderUart::onReadyRead);
    connect(sp, &QSerialPort::errorOccurred, this, &RangefinderUart::onError);
}

RangefinderUart::~RangefinderUart() {
    if (sp && sp->isOpen()) sp->close();
}

void RangefinderUart::scheduleRetry(int delay_ms)
{
    if (retryScheduled) {
        return;
    }

    retryScheduled = true;
    QTimer::singleShot(delay_ms, this, [this]() {
        retryScheduled = false;
        this->start();
    });
}

void RangefinderUart::reportUsbPorts(const QString& summary)
{
    if (summary == lastUsbSummary) {
        return;
    }

    lastUsbSummary = summary;
    emit usbDevicesChanged(summary);
    emit errorText("[USB] ttyUSB devices: " + summary);
}

bool RangefinderUart::start()
{
    if (!sp) return false;

    if (sp->isOpen()) {
        std::cout << "[RF] Port already open:" << std::endl;
        return true;
    }

    const QVector<serial_ports::PortInfo> ports = serial_ports::listUsbSerialPorts();
    reportUsbPorts(serial_ports::describeList(ports));

    const auto port = serial_ports::findPort(serial_ports::Role::Rangefinder, ports);
    if (!port) {
        emit deviceStateChanged(false, "FT232BM not found");
        emit errorText("[RF] Waiting for FT232BM rangefinder USB serial port");
        scheduleRetry(1000);
        return false;
    }

    const QString selectedPort = serial_ports::describe(*port);
    if (selectedPort != lastSelectedPort) {
        lastSelectedPort = selectedPort;
        emit errorText("[RF] Selected rangefinder: " + selectedPort);
    }

    sp->setPortName(port->devicePath);
    sp->setBaudRate(QSerialPort::Baud115200);
    sp->setDataBits(QSerialPort::Data8);
    sp->setParity(QSerialPort::NoParity);
    sp->setStopBits(QSerialPort::OneStop);
    sp->setFlowControl(QSerialPort::NoFlowControl);

    QThread::msleep(200);                 // дать чипу инициализироваться
    if (!sp->open(QIODevice::ReadWrite)) {
        emit deviceStateChanged(false, selectedPort);
        emit errorText("[RF] Can't open " + sp->portName() + ": " + sp->errorString());
        scheduleRetry(1000);
        return false;
    }

    sp->clear(QSerialPort::AllDirections);  // очистить мусор в RX/TX
    QThread::msleep(100);

    retryScheduled = false;
    emit deviceStateChanged(true, selectedPort);
    emit errorText("[RF] Opened rangefinder: " + selectedPort);
    std::cout << "[RF] Port opened: "
              << serial_ports::describe(*port).toStdString()
              << std::endl;
    this->startContinuous();
    return true;
}

void RangefinderUart::stop() {
    if (sp && sp->isOpen()) {
        emit deviceStateChanged(false, sp->portName());
        emit errorText("[RF] Port closed: " + sp->portName());
        sp->close();
        std::cout << "[RF] Port closed" << std::endl;
    }
}

void RangefinderUart::startContinuous() {
    if (!sp || !sp->isOpen()) {
        emit errorText("Port not open, can't startContinuous()");
        return;
    }
    QByteArray cmd;
    cmd.append(char(0xFA));
    cmd.append(char(0x01));
    cmd.append(char(0xFF));
    cmd.append(char(0x04));
    cmd.append(char(0x01));
    cmd.append(char(0x00));
    cmd.append(char(0x00));
    cmd.append(char(0x00));
    cmd.append(char(0x00));
    cmd.append(char(0xFF));

    uint8_t crc = 0;
    for (int i = 0; i < cmd.size(); ++i) crc += (uint8_t)cmd[i];
    cmd.append(crc);

    sp->write(cmd);
    sp->flush();
    std::cout << "[RF] Sent Start Continuous command" << std::endl;
}


void RangefinderUart::stopMeasurement() {
    if (!sp || !sp->isOpen()) {
        emit errorText("Port not open, can't stopMeasurement()");
        return;
    }
    QByteArray cmd;
    cmd.append(char(0xFA));
    cmd.append(char(0x01));
    cmd.append(char(0xFF));
    cmd.append(char(0x04));
    cmd.append(char(0x00));
    cmd.append(char(0x00));
    cmd.append(char(0x00));
    cmd.append(char(0x00));
    cmd.append(char(0x00));
    cmd.append(char(0xFE));

    uint8_t crc = 0;
    for (int i = 0; i < cmd.size(); ++i) crc += (uint8_t)cmd[i];
    cmd.append(crc);

    sp->write(cmd);
    sp->flush();
    std::cout << "[RF] Sent Stop Measurement command" << std::endl;
}

void RangefinderUart::onReadyRead()
{
    if (!sp || !sp->isOpen())
        return;

    buffer += sp->readAll();

    // если буфер слишком большой — сбрасываем мусор
    if (buffer.size() > 256) {
        //qWarning() << "[RF] Buffer overflow, clearing garbage";
        buffer.clear();
        return;
    }

    // ищем валидный пакет
    while (buffer.size() >= 9) {
        // ищем байт начала пакета
        if ((uint8_t)buffer[0] != 0xFB) {
            buffer.remove(0, 1);
            continue;
        }

        // ждём пока придёт весь пакет
        if (buffer.size() < 9)
            return;

        QByteArray pkt = buffer.left(9);
        buffer.remove(0, 9);

        // вычисляем CRC
        uint8_t crc_calc = 0;
        for (int i = 0; i < 8; ++i)
            crc_calc += (uint8_t)pkt[i];
        uint8_t crc_recv = (uint8_t)pkt[8];

        // защита от ошибочных пакетов
        static int crc_fail_count = 0;
        if (crc_calc != crc_recv) {
            crc_fail_count++;
            //qWarning() << "[RF] CRC mismatch:" << crc_calc << crc_recv;
            if (crc_fail_count > 5) {
                //qWarning() << "[RF] Too many CRC errors, restarting port...";
                crc_fail_count = 0;
                emit errorText("[RF] Too many CRC errors on " + sp->portName() + ", reopening");
                emit deviceStateChanged(false, sp->portName());
                sp->close();
                scheduleRetry(500);
            }
            continue;
        }
        crc_fail_count = 0;

        uint8_t msgType = (uint8_t)pkt[0];
        uint8_t msgCode = (uint8_t)pkt[1];
        uint16_t dist = ((uint8_t)pkt[7] << 8) | (uint8_t)pkt[6];

        if (msgType == 0xFB && msgCode == 0x03) {
            if (dist == 0) {
               emit measurementStateChanged("NO OBJECT");
               // qDebug() << "[RF] No object";
            } else if (dist == 0xFFFF) {
                emit measurementStateChanged("OPEN SPACE");
                //qDebug() << "[RF] Open space";
            } else {
                int mm = dist * 100;
                emit measurementStateChanged(QString("DATA %1 mm").arg(mm));
                emit distanceReady(mm);
            }
        } else {
            //qDebug() << "[RF RAW]" << pkt.toHex();
        }
    }
}


void RangefinderUart::onError(QSerialPort::SerialPortError err) {
    if (err == QSerialPort::NoError) return;

    const QString portName = sp ? sp->portName() : QString("?");
    const QString errorString = sp ? sp->errorString() : QString("unknown serial error");
    emit deviceStateChanged(false, portName);
    emit errorText("[RF] Serial error on " + portName + ": " + errorString + ". Rescanning USB");

    std::cerr << "[RF ERROR] " << errorString.toStdString() << std::endl;


    // Автоматически переподключимся через секунду
    if (sp) {
        sp->close();
    }
    scheduleRetry(1000);
}
