#pragma once

#include <QString>
#include <QVector>

#include <optional>

namespace serial_ports {

enum class Role
{
    Rangefinder,
    CoordinateSource
};

struct PortInfo
{
    QString ttyName;
    QString devicePath;
    QString driver;
    QString product;
    QString manufacturer;
    QString idVendor;
    QString idProduct;
    QString serial;
};

QVector<PortInfo> listUsbSerialPorts();
std::optional<PortInfo> findPort(Role role);
std::optional<PortInfo> findPort(Role role, const QVector<PortInfo>& ports);
QString describe(const PortInfo& port);
QString describeList(const QVector<PortInfo>& ports);

} // namespace serial_ports
