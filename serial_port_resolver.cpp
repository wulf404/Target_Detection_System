#include "serial_port_resolver.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QIODevice>
#include <QStringList>

namespace serial_ports {
namespace {

QString readTextFile(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return {};
    }

    return QString::fromLocal8Bit(file.readAll()).trimmed();
}

QString readAttrUp(QString sysfsPath, const QString& attrName)
{
    sysfsPath = QFileInfo(sysfsPath).canonicalFilePath();

    for (int depth = 0; depth < 8 && !sysfsPath.isEmpty(); ++depth) {
        const QString value = readTextFile(QDir(sysfsPath).filePath(attrName));
        if (!value.isEmpty()) {
            return value;
        }

        QDir dir(sysfsPath);
        if (!dir.cdUp()) {
            break;
        }

        const QString parent = dir.canonicalPath();
        if (parent.isEmpty() || parent == sysfsPath) {
            break;
        }
        sysfsPath = parent;
    }

    return {};
}

QString driverName(const QString& ttySysfsPath)
{
    const QFileInfo driverLink(QDir(ttySysfsPath).filePath("device/driver"));
    QString target = driverLink.symLinkTarget();
    if (target.isEmpty()) {
        target = driverLink.canonicalFilePath();
    }
    return QFileInfo(target).fileName();
}

bool containsCI(const QString& text, const QString& needle)
{
    return text.contains(needle, Qt::CaseInsensitive);
}

QString present(const QString& value)
{
    return value.isEmpty() ? QStringLiteral("?") : value;
}

int scoreForRole(const PortInfo& port, Role role)
{
    int score = 0;

    if (role == Role::CoordinateSource) {
        if (containsCI(port.driver, "cp210x")) score += 1000;
        if (containsCI(port.product, "cp210")) score += 500;
        if (containsCI(port.manufacturer, "silicon")) score += 150;
        if (port.idVendor.compare("10c4", Qt::CaseInsensitive) == 0) score += 150;
        if (port.idProduct.compare("ea60", Qt::CaseInsensitive) == 0) score += 150;
        if (containsCI(port.driver, "ftdi")) score -= 1000;
    } else {
        if (containsCI(port.product, "ft232bm")) score += 1000;
        if (containsCI(port.product, "ft232")) score += 500;
        if (containsCI(port.driver, "ftdi_sio")) score += 500;
        if (containsCI(port.manufacturer, "ftdi")) score += 150;
        if (port.idVendor.compare("0403", Qt::CaseInsensitive) == 0) score += 150;
        if (port.idProduct.compare("6001", Qt::CaseInsensitive) == 0) score += 150;
        if (containsCI(port.driver, "cp210x")) score -= 1000;
    }

    return score;
}

} // namespace

QVector<PortInfo> listUsbSerialPorts()
{
    QVector<PortInfo> ports;

    const QDir ttyDir("/sys/class/tty");
    const QStringList ttyNames = ttyDir.entryList(QStringList{"ttyUSB*"},
                                                  QDir::AllEntries | QDir::NoDotAndDotDot | QDir::System,
                                                  QDir::Name);

    for (const QString& ttyName : ttyNames) {
        const QString ttySysfsPath = ttyDir.filePath(ttyName);
        const QString deviceSysfsPath = QFileInfo(QDir(ttySysfsPath).filePath("device")).canonicalFilePath();
        if (deviceSysfsPath.isEmpty()) {
            continue;
        }

        PortInfo port;
        port.ttyName = ttyName;
        port.devicePath = "/dev/" + ttyName;
        port.driver = driverName(ttySysfsPath);
        port.product = readAttrUp(deviceSysfsPath, "product");
        port.manufacturer = readAttrUp(deviceSysfsPath, "manufacturer");
        port.idVendor = readAttrUp(deviceSysfsPath, "idVendor");
        port.idProduct = readAttrUp(deviceSysfsPath, "idProduct");
        port.serial = readAttrUp(deviceSysfsPath, "serial");

        ports.push_back(port);
    }

    return ports;
}

std::optional<PortInfo> findPort(Role role)
{
    const QVector<PortInfo> ports = listUsbSerialPorts();
    return findPort(role, ports);
}

std::optional<PortInfo> findPort(Role role, const QVector<PortInfo>& ports)
{
    int bestScore = 0;
    std::optional<PortInfo> best;
    for (const PortInfo& port : ports) {
        const int score = scoreForRole(port, role);
        if (score > bestScore) {
            bestScore = score;
            best = port;
        }
    }

    return best;
}

QString describe(const PortInfo& port)
{
    return QString("%1 driver=%2 product=%3 manufacturer=%4 vid=%5 pid=%6 serial=%7")
        .arg(port.devicePath)
        .arg(present(port.driver))
        .arg(present(port.product))
        .arg(present(port.manufacturer))
        .arg(present(port.idVendor))
        .arg(present(port.idProduct))
        .arg(present(port.serial));
}

QString describeList(const QVector<PortInfo>& ports)
{
    if (ports.isEmpty()) {
        return "no ttyUSB devices";
    }

    QStringList items;
    items.reserve(ports.size());
    for (const PortInfo& port : ports) {
        items.push_back(describe(port));
    }

    return items.join("; ");
}

} // namespace serial_ports
