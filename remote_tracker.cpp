#include "remote_tracker.h"

#include "target_manager.h"

RemoteTracker::RemoteTracker(QObject* parent) : QObject(parent) {}

void RemoteTracker::setEnabled(bool en)
{
    m_enabled = en;
}

void RemoteTracker::onPixels(int x, int y, bool valid)
{
    if (!m_enabled) return;
    TargetManager::submitExternalAnglesCentideg(x, y, valid);
}
