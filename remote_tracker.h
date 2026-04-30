#pragma once
#include <QObject>

class RemoteTracker : public QObject
{
    Q_OBJECT
public:
    explicit RemoteTracker(QObject* parent=nullptr);

public slots:
    // Включить/выключить remote-управление (флаг “раз”)
    void setEnabled(bool en);

    // Теперь сюда приходят НЕ пиксели, а углы *100
    // x = azimuth * 100
    // y = elevation * 100
    void onPixels(int x, int y, bool valid);

private:
    bool m_enabled = true;
};
