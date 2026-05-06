#ifndef BALANCER_H
#define BALANCER_H

#include <QObject>
#include <QString>
#include <QThreadPool>

#include "camera.h"

class Balancer : public QObject
{
    Q_OBJECT
public:
    explicit Balancer(QObject *parent = nullptr);
    ~Balancer();

public slots:
    void closeAll();
    void startAll();
    void stopAll();

    void frames_watcher(const cv::Mat &frame_);

signals:
    void closeTasks();
    void stopTasks();

    // кадр для MainWindow
    void ui_frame(const cv::Mat &frame);
    void cameraDeviceStateChanged(bool connected, const QString& description);

private:
    QThreadPool *tp = nullptr;
    Camera* camera = nullptr;
};

#endif // BALANCER_H
