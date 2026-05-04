#include "balancer.h"
#include "app_config.h"
#include <opencv2/highgui.hpp>
#include <thread>
#include <chrono>
#include <iostream>

Balancer::Balancer(QObject *parent) : QObject(parent)
{
    tp = new QThreadPool();
    tp->setMaxThreadCount(2);

    camera = new Camera(this, app_config::kCameraRequestedWidth);
    camera->setAutoDelete(false);

    connect(this, &Balancer::stopTasks,  camera, &Camera::Stop);
    connect(this, &Balancer::closeTasks, camera, &Camera::deleteLater);

    connect(camera, &Camera::output_frame, this, &Balancer::frames_watcher);
}

Balancer::~Balancer()
{
    if (tp) {
        emit stopTasks();
        tp->waitForDone(2000);
        tp->clear();
        delete tp;
    }
}

void Balancer::startAll()
{
    if (!camera || camera->isStartBefore()) {
        return;
    }
    tp->start(camera);
}

void Balancer::stopAll()
{
    std::cerr << "STOP ALL\n";
    emit stopTasks();
}

void Balancer::closeAll()
{
    std::cerr << "CLOSE\n";
    cv::destroyAllWindows();

    emit stopTasks();
    tp->waitForDone(2000);
    emit closeTasks();
}

void Balancer::frames_watcher(const cv::Mat &frame_)
{
    emit ui_frame(frame_);
}
