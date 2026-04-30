#include "balancer.h"
#include <opencv2/highgui.hpp>
#include <thread>
#include <chrono>
#include <iostream>

Balancer::Balancer(QObject *parent) : QObject(parent)
{
    tp = new QThreadPool();
    tp->setMaxThreadCount(2);

    camera = new Camera(this, 1920);   //  FullHD/4K
    camera->setAutoDelete(false);

    connect(this, &Balancer::stopTasks,  camera, &Camera::Stop);
    connect(this, &Balancer::closeTasks, camera, &Camera::deleteLater);

    connect(camera, &Camera::output_frame, this, &Balancer::frames_watcher);
}

Balancer::~Balancer()
{
    if (tp) {
        tp->waitForDone(1000);
        tp->clear();
        delete tp;
    }
}

void Balancer::startAll()
{
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
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    emit closeTasks();
}

void Balancer::frames_watcher(const cv::Mat &frame_)
{
    emit ui_frame(frame_);
}
