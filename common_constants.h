#ifndef COMMON_CONSTANTS_H
#define COMMON_CONSTANTS_H

#include <QCoreApplication>
#include <QFile>
#include <QTextCodec>
#include <QTimer>
#include <QMutex>
#include <QWaitCondition>
#include <QThreadPool>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/fast_math.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <time.h>
#include <threads.h>
#include <thread>
#include <chrono>
#include <numeric>
#include <vector>
#include <string>
#include <random>

#include <cuda_runtime.h>


#ifndef ENABLE_CUDA_PREPROCESS
#define ENABLE_CUDA_PREPROCESS 1
#endif

#ifndef OPENCV_CPU_THREADS
#define OPENCV_CPU_THREADS 1
#endif

class TaskRunner : public QObject, public QRunnable
{
    Q_OBJECT
public:
    explicit TaskRunner(QObject *parent = nullptr) : QObject(parent){}

    virtual ~TaskRunner(){}

    virtual bool isStartBefore() const final { return started.load(); }
    virtual bool isStopped() const final     { return stopped.load(); }

    virtual void setStarted(bool value) final { started.fetchAndStoreAcquire(value); }
    virtual void setStopped(bool value) final { stopped.fetchAndStoreAcquire(value); }

public slots:
    virtual void Stop(){ stopped.fetchAndStoreAcquire(1); }

signals:
    void msgFrom(const int &class_id, const int &error_state);

protected:
    virtual void run() = 0;

private:
    QAtomicInt stopped;
    QAtomicInt started;
};

struct yolo_output
{
    std::vector<cv::Point> points;
    std::vector<int> area;
};

#endif // COMMON_CONSTANTS_H
