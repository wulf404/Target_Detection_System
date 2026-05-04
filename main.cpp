#include "mainwindow.h"

#include <QApplication>
#include <QMetaType>

#include <opencv2/core.hpp>
#include <linux/can.h>   // can_frame

// Чтобы Qt точно знал типы для queued connection
Q_DECLARE_METATYPE(cv::Mat)
Q_DECLARE_METATYPE(std::vector<cv::Mat>)
Q_DECLARE_METATYPE(cv::Rect)
Q_DECLARE_METATYPE(cv::Point)
Q_DECLARE_METATYPE(std::vector<cv::Point>)
Q_DECLARE_METATYPE(std::vector<int>)
Q_DECLARE_METATYPE(yolo_output)


int main(int argc, char *argv[])
{
    std::cout << "Setting up CAN and Fan..." << std::endl;
    system("bash /home/nick/Desktop/can_run_new.sh");
    //system("bash /home/nick/Desktop/pwm_max.sh");

    QApplication a(argc, argv);
    QApplication::setApplicationName("Target_Detection_System");
    QApplication::setApplicationDisplayName("Target Detection System");

    // Регистрация метатипов для сигналов/слотов между потоками
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<std::vector<cv::Mat>>("std::vector<cv::Mat>");
    qRegisterMetaType<cv::Rect>("cv::Rect");
    qRegisterMetaType<cv::Point>("cv::Point");
    qRegisterMetaType<std::vector<cv::Point>>("std::vector<cv::Point>");
    qRegisterMetaType<std::vector<int>>("std::vector<int>");
    qRegisterMetaType<yolo_output>("yolo_output");
    qRegisterMetaType<can_frame>("can_frame");

    MainWindow w;
    w.show();
    return a.exec();
}
