#ifndef CAMERA_H
#define CAMERA_H

#include <my_yolo.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

#include <QString>
#include <string>
#include <vector>

class QTimer;

class Camera : public TaskRunner
{
    Q_OBJECT
public:
    explicit Camera(QObject *parent = nullptr, int width = 3840);
    ~Camera() override;

    void setInferEveryN(int n) { infer_every_n = (n <= 0 ? 1 : n); }
    void setYoloEnabled(bool enabled) { yolo_enabled = enabled; }
    void setRequestedWidth(int width);

signals:
    void output_frame(const cv::Mat &frame);
    void deviceStateChanged(bool connected, const QString& description);

protected:
    void run() override;

private:
    void applyUsbDefaults();

    bool openDevice();
    void closeDevice();
    std::vector<std::string> cameraDeviceCandidates() const;
    std::string buildUsbPipeline(const std::string& devicePath) const;
    void publishDeviceState(bool connected, const std::string& description);

private:
    QTimer *reconnectTimer = nullptr;

    int camera_width  = 3840;
    int camera_height = 2160;
    int camera_fps    = 60;

    int infer_every_n = 1;
    bool yolo_enabled = true;

    my_yolo *Yolo = nullptr;
    cv::VideoCapture video;
    std::string current_device_path;
    bool camera_device_connected = false;
    QString camera_device_description = "not found";

public slots:
    void checkAndReconnect();
};

#endif // CAMERA_H


//#ifndef CAMERA_H
//#define CAMERA_H

//#include <my_yolo.h>

//class Camera : public TaskRunner
//{
//    Q_OBJECT
//public:
//    explicit Camera(QObject *parent = nullptr, int width = 1920);
//    ~Camera() override;

//    // NEW: инференс раз в N кадров
//    void setInferEveryN(int n) { infer_every_n = (n <= 0 ? 1 : n); }

//signals:
//    void output_frame(const cv::Mat &frame);

//protected:
//    void run() override;

//private:
//    QTimer *reconnectTimer = nullptr;

//    int camera_width = 1920;
//    int camera_height = 1080;
//    int camera_fps = 100;

//    // NEW
//    int infer_every_n = 1;

//    my_yolo *Yolo = nullptr;
//    cv::VideoCapture video;

//    bool openDevice();
//    void closeDevice();

//public slots:
//    void checkAndReconnect();
//};

//#endif // CAMERA_H



////#ifndef CAMERA_H
////#define CAMERA_H

////#include <my_yolo.h>
////#include <opencv2/core/mat.hpp>
////#include <opencv2/videoio.hpp>

////class QTimer;

////class Camera : public TaskRunner
////{
////    Q_OBJECT
////public:
////    enum class Source
////    {
////        USB_V4L2,
////        CSI_ARGUS
////    };

////    explicit Camera(QObject *parent = nullptr, int width = 1920);
////    ~Camera() override;

////    // Инференс раз в N кадров
////    void setInferEveryN(int n) { infer_every_n = (n <= 0 ? 1 : n); }

////    // Переключение источника
////    void setSource(Source src);

////    // Настройки CSI
////    void setSensorId(int id)      { csi_sensor_id = (id < 0 ? 0 : id); }
////    void setFlipMethod(int value) { csi_flip_method = (value < 0 ? 0 : value); }

////    // Временное отключение YOLO для диагностики камеры
////    void setYoloEnabled(bool enabled) { yolo_enabled = enabled; }

////    // Переустановить желаемую ширину и автоматически подобрать режим
////    void setRequestedWidth(int width);

////signals:
////    void output_frame(const cv::Mat &frame);

////protected:
////    void run() override;

////private:
////    void applySourceDefaults();

////    bool openDevice();
////    bool openUsbDevice();
////    bool openCsiDevice();

////    void closeDevice();

////private:
////    QTimer *reconnectTimer = nullptr;

////    Source camera_source = Source::CSI_ARGUS;

////    int camera_width  = 1920;
////    int camera_height = 1080;
////    int camera_fps    = 60;

////    int infer_every_n = 1;

////    bool yolo_enabled   = true;
////    int  csi_sensor_id  = 0;
////    int  csi_flip_method = 2;

////    my_yolo *Yolo = nullptr;
////    cv::VideoCapture video;

////public slots:
////    void checkAndReconnect();
////};

////#endif // CAMERA_H
