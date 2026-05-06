#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWidget>
#include <experimental/filesystem>
#include "can_work.h"
#include <QTimer>
#include <QCloseEvent>
#include <QLabel>
#include <QImage>
#include <QPixmap>
#include <QMutex>
#include <opencv2/core.hpp>

#include <cstdint>

#include <balancer.h>
#include <rangefinder_uart.h>
#include "ui_mainwindow.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class QThread;
class UartReceiver;
class QGroupBox;

namespace fs = std::experimental::filesystem;

#define MY_CONSOLE_A ui->textBrowser->appendPlainText
#define MY_CONSOLE_I ui->textBrowser->insertPlainText

class MainWindow : public QMainWindow
{
    Q_OBJECT

protected:
    void closeEvent(QCloseEvent *ev) override
    {

        if (videoWindow) {
            videoWindow->close();
        }
        shutdownServices();
        closeWindow();
        ev->accept();
    }

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

signals:
    void setReadState(bool state);
    void putFrame(const can_frame &frame);
    void startCanRecv(bool flag);
    void closeWindow();

public slots:
    void showFrame(const can_frame &frame);
    void consoleAppendText(QString str);
    void consoleInsertText(QString str);
    void onNewFrame(const cv::Mat &frame);

private slots:
    void on_pbClear_clicked();
    void on_pbSend_clicked();
    void on_pbTest_clicked();
    void on_pbSet_clicked();
    void on_cbShowTest_stateChanged(int arg1);
    void on_pbTestSend_clicked();
    void on_pbTestReceive_clicked();
    void on_pbReceive_clicked();

    void showDistance(int mm);
    void showStatus(const QString& msg);

    void sendX10Frame();
    void sendX05Frame();
    void sendX0FFrame();
    void refreshTrackingHealth();

    void on_textBrowser_textChanged();
    void on_cmbID_currentIndexChanged(int index);

    void on_start_pressed();
    void on_stop_pressed();

private:
    Ui::MainWindow *ui = nullptr;

    QTimer *timerX10 = nullptr;
    QTimer *timerX05 = nullptr;
    QTimer *timerX0F = nullptr;
    QTimer *healthTimer = nullptr;

    void initCan();
    void shutdownServices();
    void setupSystemStatusPanel();
    void updateSystemStatusPanel();

    RangefinderUart* rf = nullptr;

    can_work *canReaderWriter = nullptr;
    QThreadPool *threadPool = nullptr;
    QThread *uartThread = nullptr;
    UartReceiver *uartReceiver = nullptr;

    std::vector<QString> can_devices;
    bool isRecieve = false;

    Balancer *balancer = nullptr;

    // ===== Video separate window =====
    QWidget* videoWindow = nullptr;
    QLabel*  view = nullptr;
    QMutex   draw_mutex;

    QGroupBox* systemStatusGroup = nullptr;
    QLabel* statusOverall = nullptr;
    QLabel* statusCan = nullptr;
    QLabel* statusCamera = nullptr;
    QLabel* statusTargetSource = nullptr;
    QLabel* statusYoloTarget = nullptr;
    QLabel* statusExternal = nullptr;
    QLabel* statusRangefinder = nullptr;
    QLabel* statusDistance = nullptr;
    QLabel* statusTurret = nullptr;

    bool externalDeviceConnected = false;
    bool rangefinderConnected = false;
    bool cameraDeviceConnected = false;
    QString externalDeviceDescription = "not found";
    QString rangefinderDescription = "not found";
    QString cameraDeviceDescription = "not found";
    uint64_t lastVideoFrameMs = 0;

    static QImage matToQImageRGB(const cv::Mat &bgr);
};

#endif // MAINWINDOW_H
