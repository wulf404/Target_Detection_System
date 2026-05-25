#ifndef CAN_WORK_H
#define CAN_WORK_H

#include <QObject>
#include <QRunnable>
#include <QThreadPool>
#include <atomic>
#include <cstdint>
#include <deque>
#include <iostream>
#include <mutex>

#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>

#include <linux/can.h>
#include <linux/can/raw.h>

#include "can_commands.h"

using namespace numbers;

#define SOCKETCAN_ERROR   -1
#define SOCKETCAN_TIMEOUT -2

typedef struct {
    uint32_t size; //!< in bytes (1=unit8_t, 2=uint16_t, 3=unit24_t, 4=uint32_t)
    uint32_t data;
} Socketcan_t;

Q_DECLARE_METATYPE(can_frame);


class can_work : public QObject, public QRunnable
{
    Q_OBJECT
public:
    can_work(const char* canName);
    ~can_work() override;

    bool openCan();
    void closeCan();
    bool isRunning() const;


    void parceMsg(const can_frame &frame);
    void analysisMsg();
    void writeCanTracked(const can_frame &frame, std::uint64_t latencyToken);

public slots:
    void setStopFlag(bool _stop);
    void writeCan(const can_frame &frame);
    void startLoop(bool startFlag);

signals:
    void appendConsole(QString msg);
    void insertConsole(QString msg);
    void frameReady(const can_frame &frame);
    void startParce(const can_frame &frame);

private:
    struct TxEntry
    {
        can_frame frame{};
        std::uint64_t latencyToken = 0;
    };

    int mSock;
    std::atomic<bool> isOpen;
    std::atomic<bool> isRecv;
    std::atomic<bool> running;
    std::string m_canName;
    struct can_frame mFrame;

    std::atomic<bool> stop_flag;
    std::mutex txMutex;
    std::deque<TxEntry> txQueue;

    uchar msgID;
    uchar msgDLC;
    std::vector<uchar> msgData;
    std::vector<uchar> msgData_bin;
    std::vector<uchar> tempData_bin;
    std::vector<QString> msgData_str;

    virtual void run() override;
    bool writeCanNow(const can_frame &frame);
    void drainTxQueue();
};


#endif // CAN_WORK_H
