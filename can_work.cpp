#include "can_work.h"
#include "tower_state.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <thread>

static uint16_t readBitsBE(const std::vector<uchar>& bits, int start, int count)
{
    if (start < 0 || count < 0 || start + count > static_cast<int>(bits.size())) {
        return 0;
    }

    uint16_t value = 0;
    for (int i = 0; i < count; ++i) {
        if (bits[start + i]) {
            value |= static_cast<uint16_t>(1u << (count - 1 - i));
        }
    }
    return value;
}

static uint16_t readU16BitsBE(const std::vector<uchar>& bits, int start)
{
    return readBitsBE(bits, start, 16);
}

static int16_t readI16BitsBE(const std::vector<uchar>& bits, int start)
{
    return static_cast<int16_t>(readU16BitsBE(bits, start));
}

can_work::can_work(const char *canName)
    : mSock(-1),
      isOpen(false),
      isRecv(false),
      running(false),
      m_canName(canName ? canName : "")
{
    stop_flag.store(0);
    isOpen.store(openCan());
    if (isOpen.load())
        std::cerr << "Can запущен\n";
    else
        std::cerr << "Can не запущен\n";
}

can_work::~can_work()
{
    setStopFlag(true);
    closeCan();
}

//Остановка работы
void can_work::setStopFlag(bool _stop)
{
    stop_flag.store(_stop);
    if (_stop) {
        isRecv.store(false);
    }
}

//Запуск бесконенчного приема сообщений
void can_work::startLoop(bool startFlag)
{
    isRecv.store(startFlag);
    if (startFlag) {
        stop_flag.store(false);
    }
}

bool can_work::isRunning() const
{
    return running.load();
}

//Получение сообщения
void can_work::run()
{
    running.store(true);
    stop_flag.store(false);

    while (!stop_flag.load())
    {
        msgID = 0;
        msgDLC = 0;
        msgData.clear();
        msgData_bin.clear();
        tempData_bin.clear();
        msgData_str.clear();

        int bytes = -1, timeout_ms = 20;
        struct pollfd p[1];

        if (!isOpen.load() || mSock < 0) {
            if (!openCan()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
        }

        drainTxQueue();

        if (!isRecv.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
            continue;
        }

        p[0].fd = mSock;
        p[0].events = POLLIN;

        const int pollResult = poll(p, 1, timeout_ms);
        if (pollResult < 0)
        {
            if (errno != EINTR) {
                emit appendConsole(QString("[CAN] poll error: %1").arg(strerror(errno)));
                closeCan();
            }
            continue;
        }

        if (pollResult == 0) {
            continue;
        }

        if (p[0].revents & (POLLERR | POLLHUP | POLLNVAL)) {
            emit appendConsole("[CAN] socket error/disconnect, reopening");
            closeCan();
            continue;
        }

        if (!(p[0].revents & POLLIN)) {
            continue;
        }

        bytes = static_cast<int>(read(mSock, &mFrame, sizeof(mFrame)));
        if (bytes != static_cast<int>(sizeof(mFrame))) {
            emit appendConsole(QString("[CAN] short read: %1 bytes").arg(bytes));
            continue;
        }

        emit frameReady(mFrame);
        parceMsg(mFrame);
    }

    {
        std::lock_guard<std::mutex> lock(txMutex);
        txQueue.clear();
    }
    running.store(false);
}

//Отправка сообщения
void can_work::writeCan(const can_frame &frame)
{
    constexpr std::size_t kMaxQueuedFrames = 256;

    std::lock_guard<std::mutex> lock(txMutex);
    if (txQueue.size() >= kMaxQueuedFrames) {
        txQueue.pop_front();
        emit appendConsole("[CAN] TX queue overflow, dropped oldest frame");
    }
    txQueue.push_back(frame);
}

//Запуск интерфейса
bool can_work::openCan()
{
    struct sockaddr_can addr;
    struct ifreq ifr;
    closeCan();

    if((mSock = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
    {
        std::cerr << "Ошибка открытия сокета: " << strerror(errno) << "\n";
        mSock = -1;
        isOpen.store(false);
        return false;
    }
    else std::cerr << "opening socket " << mSock << "\n";

    std::memset(&ifr, 0, sizeof(ifr));
    std::snprintf(ifr.ifr_name, IFNAMSIZ, "%s", m_canName.c_str());
    if (ioctl(mSock, SIOCGIFINDEX, &ifr) < 0) {
        std::cerr << "Ошибка ioctl SIOCGIFINDEX: " << strerror(errno) << "\n";
        closeCan();
        return false;
    }

    std::memset(&addr, 0, sizeof(addr));
    addr.can_family  = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    if(bind(mSock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0)
    {
        std::cerr << "Ошибка бинда сокетa: " << strerror(errno) << "\n";
        closeCan();
        return false;
    }

    isOpen.store(true);
    return true;
}

//Остановка интерфейса
void can_work::closeCan()
{
    if (mSock >= 0) {
        close(mSock);
        mSock = -1;
    }
    isOpen.store(false);
}

//Преобразование сообщения
void can_work::parceMsg(const can_frame &frame)
{
    msgData.clear();
    msgData_bin.clear();
    tempData_bin.clear();
    msgData_str.clear();

    msgID = static_cast<uchar>(frame.can_id & CAN_SFF_MASK);
    msgDLC = frame.can_dlc;
    if (msgDLC > 8) {
        emit appendConsole(QString("[CAN] invalid DLC: %1").arg(msgDLC));
        return;
    }

    for (int i = 0; i < frame.can_dlc; i++)
    {
        tempData_bin.clear();
        msgData.push_back(frame.data[i]);
        msgData_str.push_back(QString("%1").arg(frame.data[i],0,16));

        for (int j = 0; j < 8; j++)
        {
            tempData_bin.push_back(msgData[i] % 2);
            msgData[i] /= 2;
        }
        std::reverse(tempData_bin.begin(), tempData_bin.end());
        for (uint8_t k = 0; k < tempData_bin.size(); k++)
            msgData_bin.push_back(tempData_bin[k]);
    }

//    insertConsole("Парсинг сообщения: \n");
//    insertConsole("ID: ");
//    insertConsole(msgID_str);
//    insertConsole(" DLC: ");
//    insertConsole(msgDLC_str);
//    insertConsole(" DATA: \n");
//    for (int i = 0; i < msgDLC; i++)
//    {
//        insertConsole(QString("%1").arg(i,0,16));
//        insertConsole(" byte: ");
//        insertConsole(msgData_str[static_cast<size_t>(i)]);
//        insertConsole("\n");
//    }

    analysisMsg();
}

//Анализ данных из сообщения
void can_work::analysisMsg()
{
    QString msgID_str = QString("%1").arg(msgID,0,16);
    QString msgDLC_str = QString("%1").arg(msgDLC,0,16);

    switch (msgID)
    {
    case ID_2_1:
        if(msgDLC != 0x5)
        {
            appendConsole("Ошибка1 приема! Неправильная длина сообщения");
            break;
        }
        else
        {
            msgData_bin[0] ? appendConsole("БНФК исправен")        : appendConsole("БНФК не исправен");
            msgData_bin[1] ? appendConsole("Питание есть")         : appendConsole("Питания нет");
            msgData_bin[2] ? appendConsole("УПУ исправен")         : appendConsole("УПУ не исправен");
            msgData_bin[3] ? appendConsole("СО ответил")           : appendConsole("СО не ответил");
            msgData_bin[4] ? appendConsole("УПУ в исходном")       : appendConsole("УПУ не в исходном");
            msgData_bin[5] ? appendConsole("УПУ в движении")       : appendConsole("УПУ остановлен");
            msgData_bin[6] ? appendConsole("Работа заблокирована") : appendConsole("Блокировки нет");
            msgData_bin[7] ? appendConsole("Калибровка выполнена") : appendConsole("Калибровка не выполнена");
            insertConsole("\n");

            for (int i = 8; i < 16; i++)
            {
                appendConsole(QString("%1").arg(i - 7, 0, 10));
                msgData_bin[i] ? insertConsole(" БС ЛВ Патрон есть") : insertConsole(" БС ЛВ Патрона нет");
            }
            insertConsole("\n");

            for (int i = 16; i < 24; i++)
            {
                appendConsole(QString("%1").arg(i - 15, 0, 10));
                msgData_bin[i] ? insertConsole(" БС ЛН Патрон есть") : insertConsole(" БС ЛН Патрона нет");
            }
            insertConsole("\n");

            for (int i = 24; i < 32; i++)
            {
                appendConsole(QString("%1").arg(i - 23, 0, 10));
                msgData_bin[i] ? insertConsole(" БС ПВ Патрон есть") : insertConsole(" БС ПВ Патрона нет");
            }
            insertConsole("\n");

            for (int i = 32; i < 40; i++)
            {
                appendConsole(QString("%1").arg(i - 31, 0, 10));
                msgData_bin[i] ? insertConsole(" БС ПН Патрон есть") : insertConsole(" БС ПН Патрона нет");
            }
            insertConsole("\n");
            break;

//            int cnt = 0;
//            for (uint8_t j = 0; j < msgData_bin.size(); j++)
//            {
//                if (cnt == 8)
//                {
//                    cnt = 0;
//                    appendConsole("next byte: ");
//                }
//                if (msgData_bin[j])
//                    insertConsole("1 ");
//                else
//                    insertConsole("0 ");
//                cnt++;
//            }
        }
    case ID_2_2:
        if (msgDLC != 0x3)
        {
            appendConsole("Ошибка13 приема! Неправильная длина сообщения");
            break;
        }
        else
        {
            if (msgData_bin[4] && !msgData_bin[5] && msgData_bin[6] && msgData_bin[7])
                appendConsole("Данные о наработке БНФК: ");
            else if (msgData_bin[4] && msgData_bin[5] && !msgData_bin[6] && msgData_bin[7])
                appendConsole("Данные о наработке УПУ: ");
            else
            {
                appendConsole("Неизвестное устройство! Данные о наработке недоступны");
                break;
            }

            if (msgData_bin[9] || msgData_bin[8])
            {
                appendConsole("Наработка больше 9999 часов!");
                break;
            }
            else
            {
                uint16_t time = readBitsBE(msgData_bin, 10, 14);
                if (time > 9999)
                {
                    appendConsole("Наработка больше 9999 часов!");
                    break;
                }
                else
                {
                    appendConsole("Наработка ");
                    insertConsole(QString("%1").arg(time,0,10));
                    insertConsole(" часов");
                    break;
                }
            }
        }

    case ID_2_3:
        if (msgDLC != 0x1)
        {
            appendConsole("Ошибка12 приема! Неправильная длина сообщения");
            break;
        }
        else
        {
            //
            //Здесь нужно проверять сообщение 0х5
            //
            msgData_bin[0] ? appendConsole("Запрос есть") : appendConsole("Запроса нет");
        }
        break;
    case ID_2_4:
        if (msgDLC != 0x5)
        {
            appendConsole("Ошибка11 приема! Неправильная длина сообщения");
            break;
        }
        else
        {
            msgData_bin[0] ? appendConsole("БНФК неисправен!") : appendConsole("БНФК исправен");
            msgData_bin[1] ? appendConsole("Блок преобразования напряжения УПУ неисправен!") : appendConsole("Блок преообразования напряжения УПУ исправен");
            msgData_bin[2] ? appendConsole("Нет поворота двигателя ГН!") : appendConsole("Есть поворот двигателя ГН");
            msgData_bin[3] ? appendConsole("Нет поворота двигателя ВН!") : appendConsole("Есть поворот двигателя ВН");
            msgData_bin[4] ? appendConsole("Нет питания двигателя ГН!") : appendConsole("Есть питание двигателя ГН");
            msgData_bin[5] ? appendConsole("Нет питания двигателя ВН!") : appendConsole("Есть питание двигателя ВН");
            msgData_bin[7] ? appendConsole("Есть ответ от УПУ") : appendConsole("Нет уответа от УПУ!");
            break;
        }



        // ID 6!!!!!!!!!!! priem
    case ID_2_5:
    {
        if (msgDLC != 0x7)
        {
            appendConsole("Ошибка10 приема! Неправильная длина сообщения");
            break;
        }

        // ===== ДАТЧИКИ =====
        msgData_bin[0] ? appendConsole("Левый датчик сработал!")             : appendConsole("Нет срабатывания левого датчика");
        msgData_bin[1] ? appendConsole("Правый датчик сработал!")            : appendConsole("Нет срабатывания правого датчика");
        msgData_bin[2] ? appendConsole("Верхний датчик сработал!")           : appendConsole("Нет срабатывания верхнего датчика");
        msgData_bin[3] ? appendConsole("Нижний датчик сработал!")            : appendConsole("Нет срабатывания нижнего датчика");
        msgData_bin[7] ? appendConsole("Команда на срабатывание БС выдана!") : appendConsole("Команда на срабатывание БС не выдана");

        // =====================================================
        // ================== АЗИМУТ ===========================
        // bits [8..23] → int16_t (сотые градуса)
        // =====================================================
        const int16_t az = readI16BitsBE(msgData_bin, 8);

        if (std::abs(az) > 18000)
        {
            appendConsole("Ошибка! Значение угла больше 180!");
        }
        else
        {
            appendConsole(az < 0 ? "Угол по азимуту: -" : "Угол по азимуту: ");

            int az_do     = std::abs(az) / 100;
            int az_posle  = std::abs(az) % 100;

            insertConsole(QString("%1").arg(az_do));
            insertConsole(",");
            insertConsole(QString("%1").arg(az_posle, 2, 10, QChar('0')));
            insertConsole(" градусов");
        }


        const int16_t el = readI16BitsBE(msgData_bin, 24);

        if (std::abs(el) > 4000)
        {
            appendConsole("Ошибка! Значение угла больше 40!");
        }
        else
        {
            appendConsole(el < 0 ? "Угол по углу места: -" : "Угол по углу места: ");

            int el_do     = std::abs(el) / 100;
            int el_posle  = std::abs(el) % 100;

            insertConsole(QString("%1").arg(el_do));
            insertConsole(",");
            insertConsole(QString("%1").arg(el_posle, 2, 10, QChar('0')));
            insertConsole(" градусов");
        }


        setTurretState(az / 100.0, el / 100.0);


        break;
    }










    case ID_3_1:
        if (msgDLC != 0x1)
        {
            appendConsole("Ошибка9 приема! Неправильная длина сообщения");
            break;
        }
        else
        {
            msgData_bin[0] ? appendConsole("Есть неисправность СЧ СО!") : appendConsole("Неисправностей нет");
            msgData_bin[1] ? appendConsole("Загрязнение приемников сигнала!") : appendConsole("Нет загязнений приемников сигнала");
            msgData_bin[2] ? appendConsole("Цель есть!"): appendConsole("Цели нет");
            msgData_bin[3] ? appendConsole("Технологический режим СО") : appendConsole("Основной режим СО");
            break;
        }

    case ID_3_2:
        if (msgDLC != 0x5)
        {
            appendConsole("Ошибка8 приема! Неправильная длина сообщения");
            break;
        }
        else
        {
            msgData_bin[0] ? appendConsole("Есть атака!") : appendConsole("Атаки нет");
            msgData_bin[1] ? appendConsole("Пересечен рубеж 30м!") : appendConsole("Рубеж 30м не пересечен");
            msgData_bin[2] ? appendConsole("Есть запрос положения УПУ!"): appendConsole("Запроса положения УПУ нет");

            const int16_t az = readI16BitsBE(msgData_bin, 8);

            if (std::abs(az) > 18000)
            {
                appendConsole("Ошибка! Значение угла больше 180!");
            }
            else
            {
                if (az < 0) appendConsole("Угол по азимуту: -");
                else        appendConsole("Угол по азимуту: ");

                int az_do = std::abs(az) / 100;
                int az_posle = std::abs(az) % 100;


                insertConsole(QString("%1").arg(az_do));
                insertConsole(",");
                insertConsole(QString("%1").arg(az_posle,2,10, QChar('0')));
                insertConsole(" градусов");
            }

            const int16_t el = readI16BitsBE(msgData_bin, 24);

            if (std::abs(el) > 4000)
            {
                appendConsole("Ошибка! Значение угла больше 40!");
            }
            else
            {
                if (el < 0) appendConsole("Угол по азимуту: -");
                else        appendConsole("Угол по азимуту: ");
                int coord2_do = std::abs(el) / 100;
                int coord2_posle = std::abs(el) % 100;

                insertConsole(QString("%1").arg(coord2_do));
                insertConsole(",");
                insertConsole(QString("%1").arg(coord2_posle,2,10, QChar('0')));
                insertConsole(" градусов");
            }
            break;
        }
    case ID_3_3:
        if (msgDLC != 0x4)
        {
            appendConsole("Ошибка7 приема! Неправильная длина сообщения");
            break;
        }
        else
        {
            appendConsole("Радиальная скорость движения цели:");
            int speed = readU16BitsBE(msgData_bin, 0);

            int speed_do = 0;
            int speed_posle = 0;
            speed_do = speed / 10;
            speed_posle = speed % 10;

            insertConsole(QString("%1").arg(speed_do,0,10));
            insertConsole(",");
            insertConsole(QString("%1").arg(speed_posle,0,10));
            insertConsole(" м/с");

            appendConsole("Дальность до цели:");
            int distance = readU16BitsBE(msgData_bin, 16);

            int distance_do = 0;
            int distance_posle = 0;
            distance_do = distance / 10;
            distance_posle = distance % 10;

            insertConsole(QString("%1").arg(distance_do,0,10));
            insertConsole(",");
            insertConsole(QString("%1").arg(distance_posle,0,10));
            insertConsole(" м");

            break;
        }
    case ID_4_1:
        if (msgDLC != 0x5)
        {
            appendConsole("Ошибка6 приема! Неправильная длина сообщения");
            break;
        }
        else
        {
            msgData_bin[0] ? appendConsole("Неисправность 1")   : appendConsole("Исправность 1")  ;
            msgData_bin[1] ? appendConsole("Неисправность 2")    : appendConsole("Исправность 2")   ;
            msgData_bin[2] ? appendConsole("Неисправность 3")    : appendConsole("Исправность 3")   ;
            msgData_bin[3] ? appendConsole("Неисправность 4") : appendConsole("Исправность 4");
            msgData_bin[4] ? appendConsole("Неисправность 5")   : appendConsole("Исправность 5")  ;
            msgData_bin[5] ? appendConsole("Неисправность 6")  : appendConsole("Исправность 6") ;
            msgData_bin[6] ? appendConsole("Неисправность 7")   : appendConsole("Исправность 7")  ;
            msgData_bin[7] ? appendConsole("Неисправность 8") : appendConsole("Исправность 8");

            msgData_bin[8]  ? appendConsole("Неисправ. приемника сигнала №1")   : appendConsole("Исправ. приемника сигнала №1")  ;
            msgData_bin[9]  ? appendConsole("Неисправ. приемника сигнала №2")    : appendConsole("Исправ. приемника сигнала №2")   ;
            msgData_bin[10] ? appendConsole("Неисправ. приемника сигнала №3")    : appendConsole("Исправ. приемника сигнала №3")   ;
            msgData_bin[11] ? appendConsole("Неисправ. приемника сигнала №4") : appendConsole("Исправ. приемника сигнала №4");
            msgData_bin[12] ? appendConsole("Неисправ. приемника сигнала №4")   : appendConsole("Исправ. приемника сигнала №5")  ;
            msgData_bin[13] ? appendConsole("Неисправ. приемника сигнала №5")  : appendConsole("Исправ. приемника сигнала №6") ;
            msgData_bin[14] ? appendConsole("Неисправ. приемника сигнала №6")   : appendConsole("Исправ. приемника сигнала №7")  ;
            msgData_bin[15] ? appendConsole("Неисправ. приемника сигнала №7") : appendConsole("Исправ. приемника сигнала №8");

            //[16] - [23] зачем то пустые

            msgData_bin[24] ? appendConsole("Загрязнение приемника сигнала №1")   : appendConsole("Приемник сигнала №1 чистый")  ;
            msgData_bin[25] ? appendConsole("Загрязнение приемника сигнала №2")    : appendConsole("Приемник сигнала №2 чистый")   ;
            msgData_bin[26] ? appendConsole("Загрязнение приемника сигнала №3")    : appendConsole("Приемник сигнала №3 чистый")   ;
            msgData_bin[27] ? appendConsole("Загрязнение приемника сигнала №4") : appendConsole("Приемник сигнала №4 чистый");
            msgData_bin[28] ? appendConsole("Загрязнение приемника сигнала №5")   : appendConsole("Приемник сигнала №5 чистый")  ;
            msgData_bin[29] ? appendConsole("Загрязнение приемника сигнала №6")  : appendConsole("Приемник сигнала №6 чистый") ;
            msgData_bin[30] ? appendConsole("Загрязнение приемника сигнала №7")   : appendConsole("Приемник сигнала №7 чистый")  ;
            msgData_bin[31] ? appendConsole("Загрязнение приемника сигнала №8") : appendConsole("Приемник сигнала №8 чистый");

            //[32] - [39] зачем то пустые

            break;
        }

    case ID_4_2:
        if (msgDLC != 0x3)
        {
            appendConsole("Ошибка5 приема! Неправильная длина сообщения");
            break;
        }
        else
        {
            uint8_t device = static_cast<uint8_t>(readBitsBE(msgData_bin, 4, 4));

            switch (device)
            {
            case 1:
                appendConsole("Данные о наработке устройства один: ");
                break;
            case 2:
                appendConsole("Данные о наработке устройства два: ");
                break;
            case 3:
                appendConsole("Данные о наработке устройства три: ");
                break;
            case 4:
                appendConsole("Данные о наработке устройства четыре: ");
                break;
            case 5:
                appendConsole("Данные о наработке устройства пять: ");
                break;
            case 6:
                appendConsole("Данные о наработке устройства шесть: ");
                break;
            case 7:
                appendConsole("Данные о наработке устройства семь: ");
                break;
            case 8:
                appendConsole("Данные о наработке устройства восемь: ");
                break;
            default:
                appendConsole("Неизвестное устройство! Данные о наработке недоступны");
                break;
            }

            if (device < 1 || device > 8)
                break;
            else if (msgData_bin[9] || msgData_bin[8])
            {
                appendConsole("Наработка больше 9999 часов!");
                break;
            }
            else
            {
                uint16_t time = readBitsBE(msgData_bin, 10, 14);
                if (time > 9999)
                {
                    appendConsole("Наработка больше 9999 часов!");
                    break;
                }
                else
                {
                    appendConsole("Наработка ");
                    insertConsole(QString("%1").arg(time,0,10));
                    insertConsole(" часов");
                    break;
                }
            }
        }

    case ID_5_1:
        if (msgDLC != 0x1)
        {
            appendConsole("Ошибка4 приема! Неправильная длина сообщения");
            break;
        }
        else
        {
            msgData_bin[0] ? appendConsole("Включение технологического режима") : appendConsole("Отключение технологического режима");
            //вероятно надо как то отключать этот режим
            break;
        }
    case ID_5_2:
        if (msgDLC != 0x1)
        {
            appendConsole("Ошибка3 приема! Неправильная длина сообщения");
            break;
        }
        else
        {
            uint8_t device = static_cast<uint8_t>(readBitsBE(msgData_bin, 4, 4));

            switch (device)
            {
            case 1:
                appendConsole("Зпроса времени наработки устройства один: ");
                break;
            case 2:
                appendConsole("Запрос времени наработки устройства два: ");
                break;
            case 3:
                appendConsole("Запрос времени наработки устройства  три: ");
                break;
            case 4:
                appendConsole("Запрос времени наработки устройства четыре: ");
                break;
            case 5:
                appendConsole("Запрос времени наработки устройства пять: ");
                break;
            case 6:
                appendConsole("Запрос времени наработки устройства шесть: ");
                break;
            case 7:
                appendConsole("Запрос времени наработки БНФК: ");
                break;
            case 8:
                appendConsole("Запрос времени наработки УПУ: ");
                break;
            default:
                appendConsole("Неизвестное устройство! Данные о наработке недоступны");
                break;
            }
            break;
        }
    case ID_5_3:
        if (msgDLC != 0x1)
        {
            appendConsole("Ошибка2 приема! Неправильная длина сообщения");
            break;
        }
        else
        {
            msgData_bin[0] ? appendConsole("Есть шаг минус") : appendConsole("Нет шага минус")  ;
            msgData_bin[1] ? appendConsole("Есть шаг плюс")  : appendConsole("Нет шага плюс")   ;
            msgData_bin[2] ? appendConsole("Зафиксировать нулевое положение") : appendConsole("Не фиксировать нулевое положение")   ;

            if (msgData_bin[4] && !msgData_bin[5] && msgData_bin[6] && msgData_bin[7])
                appendConsole("Калибровка для привода ГН");
            else if (msgData_bin[4] && msgData_bin[5] && !msgData_bin[6] && msgData_bin[7])
                appendConsole("Калиброка для привода ВН");
            else
            {
                appendConsole("Неизвестное устройство! Калибровка недоступна");
                break;
            }
        }
    }

}

bool can_work::writeCanNow(const can_frame &frame)
{
    if (!isOpen.load() || mSock < 0) {
        emit appendConsole("[CAN] TX failed: socket is not open");
        return false;
    }

    const ssize_t bytes = write(mSock, &frame, sizeof(frame));
    if (bytes != static_cast<ssize_t>(sizeof(frame))) {
        const QString reason = (bytes < 0)
            ? QString::fromLocal8Bit(strerror(errno))
            : QString("short write: %1 bytes").arg(bytes);
        emit appendConsole("[CAN] TX failed: " + reason);
        return false;
    }

    return true;
}

void can_work::drainTxQueue()
{
    std::deque<can_frame> local;
    {
        std::lock_guard<std::mutex> lock(txMutex);
        local.swap(txQueue);
    }

    for (const can_frame& frame : local) {
        if (!writeCanNow(frame)) {
            break;
        }
    }
}
