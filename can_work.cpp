#include "can_work.h"
#include "tower_state.h"

can_work::can_work(const char *canName) : m_canName(canName)
{
    stop_flag.store(0);
    isOpen = openCan();
    connect (this, &can_work::startParce, this, &can_work::parceMsg);
    if (isOpen)
        std::cerr << "Can запущен\n";
    else
        std::cerr << "Can не запущен\n";
}

can_work::~can_work()
{
    close(mSock);
}

//Остановка работы
void can_work::setStopFlag(bool _stop)
{
    stop_flag.store(_stop);
}

//Запуск бесконенчного приема сообщений
void can_work::startLoop(bool startFlag)
{
    isRecv = startFlag;
}

//Получение сообщения
void can_work::run()
{
    while (isRecv)
    {
        sync();

        msgID = 0;
        msgDLC = 0;
        msgData.clear();
        msgData_bin.clear();
        tempData_bin.clear();
        msgData_str.clear();

        appendConsole("Ожидание сообщения...\n");

        int bytes = -1, timeout_ms = 1;
        struct pollfd p[1];

        p[0].fd = mSock;
        p[0].events = POLLIN;

        while(!stop_flag.load())
        {
            if(poll(p, 1, timeout_ms) > 0)
            {
                bytes = static_cast<int>(read(mSock, &mFrame, sizeof(mFrame))); //размер структуры или размер кадра?
                frameReady(mFrame);
                startParce(mFrame);
                break;
            }
        }
        appendConsole(bytes > 0 ? "Сообщение получено!\n" : "Ожидание сообщения прервано!\n");
    }
}

//Отправка сообщения
void can_work::writeCan(const can_frame &frame)
{
    insertConsole("Отправка сообщения...\n");
    auto bytes = write(mSock, &frame, sizeof(frame));
    insertConsole(bytes > 0 ? "Сообщение отправлено!\n" : "Не удалось отправить сообщение!\n");
}

//Запуск интерфейса
bool can_work::openCan()
{
    struct sockaddr_can addr;
    struct ifreq ifr;
    mSock = -1;

    if((mSock = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
    {
        std::cerr << "Ошибка открытия сокета!\n";
        mSock = -1;
    }
    else std::cerr << "opening socket " << mSock << "\n";

    strcpy(ifr.ifr_name, m_canName);
    ioctl(mSock, SIOCGIFINDEX, &ifr);

    addr.can_family  = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    if(bind(mSock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0)
    {
        std::cerr << "Ошибка бинда сокетa!\n";
        close(mSock);
        mSock = -1;
    }
    return mSock > 0;
}

//Остановка интерфейса
void can_work::closeCan()
{
    close(mSock);
    isOpen = false;
}

//Преобразование сообщения
void can_work::parceMsg(const can_frame &frame)
{
    msgID = frame.can_id;
    msgDLC = frame.can_dlc;

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
                //вероятно, это можно как то оптимизировать
                uint16_t time = 0;
                msgData_bin[23] ? time += 1    : time += 0;
                msgData_bin[22] ? time += 2    : time += 0;
                msgData_bin[21] ? time += 4    : time += 0;
                msgData_bin[20] ? time += 8    : time += 0;
                msgData_bin[19] ? time += 16   : time += 0;
                msgData_bin[18] ? time += 32   : time += 0;
                msgData_bin[17] ? time += 64   : time += 0;
                msgData_bin[16] ? time += 128  : time += 0;
                msgData_bin[15] ? time += 256  : time += 0;
                msgData_bin[14] ? time += 512  : time += 0;
                msgData_bin[13] ? time += 1024 : time += 0;
                msgData_bin[12] ? time += 2048 : time += 0;
                msgData_bin[11] ? time += 4096 : time += 0;
                msgData_bin[10] ? time += 8192 : time += 0;
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
        uint16_t raw_az = 0;

        if (msgData_bin[23]) raw_az |= (1 << 0);
        if (msgData_bin[22]) raw_az |= (1 << 1);
        if (msgData_bin[21]) raw_az |= (1 << 2);
        if (msgData_bin[20]) raw_az |= (1 << 3);
        if (msgData_bin[19]) raw_az |= (1 << 4);
        if (msgData_bin[18]) raw_az |= (1 << 5);
        if (msgData_bin[17]) raw_az |= (1 << 6);
        if (msgData_bin[16]) raw_az |= (1 << 7);
        if (msgData_bin[15]) raw_az |= (1 << 8);
        if (msgData_bin[14]) raw_az |= (1 << 9);
        if (msgData_bin[13]) raw_az |= (1 << 10);
        if (msgData_bin[12]) raw_az |= (1 << 11);
        if (msgData_bin[11]) raw_az |= (1 << 12);
        if (msgData_bin[10]) raw_az |= (1 << 13);
        if (msgData_bin[9])  raw_az |= (1 << 14);
        if (msgData_bin[8])  raw_az |= (1 << 15);

        int16_t az = static_cast<int16_t>(raw_az);

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


        uint16_t raw_el = 0;

        if (msgData_bin[39]) raw_el |= (1 << 0);
        if (msgData_bin[38]) raw_el |= (1 << 1);
        if (msgData_bin[37]) raw_el |= (1 << 2);
        if (msgData_bin[36]) raw_el |= (1 << 3);
        if (msgData_bin[35]) raw_el |= (1 << 4);
        if (msgData_bin[34]) raw_el |= (1 << 5);
        if (msgData_bin[33]) raw_el |= (1 << 6);
        if (msgData_bin[32]) raw_el |= (1 << 7);
        if (msgData_bin[31]) raw_el |= (1 << 8);
        if (msgData_bin[30]) raw_el |= (1 << 9);
        if (msgData_bin[29]) raw_el |= (1 << 10);
        if (msgData_bin[28]) raw_el |= (1 << 11);
        if (msgData_bin[27]) raw_el |= (1 << 12);
        if (msgData_bin[26]) raw_el |= (1 << 13);
        if (msgData_bin[25]) raw_el |= (1 << 14);
        if (msgData_bin[24]) raw_el |= (1 << 15);

        int16_t el = static_cast<int16_t>(raw_el);

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


        g_turret_az_deg = az / 100.0;
        g_turret_el_deg = el / 100.0;


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

            //[8] - [23]
            uint16_t coord1 = 0;

            msgData_bin[23] ? coord1 += 1     : coord1 += 0;
            msgData_bin[22] ? coord1 += 2     : coord1 += 0;
            msgData_bin[21] ? coord1 += 4     : coord1 += 0;
            msgData_bin[20] ? coord1 += 8     : coord1 += 0;
            msgData_bin[19] ? coord1 += 16    : coord1 += 0;
            msgData_bin[18] ? coord1 += 32    : coord1 += 0;
            msgData_bin[17] ? coord1 += 64    : coord1 += 0;
            msgData_bin[16] ? coord1 += 128   : coord1 += 0;
            msgData_bin[15] ? coord1 += 256   : coord1 += 0;
            msgData_bin[14] ? coord1 += 512   : coord1 += 0;
            msgData_bin[13] ? coord1 += 1024  : coord1 += 0;
            msgData_bin[12] ? coord1 += 2048  : coord1 += 0;
            msgData_bin[11] ? coord1 += 4096  : coord1 += 0;
            msgData_bin[10] ? coord1 += 8192  : coord1 += 0;
            msgData_bin[9]  ? coord1 += 16384 : coord1 += 0;
            msgData_bin[8]  ? coord1 += 32768 : coord1 += 0;

            int16_t az = static_cast<int16_t>(coord1);

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

            //[24] - [39]
            uint16_t coord2 = 0;
            msgData_bin[39] ? coord2 += 1     : coord2 += 0;
            msgData_bin[38] ? coord2 += 2     : coord2 += 0;
            msgData_bin[37] ? coord2 += 4     : coord2 += 0;
            msgData_bin[36] ? coord2 += 8     : coord2 += 0;
            msgData_bin[35] ? coord2 += 16    : coord2 += 0;
            msgData_bin[34] ? coord2 += 32    : coord2 += 0;
            msgData_bin[33] ? coord2 += 64    : coord2 += 0;
            msgData_bin[32] ? coord2 += 128   : coord2 += 0;
            msgData_bin[31] ? coord2 += 256   : coord2 += 0;
            msgData_bin[30] ? coord2 += 512   : coord2 += 0;
            msgData_bin[29] ? coord2 += 1024  : coord2 += 0;
            msgData_bin[28] ? coord2 += 2048  : coord2 += 0;
            msgData_bin[27] ? coord2 += 4096  : coord2 += 0;
            msgData_bin[26] ? coord2 += 8192  : coord2 += 0;
            msgData_bin[25] ? coord2 += 16384 : coord2 += 0;
            msgData_bin[24] ? coord2 += 32768 : coord2 += 0;


            int16_t el = static_cast<int16_t>(coord2);

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
            //[0] - [15]
            int speed = 0;
            msgData_bin[15] ? speed += 1     : speed += 0;
            msgData_bin[14] ? speed += 2     : speed += 0;
            msgData_bin[13] ? speed += 4     : speed += 0;
            msgData_bin[12] ? speed += 8     : speed += 0;
            msgData_bin[11] ? speed += 16    : speed += 0;
            msgData_bin[10] ? speed += 32    : speed += 0;
            msgData_bin[9]  ? speed += 64    : speed += 0;
            msgData_bin[8]  ? speed += 128   : speed += 0;
            msgData_bin[7]  ? speed += 256   : speed += 0;
            msgData_bin[6]  ? speed += 512   : speed += 0;
            msgData_bin[5]  ? speed += 1024  : speed += 0;
            msgData_bin[4]  ? speed += 2048  : speed += 0;
            msgData_bin[3]  ? speed += 4096  : speed += 0;
            msgData_bin[2]  ? speed += 8192  : speed += 0;
            msgData_bin[1]  ? speed += 16384 : speed += 0;
            msgData_bin[0]  ? speed += 32768 : speed += 0;

            int speed_do = 0;
            int speed_posle = 0;
            speed_do = speed / 10;
            speed_posle = speed % 10;

            insertConsole(QString("%1").arg(speed_do,0,10));
            insertConsole(",");
            insertConsole(QString("%1").arg(speed_posle,0,10));
            insertConsole(" м/с");

            appendConsole("Дальность до цели:");
            //[16] - [31]
            int distance = 0;
            msgData_bin[31] ? distance += 1     : distance += 0;
            msgData_bin[30] ? distance += 2     : distance += 0;
            msgData_bin[29] ? distance += 4     : distance += 0;
            msgData_bin[28] ? distance += 8     : distance += 0;
            msgData_bin[27] ? distance += 16    : distance += 0;
            msgData_bin[26] ? distance += 32    : distance += 0;
            msgData_bin[25]  ? distance += 64    : distance += 0;
            msgData_bin[24]  ? distance += 128   : distance += 0;
            msgData_bin[23]  ? distance += 256   : distance += 0;
            msgData_bin[22]  ? distance += 512   : distance += 0;
            msgData_bin[21]  ? distance += 1024  : distance += 0;
            msgData_bin[20]  ? distance += 2048  : distance += 0;
            msgData_bin[19]  ? distance += 4096  : distance += 0;
            msgData_bin[18]  ? distance += 8192  : distance += 0;
            msgData_bin[17]  ? distance += 16384 : distance += 0;
            msgData_bin[16]  ? distance += 32768 : distance += 0;

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
            uint8_t device = 0;
            msgData_bin[4] ? device += 8 : device += 0;
            msgData_bin[5] ? device += 4 : device += 0;
            msgData_bin[6] ? device += 2 : device += 0;
            msgData_bin[7] ? device += 1 : device += 0;

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
                //вероятно, это можно как то оптимизировать
                uint16_t time = 0;
                msgData_bin[23] ? time += 1    : time += 0;
                msgData_bin[22] ? time += 2    : time += 0;
                msgData_bin[21] ? time += 4    : time += 0;
                msgData_bin[20] ? time += 8    : time += 0;
                msgData_bin[19] ? time += 16   : time += 0;
                msgData_bin[18] ? time += 32   : time += 0;
                msgData_bin[17] ? time += 64   : time += 0;
                msgData_bin[16] ? time += 128  : time += 0;
                msgData_bin[15] ? time += 256  : time += 0;
                msgData_bin[14] ? time += 512  : time += 0;
                msgData_bin[13] ? time += 1024 : time += 0;
                msgData_bin[12] ? time += 2048 : time += 0;
                msgData_bin[11] ? time += 4096 : time += 0;
                msgData_bin[10] ? time += 8192 : time += 0;
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
            uint8_t device = 0;
            msgData_bin[4] ? device += 8 : device += 0;
            msgData_bin[5] ? device += 4 : device += 0;
            msgData_bin[6] ? device += 2 : device += 0;
            msgData_bin[7] ? device += 1 : device += 0;

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
