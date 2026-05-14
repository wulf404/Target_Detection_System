QT       += core gui serialport
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Target_Detection_System

CONFIG += c++17
DEFINES += QT_DEPRECATED_WARNINGS

DEFINES += ENABLE_CUDA_PREPROCESS=1
DEFINES += OPENCV_CPU_THREADS=1
DEFINES += ENABLE_DEEPSTREAM=1

CONFIG += link_pkgconfig
PKGCONFIG += gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0

OBJECTS_DIR = $$OUT_PWD/obj
MOC_DIR     = $$OUT_PWD/moc
RCC_DIR     = $$OUT_PWD/rcc
UI_DIR      = $$OUT_PWD/ui

SOURCES += \
    auto_tracker.cpp \
    balancer.cpp \
    camera.cpp \
    can_work.cpp \
    deepstream_yolo.cpp \
    main.cpp \
    mainwindow.cpp \
    my_yolo.cpp \
    rangefinder_uart.cpp \
    remote_tracker.cpp \
    serial_port_resolver.cpp \
    target_manager.cpp \
    tower_state.cpp \
    turret_command.cpp \
    tracking_state.cpp \
    uart_receiver.cpp


HEADERS += \
    auto_tracker.h \
    app_config.h \
    balancer.h \
    camera.h \
    can_commands.h \
    can_work.h \
    common_constants.h \
    cuda_preprocess.h \
    deepstream_yolo.h \
    mainwindow.h \
    my_yolo.h \
    rangefinder_uart.h \
    remote_tracker.h \
    serial_port_resolver.h \
    target_manager.h \
    tower_state.h \
    turret_command.h \
    tracking_state.h \
    uart_receiver.h

FORMS += \
    mainwindow.ui

qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

INCLUDEPATH += /usr/local/include/opencv4 \
               /usr/include/opencv4

LIBS += -L/usr/local/lib

LIBS += -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_core -lpthread \
        -lopencv_tracking -lopencv_optflow -lopencv_video -lopencv_features2d -lopencv_dnn \
        -lopencv_cudafilters -lopencv_cudabgsegm -lopencv_cudafeatures2d -lopencv_dnn_superres -lopencv_cudaoptflow \
        -lopencv_cudacodec -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudaarithm

# ===== DeepStream 7.1 / JetPack 6.2.x =====
DEEPSTREAM_DIR = /opt/nvidia/deepstream/deepstream
INCLUDEPATH += $$DEEPSTREAM_DIR/sources/includes
QMAKE_LIBDIR += $$DEEPSTREAM_DIR/lib
LIBS += -Wl,-rpath,$$DEEPSTREAM_DIR/lib \
        -lnvdsgst_meta -lnvds_meta -lnvds_inferutils

# ===== CUDA / NVCC (qmake extra compiler) =====
CUDA_DIR  = /usr/local/cuda

INCLUDEPATH += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64
LIBS += -lcudart -lcuda -lcudnn

# Jetson Orin NX
CUDA_ARCH = sm_87

NVCC = $$CUDA_DIR/bin/nvcc
NVCC_HOST_COMPILER = /usr/bin/aarch64-linux-gnu-g++

NVCCFLAGS = -std=c++17 -O3 --use_fast_math \
            --expt-relaxed-constexpr \
            -Xcompiler -fPIC \
            -ccbin $$NVCC_HOST_COMPILER \
            -gencode arch=compute_87,code=sm_87

CUDA_SOURCES += cuda_preprocess.cu

cuda_preprocess.input = CUDA_SOURCES
cuda_preprocess.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}.o
cuda_preprocess.commands = $$NVCC -c ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
    $$NVCCFLAGS \
    -I$$CUDA_DIR/include \
    -I/usr/include/opencv4 -I/usr/local/include/opencv4 \
    -I$$PWD
cuda_preprocess.dependency_type = TYPE_C
cuda_preprocess.variable_out = OBJECTS

QMAKE_EXTRA_COMPILERS += cuda_preprocess
