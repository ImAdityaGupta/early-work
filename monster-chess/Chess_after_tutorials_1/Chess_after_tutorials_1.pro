QT       += core gui
QT       += concurrent
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    board_class.cpp \
    class_declarations.cpp \
    const_setups.cpp \
    debug_help.cpp \
    game.cpp \
    main.cpp \
    more_display_stuff.cpp \
    move_generation.cpp \
    naive_original_way.cpp \
    new_ai.cpp \
    piece.cpp \
    pure_debug.cpp

HEADERS += \
    board_class.h \
    class_declarations.h \
    const_setups.h \
    debug_help.h \
    game.h \
    general_algorithm.h \
    more_display_stuff.h \
    move_generation.h \
    naive_original_way.h \
    new_ai.h \
    piece.h \
    pure_debug.h

FORMS +=

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
QMAKE_CXXFLAGS += -std=c++11
