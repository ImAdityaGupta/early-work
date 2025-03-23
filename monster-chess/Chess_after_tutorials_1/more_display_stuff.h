#ifndef MORE_DISPLAY_STUFF_H
#define MORE_DISPLAY_STUFF_H

#include <QGraphicsTextItem>
#include <QGraphicsView>
#include <QFont>
#include <QDebug>
#include <QVector>
#include <QLabel>

#include <QGraphicsRectItem>
#include <QObject>
#include <QGraphicsItem>
#include <QGraphicsView>
#include <QWidget>
#include <QGraphicsScene>
#include <QTimer>
#include <QGraphicsTextItem>
#include <QFont>
#include <QPainter>
#include <QColor>
#include <QDebug>
#include <QVector>
#include <QPixmap>
#include <QGraphicsPixmapItem>
#include <QFont>

class Game;

class Start_Button: public QGraphicsRectItem {
public:

    Game * game;

    Start_Button(Game * game, bool left);
    void mousePressEvent(QGraphicsSceneMouseEvent *event);
    QString color;




};

#endif // MORE_DISPLAY_STUFF_H
