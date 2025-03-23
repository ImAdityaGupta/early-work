#pragma once

#ifndef PIECE_H
#define PIECE_H

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

#include <QCoreApplication>
#include <QtConcurrent/QtConcurrent>
#include <QThread>
#include <QThreadPool>


#include "game.h"
#include "new_ai.h"





class Piece: public QGraphicsRectItem{
public:
    //QVector<Piece*> * all_pieces;


    QGraphicsPixmapItem * board_pic;
    Game * game;


    Piece(Game *game, int xpos, int ypos, int width, int height, int pos_in_vect, QVector<Piece*> *ac_pieces);
    void mousePressEvent(QGraphicsSceneMouseEvent *event);


    void original_color();
    void selected_color();

    void move_piece(int first_square, int second_square);




    void clear_first_selected();
    void color_potential_moves();



    int position_in_list;
    int xpos, ypos;


    QColor starting_color;
    QColor highlighted_color;
    QColor highlighted_second_color;
    QColor middling_color;
    QColor current_color;
    QColor potential_go_to_color_white;
    QColor potential_go_to_color_black;

    //QGraphicsRectItem * temp_rect;
    bool has_piece;
    bool first_selected;








};

#endif // PIECE_H
