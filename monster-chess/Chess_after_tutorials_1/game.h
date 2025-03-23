#pragma once

#ifndef GAME_H
#define GAME_H


#include <QGraphicsView>
#include <QWidget>
#include <QGraphicsScene>
#include <QGraphicsRectItem>
#include <QObject>
#include <QGraphicsItem>
#include <QTimer>
#include <QGraphicsTextItem>
#include <QFont>
#include <QDebug>
#include <QVector>
#include <QLabel>

#include "board_class.h"
#include "more_display_stuff.h"

class Piece;


class Game: public QGraphicsView {
public:
    Game(int width, int height, ChessBoard starting_position);
    QGraphicsScene * scene;

    QVector<Piece*> * all_pieces;

    int whose_turn;

    bool first_move;
    bool game_over = false;
    bool game_in_progress = false;
    bool human_is_white = false;
    bool human_is_black = true;
    bool machine_thinking = false;

    std::string * pieces_in_string;

    void make_engine_move();
    bool check_if_move_is_legal(int first_square, int second_square);
    void setup_board_from_chessboard(ChessBoard position);

    bool has_any_square_been_selected();
    int which_square_has_been_selected();

    QVector<Start_Button*> choose_colors;






};

#endif // GAME_H
