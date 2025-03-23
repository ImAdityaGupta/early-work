#include <QApplication>
#include "board_class.h"
#include "game.h"
#include <QDebug>

#include <future>

#include <iostream>
#include <general_algorithm.h>
#include <unistd.h>


#include "const_setups.h"
#include "board_class.h"
#include "pure_debug.h"
#include "move_generation.h"
#include "class_declarations.h"
#include "new_ai.h"
#include "debug_help.h"


Game * game;




int main(int argc, char *argv[]){
    srand((unsigned) time(0));
    init_ms1bTable();
    init_zobrist();




    QApplication a(argc, argv);


    ChessBoard classic_pos = ChessBoard(
                    0b0000000000000000000000000000000000000000000000000011110000000000,
                    0b0000000000000000000000000000000000000000000000000000000000000000,
                    0b0000000000000000000000000000000000000000000000000000000000001000,
                    0b0000000011111111000000000000000000000000000000000000000000000000,
                    0b1000000100000000000000000000000000000000000000000000000000000000,
                    0b0100001000000000000000000000000000000000000000000000000000000000,
                    0b0010010000000000000000000000000000000000000000000000000000000000,
                    0b0001000000000000000000000000000000000000000000000000000000000000,
                    0b0000100000000000000000000000000000000000000000000000000000000000);
    QString qstr = QString::fromStdString(classic_pos.convert_into_string());



    game = new Game(800,600,classic_pos);
    game->show();

    return a.exec();
}
