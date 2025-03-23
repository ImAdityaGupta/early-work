#pragma once

#ifndef CLASS_DECLARATIONS_H_INCLUDED
#define CLASS_DECLARATIONS_H_INCLUDED

#include <iostream>
#include <general_algorithm.h>
#include "board_class.h"
#include "move_generation.h"
#include "pure_debug.h"
#include "new_ai.h"
using namespace std;

typedef unsigned long long BB;



int find_piece(const ChessBoard &start_pos, int sq);





int manual_num_input(const ChessBoard &start_pos, const vector<ChessBoard> &pot_moves, int color);
int random_move(const ChessBoard &start_pos, const vector<ChessBoard> &pot_moves, int color);

int board_manual_input(const ChessBoard &start_pos, const vector<ChessBoard> &actual_pot_moves, int color);


class WhitePlayer {
public:
    int color = 1;

    ChessBoard move_chooser(const ChessBoard &start_pos, int time_in_seconds=20, bool time_match=false);

};


class BlackPlayer {
public:
    int color = 0;

    ChessBoard move_chooser(const ChessBoard &start_pos, int time_in_seconds=20, bool time_match=false);


};




#endif // CLASS_DECLARATIONS_H_INCLUDED
