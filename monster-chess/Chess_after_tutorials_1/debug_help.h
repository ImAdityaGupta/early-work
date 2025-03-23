#pragma once

#ifndef DEBUG_HELP_H_INCLUDED
#define DEBUG_HELP_H_INCLUDED

#include <iostream>
#include "general_algorithm.h"
#include <fstream>


#include "const_setups.h"
#include "board_class.h"
#include "pure_debug.h"
#include "move_generation.h"
#include "class_declarations.h"
#include "naive_original_way.h"


typedef unsigned long long BB;
using namespace std;
ChessBoard original_chessboard(string normal);
vector<char> random_chess_board(int color=0);
string fen_creator(string weird_way, int color);
vector<char> from_fen(string fen);


void main_debug(int color=1, string fen="");

#endif // DEBUG_HELP_H_INCLUDED
