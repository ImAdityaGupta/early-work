#pragma once

#ifndef NEW_AI_H_INCLUDED
#define NEW_AI_H_INCLUDED

#include <iostream>
#include <general_algorithm.h>
#include <fstream>

#include "const_setups.h"
#include "board_class.h"
#include "pure_debug.h"
#include "move_generation.h"

typedef unsigned long long BB;
using namespace std;

//int findAndClearSetBit(BB& mask) {
//   int sq = __builtin_ctzll(mask); // find least significant bit
//   mask &= ~(1ULL << sq); // clear least significant bit
//   return sq;
//}


string find_out_simple(ChessBoard prior, ChessBoard later);
string find_out_move(ChessBoard prior, ChessBoard later, int color);



void stress_test(ChessBoard start_pos, int color);
pair<ChessBoard,int> new_minimax_ab(function<int (const ChessBoard&, int)> func, const ChessBoard &pos, int depth, int color, int depth_to_search, int alpha, int beta);


//first value, then refutation move
//pair<int,BB> negamax(function<int (const ChessBoard&, int)> func, const ChessBoard &pos, int depth, int color, int depth_to_search, int alpha, int beta, const vector<BB> &PV, vector<bool> &PV_found){
pair<int,BB> negamax(function<int (const ChessBoard&, int)> func, const ChessBoard &pos, int depth, int color, int depth_to_search, int alpha, int beta, chrono::time_point<chrono::system_clock> start_time, int max_time);


int time_management(int time_in_seconds);




ChessBoard negamax_wrapper(function<int (const ChessBoard&, int)> func, const ChessBoard &pos, int time_in_seconds, int color, bool time_match);


void negamax_ui_wrapper(function<int (const ChessBoard&, int)> func, const ChessBoard &pos, int time_in_seconds, int color, bool time_match, ChessBoard &to_change, bool &machine_thinking, int &color_ref);

void test_func(int x);

#endif // NEW_AI_H_INCLUDED
