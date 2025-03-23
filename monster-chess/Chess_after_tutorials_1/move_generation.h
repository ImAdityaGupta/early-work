#pragma once

#ifndef MOVE_GENERATION_H_INCLUDED
#define MOVE_GENERATION_H_INCLUDED

#include <iostream>
#include <general_algorithm.h>
#include "board_class.h"
#include "pure_debug.h"
#include "const_setups.h"
using namespace std;

typedef unsigned long long BB;


int isPowerOfTwo(const BB &n);
// Returns position of the only set bit in 'n'; a8 = 63, h1 = 0
int findPosition(const BB &n);
// a8 = 63, h1 = 0
int findAndClearSetBit(BB& mask);



//bit scan reverse
int bsr(BB bb);

BB naive_zobrist(const ChessBoard &pos);

BB smart_zobrist(BB old_zobrist, const int from_sq, const int to_sq, int type_of_piece, int piece_taken);

void createMove(const int from_sq, int to_sq, vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, int type_of_piece);

void createAllMoves(vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, BB &mask, const int from_sq, int type_of_piece);

BB compute_king(const BB &king_loc, const BB &own_side);

BB one_knight(const BB &knight_loc, const BB &own_side);

void compute_knight(BB knight_locs, const BB &own_side, vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color);

BB one_white_pawn(const BB &white_pawn_loc, const BB &all_pieces, const BB &all_black_pieces);

BB two_move_white_pawn(const BB &white_pawn_loc, const BB &all_pieces, const BB &all_black_pieces, bool to_print = false);

BB white_pawn_to_queen(const BB &white_pawn_loc, const BB &all_pieces, const BB &all_black_pieces, bool to_print = false);

BB one_black_pawn_attacks(const BB &black_pawn_loc, const BB &all_pieces, const BB &all_white_pieces, bool to_print = false);


void compute_white_pawns(BB pawn_locs, const BB &own_side, vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color);

BB one_black_pawn(const BB &black_pawn_loc, const BB &all_pieces, const BB &all_white_pieces);

void compute_black_pawns(BB pawn_locs, const BB &own_side, vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color);

BB lineAttacks(const BB &const_occ, const int piece_pos, const int direc);


BB one_rook(const BB &black_rook_loc, const BB &all_pieces, const BB &own_side);

void compute_rook(BB rook_locs, const BB &own_side, vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color, const int type_of_piece);


BB one_bishop(const BB &black_bishop_loc, const BB &all_pieces, const BB &own_side);



void compute_bishop(BB bishop_locs, const BB &own_side, vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color, const int type_of_piece);


bool is_someone_in_check(const ChessBoard &start_pos, const int color);

void one_move_without_check(vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color);

void one_move_with_check(vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color);

void two_moves_with_check(vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color);


const int vals[9] = {
    4,      // WHITE PAWN
    INF,    // WHITE QUEEN
    0,      // WHITE KING (SHOULD PROBABLY NEVER BE ACCESSED)

    -1,     // BLACK PAWN
    -5,     // BLACK ROOK
    -3,     // BLACK KNIGHT
    -3,     // BLACK BISHOP
    -10,     // BLACK QUEEN
    0       // BLACK KING (SHOULD PROBABLY NEVER BE ACCESSED)
};
//map<int,int> vals = {
//    {BLACK_BISHOP_INT,-3},
//    {BLACK_KNIGHT_INT,-3},
//    {BLACK_QUEEN_INT,-9},
//    {BLACK_PAWN_INT,-1},
//    {BLACK_ROOK_INT,-5},
//
//    {WHITE_PAWN_INT,4},
//    {WHITE_QUEEN_INT,INF}
//
//};
const int black_pawns_vals[8] = {
    -0,     //
    -4,     // 2nd rank
    -3,     // 3rd rank
    -2,     // etc.
    -2,
    -1,
    -1,
    -0
};

const int white_pawns_vals[8] = {
    0,     //
    1,     // 2nd rank
    1,     // 3rd rank
    2,     // etc.
    3,
    4,
    5,
    0
};

const int val_blocked_wpawn = -2;
const int val_connected_wpawn = 1;

int eval(const ChessBoard &one_board, int color);


bool spec_sort_white_tt(const ChessBoard &c1, const ChessBoard &c2);
bool spec_sort_black_tt(const ChessBoard &c1, const ChessBoard &c2);

bool spec_sort_white(const ChessBoard &c1, const ChessBoard &c2);
bool spec_sort_black(const ChessBoard &c1, const ChessBoard &c2);


bool spec_sort_white_tt_depth(const ChessBoard &c1, const ChessBoard &c2);
bool spec_sort_black_tt_depth(const ChessBoard &c1, const ChessBoard &c2);

bool spec_sort(const ChessBoard &c1, const ChessBoard &c2);

void legal_moves(vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color, bool to_sort=true);

void whites_potential_first_moves_for_ui(vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color, bool to_sort=true);

void whites_potential_second_moves_for_ui(vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color, bool to_sort=true);



#endif // MOVE_GENERATION_H_INCLUDED
