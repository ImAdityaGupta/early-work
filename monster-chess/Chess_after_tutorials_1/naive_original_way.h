#pragma once

#ifndef NAIVE_ORIGINAL_WAY_H_INCLUDED
#define NAIVE_ORIGINAL_WAY_H_INCLUDED

#include <iostream>
#include "general_algorithm.h"

using namespace std;
const int neg_inf = -1000000;
const int pos_inf = 1000000;
//class Piece {
//public:
//
//
//
//    string printer(){
//        return to_string(x) + to_string(y);
//
//    }




//};

extern int old_nodes_checked;


class Board {
public:
    Board (vector<vector<char>>, vector<vector<int>>);
    vector<vector<char>> pieces;
    vector<vector<int>> colors;


    pair<pair<int,int>,pair<int,int>> last_move;
    pair<pair<int,int>,pair<int,int>> second_last_move;
    char last_mover;

    string board_printer();
    bool operator==(const Board& pos) const {
        return (pos.pieces == pieces and pos.colors == colors);


    }

};



pair<int,int> not_conv(string to_convert);

string into_not(pair<int,int> orig);

void eff_white_move_printer(const Board &pos);

void eff_black_move_printer(const Board &pos);

void eff_color_printer(const Board &pos, int color);


bool string_valid(string a);


bool does_piece_exist(const Board& pos, pair<int,int> square);

char which_piece_exists(const Board& pos, pair<int,int> square);

int which_color_exists(const Board& pos, pair<int,int> square);

bool is_valid(const pair<int,int>& square);

Board potential_change(Board start_pos, pair<int,int> from_square, pair<int,int> to_square);

void move_in_one_direction(int x, int y, Board start_pos, vector<Board>&to_return, int direc1, int direc2, int color);


vector<Board> possible_moves(int x, int y, const Board &start_pos);

vector<Board> one_move_without_check(const Board &start_pos, int color);

bool check_helper(const Board &pos, int color, int direc1, int direc2, int king_x, int king_y, pair<char,char> pot_checks);



bool is_white_in_check(const Board &pos, int color);

bool is_black_in_check(const Board &pos, int color);


bool is_someone_in_check_naive(const Board &pos, int color);


vector<Board> one_move_with_check(const Board &start_pos, int color);


vector<Board> two_move_with_check(const Board &start_pos, int color);

vector<Board> old_legal_moves(const Board &start_pos, int color);


Board manual_input_white(Board start_pos, int color);

Board manual_input_black(Board start_pos, int color);

Board random_move(Board start_pos, int color);

int val_of_wpawn(int x, int y, int weight);

int val_of_bpawn(int x, int y, int weight);


int val_of_king_distance(pair<int,int> wking_pos, pair<int,int> bking_pos, int weight);

int value_of_pos_bpov(const Board &start_pos, int color);

int value_of_pos_wpov(const Board &start_pos, int color);


int minimax_with_ab(function<int (const Board&, int)> func, Board &pos, int depth, int color, int depth_to_search, int alpha, int beta);



Board minimaxer_try_2(function<int (const Board&, int)> func, Board &start_pos, int color, int depth_to_search);


class white_player {
public:
    int color = 1; // (white)
    Board move_chooser(Board start_pos){



        return manual_input_white(start_pos, color);

        //return minimaxer_try_2(value_of_pos_wpov, start_pos,color,2);

        //return random_move(start_pos,color);


    }

};

class black_player {
public:
    int color = 0; // (black)
    Board move_chooser(Board start_pos){
        vector<Board> possible_moves = old_legal_moves(start_pos, color);

        return manual_input_black(start_pos, color);


        //return minimaxer_try_2(value_of_pos_bpov, start_pos,color,2);

        //return random_move(start_pos,color);


    }


};



string game_over(Board& pos, int color, string &winner);





void main_loop();


vector<Board> num_moves_in_a_position(vector<char> common_board= {
            'r','n','b','q','k','b','n','r',
            'p','p','p','p','p','p','p','p',
            '.','.','.','.','.','.','.','.',
            '.','.','.','.','.','.','.','.',
            '.','.','.','.','.','.','.','.',
            '.','.','.','.','.','.','.','.',
            '.','.','P','P','P','P','.','.',
            '.','.','.','.','K','.','.','.'

    }, int color=0);








#endif // NAIVE_ORIGINAL_WAY_H_INCLUDED
