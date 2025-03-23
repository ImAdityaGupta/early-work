#pragma once

#ifndef PURE_DEBUG_H_INCLUDED
#define PURE_DEBUG_H_INCLUDED

#include <iostream>
#include <general_algorithm.h>
#include "board_class.h"


using namespace std;

typedef unsigned long long BB;


extern vector<string> all_moves_for_debug;


extern int num_moves_for_debug;
extern map<string,int> all_mighty_square_converter;
extern map<int,string> all_mighty_to_square_converter;


string board_printer(ChessBoard board);

void semiprinter(BB to_print);




#endif // PURE_DEBUG_H_INCLUDED
