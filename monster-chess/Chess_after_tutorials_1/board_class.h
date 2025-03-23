#pragma once

#ifndef BOARD_CLASS_H_INCLUDED
#define BOARD_CLASS_H_INCLUDED

#include <iostream>
using namespace std;
typedef unsigned long long BB;


class ChessBoard{
public:
    string last_move;
    string second_last_move;
    BB ZobristValue;
    /* The white piece positions */
    BB WhitePawns;
    BB WhiteQueens;
    BB WhiteKing;

    /* The black piece positions */
    BB BlackPawns;
    BB BlackRooks;
    BB BlackKnights;
    BB BlackBishops;
    BB BlackQueens;
    BB BlackKing;

    /* Commonly derived positions */
    BB AllWhitePieces;
    BB AllBlackPieces;
    BB AllPieces;

    ChessBoard(){
        WhitePawns = 0ULL;
        WhiteQueens = 0ULL;
        WhiteKing = 0ULL;

        BlackPawns = 0ULL;
        BlackRooks = 0ULL;
        BlackKnights = 0ULL;
        BlackBishops = 0ULL;
        BlackQueens = 0ULL;
        BlackKing = 0ULL;

        AllWhitePieces = WhitePawns | WhiteQueens | WhiteKing;
        AllBlackPieces = BlackBishops | BlackKing | BlackKnights | BlackPawns | BlackQueens | BlackRooks;
        AllPieces = AllBlackPieces | AllWhitePieces;


    }


    ChessBoard(BB a, BB b, BB c, BB d, BB e, BB f, BB g, BB h, BB i, BB pot_zob=0ULL) {
        WhitePawns = a;
        WhiteQueens = b;
        WhiteKing = c;

        BlackPawns = d;
        BlackRooks = e;
        BlackKnights = f;
        BlackBishops = g;
        BlackQueens = h;
        BlackKing = i;

        AllWhitePieces = WhitePawns | WhiteQueens | WhiteKing;
        AllBlackPieces = BlackBishops | BlackKing | BlackKnights | BlackPawns | BlackQueens | BlackRooks;
        AllPieces = AllBlackPieces | AllWhitePieces;

        if (pot_zob!=0){
            ZobristValue = pot_zob;
        }

    }

    friend bool operator== (const ChessBoard &c1, const ChessBoard &c2){
        //return (c1.ZobristValue == c2.ZobristValue);
        return (c1.WhitePawns == c2.WhitePawns and
                c1.WhiteQueens == c2.WhiteQueens and
                c1.WhiteKing == c2.WhiteKing and

                c1.BlackPawns == c2.BlackPawns and
                c1.BlackRooks == c2.BlackRooks and
                c1.BlackKnights == c2.BlackKnights and
                c1.BlackBishops == c2.BlackBishops and
                c1.BlackQueens == c2.BlackQueens and
                c1.BlackKing == c2.BlackKing);
    }

    string convert_into_string();




};


ChessBoard convert_string_into_board_for_ui(string s);


#endif // BOARD_CLASS_H_INCLUDED
