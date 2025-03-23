#include "board_class.h"
using namespace std;
std::__cxx11::string ChessBoard::convert_into_string(){
    string to_return = "";


    for (int i=0;i<64;i++){
        if ((this->WhitePawns >> i) & 1ULL){
            to_return += 'P';
            continue;
        }
        if ((this->WhiteQueens >> i) & 1ULL){
            to_return += 'Q';
            continue;
        }
        if ((this->WhiteKing >> i) & 1ULL){
            to_return += 'K';
            continue;
        }
        if ((this->BlackPawns >> i) & 1ULL){
            to_return += 'p';
            continue;
        }
        if ((this->BlackRooks >> i) & 1ULL){
            to_return += 'r';
            continue;
        }
        if ((this->BlackKnights >> i) & 1ULL){
            to_return += 'n';
            continue;
        }
        if ((this->BlackBishops >> i) & 1ULL){
            to_return += 'b';
            continue;
        }
        if ((this->BlackQueens >> i) & 1ULL){
            to_return += 'q';
            continue;
        }
        if ((this->BlackKing >> i) & 1ULL){
            to_return += 'k';
            continue;
        }
        to_return += ' ';



    }

    return to_return;



}
ChessBoard convert_string_into_board_for_ui(string s){
    BB a=0ULL, b=0ULL, c=0ULL, d=0ULL, e=0ULL, f=0ULL, g=0ULL, h=0ULL, i=0ULL;
    for (int j=0;j<64;j++){
        if (s[j]=='P'){
            a |= 1ULL << j;

        } else if (s[j]=='Q'){
            b |= 1ULL << j;

        } else if (s[j]=='K'){
            c |= 1ULL << j;

        } else if (s[j]=='p'){
            d |= 1ULL << j;

        } else if (s[j]=='r'){
            e |= 1ULL << j;

        } else if (s[j]=='n'){
            f |= 1ULL << j;

        } else if (s[j]=='b'){
            g |= 1ULL << j;

        } else if (s[j]=='q'){
            h |= 1ULL << j;

        } else if (s[j]=='k'){
            i |= 1ULL << j;

        }


    }



    return ChessBoard(a,b,c,d,e,f,g,h,i);



}
