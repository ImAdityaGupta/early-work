#include "debug_help.h"

typedef unsigned long long BB;
using namespace std;
ChessBoard original_chessboard(string normal){

    BB a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0, i=0;
    int pos = 63;
    for (auto s: normal){


//            BB a = start_pos.WhitePawns;
//            BB b = start_pos.WhiteQueens;
//            BB c = start_pos.WhiteKing;
//
//            BB d = start_pos.BlackPawns;
//            BB e = start_pos.BlackRooks;
//            BB f = start_pos.BlackKnights;
//            BB g = start_pos.BlackBishops;
//            BB h = start_pos.BlackQueens;
//            BB i = start_pos.BlackKing;

        if (s==' '){
            continue;
        }
        if (s=='.'){
            pos--;

        }
        if (s=='P'){
            a ^= (-1 ^ a) & (1ULL << pos);
            pos--;
        }
        if (s=='Q'){
            b ^= (-1 ^ b) & (1ULL << pos);
            pos--;
        }
        if (s=='K'){
            c ^= (-1 ^ c) & (1ULL << pos);
            pos--;
        }
        if (s=='p'){
            d ^= (-1 ^ d) & (1ULL << pos);
            pos--;
        }
        if (s=='r'){
            e ^= (-1 ^ e) & (1ULL << pos);
            pos--;
        }
        if (s=='n'){
            f ^= (-1 ^ f) & (1ULL << pos);
            pos--;
        }
        if (s=='b'){
            g ^= (-1 ^ g) & (1ULL << pos);
            pos--;
        }
        if (s=='q'){
            h ^= (-1 ^ h) & (1ULL << pos);
            pos--;
        }
        if (s=='k'){
            i ^= (-1 ^ i) & (1ULL << pos);
            pos--;
        }

    }
    ChessBoard to_return(a,b,c,d,e,f,g,h,i);
    to_return.ZobristValue = naive_zobrist(to_return);


    return to_return;





}


vector<char> random_chess_board(int color){
    vector<char> total_list = {
            'r','n','b','q','k','b','n','r',
            'p','p','p','p','p','p','p','p',
            '.','.','.','.','.','.','.','.',
            '.','.','.','.','.','.','.','.',
            '.','.','.','.','.','.','.','.',
            '.','.','.','.','.','.','.','.',
            '.','.','P','P','P','P','.','.',
            '.','.','.','.','K','.','.','.'

    };

    while (true){
        random_shuffle(total_list.begin(), total_list.end());

        string temp_convert;
        for (auto x: total_list){
            temp_convert += x;
        }




        bool do_continue = false;
        for (int i=0;i<8;i++){
            if (total_list[i]=='p' or total_list[i]=='P'){
                do_continue = true;
            }
        }
        for (int i=56;i<64;i++){
            if (total_list[i]=='p' or total_list[i]=='P'){
                do_continue = true;
            }
        }
        if (color==1){
            for (int i=0;i<16;i++){
                if (total_list[i]=='P'){

                    do_continue = true;
                }
            }


        }


        if (do_continue){
            continue;
        }

        ChessBoard for_checking_check = original_chessboard(temp_convert);



        if (is_someone_in_check(for_checking_check,1-color)){
            //cout << "CHECK" << endl;
            continue;
        }
        break;

    }


//    auto rng = default_random_engine{};
//    shuffle(begin(total_list),end(total_list), rng);
    return total_list;



}

string fen_creator(string weird_way, int color){
    int gaps = 0;
    string to_return = "";


    for (int i=0;i<weird_way.size();i++){
        if (weird_way[i]!='.'){
            if (gaps!=0){
                to_return += to_string(gaps);
                gaps = 0;
            }
            to_return += weird_way[i];
        } else {
            gaps++;

        }
        if (i%8==7){
            if (gaps!=0){
                to_return += to_string(gaps);
                gaps = 0;
            }
            to_return += '/';
        }



    }
    to_return = to_return.substr(0,to_return.size()-1);

    if (color==0){
        to_return += " b ";
    } else {
        to_return += " w ";
    }
    to_return += "- - 0 1";

    return to_return;


}

vector<char> from_fen(string fen){
    vector<char> to_return;
    for (auto s: fen){
        if (isdigit(s)){
            int x = (int) s - '0';
            for (int i=0;i<x;i++){
                to_return.push_back('.');
            }


        } else if (s=='/'){
            continue;
        } else if (s==' '){
            break;

        } else {
            to_return.push_back(s);

        }

    }
    return to_return;


}



void main_debug(int color, string fen){
    vector<char> naive_one;
    if (fen==""){
         naive_one = random_chess_board(color);
    } else {
        naive_one = from_fen(fen);
    }

    string temp_convert;
    for (auto x: naive_one){
        temp_convert += x;
    }
    ChessBoard for_checking_check = original_chessboard(temp_convert);
    vector<ChessBoard> checking_legal_moves;

    legal_moves(checking_legal_moves,for_checking_check,color);

    vector<Board> original_way = num_moves_in_a_position(naive_one,0);


    cout << "ORIGINAL WAY: ";
    cout << original_way.size() << endl;
    cout << "NEW WAY: ";
    cout << checking_legal_moves.size() << endl;

    cout << fen_creator(temp_convert,color) << endl;

    //string printer; cin >> printer;

    ofstream myfile;

    myfile.open("debug_output.txt", ofstream::out | ofstream::trunc);




    myfile << "ORIGINAL WAY: ";
    myfile << original_way.size() << endl;
    myfile << "NEW WAY: ";
    myfile << checking_legal_moves.size() << endl;

    myfile << fen_creator(temp_convert,color) << endl;



    myfile << "NEW WAY: " << endl;

    vector<string> to_print;


    for (auto x: checking_legal_moves){
        //myfile << board_printer(x);
        to_print.push_back(find_out_move(for_checking_check,x,color));

    }

    sort(to_print.begin(),to_print.end());

    for (auto x: to_print){
        myfile << x;


    }
    for (auto x: checking_legal_moves){
        myfile << board_printer(x);
        myfile << find_out_move(for_checking_check,x,color);

    }
//    myfile << "ORIGINAL WAY: " << endl;
//    for (auto x: original_way){
//        myfile << x.board_printer();
//
//    }


    myfile.close();









}



