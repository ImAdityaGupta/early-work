#include "class_declarations.h"

using namespace std;

typedef unsigned long long BB;



int find_piece(const ChessBoard &start_pos, int sq){
    if ((start_pos.WhiteKing >> sq) & 1ULL !=0){
        return WHITE_KING_INT;
    }
    if ((start_pos.WhitePawns >> sq) & 1ULL !=0){
        return WHITE_PAWN_INT;
    }
    if ((start_pos.WhiteQueens >> sq) & 1ULL !=0){
        return WHITE_QUEEN_INT;
    }

    if ((start_pos.BlackBishops >> sq) & 1ULL !=0){
        return BLACK_BISHOP_INT;
    }

    if ((start_pos.BlackKing >> sq) & 1ULL !=0){
        return BLACK_KING_INT;
    }

    if ((start_pos.BlackKnights >> sq) & 1ULL !=0){
        return BLACK_KNIGHT_INT;
    }

    if ((start_pos.BlackPawns >> sq) & 1ULL !=0){
        return BLACK_PAWN_INT;
    }

    if ((start_pos.BlackQueens >> sq) & 1ULL !=0){
        return BLACK_QUEEN_INT;
    }

    if ((start_pos.BlackRooks >> sq) & 1ULL !=0){
        return BLACK_ROOK_INT;
    }

    return -1;



}





int manual_num_input(const ChessBoard &start_pos, const vector<ChessBoard> &pot_moves, int color){
//    int temp = (rand()%pot_moves.size());
//    cout << "TEMP: " << temp << endl;
    cout << "POS_MOVE: " << pot_moves.size() << endl;
    int num; cin >> num;
    num = num%pot_moves.size();
    all_moves_for_debug.push_back(find_out_move(start_pos, pot_moves[num],color));

//    for (int i=0;i<pot_moves.size();i++){
//        cout << "POT " << i << endl;
//        board_printer(pot_moves[i]);
//    }

    return num;
}

int random_move(const ChessBoard &start_pos, const vector<ChessBoard> &pot_moves, int color){
    int temp = (rand()%pot_moves.size());
    cout << "TEMP: " << temp << endl;
    cout << "POS_MOVE: " << pot_moves.size() << endl;
    all_moves_for_debug.push_back(find_out_move(start_pos,pot_moves[temp],color));
//    for (int i=0;i<pot_moves.size();i++){
//        cout << "POT " << i << endl;
//        board_printer(pot_moves[i]);
//    }

    return temp;
}


int board_manual_input(const ChessBoard &start_pos, const vector<ChessBoard> &actual_pot_moves, int color){
    while (true){

        if (color==0){
            string from, to; cin >> from;
            if (from=="moves"){
                for (auto x: all_moves_for_debug){
                    cout << x ;
                }
                cout << endl;
                continue;
            } else if (from=="all"){
                for (auto x: actual_pot_moves){
                    board_printer(x);
                }
                cout << "CURRENT BOARD" << endl;
                board_printer(start_pos);
                continue;

            } else if (from=="rand"){
                int rand_move = random_move(start_pos, actual_pot_moves,color);
                //all_moves_for_debug.push_back(rand_move);
                return rand_move;

            } else if (from=="num_moves"){
                cout << all_moves_for_debug.size() << endl;
                continue;

            }


            cin >> to;



            auto it = all_mighty_square_converter.find(from);
            if (it==all_mighty_square_converter.end()){
                cout << "INVALID FIRST SQUARE" << endl;
                continue;
            }
            it = all_mighty_square_converter.find(to);
            if (it==all_mighty_square_converter.end()){
                cout << "INVALID SECOND SQUARE" << endl;
                continue;
            }

            int from_sq = all_mighty_square_converter[from];
            int to_sq = all_mighty_square_converter[to];


            vector<ChessBoard> one_pot_move;

            if (find_piece(start_pos,from_sq)==-1){
                cout << "NO PIECE ON SQUARE" << endl;
                continue;
            }

            createMove(from_sq,to_sq,one_pot_move,start_pos,find_piece(start_pos,from_sq));


            for (int i=0;i<actual_pot_moves.size();i++){
                if (actual_pot_moves[i]==one_pot_move[0]){

                    all_moves_for_debug.push_back(find_out_move(start_pos,actual_pot_moves[i],color));

                    return i;

                }


            }
            cout << "ILLEGAL MOVE" << endl;



        } else if (color==1){
            string from1, to1, from2, to2; cin >> from1;
            if (from1=="moves"){
                for (auto x: all_moves_for_debug){
                    cout << x;
                }
                cout << endl;
                continue;

            } else if (from1=="all"){
                for (auto x: actual_pot_moves){
                    board_printer(x);
                }
                cout << "CURRENT BOARD" << endl;
                board_printer(start_pos);
                continue;

            } else if (from1=="rand"){
                int rand_move = random_move(start_pos, actual_pot_moves,color);
                //all_moves_for_debug.push_back(rand_move);
                return rand_move;

            } else if (from1=="num_moves"){
                cout << all_moves_for_debug.size() << endl;
                continue;

            }



            cin >> to1 >> from2 >> to2;



            auto it = all_mighty_square_converter.find(from1);
            if (it==all_mighty_square_converter.end()){
                cout << "INVALID FIRST MOVE FIRST SQUARE" << endl;
                continue;
            }
            it = all_mighty_square_converter.find(to1);
            if (it==all_mighty_square_converter.end()){
                cout << "INVALID SECOND SQUARE FIRST MOVE" << endl;
                continue;
            }
            it = all_mighty_square_converter.find(from2);
            if (it==all_mighty_square_converter.end()){
                cout << "INVALID FIRST SQUARE SECOND MOVE" << endl;
                continue;
            }
            it = all_mighty_square_converter.find(to2);
            if (it==all_mighty_square_converter.end()){
                cout << "INVALID SECOND SQUARE SECOND MOVE" << endl;
                continue;
            }



            int from_sq1 = all_mighty_square_converter[from1];
            int to_sq1 = all_mighty_square_converter[to1];

            int from_sq2 = all_mighty_square_converter[from2];
            int to_sq2 = all_mighty_square_converter[to2];

            vector<ChessBoard> one_pot_move;

            if (find_piece(start_pos,from_sq1)==-1){
                cout << "NO PIECE ON SQUARE" << endl;
                continue;
            }

            createMove(from_sq1,to_sq1,one_pot_move,start_pos,find_piece(start_pos,from_sq1));

            if (find_piece(one_pot_move[0],from_sq2)==-1){
                cout << "NO PIECE ON SQUARE" << endl;
                continue;
            }
            createMove(from_sq2,to_sq2,one_pot_move,one_pot_move[0],find_piece(one_pot_move[0],from_sq2));


            for (int i=0;i<actual_pot_moves.size();i++){
                if (actual_pot_moves[i]==one_pot_move[1]){
//                    string temp_first_move = all_mighty_to_square_converter[from_sq1]+" "+all_mighty_to_square_converter[to_sq1];
//                    string temp_second_move = all_mighty_to_square_converter[from_sq2]+" "+all_mighty_to_square_converter[to_sq2];
//
//                    all_moves_for_debug.push_back(temp_first_move);
//                    all_moves_for_debug.push_back(temp_second_move);
                    all_moves_for_debug.push_back(find_out_move(start_pos,actual_pot_moves[i],color));


                    return i;

                }


            }
            cout << "ILLEGAL MOVE" << endl;


        }

    }






}




ChessBoard WhitePlayer::move_chooser(const ChessBoard &start_pos, int time_in_seconds, bool time_match){
        vector<ChessBoard> pot_moves;
        legal_moves(pot_moves, start_pos, color);
        //return pot_moves[random_move(start_pos, pot_moves, color)];
        //return pot_moves[manual_num_input(start_pos, pot_moves, color)];

//        for (auto x: pot_moves){
//            cout << board_printer(x);
//
//        }
        //return new_minimax_ab(eval,start_pos,0,color,6,-INF,INF).first;
        return pot_moves[board_manual_input(start_pos,pot_moves,color)];
        //return negamax_wrapper(eval, start_pos, time_in_seconds, color, time_match);


    }





ChessBoard BlackPlayer::move_chooser(const ChessBoard &start_pos, int time_in_seconds, bool time_match){
        cout << "BLACK TURN" << endl;
        vector<ChessBoard> pot_moves;
        legal_moves(pot_moves, start_pos, color);


        //return new_minimax_ab(eval,start_pos,0,color,1,-INF,INF).first;
        //return pot_moves[board_manual_input(start_pos,pot_moves,color)];
        return negamax_wrapper(eval, start_pos, time_in_seconds, color, time_match);

//        pair<int,BB> from_negamax = negamax(eval,start_pos,0,color,5,-INF,INF);
//        if (from_negamax.second==0ULL){
//            return pot_moves[0];
//        } else {
//            for (auto x: pot_moves){
//                if (x.ZobristValue==from_negamax.second){
//                    cout << from_negamax.first*-1 << endl;
//                    return x;
//                }
//            }
//        }
//        cout << "YOU SHOULD NOT BE HERE" << endl;
    }







