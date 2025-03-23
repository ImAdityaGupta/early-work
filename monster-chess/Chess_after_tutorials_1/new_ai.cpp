#include "new_ai.h"

typedef unsigned long long BB;
using namespace std;

//int findAndClearSetBit(BB& mask) {
//   int sq = __builtin_ctzll(mask); // find least significant bit
//   mask &= ~(1ULL << sq); // clear least significant bit
//   return sq;
//}


string find_out_simple(ChessBoard prior, ChessBoard later){
    vector<BB> all_pieces_prior = {
        prior.WhitePawns,prior.WhiteQueens,prior.WhiteKing,prior.BlackPawns,
        prior.BlackRooks,prior.BlackKnights,prior.BlackBishops,prior.BlackQueens,
        prior.BlackKing
    };
    vector<BB> all_pieces_later = {
        later.WhitePawns,later.WhiteQueens,later.WhiteKing,later.BlackPawns,
        later.BlackRooks,later.BlackKnights,later.BlackBishops,later.BlackQueens,
        later.BlackKing
    };






        string first_square;
        string second_square;

        for (int i=0;i<all_pieces_prior.size();i++){
            for (int j=0;j<64;j++){
                bool previous = (all_pieces_prior[i] >> j) & 1ULL;
                bool afterwards = (all_pieces_later[i] >> j) & 1ULL;
                if (previous==false and afterwards == true){
                    second_square = all_mighty_to_square_converter[j];

                    for (int k=0;k<64;k++){
                        bool temp_prev = (all_pieces_prior[i] >> k) & 1ULL;
                        bool temp_after = (all_pieces_later[i] >> k) & 1ULL;
                        if (temp_prev==true and temp_after==false){
                            first_square = all_mighty_to_square_converter[k];
                            break;
                        }

                    }
                    break;




                }


            }
        }

        return first_square + " " + second_square + "\n";


}


string find_out_move(ChessBoard prior, ChessBoard later, int color){

    vector<BB> all_pieces_prior = {
        prior.WhitePawns,prior.WhiteQueens,prior.WhiteKing,prior.BlackPawns,
        prior.BlackRooks,prior.BlackKnights,prior.BlackBishops,prior.BlackQueens,
        prior.BlackKing
    };
    vector<BB> all_pieces_later = {
        later.WhitePawns,later.WhiteQueens,later.WhiteKing,later.BlackPawns,
        later.BlackRooks,later.BlackKnights,later.BlackBishops,later.BlackQueens,
        later.BlackKing
    };


    if (color==0){
        return find_out_simple(prior,later)+"\n";

    } else if (color==1){
        vector<ChessBoard> one_move_forward;
        one_move_without_check(one_move_forward,prior,color);
        for (auto x: one_move_forward){
            string pot_first_move = find_out_simple(prior,x);
            vector<ChessBoard> second_move_forward;
            one_move_with_check(second_move_forward,x,color);
            for (auto y: second_move_forward){
                if (y==later){
                    string pot_second_move = find_out_simple(x,y);
                    return pot_first_move + pot_second_move + "\n";


                }

            }

        }
        cout << "ERROR" << endl;
        return "";

    }


}





void stress_test(ChessBoard start_pos, int color){
    vector<pair<ChessBoard,int>> all_moves;
    all_moves.push_back({start_pos,0});

    for (int i=0;i<4;i++){

        vector<ChessBoard> temp;
        vector<ChessBoard> to_store;
        for (auto x: all_moves){
            legal_moves(to_store,x.first,color);
            for (auto y: to_store){
                temp.push_back(y);
            }
            to_store.clear();


        }


        all_moves.clear();
        for (auto x: temp){
            all_moves.push_back({x,eval(x,color)});

        }
        color = 1 - color;
        temp.clear();
        cout << all_moves.size() << endl;




    }





}


pair<ChessBoard,int> new_minimax_ab(function<int (const ChessBoard&, int)> func, const ChessBoard &pos, int depth, int color, int depth_to_search, int alpha, int beta){
    nodes_checked++;
    if (nodes_checked%1000000==0){
        cout << nodes_checked << endl;
    }
    bool found_move = false;

    if (depth==depth_to_search){
        int this_val = func(pos,color);
        string gap(depth,' ');
        //cout << gap << depth << " " << this_val << " " << pos.second_last_move <<  " " << pos.last_move << endl;
        return {pos,this_val};

    }

    if (color==0){
        int min_score = INF;
        ChessBoard to_return = pos;
        vector<ChessBoard> layer_deeper;
        legal_moves(layer_deeper,pos,color);
        if (layer_deeper.size()==0){
            string gap(depth,' ');

            //cout << gap << depth << " " << min_score << " " << pos.second_last_move <<  " " << pos.last_move << endl;


            return {pos,min_score};
        }

        for (auto x: layer_deeper){

            pair<ChessBoard,int> pot_deeper = new_minimax_ab(func,x,depth+1,1-color,depth_to_search,alpha,beta);
            if (pot_deeper.second<min_score){
                min_score = pot_deeper.second;
                to_return = pot_deeper.first;
                found_move = true;
            }


            beta = min(min_score, beta);

            if (beta<=alpha){
                break;
            }

        }
        //cout << min_score << endl;
        if (depth==1){
            cout << " ";
        }
        if (depth==0 or depth==1){
            cout << min_score << endl;
        }

        if (depth==0){
            //cout << min_score << " " << pos.second_last_move <<  " " << pos.last_move << endl;
            if (found_move == false){
                return {layer_deeper[0],min_score};
            }

            return {to_return,min_score};
        } else {
            string gap(depth,' ');

            //cout << gap << depth << " " << min_score << " " << pos.second_last_move <<  " " << pos.last_move << endl;

            return {pos,min_score};
        }



    } else if (color==1){
        int max_score = (-INF);
        ChessBoard to_return = pos;
        vector<ChessBoard> layer_deeper;
        legal_moves(layer_deeper,pos,color);
        if (layer_deeper.size()==0){
            string gap(depth,' ');

            //cout << gap << depth << " " << max_score << " " << pos.second_last_move <<  " " << pos.last_move << endl;

            return {pos,max_score};
        }


        for (auto x: layer_deeper){
            pair<ChessBoard,int> pot_deeper = new_minimax_ab(func,x,depth+1,1-color,depth_to_search,alpha,beta);
            if (pot_deeper.second > max_score){
                max_score = pot_deeper.second;
                to_return = pot_deeper.first;
                found_move = true;
            }


            alpha = max(alpha, max_score);

            if (beta <= alpha){
                break;
            }


        }
        if (depth==1){
            cout << " ";
            //cout << board_printer(pos);
        }
        if (depth==0 or depth==1){
            cout << max_score << endl;
        }

        if (depth==0){
            cout << "eval: " << max_score << endl;
            if (found_move == false){
                return {layer_deeper[0],max_score};
            }
            return {to_return,max_score};
            string gap(depth,' ');

            //cout << gap << depth << " " << max_score << " " << pos.second_last_move <<  " " << pos.last_move << endl;

        } else {

            string gap(depth,' ');

            //cout << gap << depth << " " << max_score << " " << pos.second_last_move <<  " " << pos.last_move << endl;

            return {pos,max_score};
        }



    }




}


//first value, then refutation move
//pair<int,BB> negamax(function<int (const ChessBoard&, int)> func, const ChessBoard &pos, int depth, int color, int depth_to_search, int alpha, int beta, const vector<BB> &PV, vector<bool> &PV_found){
pair<int,BB> negamax(function<int (const ChessBoard&, int)> func, const ChessBoard &pos, int depth, int color, int depth_to_search, int alpha, int beta, chrono::time_point<chrono::system_clock> start_time, int max_time){
    chrono::time_point<chrono::system_clock> end_time = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds = end_time - start_time;
    int millis_passed = chrono::duration_cast<chrono::milliseconds>(elapsed_seconds).count();

    if (millis_passed > max_time*1000){
        return {0,1ULL};
    }

    bool to_sort = true;
    if (depth_to_search-depth<=1){
        to_sort = false;
    }

    nodes_checked++;
    if (nodes_checked%1000000==0){
        cout << nodes_checked << endl;
    }
    int alpha_original = alpha;
    //cout << board_printer(pos);
    int mult_fact=(color*2)-1;


    HASHE * phashe = &hash_table[pos.ZobristValue%HASH_LENGTH];
    if (phashe->key==pos.ZobristValue and phashe->hash_depth >= (depth_to_search-depth)){

        //cout << "HASH " << depth_to_search << " " << depth << " " << phashe->hash_depth << endl;
        //cout << pos.ZobristValue << " " << phashe->key << endl;
        //cout << board_printer(pos);

        if (phashe->hash_flag==EXACT){
            hash_exact++;
            //cout << "EXACT" << endl;
            if (depth==0){
                if (phashe->best!=UNDEFINED){
                    return {phashe->hash_value,phashe->best};
                }
                cout << "lol" << endl;
                alpha = max(alpha, phashe->hash_value);
                beta = min(beta, phashe->hash_value);


            } else {
                return {phashe->hash_value,phashe->best};
            }

        } else if (phashe->hash_flag==ALPHA){
            alpha = max(alpha, phashe->hash_value);
            hash_alpha++;
        } else if (phashe->hash_flag==BETA){
            beta = min(beta, phashe->hash_value);
            hash_beta++;
        }

    }



    if (depth==depth_to_search){
        int this_val = func(pos,color);

        return {mult_fact*this_val,UNDEFINED};

    }

    vector<ChessBoard> layer_deeper;
    legal_moves(layer_deeper,pos,color,to_sort);
    if (layer_deeper.size()==0){
        return {-INF,UNDEFINED};
    }

    int value = -INF;
    int position = 0;

//    if (PV.size()>depth){
//        if (!PV_found[depth]){
//            for (int i=0;i<layer_deeper.size();i++){
//                if (layer_deeper[i].ZobristValue==PV[depth]){
//                    //cout << "YEAH, FOUND" << endl;
//                    PV_found[i] = true;
//                    iter_swap(layer_deeper.begin(), layer_deeper.begin()+i);
//                    break;
//
//                }
//            }
//        }
//    }





    for (int i=0;i<layer_deeper.size();i++){
        //int pot_new_val = -(negamax(func, layer_deeper[i], depth+1, 1-color, depth_to_search,-beta,-alpha,PV,PV_found).first);
        pair<int,BB> next_layer = (negamax(func, layer_deeper[i], depth+1, 1-color, depth_to_search,-beta,-alpha,start_time, max_time));

        int pot_new_val = -(next_layer).first;
        if (next_layer.second==1ULL){
            return {0,1ULL};
        }

        if (pot_new_val > value){
            value = pot_new_val;
            position = i;
        }
//        if (depth==0){
//            cout << "FIRST LAYER START" << endl;
//            cout << board_printer(layer_deeper[i]);
//            cout << pot_new_val << endl;
//            cout << "FIRST LAYER END" << endl;
//        }
//        if (depth==2){
//            cout << "SECOND LAYER START" << endl;
//            cout << board_printer(layer_deeper[i]);
//            cout << pot_new_val << endl;
//            cout << "SECOND LAYER END" << endl;
//        }
//        if (depth==4){
//            cout << "THIRD LAYER START" << endl;
//            cout << board_printer(layer_deeper[i]);
//            cout << pot_new_val << endl;
//            cout << "THIRD LAYER END" << endl;
//        }

        alpha = max(alpha, value);
        if (alpha >= beta){
            break;
        }

    }

    //always replace
    phashe->hash_value = value;
    if (depth==0){
        phashe->hash_flag==EXACT;
    } else if (value <= alpha_original){
        phashe->hash_flag = ALPHA;
    } else if (value >= beta){
        phashe->hash_flag = BETA;
    } else {
        phashe->hash_flag = EXACT;
    }
    phashe->hash_depth = depth_to_search-depth;
    phashe->best = layer_deeper[position].ZobristValue;
    phashe->key = pos.ZobristValue;




    return {value,layer_deeper[position].ZobristValue};
//    string gap(depth,' ');
//    cout << gap << depth << " " << value << endl;









}


int time_management(int time_in_seconds){
    return max(double(1),floor(time_in_seconds*0.5));


}




ChessBoard negamax_wrapper(function<int (const ChessBoard&, int)> func, const ChessBoard &pos, int time_in_seconds, int color, bool time_match){
    chrono::time_point<chrono::system_clock> start_time, end_time;

    start_time = chrono::system_clock::now();
    if (time_match){
        time_in_seconds = time_management(time_in_seconds);
    }


    //cout << "f(42) = " << fibonacci(42) << '\n';
    vector<ChessBoard> pot_moves;
    legal_moves(pot_moves, pos, color);
    //ChessBoard to_return;
    pair<int,ChessBoard> to_return;
    //vector<BB> Principal_Variation;


    for (int i=1;;i++){


//        vector<bool> temp_found;
//        for (int j=0;j<Principal_Variation.size();j++){
//            temp_found.push_back(false);
//        }

        //pair<int,BB> from_negamax = negamax(func,pos,0,color,i,-INF,INF,Principal_Variation,temp_found);
        pair<int,BB> from_negamax = negamax(func,pos,0,color,i,-INF,INF,start_time, time_in_seconds);
        if (from_negamax.second==0ULL){
            to_return = {-INF,pot_moves[0]};

        } else if (from_negamax.second==1ULL){
            end_time = chrono::system_clock::now();
            chrono::duration<double> elapsed_seconds = end_time - start_time;
            int millis_passed = chrono::duration_cast<chrono::milliseconds>(elapsed_seconds).count();


            cout << "time " << millis_passed << endl;
            cout << "depth: " << i-1 << endl;
            cout << "evaluation" << to_return.first << endl;
            cout << "nodes: " << nodes_checked << endl;
            all_moves_for_debug.push_back(find_out_move(pos,to_return.second,color));
            return to_return.second;



        } else if (abs(from_negamax.first) == INF){

            end_time = chrono::system_clock::now();
            chrono::duration<double> elapsed_seconds = end_time - start_time;
            int millis_passed = chrono::duration_cast<chrono::milliseconds>(elapsed_seconds).count();

            for (auto x: pot_moves){
                if (x.ZobristValue==from_negamax.second){
                    to_return = {from_negamax.first,x};
                }
            }


            cout << "time " << millis_passed << endl;
            cout << "depth: " << i << endl;
            cout << "evaluation" << to_return.first << endl;
            cout << "nodes: " << nodes_checked << endl;
            all_moves_for_debug.push_back(find_out_move(pos,to_return.second,color));
            return to_return.second;

        } else {
            for (auto x: pot_moves){
                if (x.ZobristValue==from_negamax.second){
                    to_return = {from_negamax.first,x};
                }
            }
        }
        //Principal_Variation.push_back(to_return.ZobristValue);



    }




}





void negamax_ui_wrapper(function<int (const ChessBoard &, int)> func, const ChessBoard &pos, int time_in_seconds, int color, bool time_match, ChessBoard &to_change, bool &machine_thinking, int &color_ref)
{
    cout << board_printer(pos) << endl;

    to_change = negamax_wrapper(func,pos,time_in_seconds,color,time_match);
    machine_thinking = false;
    color_ref = 1 - color_ref;
    cout << board_printer(to_change) << endl;

}

void test_func(int x, bool &machine_thinking, int &color_ref)
{
    x += 10;
    machine_thinking = false;
    color_ref = 1 - color_ref;
    return;
}
