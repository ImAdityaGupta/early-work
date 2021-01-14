#include <iostream>
#include <bits/stdc++.h>
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

int nodes_checked = 0;


class Board {
public:
    Board (vector<vector<char>>, vector<vector<int>>);
    vector<vector<char>> pieces;
    vector<vector<int>> colors;

    pair<pair<int,int>,pair<int,int>> last_move;
    pair<pair<int,int>,pair<int,int>> second_last_move;
    char last_mover;

    void board_printer(){
        string to_print = "";
//        for (auto x: pieces){
//            for (auto y: x){
//                cout << y << " ";
//
//
//
//            }
//            cout << endl;
//
//
//        }
//        cout << endl;

        string temp_last_mover(1,last_mover);


        to_print += (temp_last_mover + ":" + "\n");
        to_print += (to_string(last_move.first.first) + " " + to_string(last_move.first.second) + "\n");
        to_print += (to_string(last_move.second.first) + " " + to_string(last_move.second.second) + "\n");
        //cout << to_print << endl;
//        cout << last_mover << ":" << endl;
//        cout << last_move.first.first << " " << last_move.first.second << endl;
//        cout << last_move.second.first << " " << last_move.second.second << endl;

        for (int i=0;i<8;i++){
            to_print += (" " + to_string(8-i) + " | ");
            //cout << " " << 8-i << " | ";
            for (int j=0;j<8;j++){
                if (colors[i][j]==2 or colors[i][j]==1){
                    string chtstr(1,pieces[i][j]);
                    to_print += (chtstr + " | ");
                    //cout << pieces[i][j] << " | ";
                } else if (colors[i][j]==0){
                    string chtstr(1,char(tolower(pieces[i][j])));
                    to_print += (chtstr + " | ");

                    //cout << char(tolower(pieces[i][j])) << " | ";
                }




            }
            to_print += "\n";
            //cout << endl;
            to_print += ("---|---|---|---|---|---|---|---|---|");
            to_print += "\n";

            //cout << "---|---|---|---|---|---|---|---|---|" << endl;


        }
        to_print += ("   | A | B | C | D | E | F | G | H |");
        to_print += "\n";
        //cout << "   | A | B | C | D | E | F | G | H |" << endl;
        cout << to_print << endl;

    }

    bool operator==(const Board& pos) const {
        return (pos.pieces == pieces and pos.colors == colors);


    }

};


Board::Board(vector<vector<char>> a, vector<vector<int>> b){
    assert(a.size()==8);
    assert(b.size()==8);

    pieces = a;
    colors = b;



}


pair<int,int> not_conv(string to_convert){
    map<char,int> key = {
        {'a',0},{'b',1},{'c',2},{'d',3},{'e',4},{'f',5},{'g',6},{'h',7},
        {'1',7},{'2',6},{'3',5},{'4',4},{'5',3},{'6',2},{'7',1},{'8',0}
    };
    pair<int,int> to_return;

    to_return.first = key[to_convert[1]];
    to_return.second = key[to_convert[0]];
    return to_return;



}

string into_not(pair<int,int> orig){
    string first_key = "87654321";
    string second_key = "abcdefgh";

    string to_return = "";

    to_return += second_key[orig.second];
    to_return += first_key[orig.first];

    return to_return;

}

void eff_white_move_printer(const Board &pos){
    cout << into_not(pos.second_last_move.first) << " " <<into_not(pos.second_last_move.second) << endl;
    cout << into_not(pos.last_move.first) << " " <<into_not(pos.last_move.second) << endl;
    cout << endl;
}

void eff_black_move_printer(const Board &pos){
    //cout << into_not(pos.second_last_move.first) << " " <<into_not(pos.second_last_move.second) << endl;
    cout << into_not(pos.last_move.first) << " " <<into_not(pos.last_move.second) << endl;
    cout << endl;
}

void eff_color_printer(const Board &pos, int color){
    if (color==1){
        eff_black_move_printer(pos);

    } else if (color==0){
        eff_white_move_printer(pos);
    }
}


bool string_valid(string a){
    string valid_chars = "abcdefgh";
    string valid_nums = "12345678";

    if (a.size()!=2){
        return false;
    }
    bool found1 = false, found2 = false;

    for (auto x: valid_chars){
        if (x==a[0]){
            found1 = true;
        }
    }
    for (auto x: valid_nums){
        if (x==a[1]){
            found2 = true;
        }
    }
    if (found1 and found2){
        return true;
    }
    return false;

}


bool does_piece_exist(const Board& pos, pair<int,int> square){
    int x = square.first;
    int y = square.second;
    if (pos.pieces[x][y]=='.'){
        return false;
    }
    return true;



}

char which_piece_exists(const Board& pos, pair<int,int> square){
    int x = square.first;
    int y = square.second;
    return pos.pieces[x][y];

}

int which_color_exists(const Board& pos, pair<int,int> square){
    int x = square.first;
    int y = square.second;
    return pos.colors[x][y];

}


bool is_valid(const pair<int,int>& square){
    int x = square.first;
    int y = square.second;

    if (x >=0 and x<=7 and y>=0 and y<=7){
        return true;
    }
    return false;

}

Board potential_change(Board start_pos, pair<int,int> from_square, pair<int,int> to_square){
    int x1 = from_square.first;
    int y1 = from_square.second;
    int x2 = to_square.first;
    int y2 = to_square.second;
    start_pos.last_mover = start_pos.pieces[x1][y1];

    start_pos.pieces[x2][y2] = start_pos.pieces[x1][y1];
    start_pos.colors[x2][y2] = start_pos.colors[x1][y1];

    start_pos.pieces[x1][y1] = '.';
    start_pos.colors[x1][y1] = 2;

    if (start_pos.pieces[x2][y2] == 'P'){
        if (x2==0 or x2==7){
            start_pos.pieces[x2][y2] = 'Q';
        }

    }

    start_pos.second_last_move = start_pos.last_move;

    start_pos.last_move = {from_square, to_square};


    return start_pos;


}

void move_in_one_direction(int x, int y, Board start_pos, vector<Board>&to_return, int direc1, int direc2, int color){
    for (int i=1;i<8;i++){
        pair<int,int> temp_square = {x+(direc1*i),y+(direc2*i)};
        if (!is_valid(temp_square)){
            break;
        }
        if (does_piece_exist(start_pos,temp_square)){
            if (which_color_exists(start_pos,temp_square)==color){
                break;
            } else {
                Board temp = potential_change(start_pos,{x,y},temp_square);
                to_return.push_back(temp);
                break;
            }
        } else {
            Board temp = potential_change(start_pos,{x,y},temp_square);
            to_return.push_back(temp);
        }
    }
}


vector<Board> possible_moves(int x, int y, const Board &start_pos){
    vector<Board> to_return;
    char id = start_pos.pieces[x][y];
    int color = start_pos.colors[x][y];


    if (id=='.'){
        return to_return;
    }


    if (id=='P'){
        if (color==0){
            if (x==1){
                vector<pair<int,int>> pot_take_squares = {{1+x,-1+y},{1+x,1+y}};
                vector<pair<int,int>> pot_move_squares = {{1+x,0+y},{2+x,0+y}};
                for (auto square: pot_take_squares){
                    if (!is_valid(square)){
                        continue;
                    }
                    if (!does_piece_exist(start_pos,square)){
                        continue;
                    }
                    if (which_color_exists(start_pos,square)==color){
                        continue;
                    }
                    Board temp = potential_change(start_pos,{x,y},square);
                    to_return.push_back(temp);


                }
                if (does_piece_exist(start_pos,pot_move_squares[0])){
                    ;
                } else {
                    Board temp = potential_change(start_pos,{x,y},pot_move_squares[0]);
                    to_return.push_back(temp);
                    if (does_piece_exist(start_pos,pot_move_squares[1])){
                        ;

                    } else {
                        Board second_temp = potential_change(start_pos,{x,y},pot_move_squares[1]);
                        to_return.push_back(second_temp);
                    }



                }








            } else if (x==7){
                cout << "PAWN PROMOTION FAILED" << endl;

            } else {
                vector<pair<int,int>> pot_take_squares = {{1+x,-1+y},{1+x,1+y}};
                vector<pair<int,int>> pot_move_squares = {{1+x,0+y}};
                for (auto square: pot_take_squares){
                    if (!is_valid(square)){
                        continue;
                    }
                    if (!does_piece_exist(start_pos,square)){
                        continue;
                    }
                    if (which_color_exists(start_pos,square)==color){
                        continue;
                    }
                    Board temp = potential_change(start_pos,{x,y},square);
                    to_return.push_back(temp);


                }
                if (does_piece_exist(start_pos,pot_move_squares[0])){
                    ;
                } else {
                    Board temp = potential_change(start_pos,{x,y},pot_move_squares[0]);
                    to_return.push_back(temp);
                }

            }


        } else if (color==1){

            if (x==6){
                vector<pair<int,int>> pot_take_squares = {{-1+x,-1+y},{-1+x,1+y}};
                vector<pair<int,int>> pot_move_squares = {{-1+x,0+y},{-2+x,0+y}};
                for (auto square: pot_take_squares){
                    if (!is_valid(square)){
                        continue;
                    }
                    if (!does_piece_exist(start_pos,square)){
                        continue;
                    }
                    if (which_color_exists(start_pos,square)==color){
                        continue;
                    }
                    Board temp = potential_change(start_pos,{x,y},square);
                    to_return.push_back(temp);


                }
                if (does_piece_exist(start_pos,pot_move_squares[0])){
                    ;
                } else {
                    Board temp = potential_change(start_pos,{x,y},pot_move_squares[0]);
                    to_return.push_back(temp);
                    if (does_piece_exist(start_pos,pot_move_squares[1])){
                        ;

                    } else {
                        Board second_temp = potential_change(start_pos,{x,y},pot_move_squares[1]);
                        to_return.push_back(second_temp);
                    }



                }








            } else if (x==0){
                cout << "PAWN PROMOTION FAILED" << endl;

            } else {
                vector<pair<int,int>> pot_take_squares = {{-1+x,-1+y},{-1+x,1+y}};
                vector<pair<int,int>> pot_move_squares = {{-1+x,0+y}};
                for (auto square: pot_take_squares){
                    if (!is_valid(square)){
                        continue;
                    }
                    if (!does_piece_exist(start_pos,square)){
                        continue;
                    }
                    if (which_color_exists(start_pos,square)==color){
                        continue;
                    }
                    Board temp = potential_change(start_pos,{x,y},square);
                    to_return.push_back(temp);


                }
                if (does_piece_exist(start_pos,pot_move_squares[0])){
                    ;
                } else {
                    Board temp = potential_change(start_pos,{x,y},pot_move_squares[0]);
                    to_return.push_back(temp);
                }

            }

        }




    } else if (id=='N'){
        vector<pair<int,int>> pot_squares = {
            {x+2,y+1},
            {x+2,y-1},
            {x+1,y+2},
            {x+1,y-2},
            {x-2,y+1},
            {x-2,y-1},
            {x-1,y+2},
            {x-1,y-2}};
        for (auto square: pot_squares){
            if (!is_valid(square)){
                continue;
            }
            if (!does_piece_exist(start_pos,square)){
                Board temp = potential_change(start_pos,{x,y},square);
                to_return.push_back(temp);

                continue;
            }
            if (which_color_exists(start_pos,square)==color){
                continue;
            } else {
                Board temp = potential_change(start_pos,{x,y},square);
                to_return.push_back(temp);

            }


        }

    } else if (id=='R'){
        move_in_one_direction(x,y,start_pos,to_return,0,1,color);
        move_in_one_direction(x,y,start_pos,to_return,0,-1,color);
        move_in_one_direction(x,y,start_pos,to_return,1,0,color);
        move_in_one_direction(x,y,start_pos,to_return,-1,0,color);
    } else if (id=='K'){
        vector<pair<int,int>> pot_squares = {
            {x+1,y+1},
            {x+1,y},
            {x+1,y-1},
            {x,y-1},
            {x,y+1},
            {x-1,y-1},
            {x-1,y},
            {x-1,y+1}};
        for (auto square: pot_squares){
            if (!is_valid(square)){
                continue;
            }
            if (!does_piece_exist(start_pos,square)){
                Board temp = potential_change(start_pos,{x,y},square);
                to_return.push_back(temp);

                continue;
            }
            if (which_color_exists(start_pos,square)==color){
                continue;
            } else {
                Board temp = potential_change(start_pos,{x,y},square);
                to_return.push_back(temp);

            }


        }

    } else if (id=='B'){
        move_in_one_direction(x,y,start_pos,to_return,1,1,color);
        move_in_one_direction(x,y,start_pos,to_return,1,-1,color);
        move_in_one_direction(x,y,start_pos,to_return,-1,1,color);
        move_in_one_direction(x,y,start_pos,to_return,-1,-1,color);



    } else if (id=='Q'){
        move_in_one_direction(x,y,start_pos,to_return,0,1,color);
        move_in_one_direction(x,y,start_pos,to_return,0,-1,color);
        move_in_one_direction(x,y,start_pos,to_return,1,0,color);
        move_in_one_direction(x,y,start_pos,to_return,-1,0,color);
        move_in_one_direction(x,y,start_pos,to_return,1,1,color);
        move_in_one_direction(x,y,start_pos,to_return,1,-1,color);
        move_in_one_direction(x,y,start_pos,to_return,-1,1,color);
        move_in_one_direction(x,y,start_pos,to_return,-1,-1,color);

    }

    return to_return;




}

vector<Board> one_move_without_check(const Board &start_pos, int color){
    vector<Board> to_return;

    for (int i=0;i<8;i++){
        for (int j=0;j<8;j++){
            if (start_pos.colors[i][j]!=color){
                continue;
            }
            vector<Board> temp = possible_moves(i,j,start_pos);
            for (auto x: temp){
                to_return.push_back(x);
            }



        }


    }
    return to_return;



}

bool check_helper(const Board &pos, int color, int direc1, int direc2, int king_x, int king_y, pair<char,char> pot_checks){
    for (int i=1;i<8;i++){
        pair<int,int> square = {king_x+(direc1*i),(king_y+direc2*i)};

        if (!is_valid(square)){
            break;
        }
        if (does_piece_exist(pos, square)){
            if (which_color_exists(pos, square)==color){
                break;
            } else {
                if (which_piece_exists(pos, square)==pot_checks.first or which_piece_exists(pos, square)==pot_checks.second){
                    return true;
                }
            }
            break;


        }


    }
    return false;



}



bool is_white_in_check(const Board &pos, int color){

    int king_x, king_y;
    for (int i=0;i<8;i++){
        for (int j=0;j<8;j++){
            if (pos.pieces[i][j]=='K' and pos.colors[i][j]==color){
                king_x = i;
                king_y = j;
                break;

            }
        }
    }
    if (check_helper(pos, color, 1, 0, king_x, king_y, {'R','Q'})){
        return true;
    }
    if (check_helper(pos, color, -1, 0, king_x, king_y, {'R','Q'})){
        return true;
    }
    if (check_helper(pos, color, 0, 1, king_x, king_y, {'R','Q'})){
        return true;
    }
    if (check_helper(pos, color, 0, -1, king_x, king_y, {'R','Q'})){
        return true;
    }
    if (check_helper(pos, color, 1, 1, king_x, king_y, {'B','Q'})){
        return true;
    }
    if (check_helper(pos, color, 1, -1, king_x, king_y, {'B','Q'})){
        return true;
    }
    if (check_helper(pos, color, -1, 1, king_x, king_y, {'B','Q'})){
        return true;
    }
    if (check_helper(pos, color, -1, -1, king_x, king_y, {'B','Q'})){
        return true;
    }


    vector<pair<int,int>> knight_squares = {
        {king_x+2,king_y+1},
        {king_x+2,king_y-1},
        {king_x+1,king_y+2},
        {king_x+1,king_y-2},
        {king_x-2,king_y+1},
        {king_x-2,king_y-1},
        {king_x-1,king_y+2},
        {king_x-1,king_y-2}
    };

    for (auto square: knight_squares){
        if (!is_valid(square)){
            continue;
        }
        if (pos.pieces[square.first][square.second]=='N'){
            if (which_color_exists(pos,square)!=color){
                return true;
            }
        }

    }
    vector<pair<int,int>> pawn_squares = {
        {king_x-1,king_y+1},
        {king_x-1,king_y-1}
    };


    for (auto square: pawn_squares){
        if (!is_valid(square)){
            continue;
        }
        if (pos.pieces[square.first][square.second]=='P'){
            if (which_color_exists(pos,square)!=color){
                return true;
            }
        }

    }

    vector<pair<int,int>> king_squares = {
        {king_x+1,king_y+1},
        {king_x+1,king_y},
        {king_x+1,king_y-1},
        {king_x,king_y-1},
        {king_x,king_y+1},
        {king_x-1,king_y-1},
        {king_x-1,king_y},
        {king_x-1,king_y+1}
    };

    for (auto square: king_squares){
        if (!is_valid(square)){
            continue;
        }
        if (pos.pieces[square.first][square.second]=='K'){
            if (which_color_exists(pos,square)!=color){
                return true;
            }
        }


    }


    return false;


}

bool is_black_in_check(const Board &pos, int color){
    int king_x, king_y;
    int wking_x, wking_y;
    for (int i=0;i<8;i++){
        for (int j=0;j<8;j++){
            if (pos.pieces[i][j]=='K' and pos.colors[i][j]==color){
                king_x = i;
                king_y = j;

            }
            if (pos.pieces[i][j]=='K' and pos.colors[i][j]==1-color){
                wking_x = i;
                wking_y = j;
            }
            //Promotion means instant win for white
            if (pos.pieces[i][j]!='K' and pos.pieces[i][j]!='P' and pos.colors[i][j]==1-color){

                //cout << "PROMOTION" << endl;



                return true;
            }
        }
    }


    // This is a white-op version of check, but probably quite accurate
    if (abs(wking_x-king_x)<=2 and abs(wking_y-king_y)<=2){
        //cout << "WHITEOP: " << " " << wking_x << " " << wking_y << " : " << king_x << " " << king_y << endl;
        return true;
    }

    vector<pair<int,int>> immediate_take = {
        {king_x+1,king_y+1},
        {king_x+1,king_y-1}
    };


    for (auto square: immediate_take){
        if (!is_valid(square)){
            continue;
        }
        if (pos.pieces[square.first][square.second]=='P'){
            if (which_color_exists(pos,square)==1-color){

                //cout << "IMM PAWN" << endl;
                return true;
            }
        }

    }

    vector<pair<int,int>> one_up_one_diag = {
        {king_x+2,king_y+1},
        {king_x+2,king_y-1}
    };


    for (auto square: one_up_one_diag){
        if (!is_valid(square)){
            continue;
        }
        if (pos.pieces[square.first][square.second]=='P'){
            if (which_color_exists(pos,square)==1-color){
                if (!does_piece_exist(pos,{square.first-1,square.second})){
                    //cout << "1UP 1DIAG" << endl;
                    return true;
                }
            }
        }
    }

    vector<pair<int,int>> two_diag = {
        {king_x+2,king_y+2},
        {king_x+2,king_y-2}
    };




    for (auto square: two_diag){
        if (!is_valid(square)){
            continue;
        }
        if (pos.pieces[square.first][square.second]=='P'){
            if (which_color_exists(pos,square)==1-color){
                if (does_piece_exist(pos,{square.first-1,((((square.second)-king_y)/2)+king_y)})){
                    if (which_color_exists(pos,{square.first-1,((((square.second)-king_y)/2)+king_y)})==color){
                        //cout << "TWO DIAG" << endl;
                        return true;
                    }
                }
            }
        }
    }

    pair<int,int> square_to_consider = {king_x+2,king_y};

    if (is_valid(square_to_consider)){
        if (pos.pieces[square_to_consider.first][square_to_consider.second]=='P'){
            if (which_color_exists(pos,square_to_consider)==1-color){
                vector<pair<int,int>> two_pots = {{king_x+1,king_y+1},{king_x+1,king_y-1}};
                for (auto square:two_pots){
                    if (is_valid(square)){
                        if (which_color_exists(pos, square)==color){
                            //cout << "EDGECASE" << endl;
                            return true;
                        }

                    }
                }
            }
        }


    }

    //cout << "NOCHECK" << endl;

    return false;

    //return is_white_in_check(pos, color);

}

bool is_someone_in_check(const Board &pos, int color){
    if (color==1){
        return is_white_in_check(pos, color);
    } else {
        return is_black_in_check(pos, color);
    }

}

vector<Board> one_move_with_check(const Board &start_pos, int color){

    vector<Board> temp = one_move_without_check(start_pos, color);
    vector<Board> to_return;

    for (auto x: temp){
        if (is_someone_in_check(x,color)){
            continue;
        }
        to_return.push_back(x);

    }
    return to_return;

}


vector<Board> two_move_with_check(const Board &start_pos, int color){
    vector<Board> to_return;
    for (auto x: one_move_without_check(start_pos,color)){
        for (auto y: one_move_with_check(x,color)){
            to_return.push_back(y);


        }



    }
    return to_return;


}



vector<Board> legal_moves(const Board &start_pos, int color){

    if (color==0){
        return one_move_with_check(start_pos,0);


    } else if (color==1){
        return two_move_with_check(start_pos,1);

    }

}


Board manual_input_white(Board start_pos, int color){
    vector<Board> possible_moves = legal_moves(start_pos, color);
    while (true){
        cout << "WHITE MOVE REQUIRED" << endl;
        string from1; string to1; cin >> from1 >> to1;
        string from2; string to2; cin >> from2 >> to2;

        if (!string_valid(from1) or !string_valid(from2) or !string_valid(to1) or !string_valid(to2)){
            cout << "INVALID INPUT" << endl;
            continue;
        }

        pair<int,int> from_first_square = not_conv(from1);
        pair<int,int> to_first_square = not_conv(to1);
        pair<int,int> from_second_square = not_conv(from2);
        pair<int,int> to_second_square = not_conv(to2);




        cout << from_first_square.first << " " << from_first_square.second << endl;
        cout << to_first_square.first << " " << to_first_square.second << endl;
        cout << from_second_square.first << " " << from_second_square.second << endl;
        cout << to_second_square.first << " " << to_second_square.second << endl;


        Board pot_board = potential_change(potential_change(start_pos,not_conv(from1),not_conv(to1)),not_conv(from2),not_conv(to2));

        //pot_board.board_printer();

        for (int i=0;i<possible_moves.size();i++){
            if (pot_board==possible_moves[i]){
                //pot_board.board_printer();
                return possible_moves[i];
            }

        }
        cout << "ILLEGAL MOVE" << endl;

//        for (auto x: possible_moves){
//            cout << into_not(x.second_last_move.first) << " " <<into_not(x.second_last_move.second) << endl;
//            cout << into_not(x.last_move.first) << " " <<into_not(x.last_move.second) << endl;
//            cout << endl;
//        }



    }



}


Board manual_input_black(Board start_pos, int color){
    vector<Board> possible_moves = legal_moves(start_pos, color);
    while (true){
        cout << "BLACK MOVE REQUIRED" << endl;
        string from1; string to1; cin >> from1 >> to1;

        if (!string_valid(from1) or !string_valid(to1)){
            cout << "INVALID INPUT" << endl;
            continue;
        }

        pair<int,int> from_first_square = not_conv(from1);
        pair<int,int> to_first_square = not_conv(to1);



        cout << from_first_square.first << " " << from_first_square.second << endl;
        cout << to_first_square.first << " " << to_first_square.second << endl;


        Board pot_board = potential_change(start_pos,not_conv(from1),not_conv(to1));

        //pot_board.board_printer();

        for (int i=0;i<possible_moves.size();i++){
            if (pot_board==possible_moves[i]){
                //pot_board.board_printer();
                return possible_moves[i];
            }

        }
        cout << "ILLEGAL MOVE" << endl;

        for (auto x: one_move_without_check(start_pos,color)){


            //cout << into_not(x.second_last_move.first) << " " <<into_not(x.second_last_move.second) << endl;
            cout << into_not(x.last_move.first) << " " <<into_not(x.last_move.second) << endl;
            is_black_in_check(x,color);
            cout << endl;
        }



    }



}



Board random_move(Board start_pos, int color){
    vector<Board> possible_moves = legal_moves(start_pos, color);
    return possible_moves[rand()%possible_moves.size()];
}


int val_of_wpawn(int x, int y, int weight){
    int pos = 8-x;

    vector<int> vals = {1,2,3,4,5,6,8};

    return vals[pos]*weight;
}

int val_of_bpawn(int x, int y, int weight){
    int pos = x;
    vector<int> vals = {-1,-1,-1,-2,-2,-3,-5};

    return vals[pos]*weight;


}


int val_of_king_distance(pair<int,int> wking_pos, pair<int,int> bking_pos, int weight){



    return - weight*(max(abs(wking_pos.first-bking_pos.first),abs(wking_pos.second-bking_pos.second)));
}


int value_of_pos_bpov(const Board &start_pos, int color){
    map<char,int> white_vals = {
        {'Q',pos_inf},
        {'P',4},
        {'K',0}
    };

    map<char,int> black_vals = {
        {'Q',-9},
        {'R',-5},
        {'B',-3},
        {'P',-1},
        {'N',-3},
        {'K',0}
    };



    int total_val = 0;
    pair<int,int> wking_pos;
    pair<int,int> bking_pos;

    for (int i=0;i<8;i++){
        for (int j=0;j<8;j++){
            if (start_pos.colors[i][j]==0){
                total_val += black_vals[start_pos.pieces[i][j]];
                if (which_piece_exists(start_pos,{i,j})=='K'){
                    bking_pos = {i,j};
                } else if (which_piece_exists(start_pos,{i,j})=='P'){
                    total_val += val_of_bpawn(i,j,1);
                }

            } else if (start_pos.colors[i][j]==1){
                total_val += white_vals[start_pos.pieces[i][j]];
                if (which_piece_exists(start_pos,{i,j})=='P'){
                    total_val += val_of_wpawn(i,j,2);
                } else if (which_piece_exists(start_pos,{i,j})=='K'){
                    wking_pos = {i,j};
                }
            }
        }
    }

    total_val += val_of_king_distance(wking_pos,bking_pos,10);

    return total_val;


}

int value_of_pos_wpov(const Board &start_pos, int color){
    map<char,int> white_vals = {
        {'Q',100},
        {'P',4},
        {'K',0}
    };

    map<char,int> black_vals = {
        {'Q',-9},
        {'R',-5},
        {'B',-3},
        {'P',-1},
        {'N',-3},
        {'K',0}
    };



    int total_val = 0;
    pair<int,int> wking_pos;
    pair<int,int> bking_pos;

    for (int i=0;i<8;i++){
        for (int j=0;j<8;j++){
            if (start_pos.colors[i][j]==0){
                total_val += black_vals[start_pos.pieces[i][j]];
                if (which_piece_exists(start_pos,{i,j})=='K'){
                    bking_pos = {i,j};
                } else if (which_piece_exists(start_pos,{i,j})=='P'){
                    total_val += val_of_bpawn(i,j,1);
                }

            } else if (start_pos.colors[i][j]==1){
                total_val += white_vals[start_pos.pieces[i][j]];
                if (which_piece_exists(start_pos,{i,j})=='P'){
                    total_val += val_of_wpawn(i,j,1);
                } else if (which_piece_exists(start_pos,{i,j})=='K'){
                    wking_pos = {i,j};
                }
            }
        }
    }

    total_val += val_of_king_distance(wking_pos,bking_pos,100);

    return total_val;


}




int minimax_with_ab(function<int (const Board&, int)> func, Board &pos, int depth, int color, int depth_to_search, int alpha, int beta){
//    string to_print;
//    for (int i=0;i<depth;i++){
//        to_print += " ";
//    }
//    cout << to_print;
//    cout << color << " ";
//    eff_color_printer(pos,color);

    nodes_checked++;


    if (depth==depth_to_search){
        //cout << to_print << " ";
        //eff_color_printer(pos,color);
        //cout << to_print << " " << value_of_pos(pos,color) << endl;
        return func(pos, color);
    }

    if (color==1){

        int max_score = neg_inf;
        vector<Board> layer_deeper = legal_moves(pos,color);
        if (layer_deeper.size()==0){
            return max_score;
        }


        for (auto x: layer_deeper){
            max_score = max(max_score, minimax_with_ab(func,x,depth+1,1-color,depth_to_search,alpha,beta));
            alpha = max(alpha, max_score);

            if (beta <= alpha){
                break;
            }


        }

        return max_score;


    } else if (color==0){
        int min_score = pos_inf;
        vector<Board> layer_deeper = legal_moves(pos,color);
        if (layer_deeper.size()==0){
            return min_score;
        }


        for (auto x: layer_deeper){
            min_score = min(min_score, minimax_with_ab(func,x,depth+1,1-color,depth_to_search,alpha,beta));
            beta = min(min_score, beta);

            if (beta<=alpha){
                break;
            }

        }
        //cout << min_score << endl;
        return min_score;


    }


}




Board minimaxer_try_2(function<int (const Board&, int)> func, Board &start_pos, int color, int depth_to_search){
    if (color==1){
        int max_score = neg_inf;
        int max_pos = -1;
        vector<Board> layer_deeper = legal_moves(start_pos,color);
        for (int i=0;i<layer_deeper.size();i++){
            int board_val = minimax_with_ab(func, layer_deeper[i],0,1-color,depth_to_search,neg_inf,pos_inf);
            if ((board_val > max_score) or (i==layer_deeper.size()-1 and max_pos==-1)){
                max_score = board_val;
                max_pos = i;
            }


        }

        if (max_pos==-1){
            cout << "RED ALERT WHITE" << endl;
            for (auto x: layer_deeper){
                //eff_white_move_printer(x);
            }



        }


        return layer_deeper[max_pos];

    } else {
        int min_score = pos_inf;
        int min_pos = -1;
        vector<Board> layer_deeper = legal_moves(start_pos,color);
        for (int i=0;i<layer_deeper.size();i++){
            int board_val = minimax_with_ab(func, layer_deeper[i],0,1-color,depth_to_search,neg_inf,pos_inf);
            cout << board_val << endl;
            eff_black_move_printer(layer_deeper[i]);
            if ((board_val < min_score) or (i==layer_deeper.size()-1 and min_pos==-1)){
                min_score = board_val;
                min_pos = i;
            }


        }

        if (min_pos==-1){
            cout << "RED ALERT BLACK" << endl;
            start_pos.board_printer();
            for (auto x: layer_deeper){
                eff_black_move_printer(x);
            }
        }


        return layer_deeper[min_pos];

    }


}



class WhitePlayer {
public:
    int color = 1; // (white)
    Board move_chooser(Board start_pos){



        //return manual_input_white(start_pos, color);

        return minimaxer_try_2(value_of_pos_wpov, start_pos,color,2);

        return random_move(start_pos,color);


    }

};

class BlackPlayer {
public:
    int color = 0; // (black)
    Board move_chooser(Board start_pos){
        vector<Board> possible_moves = legal_moves(start_pos, color);

        //return manual_input_black(start_pos, color);


        return minimaxer_try_2(value_of_pos_bpov, start_pos,color,2);

        //return minimaxer_try_1(start_pos,color,2);

        return random_move(start_pos,color);


    }


};



string game_over(Board& pos, int color, string &winner){
    vector<Board> pot_moves = legal_moves(pos,color);
    cout << pot_moves.size() << endl;
    if (pot_moves.size()==0){
//        for (auto x: one_move_without_check(pos, color)){
//            x.board_printer();
//        }


        winner = to_string(1-color);
        if (color==1){
            winner = "black";
        } else {
            winner = "white";
        }
        return "";


    }
    return "continue";
}









void main_loop(){



    vector<vector<char>> start_board_pieces = {
        {'R','N','B','Q','K','B','N','R'},
        {'P','P','P','P','P','P','P','P'},
        {'.','.','.','.','.','.','.','.'},
        {'.','.','.','.','.','.','.','.'},
        {'.','.','.','.','.','.','.','.'},
        {'.','.','.','.','.','.','.','.'},
        {'.','.','P','P','P','P','.','.'},
        {'.','.','.','.','K','.','.','.'}
    };


    vector<vector<int>> start_board_colors = {
        {0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0},
        {2,2,2,2,2,2,2,2},
        {2,2,2,2,2,2,2,2},
        {2,2,2,2,2,2,2,2},
        {2,2,2,2,2,2,2,2},
        {2,2,1,1,1,1,2,2},
        {2,2,2,2,1,2,2,2}
    };


    Board main_board (start_board_pieces, start_board_colors);

    main_board.board_printer();

    WhitePlayer white_player;
    BlackPlayer black_player;

    string winner = "";
    int num_moves = 0;


    while (true){
        main_board = white_player.move_chooser(main_board);
        num_moves++;
        main_board.board_printer();

        if (game_over(main_board,0,winner)!="continue"){
            break;
        }


        main_board = black_player.move_chooser(main_board);
        num_moves++;
        main_board.board_printer();

        if (game_over(main_board,1,winner)!="continue"){
            break;
        }



    }

    cout << "WINNER IS: " << winner << endl;
    cout << num_moves << endl;



}


int main()
{
    srand((unsigned) time(0));
    main_loop();
    cout << nodes_checked << endl;
}

