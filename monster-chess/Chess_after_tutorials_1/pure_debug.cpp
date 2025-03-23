#include "pure_debug.h"


using namespace std;

typedef unsigned long long BB;

vector<string> all_moves_for_debug;

//vector<pair<string,string>> all_man_moves_for_debug;

int num_moves_for_debug = 0;
map<string,int> all_mighty_square_converter = {
    {"h1",0},
    {"g1",1},
    {"f1",2},
    {"e1",3},
    {"d1",4},
    {"c1",5},
    {"b1",6},
    {"a1",7},
    {"h2",8},
    {"g2",9},
    {"f2",10},
    {"e2",11},
    {"d2",12},
    {"c2",13},
    {"b2",14},
    {"a2",15},
    {"h3",16},
    {"g3",17},
    {"f3",18},
    {"e3",19},
    {"d3",20},
    {"c3",21},
    {"b3",22},
    {"a3",23},
    {"h4",24},
    {"g4",25},
    {"f4",26},
    {"e4",27},
    {"d4",28},
    {"c4",29},
    {"b4",30},
    {"a4",31},
    {"h5",32},
    {"g5",33},
    {"f5",34},
    {"e5",35},
    {"d5",36},
    {"c5",37},
    {"b5",38},
    {"a5",39},
    {"h6",40},
    {"g6",41},
    {"f6",42},
    {"e6",43},
    {"d6",44},
    {"c6",45},
    {"b6",46},
    {"a6",47},
    {"h7",48},
    {"g7",49},
    {"f7",50},
    {"e7",51},
    {"d7",52},
    {"c7",53},
    {"b7",54},
    {"a7",55},
    {"h8",56},
    {"g8",57},
    {"f8",58},
    {"e8",59},
    {"d8",60},
    {"c8",61},
    {"b8",62},
    {"a8",63}

};


map<int,string> all_mighty_to_square_converter = {
    {0,"h1"},
    {1,"g1"},
    {2,"f1"},
    {3,"e1"},
    {4,"d1"},
    {5,"c1"},
    {6,"b1"},
    {7,"a1"},
    {8,"h2"},
    {9,"g2"},
    {10,"f2"},
    {11,"e2"},
    {12,"d2"},
    {13,"c2"},
    {14,"b2"},
    {15,"a2"},
    {16,"h3"},
    {17,"g3"},
    {18,"f3"},
    {19,"e3"},
    {20,"d3"},
    {21,"c3"},
    {22,"b3"},
    {23,"a3"},
    {24,"h4"},
    {25,"g4"},
    {26,"f4"},
    {27,"e4"},
    {28,"d4"},
    {29,"c4"},
    {30,"b4"},
    {31,"a4"},
    {32,"h5"},
    {33,"g5"},
    {34,"f5"},
    {35,"e5"},
    {36,"d5"},
    {37,"c5"},
    {38,"b5"},
    {39,"a5"},
    {40,"h6"},
    {41,"g6"},
    {42,"f6"},
    {43,"e6"},
    {44,"d6"},
    {45,"c6"},
    {46,"b6"},
    {47,"a6"},
    {48,"h7"},
    {49,"g7"},
    {50,"f7"},
    {51,"e7"},
    {52,"d7"},
    {53,"c7"},
    {54,"b7"},
    {55,"a7"},
    {56,"h8"},
    {57,"g8"},
    {58,"f8"},
    {59,"e8"},
    {60,"d8"},
    {61,"c8"},
    {62,"b8"},
    {63,"a8"}
};




string board_printer(ChessBoard board){
    string to_print = "";
    //cout << bitset<64>(board.AllPieces) << endl;
    for (int i=0;i<64;i++){
        if (bitset<64>(board.AllPieces)[63-i]==0){
            to_print += ". ";
        } else if (bitset<64>(board.AllWhitePieces)[63-i]==1){
            if (bitset<64>(board.WhiteKing)[63-i]==1){
                to_print += "K ";

            } else if (bitset<64>(board.WhitePawns)[63-i]==1){
                to_print += "P ";

            } else if (bitset<64>(board.WhiteQueens)[63-i]==1){
                to_print += "Q ";

            }

        } else if (bitset<64>(board.AllBlackPieces)[63-i]==1){
            if (bitset<64>(board.BlackBishops)[63-i]==1){
                to_print +=  "b ";

            } else if (bitset<64>(board.BlackKing)[63-i]==1){
                to_print += "k ";

            } else if (bitset<64>(board.BlackKnights)[63-i]==1){
                to_print += "n ";

            } else if (bitset<64>(board.BlackPawns)[63-i]==1){
                to_print += "p ";

            } else if (bitset<64>(board.BlackQueens)[63-i]==1){
                to_print += "q ";

            } else if (bitset<64>(board.BlackRooks)[63-i]==1){
                to_print += "r ";

            }


        }



        //cout << bitset<64>(board.AllPieces)[63-i] << " ";

        if ((i+1)%8==0){
            to_print += "\n";
        }

    }
    to_print += "\n";
    to_print += to_string(board.ZobristValue);
    return to_print + "\n";

}

void semiprinter(BB to_print){
    string ac = bitset<64>(to_print).to_string<char,std::string::traits_type,std::string::allocator_type>();

    for (int i=0;i<8;i++){
        cout << ac.substr(i*8,8) << endl;

    }
    cout << endl;

}



