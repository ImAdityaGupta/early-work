#include "move_generation.h"
#include <QDebug>
using namespace std;

typedef unsigned long long BB;


int isPowerOfTwo(const BB &n){
    return n && (!(n & (n - 1)));
}

// Returns position of the only set bit in 'n'; a8 = 63, h1 = 0
int findPosition(const BB &n){
    //cout << bitset<64>(n) << endl;
    if (!isPowerOfTwo(n))
        return -1;

    BB i = 1, pos = 1;

    // Iterate through bits of n till we find a set bit
    // i&n will be non-zero only when 'i' and 'n' have a set bit
    // at same position
    while (!(i & n)) {
        // Unset current bit and set the next bit in 'i'
        i = i << 1;

        // increment position
        ++pos;
    }
    //cout << pos-1 << endl;
    return pos-1;
}


// a8 = 63, h1 = 0
int findAndClearSetBit(BB& mask) {
   int sq = __builtin_ctzll(mask); // find least significant bit
   mask &= ~(1ULL << sq); // clear least significant bit
   return sq;
}



//bit scan reverse
int bsr(BB bb){
   int result = 0;
   if (bb > 0xFFFFFFFF) {
      bb >>= 32;
      result = 32;
   }
   if (bb > 0xFFFF) {
      bb >>= 16;
      result += 16;
   }
   if (bb > 0xFF) {
      bb >>= 8;
      result += 8;
   }
   return result + ms1bTable[bb];
}


BB naive_zobrist(const ChessBoard &pos){
    BB to_return = 0ULL;
    ChessBoard work_with = pos;

    while (work_with.WhitePawns!=0){
        int position = findAndClearSetBit(work_with.WhitePawns);
        to_return ^= zobrist_table[position][0];
    }
    while (work_with.WhiteQueens!=0){
        int position = findAndClearSetBit(work_with.WhiteQueens);
        to_return ^= zobrist_table[position][1];
    }
    while (work_with.WhiteKing!=0){
        int position = findAndClearSetBit(work_with.WhiteKing);
        to_return ^= zobrist_table[position][2];
    }
    while (work_with.BlackPawns!=0){
        int position = findAndClearSetBit(work_with.BlackPawns);
        to_return ^= zobrist_table[position][3];
    }
    while (work_with.BlackRooks!=0){
        int position = findAndClearSetBit(work_with.BlackRooks);
        to_return ^= zobrist_table[position][4];
    }
    while (work_with.BlackKnights!=0){
        int position = findAndClearSetBit(work_with.BlackKnights);
        to_return ^= zobrist_table[position][5];
    }
    while (work_with.BlackBishops!=0){
        int position = findAndClearSetBit(work_with.BlackBishops);
        to_return ^= zobrist_table[position][6];
    }
    while (work_with.BlackQueens!=0){
        int position = findAndClearSetBit(work_with.BlackQueens);
        to_return ^= zobrist_table[position][7];
    }
    while (work_with.BlackKing!=0){
        int position = findAndClearSetBit(work_with.BlackKing);
        to_return ^= zobrist_table[position][8];
    }

    return to_return;



}

BB smart_zobrist(BB old_zobrist, const int from_sq, const int to_sq, int type_of_piece, int piece_taken){
//    cout << endl;
//    cout << old_zobrist << endl;
//    cout << from_sq << " " << type_of_piece << endl;
//    cout << to_sq << " " << type_of_piece << endl;
//    cout << endl;



    if (to_sq>=56 and type_of_piece==WHITE_PAWN_INT){
        old_zobrist ^= zobrist_table[from_sq][WHITE_PAWN_INT];
        old_zobrist ^= zobrist_table[to_sq][WHITE_QUEEN_INT];

    } else if (to_sq<=7 and type_of_piece==BLACK_PAWN_INT){
        old_zobrist ^= zobrist_table[from_sq][BLACK_PAWN_INT];
        old_zobrist ^= zobrist_table[to_sq][BLACK_QUEEN_INT];
    } else {




        old_zobrist ^= zobrist_table[from_sq][type_of_piece];
        old_zobrist ^= zobrist_table[to_sq][type_of_piece];

    }


    if (piece_taken!=-1){
        old_zobrist ^= zobrist_table[to_sq][piece_taken];
    }


    return old_zobrist;
}


void createMove(const int from_sq, int to_sq, vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, int type_of_piece){


    BB a = start_pos.WhitePawns;
    BB b = start_pos.WhiteQueens;
    BB c = start_pos.WhiteKing;

    BB d = start_pos.BlackPawns;
    BB e = start_pos.BlackRooks;
    BB f = start_pos.BlackKnights;
    BB g = start_pos.BlackBishops;
    BB h = start_pos.BlackQueens;
    BB i = start_pos.BlackKing;




    BB *temp[9] = {&a,&b,&c,&d,&e,&f,&g,&h,&i};


    //First figure out which square we are going to; if any piece is there it is
    //about to be taken so we send that to piece_taken
    int piece_taken = -1;
    for (int i=0;i<9;i++){

        BB bit = (*(temp[i]) >> to_sq) & 1ULL;
        if (bit!=0){
            piece_taken = i;
            break;
        }
    }


    //magic_num wipes out the from_sq and to_sq of each BB
    BB magic_num = ~(1ULL << from_sq) & ~(1ULL << to_sq);



    a &= magic_num;
    b &= magic_num;
    c &= magic_num;
    d &= magic_num;
    e &= magic_num;
    f &= magic_num;
    g &= magic_num;
    h &= magic_num;
    i &= magic_num;

    //PROMOTION
    if (to_sq>=56 and type_of_piece==WHITE_PAWN_INT){
        (*(temp[WHITE_QUEEN_INT])) |= (1ULL << to_sq);

    } else if (to_sq<=7 and type_of_piece==BLACK_PAWN_INT){
        (*(temp[BLACK_QUEEN_INT])) |= (1ULL << to_sq);
    } else {
        (*(temp[type_of_piece])) |= (1ULL << to_sq);
    }




    //cout << bitset<64>(*temp[type_of_piece]) << endl;
    //cout << bitset<64>(c) << endl;




    ChessBoard to_append(a,b,c,d,e,f,g,h,i);
    to_append.second_last_move = start_pos.last_move;
    to_append.last_move = all_mighty_to_square_converter[from_sq] + " " + all_mighty_to_square_converter[to_sq];

    to_append.ZobristValue = smart_zobrist(start_pos.ZobristValue,from_sq,to_sq,type_of_piece,piece_taken);
    //cout << board_printer(to_append);
    pot_moves.push_back(to_append);
}


void createAllMoves(vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, BB &mask, const int from_sq, int type_of_piece) {



    while(mask != 0) {
        //cout << bitset<64>(mask) << endl;
        int to_sq = findAndClearSetBit(mask);
        //cout << to_sq << endl;
        if (type_of_piece==BLACK_BISHOP_INT){
            //cout << "WOOHOO" << from_sq << " " << to_sq << endl;

        }
        createMove(from_sq, to_sq, pot_moves, start_pos, type_of_piece);
    }
}



BB compute_king(const BB &king_loc, const BB &own_side){
    /* we can ignore the rank clipping since the overflow/underflow with
        respect to rank simply vanishes. We only care about the file
        overflow/underflow. */

    BB king_clip_file_h = king_loc & ClearFile[FILE_H];
    BB king_clip_file_a = king_loc & ClearFile[FILE_A];

    /* remember the representation of the board in relation to the bitindex
        when looking at these shifts.... */
    BB spot_1 = king_clip_file_h << 7;
    BB spot_2 = king_loc << 8;
    BB spot_3 = king_clip_file_a << 9;
    BB spot_4 = king_clip_file_a << 1;

    BB spot_5 = king_clip_file_a >> 7;
    BB spot_6 = king_loc >> 8;
    BB spot_7 = king_clip_file_h >> 9;
    BB spot_8 = king_clip_file_h >> 1;

    BB king_moves = spot_1 | spot_2 | spot_3 | spot_4 | spot_5 | spot_6 |
                        spot_7 | spot_8;

    BB KingValid = king_moves & ~own_side;

    /* compute only the places where the king can move and attack. The caller
        will interpret this as a white or black king. */
    //cout << bitset<64>(KingValid) << endl;
    return KingValid;
}




BB one_knight(const BB &knight_loc, const BB &own_side){
    /* we can ignore the rank clipping since the overflow/underflow with
        respect to rank simply vanishes. We only care about the file
        overflow/underflow which is much more work for a knight. */

    BB spot_1_clip = ClearFile[FILE_A] & ClearFile[FILE_B];
    BB spot_2_clip = ClearFile[FILE_A];
    BB spot_3_clip = ClearFile[FILE_H];
    BB spot_4_clip = ClearFile[FILE_H] & ClearFile[FILE_G];

    BB spot_5_clip = ClearFile[FILE_H] & ClearFile[FILE_G];
    BB spot_6_clip = ClearFile[FILE_H];
    BB spot_7_clip = ClearFile[FILE_A];
    BB spot_8_clip = ClearFile[FILE_A] & ClearFile[FILE_B];

    /* The clipping masks we just created will be used to ensure that no
        under or overflow positions are computed when calculating the
        possible moves of the knight in certain files. */

    BB spot_1 = (knight_loc & spot_1_clip) << 10;
    BB spot_2 = (knight_loc & spot_2_clip) << 17;
    BB spot_3 = (knight_loc & spot_3_clip) << 15;
    BB spot_4 = (knight_loc & spot_4_clip) << 6;

    BB spot_5 = (knight_loc & spot_5_clip) >> 10;
    BB spot_6 = (knight_loc & spot_6_clip) >> 17;
    BB spot_7 = (knight_loc & spot_7_clip) >> 15;
    BB spot_8 = (knight_loc & spot_8_clip) >> 6;

//  cout << endl;
//  cout << bitset<64>(knight_loc) << endl;
//
//  cout << bitset<64>(spot_1) << " " << bitset<64>(spot_1_clip) << endl;
//  cout << bitset<64>(spot_2) << endl;
//  cout << bitset<64>(spot_3) << endl;
//  cout << bitset<64>(spot_4) << endl;
//  cout << bitset<64>(spot_5) << endl;
//  cout << bitset<64>(spot_6) << endl;
//  cout << bitset<64>(spot_7) << endl;
//  cout << bitset<64>(spot_8) << endl;
//  cout << endl;


    BB KnightValid = spot_1 | spot_2 | spot_3 | spot_4 | spot_5 | spot_6 |
                    spot_7 | spot_8;

    /* compute only the places where the knight can move and attack. The
        caller will determine if this is a white or black night. */
    return (KnightValid & ~own_side);
}

void compute_knight(BB knight_locs, const BB &own_side, vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color){
    //cout << bitset<64>(knight_locs) << endl;

    while (knight_locs != 0){
        //cout << bitset<64>(knight_locs) << endl;


        int temp_knight_loc = findAndClearSetBit(knight_locs);
        BB knight_loc = (1ULL << temp_knight_loc);
        BB knight_moves = one_knight(knight_loc,own_side);
        int type_of_piece;
        if (color==0){
            type_of_piece = BLACK_KNIGHT_INT;

        } else if (color==1){
            cout << "KNIGHT ERROR" << endl;
        }

        //cout << bitset<64>(knight_loc) << endl;

        createAllMoves(pot_moves, start_pos, knight_moves, findPosition(knight_loc), type_of_piece);


    }


    //BB black_king_moves = compute_king(start_pos.BlackKing, start_pos.AllBlackPieces);

    //createAllMoves(pot_moves, start_pos, black_king_moves, findPosition(start_pos.BlackKing), BLACK_KING_INT);




}


BB one_white_pawn(const BB &white_pawn_loc, const BB &all_pieces, const BB &all_black_pieces) {
    /* check the single space infront of the white pawn */
    BB white_pawn_one_step = (white_pawn_loc << 8) & ~all_pieces;

    /* for all moves that came from rank 2 (home row) and passed the above
        filter, thereby being on rank 3, check and see if I can move forward
        one more */
    BB white_pawn_two_steps =
        ((white_pawn_one_step & MaskRank[RANK_3]) << 8) & ~all_pieces;

    /* the union of the movements dictate the possible moves forward
        available */
    BB white_pawn_valid_moves = white_pawn_one_step | white_pawn_two_steps;

    /* next we calculate the pawn attacks */

    /* check the left side of the pawn, minding the underflow File A */
    BB white_pawn_left_attack = (white_pawn_loc & ClearFile[FILE_A]) << 9;

    /* then check the right side of the pawn, minding the overflow File H */
    BB white_pawn_right_attack = (white_pawn_loc & ClearFile[FILE_H]) << 7;

    /* the union of the left and right attacks together make up all the
        possible attacks */
    BB white_pawn_attacks = white_pawn_left_attack | white_pawn_right_attack;

    /* Calculate where I can _actually_ attack something */
    BB white_pawn_valid_attacks = white_pawn_attacks & all_black_pieces;

    /* then we combine the two situations in which a white pawn can legally
        attack/move. */
    BB WhitePawnValid = white_pawn_valid_moves | white_pawn_valid_attacks;

    return WhitePawnValid;
}

BB two_move_white_pawn(const BB &white_pawn_loc, const BB &all_pieces, const BB &all_black_pieces, bool to_print){
    BB white_pawn_one_step = ((white_pawn_loc << 8) & ~all_pieces) & (~MaskRank[RANK_8]);
    BB white_pawn_two_steps_in_one =
        ((white_pawn_one_step & MaskRank[RANK_3]) << 8) & ~all_pieces;

    BB white_pawn_left_attack = ((white_pawn_loc & ClearFile[FILE_A]) << 9) & all_black_pieces;
    BB white_pawn_right_attack = ((white_pawn_loc & ClearFile[FILE_H]) << 7) & all_black_pieces;

    BB white_pawn_one_step_then_left = (white_pawn_one_step & ClearFile[FILE_A]) << 9;
    BB white_pawn_one_step_then_right = (white_pawn_one_step & ClearFile[FILE_H]) << 7;

    BB white_pawn_two_steps_in_one_then_left = (white_pawn_two_steps_in_one & ClearFile[FILE_A]) << 9;
    BB white_pawn_two_steps_in_one_then_right = (white_pawn_two_steps_in_one & ClearFile[FILE_H]) << 7;


    BB white_pawn_left_then_left = ((white_pawn_left_attack & ClearFile[FILE_A]) << 9) & all_black_pieces;
    BB white_pawn_left_then_right = ((white_pawn_left_attack & ClearFile[FILE_H]) << 7) & all_black_pieces;
    BB white_pawn_right_then_left = ((white_pawn_right_attack & ClearFile[FILE_A]) << 9) & all_black_pieces;
    BB white_pawn_right_then_right = ((white_pawn_right_attack & ClearFile[FILE_H]) << 7) & all_black_pieces;

    BB PotentialAttackedSquares = white_pawn_left_attack | white_pawn_right_attack
                                    | white_pawn_one_step_then_left | white_pawn_one_step_then_right
                                    | white_pawn_two_steps_in_one_then_left | white_pawn_two_steps_in_one_then_right
                                    | white_pawn_left_then_left | white_pawn_left_then_right
                                    | white_pawn_right_then_left | white_pawn_right_then_right;

    if (to_print){
        semiprinter(white_pawn_left_attack);
        semiprinter(white_pawn_right_attack);
        semiprinter(white_pawn_one_step_then_left);
        semiprinter(white_pawn_one_step_then_right);
        semiprinter(white_pawn_two_steps_in_one_then_left);
        semiprinter(white_pawn_two_steps_in_one_then_right);
        semiprinter(white_pawn_left_then_left);
        semiprinter(white_pawn_left_then_right);
        semiprinter(white_pawn_right_then_left);
        semiprinter(white_pawn_right_then_right);
    }


    return PotentialAttackedSquares;




}

BB white_pawn_to_queen(const BB &white_pawn_loc, const BB &all_pieces, const BB &all_black_pieces, bool to_print){
    BB seventh_rank = white_pawn_loc & MaskRank[RANK_7];
    BB straight_forward = ((seventh_rank << 8) & ~all_pieces);
    BB take_left = ((seventh_rank & ClearFile[FILE_A]) << 9) & all_black_pieces;
    BB take_right = ((seventh_rank & ClearFile[FILE_H]) << 7) & all_black_pieces;

    BB potential_queen_spots = straight_forward | take_left | take_right;


    return potential_queen_spots;


}


BB one_black_pawn_attacks(const BB &black_pawn_loc, const BB &all_pieces, const BB &all_white_pieces, bool to_print){

    BB black_pawn_left_attack = ((black_pawn_loc & ClearFile[FILE_A]) >> 7);
    BB black_pawn_right_attack = ((black_pawn_loc & ClearFile[FILE_H]) >> 9);
    return black_pawn_left_attack | black_pawn_right_attack;

}


void compute_white_pawns(BB pawn_locs, const BB &own_side, vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color){
    //cout << bitset<64>(knight_locs) << endl;

    while (pawn_locs != 0){
        //cout << bitset<64>(pawn_locs) << endl;


        int temp_pawn_loc = findAndClearSetBit(pawn_locs);
        BB pawn_loc = (1ULL << temp_pawn_loc);
        BB pawn_moves = one_white_pawn(pawn_loc,start_pos.AllPieces,start_pos.AllBlackPieces);


        createAllMoves(pot_moves, start_pos, pawn_moves, findPosition(pawn_loc), WHITE_PAWN_INT);


    }



}


BB one_black_pawn(const BB &black_pawn_loc, const BB &all_pieces, const BB &all_white_pieces) {
    /* check the single space infront of the black pawn */
    BB black_pawn_one_step = (black_pawn_loc >> 8) & ~all_pieces;

    /* for all moves that came from rank 2 (home row) and passed the above
        filter, thereby being on rank 3, check and see if I can move forward
        one more */
    BB black_pawn_two_steps =
        ((black_pawn_one_step & MaskRank[RANK_6]) >> 8) & ~all_pieces;

    /* the union of the movements dictate the possible moves forward
        available */
    BB black_pawn_valid_moves = black_pawn_one_step | black_pawn_two_steps;

    /* next we calculate the pawn attacks */

    /* check the left side of the pawn, minding the underflow File A */
    BB black_pawn_left_attack = (black_pawn_loc & ClearFile[FILE_A]) >> 7;

    /* then check the right side of the pawn, minding the overflow File H */
    BB black_pawn_right_attack = (black_pawn_loc & ClearFile[FILE_H]) >> 9;

    /* the union of the left and right attacks together make up all the
        possible attacks */
    BB black_pawn_attacks = black_pawn_left_attack | black_pawn_right_attack;

    /* Calculate where I can _actually_ attack something */
    BB black_pawn_valid_attacks = black_pawn_attacks & all_white_pieces;

    /* then we combine the two situations in which a black pawn can legally
        attack/move. */
    BB BlackPawnValid = black_pawn_valid_moves | black_pawn_valid_attacks;

    return BlackPawnValid;
}


void compute_black_pawns(BB pawn_locs, const BB &own_side, vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color){
    //cout << bitset<64>(knight_locs) << endl;

    while (pawn_locs != 0){
        //cout << bitset<64>(pawn_locs) << endl;


        int temp_pawn_loc = findAndClearSetBit(pawn_locs);
        BB pawn_loc = (1ULL << temp_pawn_loc);
        BB pawn_moves = one_black_pawn(pawn_loc,start_pos.AllPieces,start_pos.AllWhitePieces);

        createAllMoves(pot_moves, start_pos, pawn_moves, findPosition(pawn_loc), BLACK_PAWN_INT);

    }


}


BB lineAttacks(const BB &const_occ, const int piece_pos, const int direc){

    BB occ = const_occ & sliding_piece_masks[piece_pos][direc];

    //semiprinter(occ);
    BB temp = ~((1ULL << piece_pos) - 1);
    //cout << piece_pos << endl;
    BB upper = (temp)&occ;
    BB lower = ~temp & occ;

    //semiprinter(upper);
    //semiprinter(lower);



    BB MS1B = -1ULL << bsr(lower | 1ULL);
    BB LS1B = upper & -upper;

    BB odiff = (2 * LS1B) + MS1B;

    BB to_return = odiff & sliding_piece_masks[piece_pos][direc];

    //semiprinter(to_return);

    return to_return;



}



BB one_rook(const BB &black_rook_loc, const BB &all_pieces, const BB &own_side){

    //cout << "ROOKTIME" << endl;
    //cout << "BRL: " << endl;
    //semiprinter(black_rook_loc);
    //cout << "end" << endl;

    BB horiz = (lineAttacks(all_pieces, findPosition(black_rook_loc), 0) & ~own_side);
    BB vertic = (lineAttacks(all_pieces, findPosition(black_rook_loc), 1) & ~own_side);


    //semiprinter(horiz);
    //semiprinter(vertic);

    return horiz | vertic;


}



void compute_rook(BB rook_locs, const BB &own_side, vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color, const int type_of_piece){
    while (rook_locs != 0){

        int temp_rook_loc = findAndClearSetBit(rook_locs);
        BB rook_loc = (1ULL << temp_rook_loc);
        BB rook_moves = one_rook(rook_loc,start_pos.AllPieces,own_side);

        if (color==0){
            ;

        } else if (color==1){
            cout << "ROOK/QUEEN ERROR" << endl;
        }

        createAllMoves(pot_moves, start_pos, rook_moves, findPosition(rook_loc), type_of_piece);

    }

}

BB one_bishop(const BB &black_bishop_loc, const BB &all_pieces, const BB &own_side){

    //cout << "ROOKTIME" << endl;
    //cout << "BRL: " << endl;
    //semiprinter(black_rook_loc);
    //cout << "end" << endl;

    BB one_diag = (lineAttacks(all_pieces, findPosition(black_bishop_loc), 2) & ~own_side);
    BB other_diag = (lineAttacks(all_pieces, findPosition(black_bishop_loc), 3) & ~own_side);


    //semiprinter(horiz);
    //semiprinter(vertic);

    return one_diag | other_diag;


}



void compute_bishop(BB bishop_locs, const BB &own_side, vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color, const int type_of_piece){
    while (bishop_locs != 0){

        int temp_bishop_loc = findAndClearSetBit(bishop_locs);
        BB bishop_loc = (1ULL << temp_bishop_loc);
        BB bishop_moves = one_bishop(bishop_loc,start_pos.AllPieces,own_side);

        if (color==0){
            ;

        } else if (color==1){
            cout << "BISHOP/QUEEN ERROR" << endl;
        }

        createAllMoves(pot_moves, start_pos, bishop_moves, findPosition(bishop_loc), type_of_piece);

    }

}


bool is_someone_in_check(const ChessBoard &start_pos, const int color){
    if (color==0){
        if (start_pos.WhiteQueens!=0){
            //return false;
            return true;
        }

        if (white_pawn_to_queen(start_pos.WhitePawns,start_pos.AllPieces,start_pos.AllBlackPieces)!=0){
            return true;
        }


        int white_king_pos = findPosition(start_pos.WhiteKing);
        int black_king_pos = findPosition(start_pos.BlackKing);
        int x1 = white_king_pos/8;
        int x2 = black_king_pos/8;
        int y1 = white_king_pos%8;
        int y2 = black_king_pos%8;

        if ((abs(x1-x2)<=2) and (abs(y1-y2)<=2)){
            return true;
        }

        BB pawn_locs = start_pos.WhitePawns;
        //board_printer(start_pos);
        //semiprinter(pawn_locs);
//        if (all_moves_for_debug.size()>=14){
//            cout << "PAWNLOCS" << endl;
//            semiprinter(pawn_locs);
//        }

        while (pawn_locs != 0){
            //cout << bitset<64>(pawn_locs) << endl;


            int temp_pawn_loc = findAndClearSetBit(pawn_locs);
            //cout << temp_pawn_loc << endl;
            BB pawn_loc = (1ULL << temp_pawn_loc);
            BB pawn_moves = two_move_white_pawn(pawn_loc,start_pos.AllPieces,start_pos.AllBlackPieces);


//            if (all_moves_for_debug.size()>=14){
//                cout << "PAWNMOVES TIME " << temp_pawn_loc << endl;
//                two_move_white_pawn(pawn_loc,start_pos.AllPieces,start_pos.AllBlackPieces, true);
//            }

            //semiprinter(pawn_moves);

            if ((pawn_moves&start_pos.BlackKing) !=0){

                return true;
            }

        }



    } else if (color==1){


        int white_king_pos = findPosition(start_pos.WhiteKing);
        int black_king_pos = findPosition(start_pos.BlackKing);
        int x1 = white_king_pos/8;
        int x2 = black_king_pos/8;
        int y1 = white_king_pos%8;
        int y2 = black_king_pos%8;

        if (abs(x1-x2)<=1 and abs(y1-y2)<=1){
            //cout << start_pos.second_last_move << " " << start_pos.last_move << " king_dist" << endl;
            return true;
        }

        BB bishop_locs = start_pos.BlackBishops | start_pos.BlackQueens;
        while (bishop_locs != 0){

            int temp_bishop_loc = findAndClearSetBit(bishop_locs);
            BB bishop_loc = (1ULL << temp_bishop_loc);
            BB bishop_moves = one_bishop(bishop_loc,start_pos.AllPieces,start_pos.AllBlackPieces);


            if ((bishop_moves&start_pos.WhiteKing) !=0){
                //cout << start_pos.second_last_move << " " << start_pos.last_move << " bishop/qn" << endl;
                return true;
            }

        }

        BB rook_locs = start_pos.BlackRooks | start_pos.BlackQueens;
        while (rook_locs != 0){

            int temp_rook_loc = findAndClearSetBit(rook_locs);
            BB rook_loc = (1ULL << temp_rook_loc);
            BB rook_moves = one_rook(rook_loc,start_pos.AllPieces,start_pos.AllBlackPieces);

            if ((rook_moves&start_pos.WhiteKing) !=0){
                //cout << start_pos.second_last_move << " " << start_pos.last_move << " rook/qn" << endl;
                return true;
            }

        }

        BB knight_locs = start_pos.BlackKnights;
        while (knight_locs != 0){
            //cout << bitset<64>(knight_locs) << endl;


            int temp_knight_loc = findAndClearSetBit(knight_locs);
            BB knight_loc = (1ULL << temp_knight_loc);
            BB knight_moves = one_knight(knight_loc,start_pos.AllBlackPieces);

//            if (start_pos.AllPieces==0b0100000110101000000000011001000000000000000010000000000000000000){
//                cout << "KNIGHT TIME" << endl;
//                semiprinter(knight_moves);
//                semiprinter(knight_moves&start_pos.WhiteKing);
//            }

            if ((knight_moves&start_pos.WhiteKing) !=0){
                //cout << start_pos.second_last_move << " " << start_pos.last_move << " knight" << endl;
                return true;
            }



        }

        BB pawn_locs = start_pos.BlackPawns;

        while (pawn_locs != 0){
            //cout << bitset<64>(pawn_locs) << endl;


            int temp_pawn_loc = findAndClearSetBit(pawn_locs);
            BB pawn_loc = (1ULL << temp_pawn_loc);
            BB pawn_moves = one_black_pawn_attacks(pawn_loc,start_pos.AllPieces,start_pos.AllWhitePieces);

            if ((pawn_moves&start_pos.WhiteKing) !=0){
                //cout << start_pos.second_last_move << " " << start_pos.last_move << " king_dist" << endl;
                return true;
            }

        }


    }
    return false;

}


void one_move_without_check(vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color){
    //qDebug() << "one_without_check";
    if (color==0){

        BB black_king_moves = compute_king(start_pos.BlackKing, start_pos.AllBlackPieces);

        createAllMoves(pot_moves, start_pos, black_king_moves, findPosition(start_pos.BlackKing), BLACK_KING_INT);


        compute_knight(start_pos.BlackKnights, start_pos.AllBlackPieces, pot_moves, start_pos, color);
        compute_black_pawns(start_pos.BlackPawns, start_pos.AllBlackPieces, pot_moves, start_pos, color);
        compute_rook(start_pos.BlackRooks, start_pos.AllBlackPieces, pot_moves, start_pos, color, BLACK_ROOK_INT);
        compute_bishop(start_pos.BlackBishops, start_pos.AllBlackPieces, pot_moves, start_pos, color, BLACK_BISHOP_INT);


        compute_rook(start_pos.BlackQueens, start_pos.AllBlackPieces, pot_moves, start_pos, color, BLACK_QUEEN_INT);
        compute_bishop(start_pos.BlackQueens, start_pos.AllBlackPieces, pot_moves, start_pos, color, BLACK_QUEEN_INT);



    } else if (color==1){
        BB white_king_moves = compute_king(start_pos.WhiteKing, start_pos.AllWhitePieces);
        createAllMoves(pot_moves, start_pos, white_king_moves, findPosition(start_pos.WhiteKing), WHITE_KING_INT);
        compute_white_pawns(start_pos.WhitePawns, start_pos.AllWhitePieces, pot_moves, start_pos, color);

        compute_rook(start_pos.WhiteQueens, start_pos.AllWhitePieces, pot_moves, start_pos,color, WHITE_QUEEN_INT);
        compute_bishop(start_pos.WhiteQueens, start_pos.AllWhitePieces, pot_moves, start_pos,color, WHITE_QUEEN_INT);



    }



}

void one_move_with_check(vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color){
    vector<ChessBoard> to_become;


    one_move_without_check(pot_moves,start_pos,color);
    for (auto x: pot_moves){
//        if (color==0){
//            board_printer(x);
//        }

        if (!is_someone_in_check(x,color)){
        //if (true){
            x.ZobristValue ^= zobrist_switch_side;
            to_become.push_back(x);
        }

    }
    pot_moves = to_become;


}


void two_moves_with_check(vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color){

    vector<ChessBoard> to_become;


    one_move_without_check(to_become,start_pos,color);

    for (auto x: to_become){
        vector<ChessBoard> second_level;
        one_move_with_check(second_level,x,color);
        for (auto y: second_level){
            pot_moves.push_back(y);
        }
    }

   // qDebug() << "two_moves_with_check";

    //pot_moves = to_become;



}




int eval(const ChessBoard &one_board, int color){

    int to_return = 0;
    to_return -= king_distance_calculator.at((one_board.WhiteKing|one_board.BlackKing));
//    int bking = findPosition(one_board.BlackKing);
//    int wking = findPosition(one_board.WhiteKing);
//
//    int b_x = bking % 8;
//    int b_y = bking / 8;
//    int w_x = wking % 8;
//    int w_y = wking / 8;
//
//    int dif = abs(b_x-w_x) + abs(b_y-w_y);
//
//    if (dif==1){
//        cout << "LMAO U  WRONG" << endl;
//    }
//    to_return -= dif;

    to_return += __builtin_popcountll(one_board.BlackBishops)*vals[BLACK_BISHOP_INT];
    to_return += __builtin_popcountll(one_board.BlackKnights)*vals[BLACK_KNIGHT_INT];
    to_return += __builtin_popcountll(one_board.BlackQueens)*vals[BLACK_QUEEN_INT];

    //to_return += __builtin_popcountll(one_board.BlackPawns)*vals[BLACK_PAWN_INT];
    to_return += __builtin_popcountll(one_board.BlackRooks)*vals[BLACK_ROOK_INT];

    //to_return += __builtin_popcountll(one_board.WhitePawns)*vals[WHITE_PAWN_INT];
    to_return += __builtin_popcountll(one_board.WhiteQueens)*vals[WHITE_QUEEN_INT];



    BB black_pawns_copy = one_board.BlackPawns;
    while (black_pawns_copy!=0){
        // lower pos means closer to bottom of chess board (wpov)
        int pos = findAndClearSetBit(black_pawns_copy) / 8;
        to_return += black_pawns_vals[pos];

    }

    BB white_pawns_copy = one_board.WhitePawns;

//    vector<int> pos_xs;

    while (white_pawns_copy!=0){
        // lower pos means closer to bottom of chess board (wpov)
        int pos = findAndClearSetBit(white_pawns_copy);
        int pos_y = pos / 8;
        int pos_x = pos % 8;
        //pos_xs.push_back(pos_x);
        BB maybe_blocked = (one_board.AllPieces >> (pos+8)) & 1ULL;
        if (maybe_blocked!=0){
            to_return += val_blocked_wpawn;
        }




        to_return += white_pawns_vals[pos_y];


    }
//    sort(pos_xs.begin(),pos_xs.end());
//    for (int i=0;i<pos_xs.size()-1;i++){
//        if (pos_xs[i+1]-pos_xs[i]==1){
//            to_return += val_connected_wpawn;
//        }
//    }






    return to_return;




}

bool spec_sort_white_tt(const ChessBoard &c1, const ChessBoard &c2){
    int first_worth;
    int second_worth;

    bool first_set = false;
    bool second_set = false;

    HASHE * phashe_1 = &hash_table[c1.ZobristValue%HASH_LENGTH];
    if (phashe_1->key==c1.ZobristValue ){
        if (phashe_1->best!=UNDEFINED){
            first_worth = phashe_1->hash_value;
            first_set = true;
        }
    }


    HASHE * phashe_2 = &hash_table[c2.ZobristValue%HASH_LENGTH];
    if (phashe_2->key==c2.ZobristValue ){
        if (phashe_2->best!=UNDEFINED){
            second_worth = phashe_2->hash_value;
            second_set = true;
        }
    }

    if (!first_set){
        first_worth = eval(c1,1);
    }
    if (!second_set){
        second_worth = eval(c2,1);
    }

    if (first_worth!=second_worth){
        return first_worth > second_worth;
    }

    return c1.ZobristValue < c2.ZobristValue;

}
bool spec_sort_black_tt(const ChessBoard &c1, const ChessBoard &c2){
    int first_worth;
    int second_worth;

    bool first_set = false;
    bool second_set = false;

    HASHE * phashe_1 = &hash_table[c1.ZobristValue%HASH_LENGTH];
    if (phashe_1->key==c1.ZobristValue ){
        if (phashe_1->best!=UNDEFINED){
            first_worth = (phashe_1->hash_value) * -1;
            first_set = true;
        }
    }


    HASHE * phashe_2 = &hash_table[c2.ZobristValue%HASH_LENGTH];
    if (phashe_2->key==c2.ZobristValue ){
        if (phashe_2->best!=UNDEFINED){
            second_worth = (phashe_2->hash_value) * -1;
            second_set = true;
        }
    }

    if (!first_set){
        first_worth = eval(c1,0);
    }
    if (!second_set){
        second_worth = eval(c2,0);
    }

    if (first_worth!=second_worth){
        return first_worth < second_worth;
    }

    return c1.ZobristValue < c2.ZobristValue;

}


bool spec_sort_white(const ChessBoard &c1, const ChessBoard &c2){
    //qDebug() << "sorting";
    int first_worth = eval(c1,1);

    int second_worth = eval(c2,1);



    if (first_worth!=second_worth){
        return first_worth > second_worth;
    }

    return c1.ZobristValue < c2.ZobristValue;

}
bool spec_sort_black(const ChessBoard &c1, const ChessBoard &c2){
    int first_worth = eval(c1,0);

    int second_worth = eval(c2,0);


    if (first_worth!=second_worth){
        return first_worth < second_worth;
    }

    return c1.ZobristValue < c2.ZobristValue;

}


bool spec_sort_white_tt_depth(const ChessBoard &c1, const ChessBoard &c2){
    int first_depth=-1;
    int second_depth=-1;
    bool first_set = false;
    bool second_set = false;

    HASHE * phashe_1 = &hash_table[c1.ZobristValue%HASH_LENGTH];
    if (phashe_1->key==c1.ZobristValue){
        first_depth = phashe_1->hash_depth;
        first_set = true;
    }


    HASHE * phashe_2 = &hash_table[c2.ZobristValue%HASH_LENGTH];
    if (phashe_2->key==c2.ZobristValue){
        second_depth = phashe_2->hash_depth;
        second_set = true;
    }

    if (first_set and second_set){
        if (first_depth==second_depth){
            return spec_sort_white_tt(c1,c2);

        } else {
            return first_depth > second_depth;
        }


    } else if (!first_set and second_set){
        return false;


    } else if (first_set and !second_set){
        return true;

    } else if (!first_set and !second_set){
        int first_worth = eval(c1,1);

        int second_worth = eval(c2,1);


        if (first_worth!=second_worth){
            return first_worth > second_worth;
        }

        return c1.ZobristValue < c2.ZobristValue;

    }

    //return c1.ZobristValue < c2.ZobristValue;

}
bool spec_sort_black_tt_depth(const ChessBoard &c1, const ChessBoard &c2){
    int first_depth=-1;
    int second_depth=-1;
    bool first_set = false;
    bool second_set = false;

    HASHE * phashe_1 = &hash_table[c1.ZobristValue%HASH_LENGTH];
    if (phashe_1->key==c1.ZobristValue){
        first_depth = phashe_1->hash_depth;
        first_set = true;
    }


    HASHE * phashe_2 = &hash_table[c2.ZobristValue%HASH_LENGTH];
    if (phashe_2->key==c2.ZobristValue){
        second_depth = phashe_2->hash_depth;
        second_set = true;
    }

    if (first_set and second_set){
        if (first_depth==second_depth){
            return spec_sort_black_tt(c1,c2);

        } else {
            return first_depth > second_depth;
        }


    } else if (!first_set and second_set){
        return false;


    } else if (first_set and !second_set){
        return true;

    } else if (!first_set and !second_set){
        int first_worth = eval(c1,1);

        int second_worth = eval(c2,1);


        if (first_worth!=second_worth){
            return first_worth < second_worth;
        }

        return c1.ZobristValue < c2.ZobristValue;

    }

}



bool spec_sort(const ChessBoard &c1, const ChessBoard &c2){
    return c1.ZobristValue < c2.ZobristValue;


}

void legal_moves(vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color, bool to_sort){

    if (color==0){
        ChessBoard tempcopy = start_pos;
       // qDebug() << QString::fromStdString(tempcopy.convert_into_string());
        one_move_with_check(pot_moves,start_pos,color);
        if (to_sort){
            sort(pot_moves.begin(),pot_moves.end(),spec_sort_black);
            auto temp_it = unique(pot_moves.begin(),pot_moves.end());
            pot_moves.resize(distance(pot_moves.begin(),temp_it));
        }
        //qDebug() << "MOVES: ";


    } else {
        two_moves_with_check(pot_moves,start_pos,color);

        if (to_sort){

            sort(pot_moves.begin(),pot_moves.end(),spec_sort_white);
            auto temp_it = unique(pot_moves.begin(),pot_moves.end());
            pot_moves.resize(distance(pot_moves.begin(),temp_it));



        }


    }



    if (!to_sort){
        sort(pot_moves.begin(),pot_moves.end(),spec_sort);
        auto temp_it = unique(pot_moves.begin(),pot_moves.end());
        pot_moves.resize(distance(pot_moves.begin(),temp_it));

    }



    //pot_moves.resize(10);


//    for (auto x: pot_moves){
//        //cout << "POTMOVE: " << endl;
//        //board_printer(x);
//    }
//
//    //pot_moves.push_back(start_pos);

}


void whites_potential_first_moves_for_ui(vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color, bool to_sort){
    if (color==0){
        qDebug() << "BLACK IN WHITE ONLY FUNCTION";
        return;
    }

    vector<ChessBoard> holding_moves;

    one_move_without_check(holding_moves,start_pos,color);

    for (auto x: holding_moves){
        vector<ChessBoard> seeing_if_any_moves;
        one_move_with_check(seeing_if_any_moves,x,color);
        if (seeing_if_any_moves.size()!=0){
            pot_moves.push_back(x);
        }


    }



}

void whites_potential_second_moves_for_ui(vector<ChessBoard> &pot_moves, const ChessBoard &start_pos, const int color, bool to_sort){
    one_move_with_check(pot_moves,start_pos,color);


}
