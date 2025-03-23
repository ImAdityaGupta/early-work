#include "piece.h"
#include "game.h"
#include "move_generation.h"

#include <stdlib.h>

Piece::Piece(Game * game, int xpos, int ypos, int width, int height, int pos_in_vect, QVector<Piece*> *ac_pieces)
{

    this->xpos = xpos;
    this->ypos = ypos;
    this->game = game;

    this->setRect(xpos,ypos,width,height);


    this->position_in_list = pos_in_vect;

    if (((this->position_in_list/8) + (this->position_in_list%8))%2==1){
        this->starting_color = QColor(38, 70, 83);
        this->highlighted_color = QColor(42, 157, 143);
        this->highlighted_second_color = QColor(24,110,48);
        this->middling_color = QColor(33,134,96);
        this->current_color = this->starting_color;
    } else {
        this->starting_color = QColor(255,255,255);
        this->highlighted_color = QColor(42, 157, 143);
        this->highlighted_second_color = QColor(24,110,48);
        this->middling_color = QColor(33,134,96);
        this->current_color = this->starting_color;

    }
    this->potential_go_to_color_white = QColor(255,0,0);
    this->potential_go_to_color_black = QColor(255,0,0);



    this->original_color();
    game->scene->addItem(this);







    this->first_selected = false;



    QString temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/Black_King.png" ;

    QChar this_piece = this->game->pieces_in_string->at(63-this->position_in_list);


    if (this_piece=='k'){
        temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/Black_King.png" ;
        this->has_piece = true;
    } else if (this_piece=='n'){
        temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/Black_Knight.png" ;
        this->has_piece = true;
    } else if (this_piece=='b'){
        temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/Black_Bishop.png" ;
        this->has_piece = true;
    } else if (this_piece=='r'){
        temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/Black_Rook.png" ;
        this->has_piece = true;
    } else if (this_piece=='p'){
        temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/Black_Pawn.png" ;
        this->has_piece = true;
    } else if (this_piece=='q'){
        temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/Black_Queen.png" ;
        this->has_piece = true;
    } else if (this_piece=='K'){
        temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/White_King.png" ;
        this->has_piece = true;
    } else if (this_piece=='P'){
        temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/White_Pawn.png" ;
        this->has_piece = true;
    } else if (this_piece=='Q'){
        temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/White_Queen.png" ;
        this->has_piece = true;
    } else if (this_piece==' '){
        this->has_piece = false;
    }

    if (this->has_piece){
        QPixmap pix(temp_address);


        QGraphicsPixmapItem *picture_ptr = game->scene->addPixmap(pix);
        picture_ptr->setPos(xpos,ypos);
        picture_ptr->setZValue(1);


        board_pic = picture_ptr;
    } else {



        QGraphicsPixmapItem *picture_ptr = game->scene->addPixmap(QPixmap());
        picture_ptr->setPos(xpos,ypos);
        picture_ptr->setZValue(1);


        board_pic = picture_ptr;


    }
//    qDebug() << (*(this->pieces_in_string));
    QString qstr = QString::fromStdString((*(this->game->pieces_in_string)));
    //qDebug() << qstr;




}

void Piece::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    QString qstr = QString::fromStdString((*(this->game->pieces_in_string)));
    qDebug() << qstr;


    if (this->game->machine_thinking){
        qDebug() << "cannot interact rn";
        return;


    }

    //qApp->processEvents();
    //this->game->make_engine_move();


//    if (this->game->human_is_white and this->game->whose_turn==0){

//        this->game->machine_thinking = true;
//        ChessBoard will_receive_engine_move;
//        qDebug() << "MACHINE IS THINKING";

//        ChessBoard to_give_to_engine = convert_string_into_board_for_ui(*(this->game->pieces_in_string));
//        to_give_to_engine.ZobristValue = naive_zobrist(to_give_to_engine);


//        negamax_ui_wrapper(eval, to_give_to_engine, 10, this->game->whose_turn, false, std::ref(will_receive_engine_move),std::ref(this->game->machine_thinking),std::ref(this->game->whose_turn));
////        //std::thread t1(test_func, 10);
////        QThreadPool temp_pool;
////        //QFuture<void> future = QtConcurrent::run(&temp_pool, test_func,10,std::ref(this->game->machine_thinking),std::ref(this->game->whose_turn));
////        QFuture<void> future = QtConcurrent::run(&temp_pool, negamax_ui_wrapper, eval, convert_string_into_board_for_ui(*(this->game->pieces_in_string)), 10, this->game->whose_turn,
////                                                 false, std::ref(will_receive_engine_move),std::ref(this->game->machine_thinking),std::ref(this->game->whose_turn));
//        this->setup_board_from_chessboard(will_receive_engine_move);

//        qDebug() << "continued";
//        return;

//    }



    //qDebug() << (this->game->pieces_in_string->at(63-this->position_in_list));



//    QString temp = *((this->pieces_in_string));
    //qDebug() << temp;


    //qDebug() << this->position_in_list;
    if (first_selected){
        first_selected = false;
        this->clear_first_selected();

    } else {

        if (this->game->has_any_square_been_selected()){
            if (!this->game->game_in_progress){
                first_selected = false;
                this->clear_first_selected();
                return;
            }


            this->move_piece(this->game->which_square_has_been_selected(),this->position_in_list);
            this->clear_first_selected();
            qApp->processEvents();
            this->game->make_engine_move();

        } else {
            if (!has_piece){
                return;
            }

            first_selected = true;
            this->selected_color();
            this->color_potential_moves();


        }

    }



}


void Piece::original_color(){


    this->setBrush(this->starting_color);

}

void Piece::selected_color()
{
    this->setBrush(this->highlighted_color);
}

void Piece::move_piece(int first_square, int second_square)
{

    //qDebug() << first_square << " " << second_square;

    if (!this->game->check_if_move_is_legal(63-first_square,63-second_square)){
        return;
    }


    //qDebug() << this->all_pieces->at(second_square)->xpos << " " << this->all_pieces->at(second_square)->ypos;
    this->game->all_pieces->at(first_square)->board_pic->setPos(this->game->all_pieces->at(second_square)->xpos,this->game->all_pieces->at(second_square)->ypos);


    this->game->scene->removeItem(this->game->all_pieces->at(second_square)->board_pic);
    this->game->all_pieces->at(second_square)->board_pic = game->scene->addPixmap(QPixmap());


    swap(this->game->all_pieces->at(first_square)->board_pic,this->game->all_pieces->at(second_square)->board_pic);


    this->game->all_pieces->at(first_square)->has_piece = false;
    this->game->all_pieces->at(second_square)->has_piece = true;

    if (this->game->whose_turn==0){
        this->game->whose_turn = 1 - this->game->whose_turn;
    } else {
        if (this->game->first_move){
            this->game->first_move = false;
        } else {
            this->game->first_move = true;
            this->game->whose_turn = 1 - this->game->whose_turn;
        }

    }





//    this->game->pieces_in_string->at(63-second_square) = ' ';
//    swap(this->game->pieces_in_string->at(63-first_square),this->game->pieces_in_string->at(63-second_square));




}


void Piece::clear_first_selected()
{
    for (int i=0;i<64;i++){
        this->game->all_pieces->at(i)->first_selected = false;
        this->game->all_pieces->at(i)->original_color();

    }






}

void Piece::color_potential_moves()
{
    int square_moving = this->game->which_square_has_been_selected();
    if (this->game->whose_turn==0){
        //qDebug() << "highlighting perhaps?";
        vector<ChessBoard> pot_moves;
        ChessBoard to_go_into_board = convert_string_into_board_for_ui(*(this->game->pieces_in_string));
        legal_moves(pot_moves,to_go_into_board,this->game->whose_turn);
        //qDebug() << "NUMBER OF AVAILABLE MOVES " << pot_moves.size();








        for (auto x: pot_moves){


            for (int j=0;j<64;j++){
                bool previous = (x.AllPieces >> (63-square_moving)) & 1ULL;
                bool other_prev = (x.AllBlackPieces >> (63-j)) & 1ULL;
                bool back_checking = (to_go_into_board.AllBlackPieces >> (63-j)) & 1ULL;
                //bool checking_to_move = (x.AllPieces >> (63-square_moving)) & 1ULL;

                //bool afterwards = (all_pieces_later[i] >> (63-j)) & 1ULL;


                if (!previous and other_prev and !back_checking){
                   // qDebug() << "to_highlight: " << j << " " << QString::fromStdString(x.convert_into_string());

                    this->game->all_pieces->at(j)->setBrush(this->game->all_pieces->at(j)->highlighted_second_color);

                }
                //break;
             }




        }


    } else if (this->game->whose_turn==1){
        if (this->game->first_move){

            QVector<int> to_first_highlight;
            QVector<int> to_second_highlight;

            vector<ChessBoard> pot_moves, ac_pot_moves, moves_in_which_relevant_piece_moves;
            ChessBoard to_go_into_board = convert_string_into_board_for_ui(*(this->game->pieces_in_string));
            one_move_without_check(pot_moves,to_go_into_board,this->game->whose_turn);
            for (auto x: pot_moves){
                vector<ChessBoard> second_level;
                one_move_with_check(second_level,x,this->game->whose_turn);
                if (second_level.size()!=0){
                    ac_pot_moves.push_back(x);
                }




            }




            for (auto x: ac_pot_moves){


                for (int j=0;j<64;j++){
                    bool previous = (x.AllPieces >> (63-square_moving)) & 1ULL;
                    bool other_prev = (x.AllWhitePieces >> (63-j)) & 1ULL;
                    bool back_checking = (to_go_into_board.AllWhitePieces >> (63-j)) & 1ULL;
                    //bool checking_to_move = (x.AllPieces >> (63-square_moving)) & 1ULL;

                    //bool afterwards = (all_pieces_later[i] >> (63-j)) & 1ULL;


                    if (!previous and other_prev and !back_checking){
                      //  qDebug() << "to_highlight: " << j << " " << QString::fromStdString(x.convert_into_string());

                        to_first_highlight.push_back(j);
                        moves_in_which_relevant_piece_moves.push_back(x);


                    }
                    //break;
                 }




            }

            for (int i=0;i<moves_in_which_relevant_piece_moves.size();i++){
                int temp_square_moving = to_first_highlight[i];

                vector<ChessBoard> second_level;
                one_move_with_check(second_level,moves_in_which_relevant_piece_moves[i],this->game->whose_turn);

                for (auto x: second_level){

                    for (int j=0;j<64;j++){
                        bool previous = (x.AllPieces >> (63-temp_square_moving)) & 1ULL;
                        bool other_prev = (x.AllWhitePieces >> (63-j)) & 1ULL;
                        bool back_checking = (moves_in_which_relevant_piece_moves[i].AllWhitePieces >> (63-j)) & 1ULL;
                        //bool checking_to_move = (x.AllPieces >> (63-square_moving)) & 1ULL;

                        //bool afterwards = (all_pieces_later[i] >> (63-j)) & 1ULL;


                        if (!previous and other_prev and !back_checking){
                          //  qDebug() << "to_highlight: " << j << " " << QString::fromStdString(x.convert_into_string());

                            to_second_highlight.push_back(j);


                        }
                        //break;
                     }




                }








            }





            for (auto square: to_first_highlight){
                bool conflict = false;
                for (auto check: to_second_highlight){
                    if (check==square){
                        conflict = true;
                    }
                }

                if (!conflict){
                    this->game->all_pieces->at(square)->setBrush(this->game->all_pieces->at(square)->highlighted_color);
                } else {
                    this->game->all_pieces->at(square)->setBrush(this->game->all_pieces->at(square)->middling_color);

                }


            }

            for (auto square: to_second_highlight){
                bool conflict = false;
                for (auto check: to_first_highlight){
                    if (check==square){
                        conflict = true;
                    }
                }

                if (!conflict){
                    this->game->all_pieces->at(square)->setBrush(this->game->all_pieces->at(square)->highlighted_second_color);
                } else {
                    this->game->all_pieces->at(square)->setBrush(this->game->all_pieces->at(square)->middling_color);

                }
            }




        //    qDebug() << "NUMBER OF AVAILABLE MOVES " << pot_moves.size();



        } else {
           // qDebug() << "highlighting perhaps?";
            vector<ChessBoard> pot_moves;
            ChessBoard to_go_into_board = convert_string_into_board_for_ui(*(this->game->pieces_in_string));
            one_move_with_check(pot_moves,to_go_into_board,this->game->whose_turn);
            for (auto x: pot_moves){


                for (int j=0;j<64;j++){
                    bool previous = (x.AllPieces >> (63-square_moving)) & 1ULL;
                    bool other_prev = (x.AllWhitePieces >> (63-j)) & 1ULL;
                    bool back_checking = (to_go_into_board.AllWhitePieces >> (63-j)) & 1ULL;
                    //bool checking_to_move = (x.AllPieces >> (63-square_moving)) & 1ULL;

                    //bool afterwards = (all_pieces_later[i] >> (63-j)) & 1ULL;


                    if (!previous and other_prev and !back_checking){
                     //   qDebug() << "to_highlight: " << j << " " << QString::fromStdString(x.convert_into_string());

                        this->game->all_pieces->at(j)->setBrush(this->game->all_pieces->at(j)->highlighted_second_color);

                    }
                    //break;
                 }




            }



        }












    }



}

