#include "game.h"
#include "piece.h"
#include "board_class.h"

Game::Game(int width, int height, ChessBoard starting_position){
    this->whose_turn = 1;
    this->first_move = true;


    // create the scene
    scene = new QGraphicsScene();
    scene->setSceneRect(0,0,width,height); // make the scene 800x600 instead of infinity by infinity (default)
    // make the newly created scene the scene to visualize (since Game is a QGraphicsView Widget,
    // it can be used to visualize scenes)
    setScene(scene);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setFixedSize(width,height);
    int square_size = 60;
    int x_offset = 100;
    int y_offset = 100;


    QVector<Piece*>* main_list = new QVector<Piece*>();





    std::string temp = starting_position.convert_into_string();

    string * try_ptr = new string(temp);

    this->pieces_in_string = try_ptr;




    for (int i=0;i<8;i++){
        for (int j=0;j<8;j++){
//            QGraphicsRectItem * temp_rect = new QGraphicsRectItem();
//            temp_rect->setRect(i*square_size+x_offset,j*square_size+y_offset,square_size,square_size);
//            scene->addItem(temp_rect);

            Piece * one_piece = new Piece(this, j*square_size+x_offset, i*square_size+y_offset, square_size, square_size, ((i*8)+j), main_list);
            main_list->push_back(one_piece);











        }
    }


    this->all_pieces = main_list;





    Start_Button * white_color = new Start_Button(this,true);
    Start_Button * black_color = new Start_Button(this,false);





    show();
}

bool Game::check_if_move_is_legal(int first_square, int second_square)
{
   // return true;
    std::string copy_of_board = *(this->pieces_in_string);
    copy_of_board[second_square] = ' ';

    if ( (second_square<8 or second_square>=56) and (this->pieces_in_string->at(first_square)=='p')){
        copy_of_board[first_square] = 'q';

    }
    if ( (second_square<8 or second_square>=56) and (this->pieces_in_string->at(first_square)=='P')){
        copy_of_board[first_square] = 'Q';

    }

    swap(copy_of_board[first_square],copy_of_board[second_square]);

    std::vector<ChessBoard> pot_moves;
    ChessBoard to_go_into_board = convert_string_into_board_for_ui(*(this->pieces_in_string));
    //qDebug() << "converted";

    if (this->whose_turn==0){
        legal_moves(pot_moves, to_go_into_board, this->whose_turn);
    } else {
        if (this->first_move){
            whites_potential_first_moves_for_ui(pot_moves, to_go_into_board, this->whose_turn);
        } else {
            whites_potential_second_moves_for_ui(pot_moves, to_go_into_board, this->whose_turn);

        }


    }



    //qDebug() << "legal_moves done";


    std::vector<string> pot_moves_into_strings;

    for (auto x: pot_moves){
        pot_moves_into_strings.push_back(x.convert_into_string());

    }

    for (auto x: pot_moves_into_strings){
        if (x==copy_of_board){
            this->pieces_in_string = new string(copy_of_board);
            return true;
        }
    }

    return false;







}


bool Game::has_any_square_been_selected()
{

   for (int i=0;i<64;i++){
       if (this->all_pieces->at(i)->first_selected){
           return true;
       }



   }
   return false;
}


int Game::which_square_has_been_selected()
{
    for (int i=0;i<64;i++){
        if (this->all_pieces->at(i)->first_selected){
            return i;
        }
    }
    qDebug() << "-1";
    return -1;
}



void Game::setup_board_from_chessboard(ChessBoard position)
{
    QString qstr = QString::fromStdString(position.convert_into_string());


    this->pieces_in_string = new string(position.convert_into_string());

    qDebug() << QString::fromStdString((*(this->pieces_in_string)));

    reverse(qstr.begin(),qstr.end());


    QString temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/Black_King.png" ;

    for (int i=0;i<64;i++){
        bool temp_has_piece;

        QChar this_piece = qstr[i];

        if (this_piece=='k'){
            temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/Black_King.png" ;
            temp_has_piece = true;
        } else if (this_piece=='n'){
            temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/Black_Knight.png" ;
            temp_has_piece = true;
        } else if (this_piece=='b'){
            temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/Black_Bishop.png" ;
            temp_has_piece = true;
        } else if (this_piece=='r'){
            temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/Black_Rook.png" ;
            temp_has_piece = true;
        } else if (this_piece=='p'){
            temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/Black_Pawn.png" ;
            temp_has_piece = true;
        } else if (this_piece=='q'){
            temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/Black_Queen.png" ;
            temp_has_piece = true;
        } else if (this_piece=='K'){
            temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/White_King.png" ;
            temp_has_piece = true;
        } else if (this_piece=='P'){
            temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/White_Pawn.png" ;
            temp_has_piece = true;
        } else if (this_piece=='Q'){
            temp_address = "C:/Users/adidh/Documents/Long Term Projects/Monster Chess/Chess_after_tutorials_1/White_Queen.png" ;
            temp_has_piece = true;
        } else if (this_piece==' '){
            temp_has_piece = false;
        }
        QPixmap pix;
        if (temp_has_piece){
            pix = QPixmap(temp_address);
            this->all_pieces->at(i)->has_piece = true;
        } else {
            pix = QPixmap();
            this->all_pieces->at(i)->has_piece = false;
        }


        this->scene->removeItem(this->all_pieces->at(i)->board_pic);
        this->all_pieces->at(i)->board_pic = this->scene->addPixmap(pix);


        this->all_pieces->at(i)->board_pic->setPos( this->all_pieces->at(i)->xpos, this->all_pieces->at(i)->ypos);
        this->all_pieces->at(i)->board_pic->setZValue(1);




    }








}




void Game::make_engine_move()
{
    if (this->human_is_white and this->whose_turn==0){

        this->machine_thinking = true;
        ChessBoard will_receive_engine_move;
        qDebug() << "MACHINE IS THINKING";

        ChessBoard to_give_to_engine = convert_string_into_board_for_ui(*(this->pieces_in_string));
        to_give_to_engine.ZobristValue = naive_zobrist(to_give_to_engine);


        negamax_ui_wrapper(eval, to_give_to_engine, 5, this->whose_turn, false, std::ref(will_receive_engine_move),std::ref(this->machine_thinking),std::ref(this->whose_turn));
//        //std::thread t1(test_func, 10);
//        QThreadPool temp_pool;
//        //QFuture<void> future = QtConcurrent::run(&temp_pool, test_func,10,std::ref(this->game->machine_thinking),std::ref(this->game->whose_turn));
//        QFuture<void> future = QtConcurrent::run(&temp_pool, negamax_ui_wrapper, eval, convert_string_into_board_for_ui(*(this->game->pieces_in_string)), 10, this->game->whose_turn,
//                                                 false, std::ref(will_receive_engine_move),std::ref(this->game->machine_thinking),std::ref(this->game->whose_turn));
        this->setup_board_from_chessboard(will_receive_engine_move);

        qDebug() << "continued";
        return;

    }

    if (this->human_is_black and this->whose_turn==1){
        this->machine_thinking = true;
        ChessBoard will_receive_engine_move;
        qDebug() << "MACHINE IS THINKING";

        ChessBoard to_give_to_engine = convert_string_into_board_for_ui(*(this->pieces_in_string));
        to_give_to_engine.ZobristValue = naive_zobrist(to_give_to_engine);


        negamax_ui_wrapper(eval, to_give_to_engine, 5, this->whose_turn, false, std::ref(will_receive_engine_move),std::ref(this->machine_thinking),std::ref(this->whose_turn));
//        //std::thread t1(test_func, 10);
//        QThreadPool temp_pool;
//        //QFuture<void> future = QtConcurrent::run(&temp_pool, test_func,10,std::ref(this->game->machine_thinking),std::ref(this->game->whose_turn));
//        QFuture<void> future = QtConcurrent::run(&temp_pool, negamax_ui_wrapper, eval, convert_string_into_board_for_ui(*(this->game->pieces_in_string)), 10, this->game->whose_turn,
//                                                 false, std::ref(will_receive_engine_move),std::ref(this->game->machine_thinking),std::ref(this->game->whose_turn));
        this->setup_board_from_chessboard(will_receive_engine_move);

        qDebug() << "continued";
        return;

    }


}


