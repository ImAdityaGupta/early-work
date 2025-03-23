#include "more_display_stuff.h"
#include "game.h"
#include "piece.h"

Start_Button::Start_Button(Game * game, bool left)
{
    int xpos;
    if (left){
        xpos = 100;
    } else {
        xpos = 340;
    }


    game->choose_colors.push_back(this);

    this->game = game;

    this->setRect(xpos,0,240,100);
    this->setBrush(QColor(255,255,255));

    //this->setZValue(1);

    game->scene->addItem(this);



    QLabel *start_label = new QLabel();
    QString color;
    if (left){
        color = "White";
    } else {
        color = "Black";
    }

    this->color = color;


    start_label->setText(color);
    QFont used_font( "Arial", 20, QFont::Bold);
    start_label->setFont(used_font);
    start_label->setAttribute(Qt::WA_TranslucentBackground);




    start_label->setGeometry(QRect(xpos,0,240,100));
    start_label->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);


    game->scene->addWidget(start_label);


}

void Start_Button::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    qDebug() << "button pressed";
    if (this->game->game_in_progress){
        return;
    }
    this->game->game_in_progress = true;

    if (this->color == "White"){
        this->game->human_is_white = true;
        this->game->human_is_black = false;
    } else if (this->color == "Black"){
        this->game->human_is_black = true;
        this->game->human_is_white = false;

    }

    this->game->make_engine_move();



}
