class Player:
    starting_elo = 2750
    starting_k_factor = 16
    all_games = []
    all_players = []

    player_index = 0



    def __init__(self, name):
        self.name = name
        self.elo_history = []

        self.current_elo = Player.starting_elo
        self.elo_history.append(self.current_elo)

        self.k_factor = Player.starting_k_factor
        self.player_index = Player.player_index

        Player.player_index += 1
        Player.all_players.append(self)


data = [
    [ 0, -1,  2,  1,  3, -1, -1, -1],
    [ 0, -1,  2,  1,  3, -1, -1, -1],
    [ 2, -1,  0,  1,  3, -1, -1, -1],
    [ 0, -1,  2,  1,  3, -1, -1, -1],
    [ 3, -1,  2,  0,  1, -1, -1, -1],

    [ 3, -1,  1,  0,  2, -1, -1, -1],
    [ 3,  0,  1,  4,  2, -1, -1, -1],
    [ 0,  2,  3,  1,  4, -1, -1, -1],
    [ 1,  2,  3, -1,  0, -1, -1, -1],
    [ 1,  4,  2,  0,  3, -1, -1, -1],
    [ 3, -1,  1,  0,  2, -1, -1, -1],
    [ 0,  1,  2,  4,  3, -1, -1, -1],

    [ 1, -1,  0,  3,  2, -1, -1, -1],

    [ 0, -1, -1,  1, -1, -1, -1, -1], #heads up

    [ 3, -1,  2,  0,  1, -1, -1, -1],

    [ 0, -1, -1,  1, -1, -1, -1, -1], #heads up

    [ 2, -1,  0,  1,  3, -1, -1, -1],
    [ 2, -1,  0,  3,  1, -1, -1, -1],

    [ 4, -1,  0,  3,  1,  2,  5, -1],

    #post-exams

    [ 1, -1,  2,  4,  3,  0,  5, -1],
    [ 3, -1,  0,  2,  1, -1, -1, -1],

    [ 1, -1,  3,  0,  2, -1, -1, -1],

    #1b michaelmas

    [ 1, -1,  0,  2,  3,  4, -1, -1],

    #hungary

    [ 0, -1,  2,  1,  3,  4, -1, -1],

    # term 2 1b

    [ 1, -1,  0,  4,  2,  3, -1, -1],

    [ 1, -1, -1,  0, -1, -1, -1, -1], #heads up

    [ 2, -1,  3,  0,  1,  4, -1, -1],

    [ 2, -1,  0,  5,  4,  1, -1,  3],

    [ 2, -1, -1,  1, -1,  0,  -1, 3] # end of 2nd term, 2nd year


]


def add_game(one_game):
    '''
    one_game = [position of player with index 0, position of player...]
    didn't play <> position = -1
    '''

    Player.all_games.append(one_game)


def rating_change_calc(player_1, player_2, who_won):
    '''
    who_won is a boolean. True => rating_1 won, False => rating_2 won.
    '''

    rating_1 = player_1.current_elo
    rating_2 = player_2.current_elo

    expected_1 = 1 / (1+pow(10,(rating_2-rating_1)/400))
    expected_2 = 1 / (1+pow(10,(rating_1-rating_2)/400))

    if who_won:

        p1_change = round(player_1.k_factor * (1 - expected_1),2)
        p2_change = round(player_2.k_factor * (0 - expected_2),2)
    else:

        p1_change = round(player_1.k_factor * (0 - expected_1),2)
        p2_change = round(player_2.k_factor * (1 - expected_2),2)

    return [p1_change,p2_change]


def calculate_elo():

    for n,game in enumerate(Player.all_games):
        rating_changes = [0 for i in range(len(Player.all_players))]

        for pos_1, player_1 in enumerate(Player.all_players):
            for pos_2, player_2 in enumerate(Player.all_players):
                if game[pos_1] == -1 or game[pos_2] == -1 or pos_1 == pos_2:
                    continue
                else:
                    if game[pos_1] < game[pos_2]: # i.e. player_1 has won.
                        temp_changes = rating_change_calc(player_1,player_2,True)
                        rating_changes[pos_1] += temp_changes[0]
                    elif game[pos_2] < game[pos_1]: # i.e. player_2 has won.
                        temp_changes = rating_change_calc(player_1,player_2,False)
                        rating_changes[pos_1] += temp_changes[0]



        for pos, player in enumerate(Player.all_players):
            player.current_elo = round(player.current_elo + rating_changes[pos],2)
            player.elo_history.append(player.current_elo)

        # for player in Player.all_players:
        #     print(f'{player.name} {player.current_elo}')
        # print("")


def main_func():
    Player("Aditya")
    Player("Alessandro")
    Player("Anthony")
    Player("Hayden")
    Player("Thomas")

    Player("George")
    Player("Lizzie")

    Player("Lingde")



    # Player.all_players[0].current_elo = 1000


    for game in data:
        add_game(game)
    calculate_elo()

    for player in Player.all_players:
        print(f'{player.name} {" ".join([str(x) for x in player.elo_history])}')

main_func()