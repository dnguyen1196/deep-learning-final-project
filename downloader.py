from basketball_reference_web_scraper import client
roster_length = 15

#get list of games
def get_games_list(start_year, stop_year):
    games = []
    for year in range(start_year, stop_year):
        games.extend(client.season_schedule(season_end_year=year))
    return games

#get players in the game
def get_players_list(game, player_list):
    players = []
    away_name = game['away_team']
    home_name = game['home_team']
    date = game['start_time']
    home_team = player_list((date.day, date.month, date.year, home_name))
    away_team = player_list((date.day, date.month, date.year, away_name))
    home_team = home_team + [None] * (roster_length - len(home_team))
    away_team = away_team + [None] * (roster_length - len(away_team))
    return home_team + away_team

def gen_player_lookup(start_year, stop_year):
    games = []
    game_player_list = {}
    player_list = {}
    for day in range(1, 32):
        for month in range(1, 13):
            for year in range(start_year, stop_year):
                games = client.player_box_scores(day=day, month=month, year=year)
                for game in games:
                    player_id = game['slug']
                    team = game['team']
                    plist = player_list.get((day, month, year, team), [])
                    plist.append(player_id)
                    player_list[(day, month, year, team)] = plist
                    game_player_list[(day, month, year, player_id)] = game
    return (player_list, game_player_list)

#get data from players
def get_backwards_player_stats(player_name, start_day, start_month, start_year, first_year, num_games, player_list):
    year = start_year
    month = start_month
    day = start_day
    games = []
    while (len(games) < num_games) and (year >= first_year):
        day -= 1;
        if day <= 0:
            day = 31
            month -= 1;
            if month <= 0:
                month = 12;
                year -= 1;
        game = player_list.get((day, month, year, player_name), None)
        if game is not None:
            games.append(game)
    return games

(player_list, game_player_list) = gen_player_lookup(1946, 2018)
for game in get_games_list():
    

print(get_backwards_player_stats('antetgi01', 31, 12, 2017, 2016, 100, game_player_list))