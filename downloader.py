from basketball_reference_web_scraper import client
import pickle
import pandas as pd
import numpy as np
import pytz
roster_length = 15
num_games = 10
first_year = 1945

name_index = 'slug'
team_index = 'team'
indices_of_interest = ['game_score', 'made_three_point_field_goals', 'attempted_three_point_field_goals', 'made_field_goals', 'attempted_field_goals', 'made_free_throws', 'attempted_free_throws', 'assists', 'offensive_rebounds', 'seconds_played', 'defensive_rebounds', 'turnovers', 'steals',  'blocks', 'personal_fouls']
date_index = 'start_time'
home_score_index = 'home_team_score'
away_score_index = 'away_team_score'


#get list of games
def get_games_list(start_year, stop_year):
    # Get the list of games from year to year
    games = []
    for year in range(start_year, stop_year):
        games.extend(client.season_schedule(season_end_year=year))
    est = pytz.timezone('US/Eastern')
    for i in range(0, len(games)):
        game = games[i]
        game[date_index] = game[date_index].astimezone(est)
        games[i] = game
    return games

#get players in the game
def get_game_participants(game, player_list):
    players = []
    away_name = game['away_team']
    home_name = game['home_team']
    date = game[date_index]
    home_team = player_list[(date.day, date.month, date.year, home_name)]
    away_team = player_list[(date.day, date.month, date.year, away_name)]
    home_team = home_team + [None] * (roster_length - len(home_team))
    away_team = away_team + [None] * (roster_length - len(away_team))
    return home_team + away_team

# Generate player look up
def gen_player_lookup(start_year, stop_year):
    games = []
    game_player_list = {}
    player_list = {}
    for year in range(start_year, stop_year):
        for month in range(1, 13):
            for day in range(1, 32):
                try:
                    print(month, day, year, 'gen_player_lookup')
                    games = client.player_box_scores(day=day, month=month, year=year)
                    for game in games:
                        player_id = game[name_index]
                        team = game[team_index]
                        plist = player_list.get((day, month, year, team), [])
                        plist.append(player_id)
                        player_list[(day, month, year, team)] = plist
                        game_player_list[(day, month, year, player_id)] = [game[index] for index in indices_of_interest]
                except Exception as e:
                    print (e)
                    continue
    return (player_list, game_player_list)

#get data from players
def get_backwards_player_stats(player_name, start_day, start_month, start_year, first_year, num_games, game_player_list):
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
        game = game_player_list.get((day, month, year, player_name), None)
        print(game, (day, month, year, player_name))
        if game is not None:
            games.append(game)
    while (len(games) < num_games):
        games.append(np.zeros(len(indices_of_interest)))
    return games


outputs = pd.DataFrame(columns=[home_score_index, away_score_index])

start_year = 2005
end_year   = 2017

game_list = get_games_list(start_year, end_year)
print('gen games')


(player_list, game_player_list) = gen_player_lookup(start_year-1, end_year)
print('gen lookup')

game_dfs = []
for game in game_list[0:1]:
    print('gen game')
    participants = get_game_participants(game, player_list)
    game_date = game[date_index]
    start_day = game_date.day
    start_month = game_date.month
    start_year = game_date.year
    outputs = outputs.append({home_score_index:game[home_score_index], away_score_index:game[away_score_index]}, ignore_index=True)
    player_dfs= []
    for player in participants:
        stats = get_backwards_player_stats(player, start_day, start_month, start_year, first_year, num_games, game_player_list)
        player_df = np.asarray(stats)
        player_dfs.append(player_df)
    game_df = np.asarray(player_dfs)
    game_dfs.append(game_df)

data = np.asarray(game_dfs)


np.save('data.pkl', data)
np.save('outputs.pkl', outputs)