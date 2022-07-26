import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from helpers import format_pts_mins

log = logging.getLogger()

games = pd.read_csv('games.csv', low_memory=False)
game_details = pd.read_csv('games_details.csv', low_memory=False)
players = pd.read_csv('players.csv', low_memory=False)
ranking = pd.read_csv('ranking.csv', low_memory=False)
teams = pd.read_csv('teams.csv', low_memory=False)

regr = LinearRegression()
sc = StandardScaler()
dt = DecisionTreeRegressor

def simple_lr(data, player_name):
    #log.info('reformatting dataframe')
    player_df = data[data['PLAYER_NAME'] == player_name].copy()
    cols = ['MIN', 'PTS']
    player_df = player_df[cols].reset_index(drop=True).dropna()
    format_pts_mins(player_df)
    X = np.array(player_df['MIN']).reshape(-1, 1)
    y = np.array(player_df['PTS']).reshape(-1, 1)
    #log.info('splitting data into training and validation sets')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    sc = StandardScaler()
    #log.info('standardizing data')
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test).astype(int)
    #plt.scatter(X_test, y_test, color='b')
    #plt.plot(X_test, y_pred, color='k')
    #plt.show()
    print("simple LR results \n"
          "r2: {}".format(r2_score(y_test, y_pred)),
          "MAE: {}".format(mean_absolute_error(y_test, y_pred)),
          "RMSE: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))

def multi_lr(data, player_name):
    player_df = data[data['PLAYER_NAME'] == player_name].copy()
    player_df = player_df[['MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT','OREB', 'AST', 'STL', 'TO', 'PF', 'PTS','PLUS_MINUS']]
    player_df = player_df.reset_index(drop=True).dropna()
    format_pts_mins(player_df)
    X = player_df[['MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OREB', 'AST', 'STL', 'TO', 'PF', 'PLUS_MINUS']]
    y = player_df.PTS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    print("multi-LR results \n"
          "r2: {}".format(r2_score(y_test, y_pred)),
          "MAE: {}".format(mean_absolute_error(y_test, y_pred)),
          "RMSE: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))

def bin_search(data, max_leaf_nodes, player_name):
    player_df = data[data['PLAYER_NAME'] == player_name].copy()
    player_df = player_df[['MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OREB', 'AST', 'STL', 'TO', 'PF', 'PTS', 'PLUS_MINUS']]
    player_df = player_df.reset_index(drop=True).dropna()
    format_pts_mins(player_df)
    X = player_df[['MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OREB', 'AST', 'STL', 'TO', 'PF', 'PLUS_MINUS']]
    y = player_df.PTS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    '''dt = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("Binary Search results \n"
          "r2: {}".format(r2_score(y_test, y_pred)),
          "MAE: {}".format(mean_absolute_error(y_test, y_pred)),
          "RMSE: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))'''

    random_forest = RandomForestRegressor(random_state=1, max_leaf_nodes=max_leaf_nodes)
    random_forest.fit(X_train, y_train)
    rf_pred = random_forest.predict(X_test)
    print("Random Forest results \n"
          "r2: {}".format(r2_score(y_test, rf_pred)),
          "MAE: {}".format(mean_absolute_error(y_test, rf_pred)),
          "RMSE: {}".format(np.sqrt(mean_squared_error(y_test, rf_pred))))

if __name__ == "__main__":
    #simple_lr(game_details, 'Karl-Anthony Towns')
    #multi_lr(game_details, 'Karl-Anthony Towns')
    #bin_search(game_details, max_leaf_nodes=17, player_name='Karl-Anthony Towns')
    for max_leaf_nodes in [71, 72, 73, 74, 75]:
        print(max_leaf_nodes, "leaf nodes")
        bin_search(game_details, max_leaf_nodes, 'Karl-Anthony Towns')
