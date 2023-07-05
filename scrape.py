# imports
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import seaborn as sns
import requests
import re
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

base = "https://www.fantasypros.com/nfl/stats/"

# https://www.fantasypros.com/nfl/games/justin-jefferson.php?season=2021&scoring=PPR

def extract_players(year: str, pos: str, scoring: str):
    """
    Reads html from fantasypros stats per position, year, and scoring specifications into a pandas df.
    Also transforms player names to be url-friendly
    Inputs: 
        Position: a string consisting of "wr", "rb", "te", "qb"
        Year: string consisting of a year (yyyy)
        Scoring: string consisting of "PPR", "HALF"
    """
    data = pd.read_html("https://www.fantasypros.com/nfl/stats/" + pos + ".php?year=" + year + "&scoring=" + scoring)[0]
    # unnesting
    data.columns = ['_'.join(col) for col in data.columns.values]
    # renaming first two columns
    data.columns.values[0] = "Rank"
    data.columns.values[1] = "Name"
    # setting rank = idx
    data = data.set_index(['Rank'])
    data['name'] = data['Name'].apply(lambda x: clean_name(x))
    return data

def clean_name(text: str):
    text = text.lower()
    if len(text.split(' ')) != 2 and text != 'amon-ra st. brown':
        text = text.rsplit(' ', 1)[0]
    text = text.replace("amon-ra", "amonra")
    text = text.replace(" jr.", "")
    text = text.replace(" sr.", "")
    text = text.replace("st. ", "st.")
    text = text.replace(".", "")
    text = text.replace("'", "")
    text = text.replace(" ", "-")
    # edge cases in top 100 (2020-2022 wrs), no rhyme/reason
    text = text.replace("gabe", "gabriel")
    text = text.replace("joshua-palmer", "josh-palmer")
    text = text.replace("valdes-scantling", "valdesscantling")
    text = text.replace("robinson-ii", "robinson")
    text = text.replace("westbrook-ikhine", "westbrook")
    text = text.replace("equanimeous-stbrown", "equanimeous-st-brown")
    text = text.replace("chosen", "robby")
    text = text.replace("ruggs-iii", "ruggs")
    text = text.replace("william-fuller-v", "will-fuller")
    text = text.replace("willie-snead-iv", "willie-snead")
    # RB edits
    text = text.replace('kenneth-walker-iii', 'kenneth-walker-rb')
    text = text.replace('jeff-wilson', 'jeffery-wilson')
    text = text.replace('brian-robinson', 'brian-robinson-jr')
    text = text.replace('zonovan', 'zonovan-bam')
    text = text.replace('ingram-ii', 'ingram')
    text = text.replace("melvin-gordon-iii", "melvin-gordon")
    text = text.replace("avery-williams", "avery-williams-cb")
    text = text.replace("pierre-strong", "pierre-strong-jr")
    text = text.replace("mike-davis", "mike-davis-rb")
    text = text.replace("david-johnson", "david-johnson-rb")
    text = text.replace("todd-gurley-ii", 'todd-gurley')
    text = text.replace("benny-snell", 'benjamin-snell-jr')
    text = text.replace("adrian-peterson", 'adrian-peterson-min')
    text = text.replace("rodney-smith", 'rodney-smith-rb')
    # TE edits
    text = text.replace("irv-smith", "irv-smith-jr")
    if text in ['dj-moore', 'mike-williams', 'michael-thomas']:
        text += '-wr'
    if text in ['najee-harris', 'michael-carter', 'damien-harris', 'justin-jackson', 'elijah-mitchell']:
        text += '-rb'
    return text

def extract_age(name: str, year_: str):
    """
    given player name and year, finds the age of that player at the specified time
    Inputs: 
        Position: a string consisting of "wr", "rb", "te", "qb"
        Year: string consisting of a year (yyyy)
        Scoring: string consisting of "PPR", "HALF"
    """
    time_diff = dt.date.today().year - eval(year_)
    url = "https://www.fantasypros.com/nfl/games/" + name + ".php?season=" + year_ + "&scoring=PPR"
    result = requests.get(url)
    doc = BeautifulSoup(result.text, "html.parser")
    deets = doc.find_all("span", attrs={'class': 'bio-detail'})
    for deet in deets:
        if "Age" in str(deet):
            current_age = int(re.findall(r'\b\d+\b', str(deet))[0])
    return current_age - time_diff


def extract_year(name: str, year: str, scoring: str):
    """
    Reads a player's game log from a past year 
    Inputs: 
        Name: player name
        Year: string consisting of a year (yyyy)
        Scoring: string consisting of "PPR", "HALF"
    """
    url = "https://www.fantasypros.com/nfl/games/" + name + ".php?season=" + year + "&scoring=" + scoring
    data = pd.read_html(url)[0]
    data.columns = ['_'.join(col) for col in data.columns.values]
    data = data.drop(data.columns[[0, 1, 2]],axis = 1)
    data = data.drop(data.columns[-1], axis=1)
    data = data.apply(pd.to_numeric, errors = "coerce")
    data = data.dropna().reset_index(drop=True)
    return data[:-1]

def reorder_class(data: pd.DataFrame, num_classes: int):
    class_num = 0
    # build mapping of classes
    idx = 0
    class_names = {}
    while class_num < num_classes:
        # get class
        result = data['class'][idx]
        # if class unseen, record it
        if result not in class_names:
            #record and increment class
            class_names[result] = class_num
            class_num += 1
        idx += 1
    data['class'] = data['class'].apply(lambda x: class_names[x])
    return data

def run_knn(data: pd.DataFrame, clusters: int):
    mat = data.values
    names = mat[:,0]
    mat = np.delete(mat, 0, 1)  # delete name column of mat
    # Using sklearn
    km = KMeans(n_clusters=clusters)
    mat = scale(mat)
    km.fit(mat)
    # Get cluster assignment labels
    labels = km.labels_
    # Format results as a DataFrame
    results = pd.DataFrame([names,labels], index=["name", "class"]).T
    results = reorder_class(results, clusters)
    return results


def extract(position : str, player : str, year : str, scoring : str, limit_: int = 100):
    """
    Reads html from fantasypros stats per position, year, and scoring specifications
    Inputs: 
        Position: a string consisting of "wr", "rb", "te", "qb"
        Year: string consisting of a year (yyyy)
        Scoring: string consisting of "PPR", "HALF"
        Limit: limits the number of rows/players
    """
    url = base + position + ".php?year=" + year + "&scoring=" + scoring
    # access url
    result = requests.get(url)
    # read html
    doc = BeautifulSoup(result.text, "html.parser")
    # tr is table row. Find all table rows with class starting with "mpb-player-..."
    # each result is a player and their stats
    rows = doc.find_all("tr", attrs={'class': re.compile('mpb\-player\-*')}, limit = limit_)
    return rows

def make_dists(names: str, year: str, pos: str):
    raw_rec = []
    raw_rb = []
    count = 1
    for name in names:
        print(name)
        data = extract_year(name=name, year=year, scoring="PPR")
        avg_rec = np.mean(data["Receiving_rec"])
        var_rec = np.std(data["Receiving_rec"])
        avg_tgt = np.mean(data["Receiving_Tgt"])
        var_tgt = np.std(data["Receiving_Tgt"])
        avg_yds = np.mean(data["Receiving_yds"])
        var_yds = np.std(data["Receiving_yds"])
        avg_ypr = np.mean(data["Receiving_Y/R"])
        var_ypr = np.std(data["Receiving_Y/R"])
        avg_lg = np.mean(data["Receiving_lg"])
        var_lg = np.std(data["Receiving_lg"])
        avg_TD = np.mean(data["Receiving_TD"])
        var_TD = np.std(data["Receiving_TD"])
        avg_rush = np.mean(data["Rushing_att"])
        var_rush = np.std(data["Rushing_att"])
        avg_ryds = np.mean(data["Rushing_yds"])
        var_ryds = np.std(data["Rushing_yds"])
        avg_rTD = np.mean(data["Rushing_TD"])
        var_rTD = np.std(data["Rushing_TD"])
        age = extract_age(name, year)
        games_played = data.shape[0]
        raw_rb.append([name, age, games_played, avg_rec, var_rec, avg_tgt, var_tgt, avg_yds, var_yds, avg_ypr, var_ypr, 
                    avg_lg, var_lg, avg_TD, var_TD, avg_rush, var_rush, avg_ryds, var_ryds,
                    avg_rTD, var_rTD])
        raw_rec.append([name, age, games_played, avg_rec, var_rec, avg_tgt, var_tgt, avg_yds, var_yds, avg_ypr, var_ypr, 
                    avg_lg, var_lg, avg_TD, var_TD])
    if pos in ['rb', 'wr']:
        data = pd.DataFrame(raw_rb, columns=["name", "age", "games_played", "avg_rec", "var_rec", "avg_tgt", "var_tgt", "avg_yds", "var_yds", 
                                        "avg_ypr", "var_ypr", "avg_lg", "var_lg", "avg_TD", "var_TD", 
                                        "avg_rush", "var_rush", "avg_ryds", "var_ryds", "avg_rTD", "var_rTD"])
        # combine aggregates with team target %
        data = pd.merge(data, read_targets(year, pos), how='inner', on=['name'])
        clusters = run_knn(data, 5)
        # combine data with knn clusters
        data = pd.merge(data, clusters, how='inner', on=['name'])
        return data
    else:
        data = pd.DataFrame(raw_rec, columns=["name", "age", "games_played", "avg_rec", "var_rec", "avg_tgt", "var_tgt", "avg_yds", "var_yds", 
                                        "avg_ypr", "var_ypr", "avg_lg", "var_lg", "avg_TD", "var_TD"])
        # combine aggregates with team target %
        data = pd.merge(data, read_targets(year, pos), how='inner', on=['name'])
        clusters = run_knn(data, 5)
        # combine data with knn clusters
        data = pd.merge(data, clusters, how='inner', on=['name'])
        return data       

def read_targets(year: str, pos: str):
    file = year + "_targets.txt"
    data = pd.read_csv(file)
    data = data[data['POS'] == pos.upper()]
    data['name'] = data['NAME'].apply(lambda x: clean_name(x))
    return data[['name', 'TM TGT %']]

def model_2023(pos: str, scoring: str, data2020: pd.DataFrame, data2021: pd.DataFrame, data2022: pd.DataFrame):
    prod20 = pd.merge(data2020, extract_players("2021", pos, scoring)[['name', 'MISC_FPTS/G']], how='inner', on=['name'])
    prod21 = pd.merge(data2021, extract_players("2022", pos, scoring)[['name', 'MISC_FPTS/G']], how='inner', on=['name'])
    # combining historical years
    main = prod21.append([prod20], ignore_index=True)
    # creating matrix
    X_train = main.values
    # seperating response
    y_train = X_train[:,-1]
    X_train = np.delete(X_train, 0, 1)  # delete name column of mat
    X_train = np.delete(X_train, -1, 1)  # delete response column of mat
    X_train = scale(X_train) # scale training data
    # Using sklearn
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['rbf']}
    svr = SVR()
    # initializing grid search
    grid = GridSearchCV(svr, param_grid, scoring='neg_root_mean_squared_error', refit = True, verbose = 3)
    grid.fit(X_train, y_train)

    # creating test matrix
    X_test = data2022.values
    # recording player names
    names2022 = X_test[:,0]
    X_test = np.delete(X_test, 0, 1)  # delete name column of mat
    X_test = scale(X_test)
    fpts_pred = grid.predict(X_test)
    results = pd.DataFrame([names2022,fpts_pred], index=["name", "proj fpts"]).T
    classes = data2022[['name', 'class']]
    results = pd.merge(results, classes, how='inner', on=['name'])
    results = results.sort_values('proj fpts', ascending=False)
    results = results.reset_index(drop=True)
    return results


    

