# imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
    data.columns.values[1] = "name"
    # setting rank = idx
    data = data.set_index(['Rank'])
    data['name'] = data['name'].apply(lambda x: clean_name(x))
    return data

def read_ngs_rec(year: str, pos: str):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chromedriver_path = '/home/user/chromedriver'
    d = webdriver.Chrome(chromedriver_path,chrome_options=chrome_options)
    d.get('https://nextgenstats.nfl.com/stats/receiving/' + year + '/REG/all#yards')
    time.sleep(3)
    html = d.page_source
    dfs = pd.read_html(html)
    ngs_data = dfs[1]
    ngs_data.columns = dfs[0].columns[:-1]
    ngs_data = ngs_data[ngs_data['POS'] == pos.upper()]
    ngs_data = ngs_data
    ngs_data['name'] = ngs_data['PLAYER NAME'].apply(lambda x: clean_name(x))
    ngs_data = ngs_data.drop(ngs_data.columns[[0,1,2,7,8,9,10,11]],axis = 1)
    return ngs_data

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
    # QB edits
    text = text.replace("patrick-mahomes-ii", "patrick-mahomes")
    text = text.replace("gardner-minshew-ii", "gardner-minshew")
    text = text.replace('pj-walker', 'phillip-walker')
    if text in ['dj-moore', 'mike-williams', 'michael-thomas']:
        text += '-wr'
    if text in ['najee-harris', 'michael-carter', 'damien-harris', 'justin-jackson', 'elijah-mitchell']:
        text += '-rb'
    if text in ['josh-allen']:
        text += '-qb'
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
    # print(data[data.isna().any(axis=1)]['name'])
    mat = data.values
    names = mat[:,0]
    mat = np.delete(mat, 0, 1)  # delete name column of mat
    # Using sklearn
    km = KMeans(n_clusters=clusters)
    mat = scale(mat)
    # PCA for 95% variance
    pca_mod = PCA(n_components=0.95)
    pca_mod.fit(mat)
    reduced_mat = pca_mod.transform(mat)
    km.fit(reduced_mat)
    # Get cluster assignment labels
    labels = km.labels_
    # Format results as a DataFrame
    results = pd.DataFrame([names,labels], index=["name", "class"]).T
    results = reorder_class(results, clusters)
    return results

def make_dists(names: str, year: str, pos: str):
    if pos == 'qb':
        return make_dists_qb(names, year, pos)
    raw = []
    for name in names:
        print(name)
        data = extract_year(name=name, year=year, scoring="PPR")
        # recieving
        avg_rec = np.mean(data["Receiving_rec"])
        std_rec = np.std(data["Receiving_rec"])
        avg_tgt = np.mean(data["Receiving_Tgt"])
        std_tgt = np.std(data["Receiving_Tgt"])
        avg_yds = np.mean(data["Receiving_yds"])
        std_yds = np.std(data["Receiving_yds"])
        avg_ypr = np.mean(data["Receiving_Y/R"])
        std_ypr = np.std(data["Receiving_Y/R"])
        avg_lg = np.mean(data["Receiving_lg"])
        std_lg = np.std(data["Receiving_lg"])
        avg_TD = np.mean(data["Receiving_TD"])
        std_TD = np.std(data["Receiving_TD"])
        # rushing
        avg_rush = np.mean(data["Rushing_att"])
        std_rush = np.std(data["Rushing_att"])
        avg_ryds = np.mean(data["Rushing_yds"])
        std_ryds = np.std(data["Rushing_yds"])
        avg_rypa = np.mean(data["Rushing_Y/A"])
        std_rypa = np.std(data["Rushing_Y/A"])
        avg_rlg = np.mean(data["Rushing_lg"])
        std_rlg = np.std(data["Rushing_lg"])
        avg_rTD = np.mean(data["Rushing_TD"])
        std_rTD = np.std(data["Rushing_TD"])
        age = extract_age(name, year)
        games_played = data.shape[0]
        raw.append([name, age, games_played, avg_rec, std_rec, avg_tgt, std_tgt, avg_yds, std_yds, avg_ypr, std_ypr, 
                    avg_lg, std_lg, avg_TD, std_TD, avg_rush, std_rush, avg_ryds, std_ryds, avg_rypa, std_rypa, avg_rlg,
                    std_rlg, avg_rTD, std_rTD])
        
    data = pd.DataFrame(raw, columns=["name", "age", "games_played", "avg_rec", "std_rec", "avg_tgt", "std_tgt", "avg_yds", "std_yds", 
                                    "avg_ypr", "std_ypr", "avg_lg", "std_lg", "avg_TD", "std_TD", 
                                    "avg_rush", "std_rush", "avg_ryds", "std_ryds","avg_rypa", "std_rypa", "avg_rlg", "std_rlg", "avg_rTD", "std_rTD"])
    
    # combine aggregates with team target %
    data = pd.merge(data, read_targets(year, pos), how='inner', on=['name'])

    # combine aggregates with nfl next gen recieving stats
    if pos in ['wr']:
        print("merging with Next Gen Stats")
        data = pd.merge(data, read_ngs_rec(year, pos), how='inner', on=['name'])

    clusters = run_knn(data, 5)
    # combine data with knn clusters
    data = pd.merge(data, clusters, how='inner', on=['name'])
    print("Done for " + year)
    return data

def make_dists_qb(names: str, year: str, pos: str):
    raw_qb = []
    for name in names:
        print(name)
        data = extract_year(name=name, year=year, scoring="PPR")
        # passing
        avg_qbr = np.mean(data["Passing_QB Rat"])
        std_qbr = np.std(data["Passing_QB Rat"])
        avg_cmp = np.mean(data["Passing_cmp"])
        std_cmp = np.std(data["Passing_cmp"])
        avg_att = np.mean(data["Passing_att"])
        std_att = np.std(data["Passing_att"])
        avg_pct = np.mean(data["Passing_pct"])
        std_pct = np.std(data["Passing_pct"])
        avg_pyds = np.mean(data["Passing_yds"])
        std_pyds = np.std(data["Passing_yds"])
        avg_ypa = np.mean(data["Passing_Y/A"])
        std_ypa = np.std(data["Passing_Y/A"])
        avg_TD = np.mean(data["Passing_TD"])
        std_TD = np.std(data["Passing_TD"])
        avg_INT = np.mean(data["Passing_INT"])
        std_INT = np.std(data["Passing_INT"])
        avg_SACK = np.mean(data["Passing_Sacks"])
        std_SACK = np.std(data["Passing_Sacks"])
        # rushing
        avg_rush = np.mean(data["Rushing_att"])
        std_rush = np.std(data["Rushing_att"])
        avg_ryds = np.mean(data["Rushing_yds"])
        std_ryds = np.std(data["Rushing_yds"])
        avg_rypa = np.mean(data["Rushing_Y/A"])
        std_rypa = np.std(data["Rushing_Y/A"])
        avg_rlg = np.mean(data["Rushing_lg"])
        std_rlg = np.std(data["Rushing_lg"])
        avg_rTD = np.mean(data["Rushing_TD"])
        std_rTD = np.std(data["Rushing_TD"])
        age = extract_age(name, year)
        games_played = data.shape[0]
        raw_qb.append([name, age, games_played, avg_qbr, std_qbr, avg_cmp, std_cmp, avg_att, std_att, avg_pct, std_pct, 
                        avg_pyds, std_pyds, avg_ypa, std_ypa, avg_TD, std_TD, avg_INT, std_INT, avg_SACK, std_SACK, avg_rush, std_rush, avg_ryds, std_ryds, avg_rypa, std_rypa, avg_rlg,
                        std_rlg, avg_rTD, std_rTD])
    data = pd.DataFrame(raw_qb, columns=['name', 'age', 'games_played', 'avg_qbr', 'std_qbr', 'avg_cmp', 'std_cmp', 'avg_att', 'std_att', 'avg_pct', 'std_pct', 
                    'avg_pyds', 'std_pyds', 'avg_ypa', 'std_ypa', 'avg_TD', 'std_TD', 'avg_INT', 'std_INT', 'avg_SACK', 'std_SACK', 'avg_rush', 'std_rush', 'avg_ryds', 'std_ryds', 'avg_rypa', 'std_rypa', 'avg_rlg',
                    'std_rlg', 'avg_rTD', 'std_rTD'])
    # marcus mariota in 2020 gives an error
    if year == '2020':
        data = data[data.name != 'marcus-mariota']
    
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

def model_2023(pos: str, scoring: str, data2020: pd.DataFrame, data2021: pd.DataFrame, data2022: pd.DataFrame, pca: bool = True):
    prod20 = pd.merge(data2020, extract_players("2021", pos, scoring)[['name', 'MISC_FPTS/G']], how='inner', on=['name'])
    prod21 = pd.merge(data2021, extract_players("2022", pos, scoring)[['name', 'MISC_FPTS/G']], how='inner', on=['name'])
    # combining historical years
    main = prod21.append([prod20], ignore_index=True)
    features = main.columns[1:-1]
    # creating train matrix
    X_train = main.values
    # seperating response
    y_train = X_train[:,-1]
    X_train = np.delete(X_train, 0, 1)  # delete name column of mat
    X_train = np.delete(X_train, -1, 1)  # delete response column of mat
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train) # scale training data

    # creating test matrix
    X_test = data2022.values
    # recording player names
    names2022 = X_test[:,0]
    X_test = np.delete(X_test, 0, 1)  # delete name column of mat
    X_test = scaler.transform(X_test)

    # Using sklearn
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['rbf']}
    svr = SVR()
    # initializing grid search model
    grid = GridSearchCV(svr, param_grid, scoring='neg_root_mean_squared_error', refit = True, verbose = 3)

    if pca:
        print("PCA Selected")
        pca_mod = PCA(n_components=0.95)
        print("Train dimensions (pre PCA): ", X_train.shape)
        print("Test dimensions (pre PCA): ", X_test.shape)
        pca_mod.fit(X_train)
        X_train = pca_mod.transform(X_train)
        X_test = pca_mod.transform(X_test)
        print("Train dimensions (post PCA): ", X_train.shape)
        print("Test dimensions (post PCA): ", X_test.shape)

    # fit model to training data
    grid.fit(X_train, y_train)

    if not pca:
        # perform permutation importance
        res = permutation_importance(grid, X_train, y_train, scoring='neg_mean_squared_error')
        # get importance
        importance_indices = np.argsort(res["importances_mean"])[::-1]
        sorted_important_features = features[importance_indices]
        print(f"Feature importances: {sorted_important_features}")

    fpts_pred = grid.predict(X_test)
    results = pd.DataFrame([names2022,fpts_pred], index=["name", "proj fpts"]).T
    classes = data2022[['name', 'class']]
    results = pd.merge(results, classes, how='inner', on=['name'])
    results = results.sort_values('proj fpts', ascending=False)
    return results


def run_svr_2023(pos: str, scoring: str, pca: bool):
    if pos in ['wr', 'rb']:
        num = 100
    else:
        num = 50
    # 2022
    df2022 = extract_players('2022', pos, scoring)
    names2022 = list(df2022["name"].head(num))
    dists2022 = make_dists(names2022, "2022", pos)
    # 2021
    df2021 = extract_players("2021", pos, scoring)
    names2021 = list(df2021["name"].head(num))
    dists2021 = make_dists(names2021, "2021", pos)
    # 2020
    df2020 = extract_players("2020", pos, scoring)
    names2020 = list(df2020["name"].head(num))
    dists2020 = make_dists(names2020, "2020", pos)
    # model
    res = model_2023(pos, scoring, dists2020, dists2021, dists2022, pca=pca)
    result = {'2020_df' : dists2020,
              '2021_df' : dists2021,
              '2022_df' : dists2022,
              'projections' : res
    }
    return result

