# imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
import numpy as np
import xgboost as xg
from bs4 import BeautifulSoup
import requests
import re
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNetCV


base = "https://www.fantasypros.com/nfl/stats/"

# https://www.fantasypros.com/nfl/games/justin-jefferson.php?season=2021&scoring=PPR

def extract_players(year: str, pos: str, scoring: str) -> pd.DataFrame:
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

def read_ngs_rec(year: str, pos: str) -> pd.DataFrame:
    """
    Reads html from NFL Next Gen Recieving Stats per position, year, and scoring specifications into a pandas df.
    Also transforms player names to be url-friendly
    Inputs: 
        Position: a string consisting of "wr", "rb", "te", "qb"
        Year: string consisting of a year (yyyy)
    """
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

def clean_name(text: str) -> str:
    """
    Transforms a player name into a format that can be callable in html requests on fantasypros
    Inputs: 
        text: a player name 
    """
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
    # edge cases in top 100 (2019-2022 wrs), no rhyme/reason
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
    text = text.replace('phillip-dorsett-ii', 'phillip-dorsett')
    text = text.replace('ted-ginn', 'ted-ginn-jr')
    text = text.replace('bisi-johnson', 'olabisi-johnson')
    text = text.replace('bennie-fowler-iii', 'bennie-fowler')
    # RB edits
    text = text.replace('kenneth-walker-iii', 'kenneth-walker')
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
    text = text.replace('chris-ivory', 'christopher-ivory')
    # TE edits
    text = text.replace("irv-smith", "irv-smith-jr")
    text = text.replace('zach-miller', 'zach-miller-chi')
    # QB edits
    text = text.replace("patrick-mahomes-ii", "patrick-mahomes")
    text = text.replace("gardner-minshew-ii", "gardner-minshew")
    text = text.replace('pj-walker', 'phillip-walker')
    text = text.replace('tj-yates', 'taylor-yates')

    # WRS
    if text in ['dj-moore', 'mike-williams', 'michael-thomas']:
        text += '-wr'
    # RBS
    if text in ['najee-harris', 'michael-carter', 'damien-harris', 'justin-jackson', 'elijah-mitchell', 'kenneth-walker']:
        text += '-rb'
    # QBS
    if text in ['josh-allen']:
        text += '-qb'
    # TES
    if text in ['josh-hill']:
        text += '-te'
    return text

def extract_age(name: str, year_: str) -> int:
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

def extract_years_played(name: str, year_: str, pos: str) -> int:
    # get year difference between current year and year specified
    time_diff = dt.date.today().year - eval(year_)
    url_ = 'https://www.fantasypros.com/nfl/stats/' + name + '.php'
    years_played = pd.read_html(url_)[0].shape[0] - (time_diff + 1)
    return years_played

def extract_year(name: str, year: str, scoring: str) -> pd.DataFrame:
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

def reorder_class(data: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    """
    Reorders k-means classes to represent player pedigree
    Inputs: 
        data: a pd.Dataframe containing name and associated classes
        num_classes: int of the number of classes data is split into
    """
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

def run_k_means(data: pd.DataFrame, clusters: int) -> pd.DataFrame:
    """
    Applies k-means algo to classify player types within a position
    Inputs: 
        data: a pd.Dataframe containing name and associated classes
        num_classes: int of the number of classes data is split into
    """
    if 'kenneth-walker-rb' in data['name']:
        print('check!')
        print(data[data['name'] == 'kenneth-walker-rb'])
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

def make_dists(names: str, year: str, pos: str, scoring: str = 'PPR') -> pd.DataFrame:
    """
    Reads html from fantasypros stats for each name and makes scoring distributions based on that year's game log and scoring type.
    Inputs: 
        Position: a string consisting of "wr", "rb", "te", "qb"
        Year: string consisting of a year (yyyy)
    """
    if pos == 'qb':
        return make_dists_qb(names, year, pos, scoring=scoring)
    raw = []
    for name in names:
        print(name)
        data = extract_year(name=name, year=year, scoring=scoring)
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
        years_played = extract_years_played(name, year, pos)
        games_played = data.shape[0]
        raw.append([name, age, years_played, games_played, avg_rec, std_rec, avg_tgt, std_tgt, avg_yds, std_yds, avg_ypr, std_ypr, 
                    avg_lg, std_lg, avg_TD, std_TD, avg_rush, std_rush, avg_ryds, std_ryds, avg_rypa, std_rypa, avg_rlg,
                    std_rlg, avg_rTD, std_rTD])
        
    data = pd.DataFrame(raw, columns=["name", "age", "years_played", "games_played", "avg_rec", "std_rec", "avg_tgt", "std_tgt", "avg_yds", "std_yds", 
                                    "avg_ypr", "std_ypr", "avg_lg", "std_lg", "avg_TD", "std_TD", 
                                    "avg_rush", "std_rush", "avg_ryds", "std_ryds","avg_rypa", "std_rypa", "avg_rlg", "std_rlg", "avg_rTD", "std_rTD"])
    
    # combine aggregates with team target %
    data = pd.merge(data, read_targets(year, pos), how='inner', on=['name'])
    print('Merging ' + year + ' data with ' + str(eval(year)+1) + ' ADPs')
    data = pd.merge(data, read_adps(str(eval(year)+1), scoring), how='inner', on=['name'])

    # combine aggregates with nfl next gen recieving stats
    if pos in ['wr']:
        print("merging with Next Gen Stats")
        data = pd.merge(data, read_ngs_rec(year, pos), how='inner', on=['name'])

    # hunter renfrow and tyrell williams in 2019 gives an error
    if year == '2019' and pos == 'wr':
        data = data[data.name != 'hunter-renfrow']
        data = data[data.name != 'tyrell-williams']
    
    if year == '2018' and pos == 'wr':
        data = data[data.name != 'jordy-nelson']
        data = data[data.name != 'seth-roberts']
        data = data[data.name != 'ryan-grant']

    # michael crabtree, amari cooper, seth roberts, and ryan grant in 2018 gives an error
    if year == '2017' and pos == 'wr':
        data = data[data.name != 'michael-crabtree']
        data = data[data.name != 'amari-cooper']
        data = data[data.name != 'ryan-grant']
        data = data[data.name != 'seth-roberts']
    
    # josh-jacobs, deandre-washington, and jalen-richard in 2019 gives an error
    if year == '2019' and pos == 'rb':
        data = data[data.name != 'josh-jacobs']
        data = data[data.name != 'deandre-washington']
        data = data[data.name != 'jalen-richard']
    
    if year == '2018' and pos == 'rb':
        data = data[data.name != 'jalen-richard']
        data = data[data.name != 'doug-martin']
        data = data[data.name != 'marshawn-lynch']
    
    if year == '2017' and pos == 'rb':
        data = data[data.name != 'jalen-richard']
        data = data[data.name != 'deandre-washington']
        data = data[data.name != 'marshawn-lynch']

    if year == '2019' and pos == 'te':
        data = data[data.name != 'darren-waller']
        data = data[data.name != 'foster-moreau']

    if year in ['2018', '2017'] and pos == 'te':
        data = data[data.name != 'jared-cook']

    if data.shape[0] > 1:
        clusters = run_k_means(data, 5)
        # combine data with knn clusters
        data = pd.merge(data, clusters, how='inner', on=['name'])
    print("Done for " + year)
    return data

def recover_kw3():
    return make_dists(['kenneth-walker-rb'], '2022', 'rb', scoring = 'PPR')

def make_dists_qb(names: str, year: str, pos: str, scoring: str = 'PPR') -> pd.DataFrame:
    """
    Reads html from fantasypros stats for each name and makes scoring distributions based on that year's game log and scoring type.
    Inputs: 
        Position: a string consisting of "wr", "rb", "te", "qb"
        Year: string consisting of a year (yyyy)
    """
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
        years_played = extract_years_played(name, year, pos)
        games_played = data.shape[0]
        raw_qb.append([name, age, years_played, games_played, avg_qbr, std_qbr, avg_cmp, std_cmp, avg_att, std_att, avg_pct, std_pct, 
                        avg_pyds, std_pyds, avg_ypa, std_ypa, avg_TD, std_TD, avg_INT, std_INT, avg_SACK, std_SACK, avg_rush, std_rush, avg_ryds, std_ryds, avg_rypa, std_rypa, avg_rlg,
                        std_rlg, avg_rTD, std_rTD])

    data = pd.DataFrame(raw_qb, columns=['name', 'age', 'years_played', 'games_played', 'avg_qbr', 'std_qbr', 'avg_cmp', 'std_cmp', 'avg_att', 'std_att', 'avg_pct', 'std_pct', 
                    'avg_pyds', 'std_pyds', 'avg_ypa', 'std_ypa', 'avg_TD', 'std_TD', 'avg_INT', 'std_INT', 'avg_SACK', 'std_SACK', 'avg_rush', 'std_rush', 'avg_ryds', 'std_ryds', 'avg_rypa', 'std_rypa', 'avg_rlg',
                    'std_rlg', 'avg_rTD', 'std_rTD'])
    # marcus mariota in 2020 gives an error
    if year == '2020':
        data = data[data.name != 'marcus-mariota']

    if year == '2019':
        data = data[data.name != 'derek-carr']
    
    if year == '2018':
        data = data[data.name != 'derek-carr']
        data = data[data.name != 'matt-barkley']

    if year == '2017':
        data = data[data.name != 'derek-carr']
        data = data[data.name != 'ej-manuel']
    print('Merging ' + year + ' data with ' + str(eval(year)+1) + ' ADPs')
    data = pd.merge(data, read_adps(str(eval(year)+1), scoring), how='inner', on=['name'])
    clusters = run_k_means(data, 5)
    # combine data with knn clusters
    data = pd.merge(data, clusters, how='inner', on=['name'])
    print("Done for " + year)
    return data

def read_targets(year: str, pos: str) -> pd.DataFrame:
    """
    Reads CSV data for each player's target share in a year
    Inputs: 
        Position: a string consisting of "wr", "rb", "te", "qb"
        Year: string consisting of a year (yyyy)
    """
    file = 'target_data/' + year + "_targets.csv"
    data = pd.read_csv(file)
    data = data[data['POS'] == pos.upper()]
    data['name'] = data['NAME'].apply(lambda x: clean_name(x))
    return data[['name', 'TM TGT %']]

def train_test(pos: str, scoring: str = 'PPR', num_years: int = 5, year_for: int = 2022, local: bool = True) -> pd.DataFrame:
    """
    Gives train, test data given specifications
    Inputs: 
        Position: a string consisting of "wr", "rb", "te", "qb"
        Scoring: fantasy scoring type
        num_years: number of years to look back at (earliest year is 2017)
        year_for: year of data you are using as testing
        local: indicates whether data should be grabbed locally (keep true unless no data)
        bootstrap: number of times to bootstrap results
        csv_: option to save projections into a csv
    """
    # grab 2022, 2021, 2020, 2019, 2018, 2017 data per position
    # grab 2022, 2021, 2020, 2019, 2018 projections per position

    if local:
        # locally grab data and format as get_data output
        datas = {}
        for back in range(num_years+1):
            year = year_for-back
            # read year's worth of player data from local data folder
            data = pd.read_csv('data/' + str(year) + '_' + pos + '_' + scoring  + '_data.csv')
            datas[str(year)] = data
        # locally grab results and format as get_data output
        res_s = {}
        for back in range(num_years):
            year = year_for-back
            res_s[str(year)] = pd.read_csv('results/' + str(year) + '_' + pos + '_' + scoring + '_results.csv')
    else:
        datas = get_data(pos, num_years=num_years+1, year_for=year_for, save_csv=True, scoring=scoring)
        res_s = get_results(pos, num_years=num_years, year_for=year_for, save_csv=True, scoring=scoring)
    
    # getting test data (most recent yearly records)
    most_recent = datas[str(year_for)]

    # historic data will have identical columns, joined upon 'MISC_FPTS/G'
    df_cols = list(most_recent.columns)
    df_cols.append('MISC_FPTS/G')
    # start historic dataframe with previously specified column names
    historic = pd.DataFrame(columns=df_cols)

    # iteratively build historic data for number of years to look back upon
    for back in range(num_years):
        year = year_for-back
        # merge 2021 with 2022 results, 2020 with 2021 results, ...
        combined = pd.merge(datas[str(year-1)], res_s[str(year)], how='inner', on=['name'])
        # append year
        historic = historic.append([combined], ignore_index=True)
    return historic, most_recent

def get_data(pos: str, num_years: int, year_for: int = 2022, save_csv: bool = False, scoring: str = 'PPR') -> dict:
    """
    Fetches dictionary of fantasypros data for multiple years
    Inputs: 
        pos: a string consisting of "wr", "rb", "te", "qb"
        scoring: fantasy scoring type
        num_years: number of years total, including year_for
        year_for: int of target year to make predictions on
        save_csv: indicate if you want to save csv
    """
    # define num players to grab
    if pos in ['wr', 'rb']:
        num = 100
    else:
        num = 50
    
    # get dict of yearly data
    years = {}
    for back in range(num_years):
        year = year_for-back
        df = extract_players(str(year), pos, scoring)
        names = list(df["name"].head(num))
        data = make_dists(names, str(year), pos, scoring)
        if save_csv:
            data.to_csv(f'data/{(year)}_' + pos + '_' + scoring + '_data.csv', index=False)
        years[str(year)] = data
    return years
    
def get_results(pos: str, num_years: int, year_for: int = 2022, save_csv: bool = False, scoring: str = 'PPR') -> pd.DataFrame:
    years = {}
    for back in range(num_years):
        year = year_for-back
        df = extract_players(str(year), pos, scoring)[['name', 'MISC_FPTS/G']]
        data = df[['name', 'MISC_FPTS/G']]
        if save_csv:
            data.to_csv(f'results/{(year)}_' + pos + '_' + scoring + '_results.csv', index=False)
        years[str(year)] = data
    return years

def grab_years_played(pos: str, years_played: int, num_years: int = 5, year_for: int = 2022, scoring: str = 'PPR'):
    # get most recent data
    most_recent = pd.read_csv('data/' + str(year_for) + '_' + pos + '_' + scoring + '_data.csv')
    df_cols = list(most_recent.columns)
    df_cols.append('MISC_FPTS/G')
    # start historic dataframe with previously specified column names
    historic = pd.DataFrame(columns=df_cols)
    # iteratively build historic data for number of years to look back upon
    for back in range(num_years):
        year = year_for-back
        # merge 2021 with 2022 results, 2020 with 2021 results, ...
        year_data = pd.read_csv('data/' + str(year-1) + '_' + pos + '_' + scoring  + '_data.csv')
        result = pd.read_csv('results/' + str(year) + '_' + pos + '_' + scoring + '_results.csv')
        combined = pd.merge(year_data, result, how='inner', on=['name'])
        # append year
        historic = historic.append([combined], ignore_index=True)
    filtered_historic = historic[historic['years_played'] == years_played]
    filtered_historic.reset_index(inplace = True,drop = True)
    filtered_most_recent = most_recent[most_recent['years_played'] == years_played]
    return filtered_historic, filtered_most_recent

def revised_run(train: pd.DataFrame, test: pd.DataFrame, pos: str, year_for: str = '2022', scoring: str = 'PPR', model_: str = 'xgb', bootstrap: int = 5, save_csv = True, pca: bool = True):
    # making copies to avoid mutations
    X_train_df = train.copy()
    X_train_cols = list(X_train_df.columns)[1:-1]
    #print('train dim: ', X_train_df.shape)
    X_test_df = test.copy()
    #print('test dim: ', X_test_df.shape)

    X_train = X_train_df.values
    # seperating response
    y_train = X_train[:,-1]
    #print('train mat dim: ', X_train.shape)
    #print('deleting names and response')
    X_train = np.delete(X_train, 0, 1)  # delete name column of mat
    X_train = np.delete(X_train, -1, 1)  # delete response column of mat
    #print('train mat dim: ', X_train.shape)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train) # scale training data

    # creating test matrix
    X_test = X_test_df.values
    #print('test mat dim: ', X_test.shape)
    # recording player names
    names2022 = X_test[:,0]
    X_test = np.delete(X_test, 0, 1)  # delete name column of mat
    #print('test mat dim: ', X_test.shape)
    # scale test according to train data
    X_test = scaler.transform(X_test)

    if pca:
        # apply PCA to train/test
        pca_mod = PCA(n_components=0.95)
        pca_mod.fit(X_train)
        X_train = pca_mod.transform(X_train)
        X_test = pca_mod.transform(X_test)

    # initialize k-folds
    kfold = KFold()

    # operate on folds
    fold = 0
    errors = []
    for train_idx, val_idx in kfold.split(X_train, y_train):
        X_tr = X_train[train_idx, :]
        y_tr = y_train[train_idx]
        
        X_val = X_train[val_idx, :]
        y_val = y_train[val_idx]

        if model_ == 'xgb':
            # initialize XGB regressor
            mod = xg.XGBRegressor(objective ='reg:squarederror', booster = 'gblinear',
                                    n_estimators = 10, eval_metric = 'rmse')
        if model_ == 'svr':
            param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['rbf']}
            svr = SVR()
            # initializing grid search model
            mod = GridSearchCV(svr, param_grid, scoring='neg_root_mean_squared_error', refit = True, verbose = 0)

        if model_ == 'rf':
            mod = RandomForestRegressor()

        if model_ == 'enet':
            mod = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], max_iter=100000)

        mod.fit(X_tr, y_tr)
        pred = mod.predict(X_val)
        rmse = mean_squared_error(y_val, pred)
        print(f"======= Fold {fold} ========")
        print(
            f"Our accuracy on the validation set is {np.sqrt(rmse)}"
        )
        fold += 1
        errors.append(np.sqrt(rmse))
    print(f"Overall accuracy: {np.mean(errors)}")

    # classes = X_test_df[['name', 'class']]
    results = pd.DataFrame(columns=['name', 'proj fpts'])

    for i in range(bootstrap):
        # Fitting the model
        mod.fit(X_train, y_train)
        
        # Predict the model
        fpts_pred = mod.predict(X_test)
        result = pd.DataFrame([names2022,fpts_pred], index=["name", "proj fpts"]).T
        results = results.append([result], ignore_index=True)
        print(f"iteration {i+1}: Dimensions = {results.shape}")
    
    if not pca:
        perm_importance = permutation_importance(mod, X_train, y_train, scoring='neg_root_mean_squared_error', n_repeats=10)
        sorted_idx = perm_importance.importances_mean.argsort()[-10:]
        plt.barh(np.array(X_train_cols)[sorted_idx], perm_importance.importances_mean[sorted_idx])
        plt.xlabel("Permutation Importance")

    # group bootstrapped results by player name
    results_grouped = results.groupby('name', as_index=False)
    # record mean
    mean_result = results_grouped.mean()
    mean_result = mean_result.sort_values('proj fpts', ascending=False)
    mean_result = mean_result.reset_index(drop=True)
    # join results and classes
    # mean_result = pd.merge(mean_result, classes, how='inner', on=['name'])
    # maintain 2022 ranks and join
    ranks_2022 = X_test_df.copy()
    ranks_2022['last rank'] = ranks_2022.index + 1
    mean_result['rank'] = mean_result.index + 1
    print(mean_result.columns)
    mean_result = pd.merge(mean_result, ranks_2022[['name', 'last rank']], how='inner', on=['name'])
    if save_csv:
        mean_result.to_csv('projections/' + pos + '_' + scoring + '_' + year_for + '_' + model_ + '_projections.csv', index = False)
    return mean_result

def read_adps(year: str, scoring: str) -> pd.DataFrame:
    if scoring == 'PPR': 
        data = pd.read_html(f'https://www.fantasypros.com/nfl/adp/ppr-overall.php?year={year}')[0]
    else:
        data = pd.read_html(f'https://www.fantasypros.com/nfl/adp/half-point-ppr-overall.php?year={year}')[0]

    data['name'] = data['Player Team (Bye)'].apply(lambda x: process_adp_name(str(x)))

    if year == '2023':
        data = data[['name', 'AVG']]
    else:
        # ESPN for PPR
        if scoring == 'PPR':
            data = data[['name', 'ESPN']]
        # Yahoo for HALF
        else:
            data = data[['name', 'Yahoo']]
    data.columns.values[1] = "ADP"
    data = data.dropna()
    return data

def process_adp_name(text: str) -> str:
    if len(text.split(' ')) == 2:
        return clean_name(text)
    else:
        # section last word of name
        last_word = text.split(' ')[-1]
        # if the last word doesn't start with "(" it's a injury designation {"O", "Q", "D", "IR"}. Remove it
        if last_word[0] != "(":
            text = text.rsplit(' ', 1)[0]
        # now take away bye week designation because it doesn't match previous convention
        text = text.rsplit(' ', 1)[0]
        return clean_name(text)

def summarize_proj(pos: str, year: str = '2022', scoring: str = 'PPR', save_csv: bool = True):
    rf = pd.read_csv(f"projections/{pos}_{scoring}_{year}_rf_projections.csv")[['name', 'proj fpts']]
    rf.columns.values[1] = "rf proj"

    names_ranks = pd.read_csv(f"projections/{pos}_{scoring}_{year}_rf_projections.csv")[['name', 'last rank']]

    enet = pd.read_csv(f"projections/{pos}_{scoring}_{year}_enet_projections.csv")[['name', 'proj fpts']]
    enet.columns.values[1] = "enet proj"

    xgb = pd.read_csv(f"projections/{pos}_{scoring}_{year}_xgb_projections.csv")[['name', 'proj fpts']]
    xgb.columns.values[1] = "xgb proj"

    svr = pd.read_csv(f"projections/{pos}_{scoring}_{year}_svr_projections.csv")[['name', 'proj fpts']]
    svr.columns.values[1] = "svr proj"

    mean = pd.read_csv(f"projections/{pos}_{scoring}_{year}_mean_projections.csv")
    mean.columns.values[1] = "mean proj"

    sum_ = pd.merge(svr, mean, how='inner', on=['name'])
    sum_ = pd.merge(enet, mean, how='inner', on=['name'])
    sum_ = pd.merge(xgb, sum_, how='inner', on=['name'])
    sum_ = pd.merge(rf, sum_, how='inner', on=['name'])
    sum_ = pd.merge(sum_, names_ranks, how='inner', on=['name'])
    sum_ = sum_.sort_values('rank', ascending=True)
    
    if save_csv:
        sum_.to_csv(f"projections/{pos}_{scoring}_{year}_summary.csv", index=False)
    return sum_