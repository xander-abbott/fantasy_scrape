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
    data['Name'] = data['Name'].apply(lambda x: clean_name(x))
    return data

def clean_name(text: str):
    text = text.rsplit(' ', 1)[0].lower()
    text = text.replace("amon-ra", "amonra")
    text = text.replace(" jr.", "")
    text = text.replace(" sr.", "")
    text = text.replace("st. ", "st.")
    text = text.replace(".", "")
    text = text.replace("'", "")
    text = text.replace(" ", "-")
    # edge cases in top 100 (2020-2022 wrs), no rhyme/reason
    text = text.replace("gabe", "gabriel")
    text = text.replace("joshua", "josh")
    text = text.replace("valdes-scantling", "valdesscantling")
    text = text.replace("robinson-ii", "robinson")
    text = text.replace("westbrook-ikhine", "westbrook")
    text = text.replace("equanimeous-stbrown", "equanimeous-st-brown")
    text = text.replace("chosen", "robby")
    text = text.replace("ruggs-iii", "ruggs")
    text = text.replace("william-fuller-v", "will-fuller")
    text = text.replace("willie-snead-iv", "willie-snead")
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

def make_dists(names, year):
    raw = []
    count = 1
    for name in names:
        print(name)
        if name in ['dj-moore', 'mike-williams', 'michael-thomas']:
            name += '-wr'
        try:
            data = extract_year(name=name, year=year, scoring="PPR")
        except:
            return "couldn't extract for " + name
        else:
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
            raw.append([name, age, avg_rec, var_rec, avg_tgt, var_tgt, avg_yds, var_yds, avg_ypr, var_ypr, 
                        avg_lg, var_lg, avg_TD, var_TD, avg_rush, var_rush, avg_ryds, var_ryds,
                        avg_rTD, var_rTD])
    df = pd.DataFrame(raw, columns=["name", "age", "avg_rec", "var_rec", "avg_tgt", "var_tgt", "avg_yds", "var_yds", 
                                        "avg_ypr", "var_ypr", "avg_lg", "var_lg", "avg_TD", "var_TD", 
                                        "avg_rush", "var_rush", "avg_ryds", "var_ryds", "avg_rTD", "var_rTD"])
    return df
    
    