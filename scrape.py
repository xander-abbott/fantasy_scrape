# imports
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import sklearn
import seaborn as sns
import requests
import re
import pandas as pd

base = "https://www.fantasypros.com/nfl/stats/"

nextgen = "https://nextgenstats.nfl.com/stats/receiving#yards"

# https://www.fantasypros.com/nfl/games/justin-jefferson.php?season=2021&scoring=PPR

def extract_players():
    data = pd.read_html("https://www.fantasypros.com/nfl/stats/wr.php?year=2022&scoring=PPR")[0]
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
    text = text.replace(". ", "")
    text = text.replace(".", "")
    text = text.replace(" ", "-")
    return text
    

def extract_previous_year(name: str, position: str, scoring: str):
    url = "https://www.fantasypros.com/nfl/games/" + name + ".php?season=2021&scoring=" + scoring
    



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