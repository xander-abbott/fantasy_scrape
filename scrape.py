# imports
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import sklearn
import seaborn as sns
import requests
import re

url = "https://www.fantasypros.com/nfl/stats/"

def extract(position : str, player : str, year : str, scoring : str):
    """
    Reads html from fantasypros stats per position, year, and scoring specifications
    Inputs: 
        Position: a string consisting of "wr", "rb", "te", "qb"
        Year: string consisting of a year
        Scoring: string consisting of "PPR", "HALF"
    """
    url = "https://www.fantasypros.com/nfl/stats/" + position + ".php?year=" + year + "&scoring=" + scoring
    # access url
    result = requests.get(url)
    # read html
    doc = BeautifulSoup(result.text, "html.parser")
    # tr is table row. Find all table rows with class starting with "mpb-player-..."
    # each result is a player and their stats
    rows = doc.find_all("tr", attrs={'class': re.compile('mpb\-player\-*')}, limit=1)
    return rows
print(extract("wr", "_", "2022", "PPR"))