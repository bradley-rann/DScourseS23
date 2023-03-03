#Webscraping from HTML
from bs4 import BeautifulSoup
import requests
import pandas as pd

url = "https://en.wikipedia.org/wiki/List_of_Wimbledon_gentlemen%27s_singles_champions"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

tennis = soup.select("#mw-content-text > div.mw-parser-output > table:nth-child(28)")[0]
df = pd.read_html(str(tennis))[0]

print(df)

champions = df['Champion']
print(champions)

#Grabbing a dataset with an API

url = "https://healthdata.gov/resource/g62h-syeh.csv"
response = requests.get(url)

df = pd.read_csv(url)
print(df.head())

columns = ["state", "inpatient_beds"]
df_columns = df[columns]
print(df_columns.head())

