#Webscraping from HTML
from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

url = "https://en.wikipedia.org/wiki/List_of_Wimbledon_gentlemen%27s_singles_champions"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

tennis = soup.select("#mw-content-text > div.mw-parser-output > table:nth-child(28)")[0]
df = pd.read_html(str(tennis))[0]

print(df)

champions = df['Champion']
print(champions)

#Remove the COVID year as it is not relevant in our graph or data

filtered_df = df[df['Champion'] != 'No competition (due to COVID-19 pandemic)[6]']
pd.set_option('display.max_columns', 10)
print(filtered_df)

#Start creating some graphs

nationality_count = filtered_df['Country'].value_counts()
print(nationality_count)

#Pie Chart
plt.pie(nationality_count.values, labels=nationality_count.index)
plt.title('Wimbledon Winners By Country')
plt.show()

#Bar Chart
plt.bar(nationality_count.index, nationality_count.values)
plt.title('Wimbledon Winners by Country')
plt.xlabel('Country')
plt.ylabel('Wins')
plt.show()

#Find out who has the most championships
Champion_count = filtered_df['Champion'].value_counts()
champion_counts = Champion_count[Champion_count >= 3]
print(Champion_count)

#Do a bar plot for how many champions there are

plt.bar(champion_counts.index, champion_counts.values)
plt.title('Wimbledon Wins by Player')
plt.xlabel('Champion', fontsize=12)
plt.xticks(fontsize=7)
plt.ylabel('Wins')
plt.show()
