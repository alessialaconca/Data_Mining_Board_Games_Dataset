#Librerie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Stile generale per i grafici
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)

#Caricamento DataFrame
df = pd.read_csv("/Users/ale/Documents/primo anno/datamining/dm1_25_26_dataset/DM1_game_dataset.csv")

# Sommario del dataset
print(df.info())
print(df.describe(include='all'))

#YearPublished
#Valori uguali a 0
prova_year0 = df[df["YearPublished"] == 0]
print(prova_year0)
#Istogramma
sns.histplot(df[(df["YearPublished"] >= 1600) & (df["YearPublished"] <= 2100)],
             x="YearPublished", color="red", alpha=1)
plt.title("Densità YearPublished")
plt.xlabel("YearPublished")
plt.ylabel("Densità")
plt.show()

#GameWeight
#Conteggio valori
print(df["GameWeight"].value_counts().sort_index())
#Valori fuoriscala
prova_gameweight = df[(df["GameWeight"] < 1) | (df["GameWeight"] > 5)]
print(prova_gameweight)
#Grafico
sns.histplot(df, x="GameWeight", color="red", alpha=1)
plt.title("Densità GameWeight")
plt.xlabel("GameWeight")
plt.ylabel("Densità")
plt.show()

#ComWeight
print(df["ComWeight"].value_counts().sort_index())
prova_comweight = df[(df["ComWeight"] < 1) | (df["ComWeight"] > 5)]
print(prova_comweight)
#Grafico
sns.histplot(df, x="ComWeight", color="red", alpha=1)
plt.title("Densità ComWeight")
plt.xlabel("ComWeight")
plt.ylabel("Densità")
plt.show()

#MinPlayers
prova_minplayers = df[df["MinPlayers"] == 0]
print(prova_minplayers)
sns.histplot(df, x="MinPlayers", color="red", alpha=1)
plt.title("Densità MinPlayers")
plt.xlabel("MinPlayers")
plt.ylabel("Densità")
plt.show()

#MaxPlayers
prova_maxplayers_0 = df[df["MaxPlayers"] == 0]
prova_maxplayers_high = df[df["MaxPlayers"] > 99]
print(prova_maxplayers_0)
print(prova_maxplayers_high)
#Grafico
sns.histplot(df, x="MaxPlayers", color="red", alpha=1)
plt.title("Densità MaxPlayers")
plt.xlabel("MaxPlayers")
plt.ylabel("Densità")
plt.show()

#ComAgeRec
sns.histplot(df, x="ComAgeRec", color="red", alpha=1)
plt.title("Densità ComAgeRec")
plt.xlabel("ComAgeRec")
plt.ylabel("Densità")
plt.show()

#LanguageEase
print(df["LanguageEase"].value_counts().sort_index())
#Grafico
sns.histplot(df, x="LanguageEase", color="red", alpha=1)
plt.title("Densità LanguageEase")
plt.xlabel("LanguageEase")
plt.ylabel("Densità")
plt.show()


