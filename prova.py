# LIBRERIE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Stile generale per i grafici
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)


# CARICAMENTO DATAFRAME
df = pd.read_csv("/Users/ale/Documents/primo anno/dm1_25_26_dataset/df/DM1_game_dataset.csv")


# -----------------------------------------------
# -----------------------------------------------
# ANALISI ESPLORATIVA

'''
# Describe generale
df_describe = df.describe(include="all")


# Funzione di dispersione
def dispersione(Disp):
    return pd.Series({
        "Varianza": Disp.var(),
        "Deviazione std": Disp.std(),
        "Range": Disp.max() - Disp.min(),
        "IQR": Disp.quantile(0.75) - Disp.quantile(0.25),
        "Coeff. Variazione": (Disp.std() / Disp.mean())
    })


#GameWeight
print(f"STATISTICHE DESCRITTIVE:\n{df['GameWeight'].describe(include='all')}")
print(f"MODA:\n{df['GameWeight'].mode()}")
print(f"STATISTICHE DI DISPERSIONE:\n{dispersione(df["GameWeight"])}")
oppa1= [df['GameWeight'].describe(include='all'), df['GameWeight'].mode(), dispersione(df["GameWeight"])]
df_gameweight_stats = pd.concat(oppa1)

sns.histplot(df, x="GameWeight", color="red", alpha=1)
plt.title("Istogramma GameWeight")
plt.xlabel("GameWeight")
plt.ylabel("Numerosità")
plt.show()

sns.kdeplot(df, x="GameWeight", color="red", alpha=1)
plt.title("Densità GameWeight")
plt.xlabel("GameWeight")
plt.ylabel("Densità")
plt.show()


# MfgPlaytime
df_mfgplaytime = df[df["MfgPlaytime"] > 0]
print(
    f"STATISTICHE DESCRITTIVE:\n{df_mfgplaytime['MfgPlaytime'].describe(include='all')}")
print(f"MODA:\n{df_mfgplaytime['MfgPlaytime'].mode()}")
print(
    f"STATISTICHE DI DISPERSIONE:\n{dispersione(df_mfgplaytime["MfgPlaytime"])}")
oppa_mfgplaytime = [df_mfgplaytime['MfgPlaytime'].describe(
    include='all'), df_mfgplaytime['MfgPlaytime'].mode(), dispersione(df_mfgplaytime["MfgPlaytime"])]
df_mfgplaytime_stats = pd.concat(oppa_mfgplaytime)
count_mfgplaytime = df_mfgplaytime["MfgPlaytime"].value_counts()
print(count_mfgplaytime)
# ho rimosso i valori 0
# ci sono alcuni (pochi) giochi con un tempo di completamento molto alto (qualcosa come 48h+)


# ComMinPlaytime
df_comminplaytime = df[df["ComMinPlaytime"] > 0]
print(
    f"STATISTICHE DESCRITTIVE:\n{df_comminplaytime['ComMinPlaytime'].describe(include='all')}")
print(f"MODA:\n{df_comminplaytime['ComMinPlaytime'].mode()}")
print(
    f"STATISTICHE DI DISPERSIONE:\n{dispersione(df_comminplaytime["ComMinPlaytime"])}")
oppa_comminplaytime = [df_comminplaytime['ComMinPlaytime'].describe(
    include='all'), df_comminplaytime['ComMinPlaytime'].mode(), dispersione(df_comminplaytime["ComMinPlaytime"])]
df_comminplaytime_stats = pd.concat(oppa_comminplaytime)
count_comminplaytime = df_comminplaytime["ComMinPlaytime"].value_counts()
print(count_comminplaytime)
# secondo la comunity ci vuole in generale meno tempo per finire i giochi
# ho rimosso i valori 0
# ci sono alcuni (pochi) giochi con un tempo di completamento molto alto (qualcosa come 48h+)


# ComMaxPlaytime
df_commaxplaytime = df[df["ComMaxPlaytime"] > 0]
print(
    f"STATISTICHE DESCRITTIVE:\n{df_commaxplaytime['ComMaxPlaytime'].describe(include='all')}")
print(f"MODA:\n{df_commaxplaytime['ComMaxPlaytime'].mode()}")
print(
    f"STATISTICHE DI DISPERSIONE:\n{dispersione(df_commaxplaytime["ComMaxPlaytime"])}")
oppa_commaxplaytime = [df_commaxplaytime['ComMaxPlaytime'].describe(
    include='all'), df_commaxplaytime['ComMaxPlaytime'].mode(), dispersione(df_commaxplaytime["ComMaxPlaytime"])]
df_commaxplaytime_stats = pd.concat(oppa_commaxplaytime)
count_commaxplaytime = df_commaxplaytime["ComMaxPlaytime"].value_counts()
print(count_commaxplaytime)

gne = df[["ComMaxPlaytime", "MfgPlaytime"]]
print(
    f"Sono le stesse colonne?:\n {(df["ComMaxPlaytime"] == df["MfgPlaytime"]).all()}")
# ComMaxPlaytime e MfgPlaytime SONO IDENTICHE? HO SBAGLIATO QUALCOSA?


# MfgAgeRec
df_mfgagerec = df[df["MfgAgeRec"] > 0]
print(
    f"STATISTICHE DESCRITTIVE:\n{df_mfgagerec['MfgAgeRec'].describe(include='all')}")
print(f"MODA:\n{df_mfgagerec['MfgAgeRec'].mode()}")
print(f"STATISTICHE DI DISPERSIONE:\n{dispersione(df_mfgagerec["MfgAgeRec"])}")
oppa_mfgagerec = [df_mfgagerec['MfgAgeRec'].describe(
    include='all'), df_mfgagerec['MfgAgeRec'].mode(), dispersione(df_mfgagerec["MfgAgeRec"])]
df_mfgagerec_stats = pd.concat(oppa_mfgagerec)
count_mfgagerec = df_mfgagerec["MfgAgeRec"].value_counts()
print(count_mfgagerec)

# valori (pochi) =0 esclusi, tendenza centrale con 3 picchi intorno alla media/mediana

# LanguageEase
# https://boardgamegeek.com/wiki/page/language-independent
df_languageease = df[df["LanguageEase"] > 0]
print(
    f"STATISTICHE DESCRITTIVE:\n{df_languageease['LanguageEase'].describe(include='all')}")
print(f"MODA:\n{df_languageease['LanguageEase'].mode()}")
print(
    f"STATISTICHE DI DISPERSIONE:\n{dispersione(df_languageease["LanguageEase"])}")
oppa_languageease = [df_languageease['LanguageEase'].describe(
    include='all'), df_languageease['LanguageEase'].mode(), dispersione(df_languageease["LanguageEase"])]
df_languageease_stats = pd.concat(oppa_languageease)
count_languageease = df_languageease["LanguageEase"].value_counts()
print(count_languageease)
# boh, i valori mi sembrano tutti sballati
# dopo qualche ricerca: COMPLETAMENTE SBALLATO

# NumWeightVotes
df_numweightvotes = df[df["NumWeightVotes"] > 0]
print(
    f"STATISTICHE DESCRITTIVE:\n{df_numweightvotes['NumWeightVotes'].describe(include='all')}")
print(f"MODA:\n{df_numweightvotes['NumWeightVotes'].mode()}")
print(
    f"STATISTICHE DI DISPERSIONE:\n{dispersione(df_numweightvotes["NumWeightVotes"])}")
oppa_numweightvotes = [df_numweightvotes['NumWeightVotes'].describe(
    include='all'), df_numweightvotes['NumWeightVotes'].mode(), dispersione(df_numweightvotes["NumWeightVotes"])]
df_numweightvotes_stats = pd.concat(oppa_numweightvotes)
count_numweightvotes = df_numweightvotes["NumWeightVotes"].value_counts()
print(count_numweightvotes)
# NUmero di persone che hanno votato per la complessità
# Tutto normale, qualche outlier in eccesso (Carcassone es.)


# Kickstarted
print(
    f"Variabile kickstarter binaria, conta:\n{df["Kickstarted"].value_counts()}")
# 3362 giochi sono stati finanziati dai fan


# CATEGORIE e RANKING
categorie = ["Cat:Thematic", "Cat:Strategy", "Cat:War", "Cat:Family",
             "Cat:CGS", "Cat:Abstract", "Cat:Party", "Cat:Childrens"]
df["multi_categoria"] = df[categorie].sum(axis=1) > 1
multi_categorie = df[df[categorie].sum(axis=1) > 1]
# 1574 giochi fanno parte di più di una categoria

pairs = list(itertools.combinations(categorie, 2))
pair_counts = {}
for a, b in pairs:
    pair_counts[(a, b)] = ((multi_categorie[a] == 1)
                           & (multi_categorie[b] == 1)).sum()

pair_df = pd.DataFrame(pair_counts.items(), columns=["Pair", "Count"])
# in caso te ne scordi, il >0 vuol dire che teniamo solo le coppie esistenti
pair_df = pair_df[pair_df["Count"] > 0]


#cambio i valori inutili nel ranking con null
colonne_rank = [
    "Rank:strategygames",
    "Rank:abstracts",
    "Rank:familygames",
    "Rank:thematic",
    "Rank:cgs",
    "Rank:wargames",
    "Rank:partygames",
    "Rank:childrensgames"
]
df[colonne_rank] = df[colonne_rank].replace(21926, np.nan)
'''


# -----------------------------------------------
# -----------------------------------------------
# DROPPING VARIABILI & CAMBIO


# Dropping explanation
print(
    f"\nCorrelazione tra GameWeight e ComWeight:\n {df["GameWeight"].corr(df["ComWeight"])}")
print(
    f"\nCorrelazione tra MfgPlaytime e ComMaxPlaytime:\n {df["MfgPlaytime"].corr(df["ComMaxPlaytime"])}")


# Dropping worthless variables
df = df.drop(["ComWeight", "LanguageEase", "BestPlayers", "GoodPlayers",
              "ComMaxPlaytime", "NumComments", "ImagePath"], axis=1)


# Cambio di RATING
'''df["Rating"] = df["Rating"].replace("Low", "1")
df["Rating"] = df["Rating"].replace("Medium", "2")
df["Rating"] = df["Rating"].replace("High", "3")  '''


# Parentesi Quadre come null
'''df = df.replace("[]", np.nan)'''


# -----------------------------------------------
# -----------------------------------------------
# MISSING VALUES


# Missing Values: Episode I - The Phantom Value

# YearPublished -> missing values si lehano
df = df[df["YearPublished"] != 0]
# Players -> missing values si lehano
df = df[df["MinPlayers"] != 0]
df = df[df["MaxPlayers"] != 0]
# Family -> Rinomino missing values to a category
df["Family"] = df["Family"].fillna("No_Family")
# df["Family"] = df["Family"].replace(np.nan, "No_Family")  #Alternativa alla riga sopra
# Playtime-> Si usa la moda
df["MfgPlaytime"] = df["MfgPlaytime"].replace(0, df["MfgPlaytime"].mode()[0])
df["ComMinPlaytime"] = df["ComMinPlaytime"].replace(
    0, df["ComMinPlaytime"].mode()[0])


# Missing Values: Episode II - Attack of the imputation

# In pratica ho usato due tecniche basilari (Random Forest, Knn)
# e una avanzata (MICE) per generare valori mancanti di GameWeight
# Chiaramente, solo un modello è da utilizzare, se runnati insieme il primo
# sostituisce i valori mancanti, e gli altri non fanno nulla
# GameWeight ha circa 500 missing values (in realtà sono valori 0, ma è lo stesso discorso)


# Random_Forest
df["GameWeight"] = df["GameWeight"].replace(0, np.nan)
df_solo_numeriche = df.select_dtypes(include=["number"])
df_solo_numeriche = df_solo_numeriche.drop([
    "Rank:strategygames",
    "Rank:abstracts",
    "Rank:familygames",
    "Rank:thematic",
    "Rank:cgs",
    "Rank:wargames",
    "Rank:partygames",
    "Rank:childrensgames"
], axis=1)
valori = df_solo_numeriche[df_solo_numeriche["GameWeight"].notna()]
missing = df_solo_numeriche[df_solo_numeriche["GameWeight"].isna()]
X_train = valori.drop(columns=["GameWeight"])
y_train = valori["GameWeight"]
X_missing = missing.drop(columns=["GameWeight"])
modello_RF = RandomForestRegressor(random_state=222)
modello_RF.fit(X_train, y_train)
predizione = modello_RF.predict(X_missing)
df.loc[df["GameWeight"].isna(), "GameWeight"] = predizione


'''# KNN_Imputer
predittori = df_solo_numeriche.drop(columns=["GameWeight"])
# per riorganizzare il dataset
df_knn = pd.concat([df_solo_numeriche["GameWeight"], predittori], axis=1)
# si potrebbe aumentare, ma non ne vedo il motivo
imputer = KNNImputer(n_neighbors=5)
df_imputato = imputer.fit_transform(df_knn)
df_solo_numeriche["GameWeight"] = df_imputato[:, 0]'''


'''# MICE usando linear regression (mi sembra la più opportuna)
imputer = IterativeImputer(random_state=222, max_iter=100)
imputer.fit(df_solo_numeriche)
df_imputato = imputer.transform(df_solo_numeriche)
df["GameWeight"] = df_imputato[:,
                               df_solo_numeriche.columns.get_loc("GameWeight")]
# Sfortunatamente, con MICE escono fuori valori <1, quindi lo scarterei o si cambia qualcosa tra i parametri
df["GameWeight"] = df["GameWeight"].clip(lower=1)
# soluzione: si clippano i valori minori di 1 con 1
# in realtà, si potrebbe usare un MICE Random Forest, ma diventa pesante'''

count_peso = df["GameWeight"].value_counts()


# -----------------------------------------------
# -----------------------------------------------
# TRASFORMAZIONE VARIABILI


numeriche = df.select_dtypes(include=["int64", "float64"]).columns
categoriche = df.select_dtypes(include=["object", "category"]).columns
numeriche = numeriche.drop([
    "BGGId",
    "Rank:strategygames",
    "Rank:abstracts",
    "Rank:familygames",
    "Rank:thematic",
    "Rank:cgs",
    "Rank:wargames",
    "Rank:partygames",
    "Rank:childrensgames",
    "Cat:Thematic", "Cat:Strategy", "Cat:War", "Cat:Family", "Cat:CGS",
    "Cat:Abstract", "Cat:Party", "Cat:Childrens",
    "Kickstarted", "IsReimplementation"
])


# Grafici per ogni quantitativa
'''for col in numeriche:
    sns.histplot(df[col], kde=True)
    plt.title(col)
    plt.show()
'''

# Skewness
valori_skew = df[numeriche].skew()
# print(f"\nSkewness originale:\n{valori_skew}\n")


def trasformatore(dataframe: pd.DataFrame, colonne: list, soglia: float = 1.0) -> pd.DataFrame:

    df_copy = dataframe.copy()
    skew_values = df_copy[colonne].skew()
    colonne_da_trasformare = skew_values[abs(skew_values) > soglia].index

    if colonne_da_trasformare.empty:
        print(f"\nFINE: nessuna colonna con skewness > |{soglia}|.")
        return df_copy

    col_log = []
    col_cbrt = []

    for col in colonne_da_trasformare:

        if df_copy[col].min() < 0:
            df_copy[col] = np.cbrt(df_copy[col])
            col_cbrt.append(col)
        else:
            df_copy[col] = np.log1p(df_copy[col])
            col_log.append(col)

    print("\nFINE: trasformazione applicata")
    if col_log:
        print(f"\n  - Trasformazione Logaritmica applicata a: {col_log}")
    if col_cbrt:
        print(f"\n  - Trasformazione Radice Cubica applicata a: {col_cbrt}")

    return df_copy


df_corretto = trasformatore(df, numeriche.tolist())

  
# -----------------------------------------------
# -----------------------------------------------
# OUTLIERS


iforest = IsolationForest(
    n_estimators=100, contamination=0.05, random_state=222)

numeriche_corretto = df_corretto[numeriche]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeriche_corretto)
y_pred_iforest = iforest.fit_predict(X_scaled)

df_corretto["outlier"] = y_pred_iforest

print(f"\nVi sono {np.sum(y_pred_iforest == -1)} possibili outlier")

df_outliers = df_corretto[df_corretto["outlier"] == -1]

print(
    f"\nPercentuale di outlier:\n {len(df_outliers) / len(df_corretto) * 100:.2f}%")


df_capped = df_corretto.copy()
LOWER_LIMIT = 0.05
UPPER_LIMIT = 0.95
#bisogna giocare con questi valori, sono alti ma ci sono anche tanti outliers

for col in numeriche:

    lower_bound = df_capped[col].quantile(LOWER_LIMIT)
    upper_bound = df_capped[col].quantile(UPPER_LIMIT)

    df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)

df_capped = df_capped.drop(columns=["outlier"    ])


# Commentino tattico: ho trovato i valori estremi con IsolationForest, in questo modo ho analizzato i vettori di riga invece che i singoli valori per ogni riga/colonna;
# poi, ho deciso di clippare i valori invece di rimuoverli poichè non sono errori in se per se, e credo possa valerne la pena di mantenerli (consideriamo che alcuni degli outliers
# sono giochi famosissimi, sarebbe come togliere Messi da un database di calcio)
# al posto di isolation forest si potrebbe usare LOF o addirittura PCA

# ATTENZIONE ALL'ORDINE: può avere senso rimuovere gli outliers prima della trasformazione delle variabili e dell'imputazione di dati mancanti


# -----------------------------------------------
# -----------------------------------------------
# CORRELAZIONE


df_capped_num= df_capped.drop(["BGGId", "Name" , "Description", "Family", "Rating",], axis=1)
M_Cor = df_capped_num.corr()
sns.heatmap(M_Cor)

#Variabili con alta correlazione: (Gameweight, MfgPLaytime, comeagerec, comminplaytime); (NumOwned, NumWeightVotes, NumWant, NumWIsh, NumUserRating)
#(NUmReimplementation, isReimplementation); (ongi categotia ed il corrispettivo ranking)
#Ci si fa un grafico I guess? Sicuramente si può fare un discorso ganzo nel report


# -----------------------------------------------
# -----------------------------------------------
#CLUSTERING CENTROID-BASED +



from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd # Assicurati che pandas sia importato

# 1. Definisci le colonne che vuoi scalare
colonne_numeriche = df_corretto.select_dtypes(include=["int64", "float64"]).columns

colonne_da_escludere = [
    "BGGId",
    "Rank:strategygames", "Rank:abstracts", "Rank:familygames",
    "Rank:thematic", "Rank:cgs", "Rank:wargames", "Rank:partygames",
    "Rank:childrensgames", "Cat:Thematic", "Cat:Strategy", "Cat:War",
    "Cat:Family", "Cat:CGS", "Cat:Abstract", "Cat:Party", "Cat:Childrens",
    "Kickstarted", "IsReimplementation", "ComAgeRec"
]

# 'errors='ignore'' evita un errore se una colonna da escludere non è tra quelle numeriche
colonne_da_scalare = colonne_numeriche.drop(colonne_da_escludere, errors='ignore')

# 2. Seleziona i DATI da quelle colonne
# Questa è la parte che mancava:
X = df_corretto[colonne_da_scalare]

# 3. Inizializza lo scaler e adatta i DATI (X), non la lista dei nomi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# (Opzionale, ma utile) Puoi riconvertire l'array scalato in un DataFrame
# X_scaled_df = pd.DataFrame(X_scaled, columns=colonne_da_scalare, index=X.index)


# 4. Applica la PCA
# Questa parte era già corretta
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

import matplotlib.pyplot as plt
import seaborn as sns # <-- 1. Assicurati di importare Seaborn

# --- Il tuo codice PCA (che è corretto) va qui sopra ---
# ...
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
# ...

# --- 2. Crea un nuovo DataFrame per il plotting ---
# Questo rende il plotting con Seaborn molto più semplice

# Crea il DataFrame con i risultati della PCA
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', "PC3"])

# Aggiungi la colonna per il colore dal tuo DataFrame originale
# USA IL NOME CORRETTO DELLA COLONNA CHE VUOI USARE PER I COLORI
# (Sostituisci 'NOME_COLONNA_PER_COLORE' con 'outliers', 'cluster', ecc.)
# Usiamo l'indice di 'X' per garantire che le righe corrispondano

total_explained_variance = np.sum(pca.explained_variance_ratio_)
k = 3
print(f"\nVarianza totale spiegata dalle {k} componenti: {total_explained_variance * 100:.2f}%")


pca_df['colore'] = df_corretto.loc[X.index, 'outlier'].values


# --- 3. Calcola la varianza (già fatto da te) ---
pc1_variance = pca.explained_variance_ratio_[0] * 100
pc2_variance = pca.explained_variance_ratio_[1] * 100


# --- 4. Disegna il grafico con Seaborn (ora con i dati!) ---
plt.figure(figsize=(10, 7)) # Imposta una dimensione leggibile

sns.scatterplot(
    data=pca_df,       # Usa il DataFrame appena creato
    x='PC1',           # Colonna per l'asse X
    y='PC2',           # Colonna per l'asse Y
    hue='colore',      # Colonna per definire i colori
    palette='viridis', # Mappa colori (puoi cambiarla)
    s=50,              # Dimensione dei punti
    alpha=0.7          # Trasparenza
)

# Aggiungi etichette e titolo (il tuo codice era già corretto)
plt.xlabel(f'Principal Component 1 ({pc1_variance:.2f}%)')
plt.ylabel(f'Principal Component 2 ({pc2_variance:.2f}%)')
plt.title('PCA del Game Dataset')
plt.grid()
plt.show()


# --- Importa il toolkit 3D (assicurati che sia all'inizio del file) ---
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np # Assicurati che numpy sia importato

# --- Assumendo che 'pca' e 'pca_df' esistano dal tuo codice precedente ---
# pca = PCA(n_components=3)
# X_pca = pca.fit_transform(X_scaled)
# pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', "PC3"])
# pca_df['colore'] = df_corretto.loc[X.index, 'outlier'].values
# ---

# --- 1. Calcola la varianza per TUTTE e 3 le componenti ---
pc1_variance = pca.explained_variance_ratio_[0] * 100
pc2_variance = pca.explained_variance_ratio_[1] * 100
pc3_variance = pca.explained_variance_ratio_[2] * 100 # <-- Aggiungi questa

# --- 2. Crea la figura e l'asse 3D ---
fig = plt.figure(figsize=(12, 9)) # Un po' più grande per il 3D
ax = fig.add_subplot(111, projection='3d')

# --- 3. Disegna il grafico a dispersione 3D ---
# Usiamo ax.scatter() di matplotlib invece di sns.scatterplot()
scatter = ax.scatter(
    xs=pca_df['PC1'],      # Dati per l'asse X
    ys=pca_df['PC2'],      # Dati per l'asse Y
    zs=pca_df['PC3'],      # Dati per l'asse Z (la tua richiesta)
    c=pca_df['colore'],    # Colore basato sulla colonna 'colore'
    cmap='viridis',      # Mappa colori (equivalente di 'palette' in seaborn)
    s=50,                  # Dimensione dei punti
    alpha=0.7              # Trasparenza
)

# --- 4. Aggiungi etichette e titolo ---
ax.set_xlabel(f'Principal Component 1 ({pc1_variance:.2f}%)')
ax.set_ylabel(f'Principal Component 2 ({pc2_variance:.2f}%)')
ax.set_zlabel(f'Principal Component 3 ({pc3_variance:.2f}%)') # <-- Etichetta per l'asse Z
ax.set_title('PCA 3D del Game Dataset')

# --- 5. (Opzionale) Aggiungi una legenda ---
# Utile se 'colore' ha poche categorie (come 0 e 1 per 'outlier')
legend1 = ax.legend(*scatter.legend_elements(), title="Gruppi")
ax.add_artist(legend1)

plt.grid(True) # Aggiunge una griglia per la leggibilità
plt.show()