# %%
# Loaded variable 'df' from URI: /Users/ale/Documents/primo anno/datamining/dm1_25_26_dataset/DM1_game_dataset.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r'/Users/ale/Documents/primo anno/datamining/dm1_25_26_dataset/DM1_game_dataset.csv')

#Parentesi Quadre come null
#df = df.replace("[]", np.nan)

#Describe generale
df_describe=df.describe(include="all")
df_describe

#Funzione di dispersione
def dispersione(Disp):
    return pd.Series({
        "Varianza": Disp.var(),
        "Deviazione std": Disp.std(),
        "Range": Disp.max() - Disp.min(),
        "IQR": Disp.quantile(0.75) - Disp.quantile(0.25),
        "Coeff. Variazione": (Disp.std() / Disp.mean())
    })

# %%
righe_totali = len(df)
conteggio_numowned = df['NumOwned'].count()
valori_mancanti = df['NumOwned'].isnull().sum()
print("\n--- Analisi Valori Mancanti per 'NumOwned' ---")
print(f"Numero totale di righe nel DataFrame: {righe_totali}")
print(f"Conteggio dei valori presenti in 'NumOwned': {conteggio_numowned}")
print(f"Numero di valori mancanti (celle vuote) in 'NumOwned': {valori_mancanti}")
print(f"Verifica: {conteggio_numowned} + {valori_mancanti} = {conteggio_numowned + valori_mancanti}\n")

# %%
#NumOwned
#valore 0 ma con commenti
df=df[df['NumOwned']<4000]
print(f"STATISTICHE DESCRITTIVE:\n{df['NumOwned'].describe(include='all')}")
print(f"MODA:\n{df['NumOwned'].mode()}")
print(f"STATISTICHE DI DISPERSIONE:\n{dispersione(df["NumOwned"])}")
oppa1= [df['NumOwned'].describe(include='all'), df['NumOwned'].mode(), dispersione(df["NumOwned"])]
df_NumOwned_stats = pd.concat(oppa1)

sns.histplot(df, x="NumOwned", color="red", alpha=1)
plt.title("Istogramma NumOwned")
plt.xlabel("NumOwned")
plt.ylabel("Numerosità")
plt.show()

# %%
#NumWant
#20917 valori da 0 a 200, 1938=0 ma possibile che nessuno lo voglia
df = df[df['NumWant'] < 200]
print(f"STATISTICHE DESCRITTIVE:\n{df['NumWant'].describe(include='all')}")
print(f"MODA:\n{df['NumWant'].mode()}")
print(f"STATISTICHE DI DISPERSIONE:\n{dispersione(df["NumWant"])}")
oppa1= [df['NumWant'].describe(include='all'), df['NumWant'].mode(), dispersione(df["NumWant"])]
df_NumOwned_stats = pd.concat(oppa1)

sns.histplot(df, x="NumWant", color="red", alpha=1)
plt.title("Istogramma NumWant")
plt.xlabel("NumWant")
plt.ylabel("Numerosità")
plt.show()

# %%
#NumWish
#statistiche per <1000
#SBAGLIATO
'''STATISTICHE DESCRITTIVE:
count    20149.000000
mean        83.823813
std        131.843840
min          0.000000
25%         13.000000
50%         33.000000
75%         91.000000
max        998.000000
Name: NumWish, dtype: float64
MODA:
0    7
Name: NumWish, dtype: int64
STATISTICHE DI DISPERSIONE:
Varianza             17382.798220
Deviazione std         131.843840
Range                  998.000000
IQR                     78.000000
Coeff. Variazione        1.572869
dtype: float64'''

df = df[df['NumWish'] < 600]
print(f"STATISTICHE DESCRITTIVE:\n{df['NumWish'].describe(include='all')}")
print(f"MODA:\n{df['NumWish'].mode()}")
print(f"STATISTICHE DI DISPERSIONE:\n{dispersione(df["NumWish"])}")
oppa1= [df['NumWish'].describe(include='all'), df['NumWish'].mode(), dispersione(df["NumWish"])]
df_NumWish_stats = pd.concat(oppa1)

sns.histplot(df, x="NumWish", color="red", alpha=1)
plt.title("Istogramma NumWish")
plt.xlabel("NumWish")
plt.ylabel("Numerosità")
plt.show()

sns.kdeplot(df, x="NumWish", color="blue", alpha=1)
plt.title("Densità NumWish")
plt.xlabel("NumWish")
plt.ylabel("Densità")
plt.show()

# %%
#NumUserRatings
df=df[df['NumUserRatings']<2000]
print(f"STATISTICHE DESCRITTIVE:\n{df['NumUserRatings'].describe(include='all')}")
print(f"MODA:\n{df['NumUserRatings'].mode()}")
print(f"STATISTICHE DI DISPERSIONE:\n{dispersione(df["NumUserRatings"])}")
oppa1= [df['NumUserRatings'].describe(include='all'), df['NumUserRatings'].mode(), dispersione(df["NumUserRatings"])]
df_NumUserRatings_stats = pd.concat(oppa1)

sns.histplot(df, x="NumUserRatings", color="red", alpha=1)
plt.title("Istogramma NumUserRatings")
plt.xlabel("NumUserRatings")
plt.ylabel("Numerosità")
plt.show()

# %%
#NumComments
#nullo

# %%
#NumAlternates
df=df[df['NumAlternates']<20]
print(f"STATISTICHE DESCRITTIVE:\n{df['NumAlternates'].describe(include='all')}")
print(f"MODA:\n{df['NumAlternates'].mode()}")
print(f"STATISTICHE DI DISPERSIONE:\n{dispersione(df["NumAlternates"])}")
oppa1= [df['NumAlternates'].describe(include='all'), df['NumAlternates'].mode(), dispersione(df["NumAlternates"])]
df_NumAlternates_stats = pd.concat(oppa1)

sns.countplot(
    data=df, 
    x="NumAlternates", 
    color="red")
plt.title("Grafico a Barre NumAlternates")
plt.xlabel("NumAlternates")
plt.ylabel("Conteggio") 
plt.show()

sns.countplot(
    data=df, 
    x="NumAlternates", 
    color="red")
plt.title("Grafico a Barre NumAlternates")
plt.xlabel("NumAlternates")
plt.ylabel("Conteggio") 
plt.xlim(-0.5, 10.5)
plt.show()

'''sns.countplot(
    data=df, 
    x="NumAlternates", 
    color="red")
plt.yscale('log')
plt.title("Grafico a Barre NumAlternates")
plt.xlabel("NumAlternates")
plt.ylabel("Conteggio") 
plt.show()'''

# %%
#NumExpansions
'''df_sorted = df.sort_values(['NumExpansions'], ascending=[False])
print(df[['Name', 'NumExpansions']].head(9))

df_filtrato=df[df['NumExpansions']<10]
print(f"STATISTICHE DESCRITTIVE:\n{df_filtrato['NumExpansions'].describe(include='all')}")
print(f"MODA:\n{df_filtrato['NumExpansions'].mode()}")
print(f"STATISTICHE DI DISPERSIONE:\n{dispersione(df_filtrato["NumExpansions"])}")
oppa1= [df['NumExpansions'].describe(include='all'), df_filtrato['NumExpansions'].mode(), dispersione(df["NumExpansions"])]
df_NumExpansions_stats = pd.concat(oppa1)

sns.countplot(
    data=df, 
    x="NumExpansions", 
    color="red")
plt.title("Grafico a Barre NumAlternates")
plt.xlabel("NumExpansions")
plt.ylabel("Conteggio") 
plt.show()'''

# --- Parte 1: Ordinamento e visualizzazione dei top 9 ---

# Crea un DataFrame ordinato senza modificare l'originale 'df'
df_sorted = df.sort_values(by='NumExpansions', ascending=False)

# Stampa le prime 9 righe dal DataFrame ORDINATO ('df_sorted')
print("--- Top 9 giochi per NumExpansions ---")
print(df_sorted[['Name', 'NumExpansions']].head(9))

# --- Parte 2: Analisi e grafico per giochi con meno di 10 espansioni ---

# Crea un NUOVO DataFrame per l'analisi, lasciando 'df' intatto
df_filtered = df[df['NumExpansions'] < 10].copy() # .copy() previene SettingWithCopyWarning

# Ora esegui le statistiche sul DataFrame filtrato
print(f"--- Statistiche per giochi con NumExpansions < 10 ---")
print(f"STATISTICHE DESCRITTIVE:\n{df_filtered['NumExpansions'].describe(include='all')}")
print(f"MODA:\n{df_filtered['NumExpansions'].mode()}")

# Assicurati che la funzione 'dispersione' sia definita prima di usarla
# Presumo sia definita in un'altra cella o file, come in inutile.descr.var.py
# from inutile.descr.var import dispersione 
# print(f"STATISTICHE DI DISPERSIONE:\n{dispersione(df_filtered['NumExpansions'])}")


# --- Parte 3: Grafico sui dati filtrati ---

# Il grafico ora usa correttamente il DataFrame filtrato 'df_filtered'
sns.countplot(
    data=df_filtered, 
    x="NumExpansions", 
    color="red")
# Corretto il titolo del grafico
plt.title("Conteggio Giochi per Numero di Espansioni (NumExpansions < 10)")
plt.xlabel("Numero di Espansioni")
plt.ylabel("Conteggio Giochi") 
plt.show()


# %%
#NumImplementations

df_sorted = df.sort_values(by='NumImplementations', ascending=False)
print(df_sorted[['Name', 'NumImplementations']].head(9))

df_filtered = df[df['NumImplementations'] < 5].copy() # .copy() previene SettingWithCopyWarning
print(f"STATISTICHE DESCRITTIVE:\n{df_filtered['NumImplementations'].describe(include='all')}")
print(f"MODA:\n{df_filtered['NumImplementations'].mode()}")

category_counts = df_filtered['NumImplementations'].value_counts()
labels = category_counts.index
sizes = category_counts.values
plt.figure(figsize=(6,6 )) 
plt.pie(sizes,
        labels=labels,
        autopct='%1.1f%%', # Mostra le percentuali con una cifra decimale
        startangle=90,     # Inizia la prima fetta dall'alto
        pctdistance=0.85,  # Distanza delle percentuali dal centro
        wedgeprops={'edgecolor': 'black'}) # Aggiunge un bordo nero alle fette per chiarezza


plt.axis('equal')

plt.title("Distribuzione delle NumImplementations (< 5)")

plt.show()


# %%
#IsReimplementation
uno= (df['IsReimplementation']==1).sum()
print(uno, 'game are presenting a reimplementation')

category_counts = df_filtered['IsReimplementation'].value_counts()
labels = category_counts.index
sizes = category_counts.values
plt.figure(figsize=(6,6 )) 
plt.pie(sizes,
        labels=labels,
        autopct='%1.1f%%', # Mostra le percentuali con una cifra decimale
        startangle=90,     # Inizia la prima fetta dall'alto
        pctdistance=0.85,  # Distanza delle percentuali dal centro
        wedgeprops={'edgecolor': 'black'}) # Aggiunge un bordo nero alle fette per chiarezza


plt.axis('equal')

plt.title("Games presenting a reimplementation")

plt.show()


