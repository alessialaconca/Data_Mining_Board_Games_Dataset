# Loaded variable 'df' from URI: /Users/ale/Documents/primo anno/datamining/dm1_25_26_dataset/DM1_game_dataset.csv
import pandas as pd
import numpy as np
df = pd.read_csv(r'/Users/ale/Documents/primo anno/datamining/dm1_25_26_dataset/df2.csv')

#Librerie
import matplotlib.pyplot as plt
import seaborn as sns


#Funzione di dispersione
def dispersione(Disp):
    return pd.Series({
        "Varianza": Disp.var(),
        "Deviazione std": Disp.std(),
        "Range": Disp.max() - Disp.min(),
        "IQR": Disp.quantile(0.75) - Disp.quantile(0.25),
        "Coeff. Variazione": (Disp.std() / Disp.mean())
    })

risultati = {}

for col in df.columns:
    if col=='Name':
        continue
    print(f"\n===== STATISTICHE PER {col} =====")
    print(f"STATISTICHE DESCRITTIVE:\n{df[col].describe(include='all')}")
    print(f"MODA:\n{df[col].mode()}")
    print(f"STATISTICHE DI DISPERSIONE:\n{dispersione(df[col])}")

# Salva i risultati in un dizionario
    oppa = [
        df[col].describe(include='all'),
        df[col].mode(),
        dispersione(df[col])
    ]
    risultati[col] = pd.concat(oppa)

# Combina tutto in un unico DataFrame finale
df_stats_totali = pd.concat(risultati, axis=1)