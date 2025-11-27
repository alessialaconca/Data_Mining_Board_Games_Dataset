from sklearn.preprocessing import StandardScaler
from pyclustering.cluster.xmeans import xmeans
from sklearn.decomposition import PCA
# DATA MINING 1

from yellowbrick.cluster import SilhouetteVisualizer
# -----------------------------------------------
# -----------------------------------------------
# LIBRERIE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, BisectingKMeans, OPTICS, cluster_optics_dbscan
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '12'

# -----------------------------------------------
# -----------------------------------------------
# LIBRERIE
#scaling, normalization
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#kmeans, dbscan, hierarchical (sklearn)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
#evaluation
from sklearn.metrics import silhouette_score
#import dataset

#distance matrix (dbscan elbow, hierarchical)
from scipy.spatial.distance import pdist, squareform

# hierarchical (scipy)
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, BisectingKMeans, OPTICS, cluster_optics_dbscan
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '12'

# Stile generale per i grafici
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8, 4)

# CARICAMENTO DATAFRAME
df = pd.read_csv("/Users/ale/Documents/primo anno/dm1_25_26_dataset/df/DM1_game_dataset.csv")

plt.rcParams['figure.figsize'] = (6, 4)


# Dataset_cluster
train_cluster = train_imputed.copy()
test_cluster = test_imputed.copy()
colonne_cluster = ["MinPlayers", "GameWeight", "ComAgeRec",
                   "NumWant", "NumUserRatings", "NumImplementations"]
train_cluster = train_cluster[colonne_cluster]
test_cluster = test_cluster[colonne_cluster]
X = train_cluster
y = np.array(train_imputed['Rating'])

scaler = MinMaxScaler()
X_minmax = scaler.fit_transform(X)

# PCA
scalerStan = StandardScaler()
features = train_cluster.select_dtypes(include=[np.number])
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=6, random_state=222)
principal_components = pca.fit_transform(features_scaled)
pca_df = pd.DataFrame(data=principal_components,
                      columns=['PC1', 'PC2', "PC3", "PC4", "PC5", "PC6"])

explained_variance = pca.explained_variance_ratio_
total_explained_variance = np.sum(explained_variance)

print("\n--- Varianza Spiegata ---")
print(
    f"Varianza spiegata da PC1: {explained_variance[0]:.4f} ({explained_variance[0]*100:.2f}%)")
print(
    f"Varianza spiegata da PC2: {explained_variance[1]:.4f} ({explained_variance[1]*100:.2f}%)")
print(
    f"Varianza spiegata da PC3: {explained_variance[2]:.4f} ({explained_variance[2]*100:.2f}%)")
print(
    f"Varianza spiegata da PC4: {explained_variance[3]:.4f} ({explained_variance[3]*100:.2f}%)")

print(
    f"Varianza spiegata da PC5: {explained_variance[4]:.4f} ({explained_variance[4]*100:.2f}%)")

print(
    f"Varianza spiegata da PC6: {explained_variance[5]:.4f} ({explained_variance[5]*100:.2f}%)")

print(
    f"Varianza Totale Spiegata (6 componenti): {total_explained_variance:.4f} ({total_explained_variance*100:.2f}%)")



# K_Means
sse = []
k_range = range(1, 20)

for k in k_range:

    kmeans_model = KMeans(n_clusters=k,
                          n_init=10,
                          max_iter=100,
                          random_state=222)

    kmeans_model.fit(X_minmax)

    sse.append(kmeans_model.inertia_)

plt.plot(k_range, sse, marker='o', linestyle='--')
plt.xlabel("Numero di Cluster (K)")
plt.ylabel("Somma dei quadrati (SSE / Inertia)")
plt.title("Metodo del Gomito (Elbow Method)")
plt.xticks(k_range)
plt.grid(True)
plt.show()


kmeans = KMeans(n_clusters=4, n_init=10, max_iter=100, random_state=222)
kmeans.fit(X_minmax)

sns.scatterplot(data=pca_df,
                x="PC1",
                y="PC2",
                hue=kmeans.labels_,
                style=kmeans.labels_,
                palette="bright")
plt.show()



# Bisecting K_Means
sse = []
k_range = range(1, 20)

for k in k_range:

    bkmeans_model = BisectingKMeans(n_clusters=k,
                          n_init=10,
                          max_iter=100,
                          random_state=222)

    bkmeans_model.fit(X_minmax)

    sse.append(bkmeans_model.inertia_)

plt.plot(k_range, sse, marker='o', linestyle='--')
plt.xlabel("Numero di Cluster (K)")
plt.ylabel("Somma dei quadrati (SSE / Inertia)")
plt.title("Metodo del Gomito (Elbow Method)")
plt.xticks(k_range)
plt.grid(True)
plt.show()


bkmeans = BisectingKMeans(n_clusters=4, n_init=10,
                          max_iter=100, random_state=222)
bkmeans.fit(X_minmax)

sns.scatterplot(data=pca_df,
                x="PC1",
                y="PC2",
                hue=bkmeans.labels_,
                style=bkmeans.labels_,
                palette="bright")
plt.show()


#X_Means
K_MAX = 20
xmeans_instance = xmeans(data=X_minmax,
                         kmax=K_MAX,
                         random_state=222)

xmeans_instance.process()
clusters = xmeans_instance.get_clusters()
n_clusters_found = len(clusters)
labels_xmeans = np.zeros(len(X_minmax), dtype=int)
for cluster_id, points_in_cluster in enumerate(clusters):
    for point_index in points_in_cluster:
        labels_xmeans[point_index] = cluster_id

plt.figure(figsize=(8, 6))  # Puoi regolare la dimensione
sns.scatterplot(data=pca_df,
                x="PC1",
                y="PC2",
                hue=labels_xmeans,
                style=labels_xmeans,
                # Adatta la palette al K trovato
                palette=sns.color_palette("bright", n_colors=n_clusters_found))
plt.title(f'X-Means Clustering (K={n_clusters_found} trovato)')
plt.show()