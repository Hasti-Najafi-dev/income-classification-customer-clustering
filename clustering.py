#%%
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
#%%
print('STEP1: loading dataset...')
dataset = pd.read_csv('customer_personality_analysis_dataset.csv', sep='\t')
dataset.replace('?', np.nan, inplace=True)
print('STEP2: Handling missing values...')
# %%
print('=================Dataset Info (before cleaning)=======================')
info = dataset.info()
print(info)
#%%
print('=================Dataset Describe (before cleaning)=======================')
describe = dataset.describe(include='all')
print(describe)
#%%
print('Replace NaN values with mode (for object) or mean (for numeric) columns')
for col in tqdm(dataset.columns , desc='cleaning...'):
    if dataset[col].isnull().sum() > 0: 
        if dataset[col].dtype == 'object':
            dataset[col] = dataset[col].fillna(dataset[col].mode()[0])
        else:
            dataset[col] = dataset[col].fillna(dataset[col].mean())

# %%
print('=================Dataset Info (after cleaning)=======================')
info = dataset.info()
print(info)
#%%
print('=================Dataset Describe (after cleaning)=======================')
describe = dataset.describe(include='all')
print(describe)
#%%
print('STEP3: Encoding categorical features')
object_cols = dataset.select_dtypes(include='object').columns

label_encoders = {}

ordinal_mappings = {
    'Education': ['2n Cycle', 'Basic', 'Graduation', 'Master', 'PhD']
}

for col in tqdm(object_cols, desc='Encoding...'):
    if col == 'Dt_Customer':
        dataset['Dt_Customer'] = pd.to_datetime(
            dataset['Dt_Customer'],
            format='%d-%m-%Y',
            errors='coerce'
        )
        dataset['Customer_Year'] = dataset['Dt_Customer'].dt.year
        dataset['Customer_Month'] = dataset['Dt_Customer'].dt.month
    elif col in ordinal_mappings:
        mapping = {val: idx for idx, val in enumerate(ordinal_mappings[col])}
        dataset[col] = dataset[col].map(mapping)
    else:
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col])
        label_encoders[col] = le
        
# %%
print('=================Dataset Info (after encoding)=======================')
info = dataset.info()
print(info)
#%%
print('=================Dataset Describe (after encoding)=======================')
describe = dataset.describe(include='all')
print(describe)
# %%
print('STEP4: normalizing features ...')
df_customers_clean = dataset.drop(['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer'], axis=1, errors='ignore')
scaler_clust = StandardScaler()
scaled_features = scaler_clust.fit_transform(df_customers_clean.select_dtypes(include=[np.number]))
df_scaled = pd.DataFrame(scaled_features, columns=df_customers_clean.select_dtypes(include=[np.number]).columns)
#%%
print('STEP5: PCA')
pca = PCA(n_components=0.95)  
X_pca = pca.fit_transform(df_scaled)
print(f"pca.n_components_ : {pca.n_components_}")

#%%
print('STEP6: Apply KMeans')
wcss = [] #cluster compactness
silhouette_scores = []
k_range = list(range(2, 11))

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))
#%%
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(k_range, wcss, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')

plt.subplot(1,2,2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')
plt.show()
#%%
optimal_k = k_range.pop(silhouette_scores.index(max(silhouette_scores)))
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_pca)

#%%
print('STEP6: Apply AgglomerativeClustering')
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
agg_labels = agg_clustering.fit_predict(X_pca)

#%%
print('STEP7:  Evaluate Clustering Quality')
kmeans_silhouette = silhouette_score(X_pca, kmeans_labels)
kmeans_db = davies_bouldin_score(X_pca, kmeans_labels)

agg_silhouette = silhouette_score(X_pca, agg_labels)
agg_db = davies_bouldin_score(X_pca, agg_labels)

print(f"K-Means - Silhouette: {kmeans_silhouette:.3f}, DB Index: {kmeans_db:.3f}")
print(f"Hierarchical - Silhouette: {agg_silhouette:.3f}, DB Index: {agg_db:.3f}")

#%%
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(df_scaled)
#%%
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(X_2d[:,0], X_2d[:,1], c=kmeans_labels, cmap='viridis', alpha=0.6)
plt.scatter(kmeans_final.cluster_centers_[:,0], kmeans_final.cluster_centers_[:,1], 
            marker='X', s=200, c='red', label='Centroids')
plt.title(f'K-Means Clustering (k={optimal_k})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()

plt.subplot(1,2,2)
plt.scatter(X_2d[:,0], X_2d[:,1], c=agg_labels, cmap='viridis', alpha=0.6)
plt.title(f'Hierarchical Clustering (k={optimal_k})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.tight_layout()
plt.show()

#%%
df_customers_clean['Cluster_KMeans'] = kmeans_labels
df_customers_clean['Cluster_Hierarchical'] = agg_labels

cluster_profile_kmeans = df_customers_clean.groupby('Cluster_KMeans').mean()
cluster_profile_hierarchical = df_customers_clean.groupby('Cluster_Hierarchical').mean()

print("K-Means Cluster Profiles:")
print(cluster_profile_kmeans[['Year_Birth' , 'Income']])

print("\nHierarchical Cluster Profiles:")
print(cluster_profile_hierarchical[['Year_Birth' , 'Income']])
# %%
