"""
## Research Question:

### Data Source:
The data is sourced from the ASQ (Adolescent Sleep Questionnaire).

### Variables of Interest:
- `mdhx_sleep_problem`:
    - I do not have a sleep problem
    - Snoring
    - My breathing stops at night
    - Sleepiness during the day
    - Unrefreshing sleep
    - Difficulty falling asleep
    - Difficulty staying asleep
    - Difficulty keeping a normal sleep schedule
    - Talk, walk, and/or other behavior in my sleep
    - Nightmares or abnormal dreaming
    - Act out dreams in my sleep
    - Teeth grinding or sore teeth/jaw after sleeping
    - Restless or unpleasant sensations in my legs
    - Weakness in muscles of the face, neck, arms or legs when you laugh or are surprised

### Variables of Interest ASQ Dictionary Names:
- Table Name: `mdhx_sleep_problem`

### Analysis:
K-means Clustering
- With ASQ and without
- Pure phenotyping to see if clusters together
"""
import pathlib
import pandas as pd
from config.data_paths import data_paths, multi_response_col
from library.utils import compute_sparse_encoding, FuzzySearch, compute_sparse_encoding
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from library.factorial_analysis_wrapper import FactorAnalysisWrapper
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
import matplotlib.cm as cm
import numpy as np
from sklearn.model_selection import GridSearchCV
from scipy.stats import f_oneway
from scipy.stats import ttest_ind
from scipy.stats import kruskal
import itertools
from tabulate import tabulate
from typing import Optional, Tuple
import json
from typing import Dict, List
from sklearn.preprocessing import StandardScaler
from config.columns_use import columns_interest, col_ehr

if __name__ == "__main__":
    columns_hypothesis = [key for key in columns_interest.keys() if 'mdhx_sleep_' in key]
    # selected_columns = {col: columns_interest[col] for col in columns_hypothesis}
    # %% read the  data
    df_data = pd.read_csv(data_paths.get('pp_data').joinpath('pp_data.csv'))
    # %% select the column for this hypothesis
    columns_asq_ehr_interest = {**columns_interest, **col_ehr}
    df_features = df_data[columns_hypothesis].copy()

    # %% k-means
    kmeans = KMeans(n_clusters=3)  # Specify the number of clusters (you can adjust this)
    cluster_labels = kmeans.fit_predict(df_features)

    # Add cluster labels to the original data
    data_with_clusters = df_features.copy()
    data_with_clusters['Cluster'] = cluster_labels

    # Analyze cluster sizes
    cluster_sizes = data_with_clusters['Cluster'].value_counts()
    print("Cluster Sizes:")
    print(cluster_sizes)

    # Plot clusters (for binary features, you might want to use bar plots) The plot showing is a visual
    # representation of the distribution of binary features within each cluster. It helps you understand how the
    # binary features are distributed among different clusters identified by the K-means algorithm
    for feature in df_features.columns:
        if feature != 'Cluster':  # Skip the cluster column
            plt.figure(figsize=(8, 6))
            data_with_clusters.groupby(['Cluster', feature]).size().unstack().plot(kind='bar', stacked=True)
            plt.title(f'Cluster Analysis for {feature}')
            plt.xlabel('Cluster')
            plt.ylabel('Frequency')
            plt.xticks(rotation=0)
            plt.legend(title=feature)
            plt.grid(0.7)
            plt.tight_layout()
            plt.show()

    #%% sprase pca - k-means
    sparse_pca = SparsePCA(n_components=2)
    sparse_pca.fit(df_features)
    sparse_pca_result = sparse_pca.transform(df_features)

    eigenvalues = sparse_pca.components_
    total_variance = sum(eigenvalues)
    explained_variance = eigenvalues / total_variance

    for i, variance in enumerate(explained_variance):
        print(f"Variance of Principal Component {i + 1}: {variance:.4f}")



    # Perform KMeans clustering on the transformed data
    kmeans = KMeans(n_clusters=3)  # Adjust the number of clusters as needed
    cluster_labels = kmeans.fit_predict(sparse_pca_result)

    # Visualize the clusters in 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(sparse_pca_result[:, 0], sparse_pca_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
    plt.title('Sparse PCA + KMeans Clustering')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()