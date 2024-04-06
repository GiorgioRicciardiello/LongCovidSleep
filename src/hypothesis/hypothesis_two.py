"""
## Research Question
Are sleep disturbance symptoms connected with other symptoms? Do they commonly occur with certain other symptoms?

### Data Source
- Electronic Medical Record — Likert Scale (1 to 5) Graded Symptoms taken at the last clinic visit (Expanded Database for PCA 3/29)

### Variables of Interest
- Headaches
- Nasal congestion
- Fatigue
- Brain fog
- Unrefreshing sleep
- Insomnia
- Lethargy
- Post-exertional malaise
- Anosmia
- Dysgeusia
- Anxiety and depression
- Cough
- Shortness of breath
- Lightheadedness
- GI symptoms

### Analysis
PCA (Principal Component Analysis)

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
    columns_hypothesis = [
    # 'age',
    # 'bmi',
    # 'gender',
    'Headaches',
    'Nasal Congestion',
    'Fatigue',
    'Brain fog',
    'Unrefreshing sleep',
    'Insomnia',
    'Lethargy',
    'Post-exertional Malaise',
    'Change in Smell',
    'Change in Taste',
    'Anxiety/Depression',
    'Cough',
    'Shortness of Breath',
    'Lightheadedness',
    'GI Symptoms'
    ]

    # selected_columns = {col: columns_interest[col] for col in columns_hypothesis}
    # %% read the  data
    df_data = pd.read_csv(data_paths.get('pp_data').joinpath('pp_data.csv'))
    # %% select the column for this hypothesis
    columns_asq_ehr_interest = {**columns_interest, **col_ehr}
    df_features = df_data[columns_hypothesis].copy()
    # Drop rows with all NaN values
    df_features.dropna(how='all', inplace=True)
    # %% standardize Z-scores
    # continuous_variables = [key for key, value in columns_asq_ehr_interest.items() if value == 'continuous']
    # continuous_variables = [var for var in continuous_variables if var in df_features.columns]
    # if len(continuous_variables) > 1:
    #     scaler = StandardScaler()
    #     df_continuous_standardized = scaler.fit_transform(df_features[continuous_variables])
    #     df_continuous_standardized = pd.DataFrame(df_continuous_standardized, columns=continuous_variables)
    #     df_features[continuous_variables] = df_continuous_standardized
    # %% Apply PCA
    pca = PCA(n_components=8)
    df_pca = pca.fit_transform(df_features)
    explained_variance_ratio = pca.explained_variance_ratio_
    # Scree plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1),
             explained_variance_ratio, marker='o', linestyle='-')
    plt.title('Scree Plot')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig((data_paths.get('results').joinpath(f'hyp_2_scree_plot_eight_factors.png')), dpi=300)
    plt.show()

    pca = PCA(n_components=3)
    component_combinations = [(0, 1), (0, 2), (1, 3)]

    # Loop over each combination and create scatter plot
    for i, (pc1, pc2) in enumerate(component_combinations):
        plt.figure(figsize=(8, 6))
        plt.scatter(df_pca[:, pc1], df_pca[:, pc2], alpha=0.5)
        plt.title(f'Scatter plot of PCA dimensions (PC{pc1 + 1} vs PC{pc2 + 1})')
        plt.xlabel(f'Principal Component {pc1 + 1}')
        plt.ylabel(f'Principal Component {pc2 + 1}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig((data_paths.get('results').joinpath(f'hyp_2_pca_{pc1}_vs_{pc2}.png')), dpi=300)
        plt.show()

    # apply k-means

    df_features.to_csv('covid_sleep.csv', index=False)
