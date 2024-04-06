"""
# Research Question: Do specific sleep disturbances tend to co-occur?

## Data Source: ASQ
The data is sourced from the ASQ (Adolescent Sleep Questionnaire).

## Variables of Interest:
The variables of interest are scores from the validated sleep scales, including:
- ESS (Epworth Sleepiness Scale)
- ISI (Insomnia Severity Index)
- RLS (Restless Legs Syndrome)
- Sleepwalking
- Sleep eating
- Acting out dreams
- Violent behavior during sleep
- Sex with no memory

## Variables of Interest ASQ Dictionary Names:
- ESS: `ess_0900_score`
- ISI: `score`
- RLS: `rls_probability`
- Sleepwalking: `par_0205`
- Sleep eating: `par_0305`
- Acting out dreams: `par_0505`
- Violent behavior during sleep: `par_0605`
- Sex with no memory: `par_1005`

## Analysis:
The analysis will involve performing sparse PCA (Principal Component Analysis) to explore patterns and relationships
among the sleep disturbances. Other methods may also be explored to validate findings.

## Interpretation:
If the answer to all the parasomnia questions is "never," it indicates that no parasomnia is reported. If there is a
frequency reported for any of the parasomnia questions, it suggests that some kind of parasomnia activity is reported.
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
    'age',
    'bmi',
    'gender',
    'ess_0900',
    'isi_score',
    'rls_probability',
    'par_0205',
    'par_0505',
    'par_0605',
    'par_1005'
    ]

    # selected_columns = {col: columns_interest[col] for col in columns_hypothesis}
    # %% read the  data
    df_data = pd.read_csv(data_paths.get('pp_data').joinpath('pp_data.csv'))
    # %% select the column for this hypothesis
    columns_asq_ehr_interest = {**columns_interest, **col_ehr}
    df_features = df_data[columns_hypothesis].copy()
    # %% standardize Z-scores
    col_standardize = ['age',
                       'bmi',
                       'ess_0900',
                       'isi_score',
                       ]
    scaler = StandardScaler()
    df_continuous_standardized = scaler.fit_transform(df_features[col_standardize])
    df_continuous_standardized = pd.DataFrame(df_continuous_standardized, columns=col_standardize)
    df_features[col_standardize] = df_continuous_standardized
    df_features.to_csv('sleep_data_hyp3.csv', index=False)