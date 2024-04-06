"""
Research Question: Are certain demographic characteristics associated with moderate vs
severe sleep symptom presentations?

Two groups- moderate symptoms, severe symptoms

CSV - which column is each one
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
from sklearn.decomposition import PCA, SparsePCA
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
    col_predictors = [
        'Hospitalized (no = 0, yes = 1)',
        'Days the Person Has Had Long COVID',
        'bmi',
        'age',
        'gender',
        'race',
    ]

    outcome_vars = [
        'isi_score',
        'ess_0900',
        'rls_probability',
    ]
    all_columns = col_predictors + outcome_vars
    # selected_columns = {col: columns_interest[col] for col in columns_hypothesis}
    # %% read the  data
    df_data = pd.read_csv(data_paths.get('pp_data').joinpath('pp_data.csv'))
    # %% get the medical history columns
    col_mdhx_sleep_problem = [col for col in df_data.columns if 'mdhx_sleep_problem' in col]
    all_columns.extend(col_mdhx_sleep_problem)

    # %% select the column for this hypothesis
    # merge the dictionary questions of the two datasets into a single one
    columns_asq_ehr_interest = {**columns_interest, **col_ehr}
    df_features = df_data[all_columns].copy()
    # Drop rows with all NaN values
    df_features = df_features.dropna(how='any')

    # %% standardize Z-scores
    # continuous_variables = [key for key, value in columns_asq_ehr_interest.items() if value == 'continuous']
    # continuous_variables = [var for var in continuous_variables if var in df_features.columns]
    # continuous_variables = ['age', 'bmi']
    # if len(continuous_variables) > 1:
    #     scaler = StandardScaler()
    #     df_continuous_standardized = scaler.fit_transform(df_features[continuous_variables])
    #     df_continuous_standardized = pd.DataFrame(df_continuous_standardized, columns=continuous_variables)
    #     df_features[continuous_variables] = df_continuous_standardized
    # %% Apply PCA - to nly mdhx sleep questions
    n_components = 5

    data_pca = df_features[col_mdhx_sleep_problem]

    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(data_pca)
    explained_variance_ratio = pca.explained_variance_

    # Scree plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1),
             explained_variance_ratio, marker='o', linestyle='-')
    plt.title(f'Scree Plot - Components {n_components}')
    plt.xlabel('Component Number')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig((data_paths.get('results').joinpath(f'hyp_7_scree_plot_eight_factors.png')), dpi=300)
    plt.show()

    # Loop over each combination and create scatter plot
    component_combinations = list(itertools.combinations(range(df_pca.shape[1]), 2))
    # Loop over each combination and create scatter plot
    for i, (pc1, pc2) in enumerate(component_combinations):
        plt.figure(figsize=(8, 6))
        plt.scatter(df_pca[:, pc1], df_pca[:, pc2], alpha=0.5)
        plt.title(f'Scatter plot of PCA dimensions (PC{pc1 + 1} vs PC{pc2 + 1})')
        plt.xlabel(f'Principal Component {pc1 + 1}')
        plt.ylabel(f'Principal Component {pc2 + 1}')
        plt.grid(True)
        plt.tight_layout()

        # Save the scatter plot to a file
        filename = f'hyp_2_pca_{pc1}_vs_{pc2}.png'
        plt.savefig(data_paths.get('results').joinpath(filename), dpi=300)

        # Display the scatter plot
        plt.show()

    # apply k-means
    num_clusters = 2
    data_kmeans = df_features[col_mdhx_sleep_problem]
    df_kmeans = data_kmeans.copy()
    OMP_NUM_THREADS = 1
    kmeans = KMeans(n_clusters=num_clusters)  # choose the number of clusters
    kmeans.fit(data_kmeans)
    df_kmeans['cluster'] = kmeans.labels_

    # merge the ehr with the dem

    df_merged = pd.merge(
        left=df_features[['age', 'bmi', 'race']],
        right=df_kmeans,
        left_index=True,  # Use the index of the left DataFrame (df_features[['age', 'bmi', 'race']])
        right_index=True  # Use the index of the right DataFrame (df_kmeans)
    )

    df_merged.to_csv(f'asq_ehr_kmeans.csv', index=False)
    # statistical test on the demographics groups
    cluster_summary = df_merged.groupby('cluster').agg({
    'age': ['mean', 'std', 'count'],
    'bmi': ['mean', 'std', 'count']})

    table = tabulate(cluster_summary, headers='keys')
    print(table)


    # statistical test
    # Perform chi-square test of independence
    from scipy.stats import chi2_contingency
    contingency_table = pd.crosstab(df_merged['race'], df_merged['cluster'])
    table = tabulate(contingency_table, headers='keys')
    print(table)

    chi2, p_value, _, expected = chi2_contingency(contingency_table,
                                                  correction=True)
    print(f"Chi Square Exact Test:\n\tstatistic: {chi2}\n\tP-value: {p_value}")


    # t-test for continuous
    age_by_cluster = [df_merged[df_merged['cluster'] == cluster_id]['age'] for cluster_id in
                      df_merged['cluster'].unique()]
    # Perform one-way ANOVA to test for differences in age among clusters
    f_statistic, p_value = f_oneway(*age_by_cluster)
    print(f"Anova Test Results:\n\tstatistic: {f_statistic}\n\tP-value: {p_value}")

    bmi_by_cluster = [df_merged[df_merged['cluster'] == cluster_id]['bmi'] for cluster_id in
                      df_merged['cluster'].unique()]
    # Perform one-way ANOVA to test for differences in age among clusters
    f_statistic, p_value = f_oneway(*bmi_by_cluster)
    print(f"Anova Test Results:\n\tstatistic: {f_statistic}\n\tP-value: {p_value}")

    # visualize the cluster in 2D using the clusters
    feature1 = 'age'
    feature2 = 'bmi'
    plt.figure(figsize=(8, 6))
    for cluster_id in df_merged['cluster'].unique():
        cluster_data = df_merged[df_merged['cluster'] == cluster_id]
        plt.scatter(cluster_data[feature1], cluster_data[feature2], label=f'Cluster {cluster_id}', alpha=0.5)

    plt.title('Clusters in Two Dimensions (Original Features)')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




    # do PCA to visualize the cluster in 2D
    # Create a DataFrame for the two dimensions and code by cluster
    df_pca_with_cluster = pd.DataFrame(data=df, columns=['PCA 1', 'PCA 2'])
    df_pca_with_cluster['cluster'] = df_kmeans['cluster']
    pca.explained_variance_.round(2)
    # Create a scatter plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(f'PCA 1 ({pca.explained_variance_.round(2)[0]}%)', fontsize=15)
    ax.set_ylabel(f'PCA 2 ({pca.explained_variance_.round(2)[1]}%)', fontsize=15)
    ax.set_title('PCA Decomposition Two Factors \nCoded By Kmeans', fontsize=20)
    # Create targets and colors
    targets = df_kmeans['cluster'].unique()
    colors = [cmap(i) for i in np.linspace(0, 1, len(targets))]
    for target, color in zip(targets, colors):
        indices = df_pca_with_cluster['cluster'] == target
        ax.scatter(df_pca_with_cluster.loc[indices, 'PCA 1'],
                   df_pca_with_cluster.loc[indices, 'PCA 2'],
                   color=np.array([color]),
                   s=50)
    ax.legend(targets)
    ax.grid()
    plt.tight_layout()
    fig.savefig(data_paths.get('results').joinpath(f'fig_k_means_pca_plot.png'), dpi=300)
    plt.show()

    print('Dimension in each cluster:')
    for cluster_ in df_kmeans['cluster'].unique():
        df_cluster = df_kmeans.loc[df_kmeans['cluster'] == cluster_, :]
        print(f'Cluster {cluster_}: {df_cluster.shape[0]}')
    df_kmeans.to_excel(data_paths.get('results').joinpath('df_k_means.xlsx'), index=False)
    # %% analyse the clusters
    # continuous features
    sns.set(style="whitegrid")
    palette = sns.color_palette("Set3")[0:df_kmeans['cluster'].nunique()]
    continuous_features = col_scores
    continuous_features.extend(['age', 'bmi'])
    for continuous_feature_ in continuous_features:
        # Create box plot
        df_results, anova_results = hypothesis_testing_continuous(df_kmeans,
                                                                  continuous_column=continuous_feature_,
                                                                  # save_path=data_paths.get('results')
                                                                  )

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_kmeans,
                    x='cluster',
                    y=continuous_feature_,
                    hue='cluster',
                    palette=palette)

        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel(f'{continuous_feature_}', fontsize=12)
        # rounded_anova_results = {key: round(value, 5) for key, value in anova_results.items()}
        plt.title(f'Distribution of {continuous_feature_} Between The Groups\nAnova: {anova_results}', fontsize=14)
        plt.tight_layout()
        plt.savefig(data_paths.get('results').joinpath(f'fig_box_plot_{continuous_feature_}.png'), dpi=300)
        plt.show()

    # discrete features
    for discrete_feature in discrete_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df_kmeans, x=discrete_feature, hue='cluster', palette='Set3')
        plt.xlabel(discrete_feature, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(f'Distribution of {discrete_feature} by Cluster', fontsize=14)
        plt.legend(title='Cluster')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        severity_counts = df_raw_data['rls_severity'].value_counts()

        # Plot the bar plot
        severity_counts.plot(kind='bar')
        plt.title('Value Counts of rls_severity')
        plt.xlabel('Severity')
        plt.ylabel('Count')
        plt.show()

