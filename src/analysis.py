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

def optimize_sparse_pca(df: pd.DataFrame,
                        columns: list,
                        alpha_range: tuple = (0.01, 1.0),
                        ridge_alpha_range: tuple = (0.01, 1.0),
                        random_state: int = 42):
    """
    Optimizie the sparse PCA algorithm to fin the best paramaters that results in the highest summed variances
    of the components.
    :param df:
    :param columns:
    :param alpha_range:
    :param ridge_alpha_range:
    :param random_state:
    :return:
    """

    def custom_explained_variance_scorer(estimator, X):
        """
        Custom scorer to compute the explained variance ratio directly from the Sparse PCA model.
        """
        explained_variance_ratio = np.var(estimator.components_, axis=1)
        return sum(explained_variance_ratio)

    # Define parameter grid
    param_grid = {
        'alpha': np.linspace(alpha_range[0], alpha_range[1], 5),  # 5 values between alpha_range
        'ridge_alpha': np.linspace(ridge_alpha_range[0], ridge_alpha_range[1], 5)
        # 5 values between ridge_alpha_range
    }

    # Initialize SparsePCA object
    sparse_pca = SparsePCA(n_components=2,
                           random_state=random_state)

    # Perform grid search
    grid_search = GridSearchCV(estimator=sparse_pca,
                               param_grid=param_grid,
                               cv=5,
                               scoring=custom_explained_variance_scorer)
    grid_search.fit(df[columns])

    # Get the best parameters
    best_params = grid_search.best_params_

    # Initialize SparsePCA with best parameters
    best_sparse_pca = SparsePCA(n_components=2,
                                alpha=best_params.get('alpha'),
                                ridge_alpha=best_params.get('ridge_alpha'),
                                random_state=random_state)

    # Fit Sparse PCA with the best parameters
    best_sparse_pca.fit(df[columns])

    # Compute explained variance ratio for each component
    variance_explained = np.var(best_sparse_pca.components_, axis=1)

    # Print the variance explained by each component
    for i, var in enumerate(variance_explained):
        print(f"Variance explained by Sparse PCA component {i + 1}: {var:.2f}")

    return best_sparse_pca, explained_variance_ratio, best_params


def hypothesis_testing_continuous(df: pd.DataFrame,
                                  continuous_column: str,
                                  save_path:Optional[pathlib.Path] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Perform hypothesis testing on continuous variables
    :param df:
    :param continuous_column:
    :param save_path: save path of the tables
    :return:
        t-test and kruskal pair results and the anova result
    """
    print(f'\n\nHypothesis testing evaluation for {continuous_column}___________________________________')
    # Create clusters
    clusters = {}
    for i, (name, group) in enumerate(df.groupby('cluster')[continuous_column]):
        clusters[f'cluster_{i + 1}'] = group

    # Perform ANOVA
    f_statistic, p_value_anova = f_oneway(*clusters.values())
    anova_result = {
        'f_statistic': f_statistic,
        'p_value': p_value_anova
    }
    # Perform t-tests and Kruskal-Wallis tests
    hypothesis_tests = {
        't-test': {},
        'kruskal': {},
    }

    for cluster_name_1, cluster_name_2 in itertools.combinations(clusters.keys(), 2):
        # Get data for the two clusters
        cluster_data_1 = clusters[cluster_name_1]
        cluster_data_2 = clusters[cluster_name_2]

        # Compute t-test results
        t_statistic, p_value_ttest = ttest_ind(cluster_data_1, cluster_data_2)

        # Compute Kruskal-Wallis test results
        h_statistic, p_value_kruskal = kruskal(cluster_data_1, cluster_data_2)

        # Store results
        hypothesis_tests['t-test'][f'{cluster_name_1}_vs_{cluster_name_2}'] = {
            't_statistic': t_statistic,
            'p_value': p_value_ttest,
            'sample_size': (len(cluster_data_1), len(cluster_data_2))
        }

        hypothesis_tests['kruskal'][f'{cluster_name_1}_vs_{cluster_name_2}'] = {
            'h_statistic': h_statistic,
            'p_value': p_value_kruskal,
            'sample_size': (len(cluster_data_1), len(cluster_data_2))
        }

    # Save results
    df_results = pd.concat({k: pd.DataFrame.from_dict(v, 'index') for k, v in hypothesis_tests.items()}, axis=1)
    if save_path:
        df_results.to_excel(save_path.joinpath(f'hypothesis_testing_{continuous_column}.xlsx'),
                            index=True)
        with open(save_path.joinpath(f'hypothesis_testing_anova_{continuous_column}'), 'w') as json_file:
            json.dump(anova_result, json_file)

    # print results
    print("ANOVA Results:")
    print(f"\tF-statistic: {f_statistic}")
    print(f"\tP-value: {p_value_anova}")
    table = tabulate(df_results, headers='keys', tablefmt='pretty')
    # Print the tabular format
    print(table)
    return df_results, anova_result

if __name__ == "__main__":
    do_analysis = {
        'plot_dist': True,
        'tnse': True,
        'fca': True,
        'k_means': True,
    }

    # read the raw data
    df_raw_data = pd.read_excel(data_paths.get('raw_data').joinpath('results_data_pull.xlsx'))


    # TODO: remove fosq, keep isi
    # get pre-process data
    df_data = pd.read_csv(data_paths.get('pp_data').joinpath('pp_data.csv'))
    # columns of interest
    column_interest = [
        "irls_score",
        "isi_0100",
        "isi_0200",
        "isi_0300",
        "isi_0400",
        "isi_0500",
        "isi_0600",
        "isi_0700",
        "isi_score",
        "ess_0100",
        "ess_0200",
        "ess_0300",
        "ess_0400",
        "ess_0500",
        "ess_0600",
        "ess_0700",
        "ess_0800",
        "ess_0900",
        # "irls_score",  # removed because many missing values
        # "rls_0100",
        # "rls_0200",
        # "rls_0300",
        # "rls_0310",
        # "rls_0400",
        # "rls_0410",
        # "rls_0500",
        # "rls_0510",
        # "rls_0600",
        # "rls_0610",
        # "rls_0700",
        # "rls_0710",
        # "rls_0800",
        # "rls_0801",
        # "rls_0850",
        # "rls_0851",
        # "rls_0900",
        # "rls_0910",
        # "rls_1000",
        # "rls_1100",
        # "rls_1200",
        # "rls_1300",
        # "rls_1400",
        # "rls_1500",
        # "rls_1600",
        # "rls_1700",
        # "rls_probability",
        "rls_severity",
        "fosq_0100",
        "fosq_0200",
        "fosq_0300",
        "fosq_0400",
        "fosq_0500",
        "fosq_0600",
        "fosq_0700",
        "fosq_0800",
        "fosq_0900",
        "fosq_1000",
        "fosq_1100"]

    identifiers = [
        # 'survey_id',
        'age',
        'gender',
        'bmi',
        'race',
    ]

    comorbidities = [
        'mdhx_5700',
        'mdhx_5720',
        'mdhx_5810',
        'mdhx_5800',
        'mdhx_6230',
        'mdhx_6310',

    ]

    sleep_related = [
        'bthbts_sleep_disruption_nan',
        'bthbts_sleep_disruption_0',
        'bthbts_sleep_disruption_1',
        'bthbts_sleep_disruption_2',
        'bthbts_sleep_disruption_3',
        'bthbts_sleep_disruption_4',
        'bthbts_sleep_disruption_5',
        'bthbts_sleep_disruption_6',
        'bthbts_sleep_disruption_7',
        'bthbts_sleep_disruption_8',
        'bthbts_sleep_disruption_9',
        'bthbts_sleep_disruption_10',
        'bthbts_sleep_disruption_11',
    ]

    col_scores = ['ess_0900', 'fosq_1100', 'rls_severity', 'isi_score']

    columns = identifiers + column_interest + sleep_related
    df_subset = df_data[columns].copy()

    # %%
    df_subset['rls_severity'].fillna(0, inplace=True)
    df_subset.replace(-44, 0, inplace=True)  # not applicable can be understood as zero
    df_subset.drop(columns=['irls_score'],
                   inplace=True)
    # %% dataset distribution
    if do_analysis.get('plot_dist'):
        font_size = 12
        # Create a figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

        # Plot histogram for age
        sns.histplot(df_subset['age'], ax=axes[0])
        axes[0].set_title('Age Histogram', fontsize=font_size)
        axes[0].set_xlim(df_subset['age'].min(), df_subset['age'].max())
        axes[0].set_ylim(0, None)

        # Plot histogram for BMI
        sns.histplot(df_subset['bmi'], ax=axes[1])
        axes[1].set_title('BMI Histogram', fontsize=font_size)
        axes[1].set_xlim(df_subset['bmi'].min(), df_subset['bmi'].max())
        axes[1].set_ylim(0, None)

        # Plot bar plot for gender
        sns.countplot(data=df_subset, x='gender', ax=axes[2])
        axes[2].set_title('Gender Distribution', fontsize=font_size)
        axes[2].set_ylim(0, None)

        plt.tight_layout()
        plt.grid(.7)
        fig.savefig(data_paths.get('results').joinpath(f'fig_dem_distribution.png'), dpi=300)

        plt.show()
    # %% Research Question 3 :
    # Do specific sleep disturbances tend to co-occur?
    sleep_related_sum = df_subset[sleep_related].sum(axis=0)
    # Convert to DataFrame for Seaborn plotting and sort by sum
    sleep_related_sum_df = pd.DataFrame({'Column Names': sleep_related_sum.index, 'Sum': sleep_related_sum.values})
    sleep_related_sum_df = sleep_related_sum_df.sort_values(by='Sum', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=sleep_related_sum_df, x='Column Names', y='Sum', color='skyblue')
    plt.title('Sum of Sleep-related Columns')
    plt.xlabel('Column Names')
    plt.ylabel('Sum')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(data_paths.get('results').joinpath(f'fig_sleep_disturbance_response.png'), dpi=300)
    plt.show()

    # Plot histograms for each score column
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    for i, col in enumerate(col_scores):
        sns.histplot(df_subset[col], ax=axes[i], kde=True)
        axes[i].set_title(f'Histogram of {col}', fontsize=12)
        axes[i].set_xlabel('Score', fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
    plt.tight_layout()
    plt.savefig(data_paths.get('results').joinpath(f'fig_scores_disturbance_response.png'), dpi=300)
    plt.show()

    #%% TSNE
    if do_analysis.get('tsn'):
        df_stne = df_subset.copy()
        # Normalize the data
        normalized_df = (df_stne - df_stne.mean()) / df_stne.std()
        tsne = TSNE(n_components=2,
                    random_state=42)
        tsne_results = tsne.fit_transform(normalized_df)

        # Create a DataFrame for plotting
        df_tsne_results = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])

        # Plot t-SNE results
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_tsne_results, x='tsne1', y='tsne2')
        plt.title('t-SNE Visualization of the dataset')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(alpha=.7)
        plt.show()

        # Merge t-SNE results with sleep-related columns using indexes    #
        df_tsne_results_sleep_dist = pd.concat([df_tsne_results, df_subset[sleep_related]], axis=1)

        for sleep_question_ in sleep_related:
            # sleep_question_ = sleep_related[0]
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df_tsne_results_sleep_dist,
                            x='tsne1',
                            y='tsne2',
                            hue=sleep_question_)
            plt.title(f't-SNE Visualization of the dataset - Coded by {sleep_question_}')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.grid(alpha=.7)
            plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_tsne_results_sleep_dist,
                        x='tsne1',
                        y='tsne2',
                        hue=df_subset.gender)
        plt.title(f't-SNE Visualization of the dataset - Coded by gender')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(alpha=.7)
        plt.show()

    #%% Factor analysis
    if do_analysis.get('fca'):
        factor_analysis = FactorAnalysisWrapper(data=df_subset,
                                                n_factors=6,
                                                rotation='varimax')
        # factor_analysis.plot_factors_features()
        factor_analysis.plot_biplot(folder_path=data_paths.get('results'))
        factor_analysis.plot_heatmap(save_path=data_paths.get('results').joinpath(f'fig_fca_heat_map.png'))
        df_factors:pd.DataFrame = factor_analysis.factors_df
        df_factors.to_excel(data_paths.get('results').joinpath(f'df_fca_factors.xlsx'))

    # %% K mean clustering
    if do_analysis.get('k_means'):
        sparse_pca = False
        cmap = cm.get_cmap('viridis')
        # k means on the subset dataset
        num_clusters = 3
        df_kmeans = df_subset.copy()
        OMP_NUM_THREADS = 1
        kmeans = KMeans(n_clusters=num_clusters)  # choose the number of clusters
        kmeans.fit(df_subset)
        df_kmeans['cluster'] = kmeans.labels_

        # Group by clusters
        df_grouped = df_kmeans.groupby('cluster')
        df_grouped.describe()

        # compute pca to visualize the clusters in 2 dimensions
        df_pca = df_subset.copy()
        # normalize all columns expect binary ones
        bin_columns = [col for col in df_subset.columns if df_subset[col].isin([0, 1]).all()]
        # Normalize the data
        for column in df_pca.columns:
            if column not in bin_columns:
                df_pca[column] = (df_pca[column] - df_pca[column].mean()) / df_pca[column].std()

        if sparse_pca:
            # Compute Sparse PCA on binary columns
            # find best params
            best_sparse_pca, explained_variance_ratio, best_params = optimize_sparse_pca(df=df_pca, columns=bin_columns)
            print("Best parameters:", best_params)
            print("Explained variance ratio:", explained_variance_ratio)
            # do sparse PCA with the best params
            sparse_pca = SparsePCA(n_components=2,
                                        alpha=best_params.get('alpha'),
                                        ridge_alpha=best_params.get('ridge_alpha'),
                                        random_state=42)

            sparse_pca.fit(df_pca[bin_columns])
            sparse_pca_components = sparse_pca.transform(df_pca[bin_columns])

            sparse_pca_df = pd.DataFrame(data=sparse_pca.components_.T,
                                         columns=['SparsePCA_1', 'SparsePCA_2'],
                                         index=bin_columns)

            # Access explained variance ratio
            sparse_components = sparse_pca.components_
            # Compute the variance explained by each component
            variance_explained = np.var(sparse_components, axis=1)
            # Print the variance explained by each component
            for i, var in enumerate(variance_explained):
                print(f"Variance explained by Sparse PCA component {i + 1}: {var:.2f}")
            # visualization
            plt.figure(figsize=(8, 6))
            sns.heatmap(sparse_pca_df, cmap='coolwarm', annot=True, fmt=".2f")
            plt.title('Sparse PCA Components Heatmap (Binary Columns)', fontsize=20)
            plt.xlabel('Component', fontsize=15)
            plt.ylabel('Binary Column', fontsize=15)
            plt.tight_layout()
            plt.show()

        # do PCA to visualize the cluster in 2D
        pca = PCA(2)
        df = pca.fit_transform(df_pca)

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
        #%% analyse the clusters
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
