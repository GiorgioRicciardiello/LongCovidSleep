"""

Author: Giorgio Ricciardiello
email: giocrm@stanford.edu

Compute factor analysis with rotations and perform the plots of the factors, biplots and scatter. All in a single class.
Some reference where obtained from https://www.youtube.com/watch?v=jRjeC9Bre8M
"""

import pathlib

import pandas as pd
import numpy as np
import seaborn as sns
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from typing import Optional

class FactorAnalysisWrapper:
    def __init__(self, data: pd.DataFrame,
                 n_factors: int = 3,
                 rotation: str = 'varimax',
                 method: str = 'minres'):
        """
        Compute the factor analysis with rotation and create the factor loading dataframe and the features
        dataframe, so we do not have problem in confussing the dimentions or which one is which as it returns
        labeled data
        https://www.youtube.com/watch?v=jRjeC9Bre8M

        Features frames:
            rows: features
            col: original_eigen  common_factors_eigen  comunalities  uniqueness

        Lading frames:
            rows: factors,   factor_1  factor_2 ... factor_7  factor_N
            col: original_eigen  factor_1 ... factor_N, SS Loadings, Proportional Variances, Cumulative Variances
        :param data:pd.DataFrame, frame to apply the factor rotation
        :param n_factors: int, number of factors
        """
        if n_factors > data.shape[1]:
            raise ValueError(f'n_factors can not be greater than the number of features in the data')

        self.data = data
        self.n_factors = n_factors
        self.fa = FactorAnalyzer(n_factors,
                                 rotation=rotation,
                                 method=method)
        self.fa.fit(self.data)
        self.scores_df = None
        self.factors_df = None
        self.features_df = None
        self._features_lbls = self.data.columns.to_list()  # [f'feature_{i + 1}' for i in range(self.data.shape[1])]
        self._factors_lbls = [f'factor_{i + 1}' for i in range(self.fa.loadings_.shape[1])]
        self._set_factors_df()
        self._set_features_df()
        self._set_scores()

    def _set_factors_df(self):
        """
        Create the factor loading matrix
        Each feature should prove 1 std for the variance, there if we have 10 features, we should have a total
        variance of 10 in the model. For this assumption, we must normalize each feature so they have 1 std.

        ss_loadings: variance explained by each factor
        ss_loadings_perc: variance explained by each factor in percentage of the total of features (total var)
        proportional_var: explained variances, proportion of the total variance explained by each factor. They are
            ratio of the variance of each factor to the total variance in the dataset
        cumulative_variances: running total of the Proportional Variances. gives you the cumulative proportion of the
            total variance explained by the current and all preceding factors
        """
        loadings_df = pd.DataFrame(data=self.fa.loadings_,
                                   index=[feat for feat in self.data.columns],
                                   columns=self._factors_lbls)
        ss_loadings, prop_variances, cum_variances = self.fa.get_factor_variance()
        # var, prop_var, cum_var = self.fa.get_factor_variance()
        # print("Factor Variance:\n", var)
        # print("Proportional Variance:\n", prop_var)
        # print("Cumulative Variance:\n", cum_var)

        factor_variance_df = pd.DataFrame(data={
            'ss_loadings': ss_loadings,
            'ss_loadings_perc': (ss_loadings * 100) / self.data.shape[1],
            'proportional_var': prop_variances,
            # 'proportional_var_perc': (prop_variances * 100) / self.data.shape[1],
            'cumulative_variances': cum_variances,
            # 'cumulative_variances_perc': (cum_variances * 100) / self.data.shape[1]
        },
            index=self._factors_lbls).T
        nan_df = pd.DataFrame(np.nan, index=['------'], columns=loadings_df.columns)
        self.factors_df = pd.concat([loadings_df, nan_df, factor_variance_df])

    def _set_features_df(self):
        """Create the feature matrix"""
        communalities_df = pd.DataFrame(data=self.fa.get_communalities(),
                                        columns=['comunalities'],
                                        index=self._features_lbls)
        eigen_original, eigen_common_factors = self.fa.get_eigenvalues()
        eigenvalues_df = pd.DataFrame({'original_eigen': eigen_original,
                                       'common_factors_eigen': eigen_common_factors},
                                      index=self._features_lbls)
        uniqueness_df = pd.DataFrame(data=self.fa.get_uniquenesses(),
                                     columns=['uniqueness'],
                                     index=self._features_lbls)
        for frame in [eigenvalues_df, communalities_df, uniqueness_df]:
            frame.reset_index(inplace=True, drop=False)
            frame.rename(columns={'index': 'features'}, inplace=True)
        self.features_df = eigenvalues_df.merge(communalities_df,
                                                on='features',
                                                how='inner').merge(uniqueness_df,
                                                                   on='features',
                                                                   how='inner')
        self.features_df = self.features_df.round(3)

    def _set_scores(self):
        """Compute the factor scores DataFrame (n_obs x n_factors)"""
        factor_scores = self.fa.transform(self.data)
        self.scores_df = pd.DataFrame(factor_scores,
                                      columns=self._factors_lbls,
                                      index=self.data.index)

    def get_factors_df(self) -> pd.DataFrame:
        return self.factors_df

    def get_features_df(self) -> pd.DataFrame:
        return self.features_df

    def get_scores(self) -> pd.DataFrame:
        return self.scores_df

    def plot_factors_features(self):
        """
        Subplot where the axes are the factors and the markers are color-coded by the features.
        The factors_df contains this information
        """
        plot_factors_df = self.factors_df.iloc[0:self.data.shape[1]].copy()
        plot_factors_df.reset_index(inplace=True,
                                    drop=False)
        plot_factors_df.rename(columns={'index': 'features'},
                               inplace=True)
        # Initialize a grid of subplots
        g = sns.PairGrid(plot_factors_df,
                         vars=self._factors_lbls,
                         hue='features',
                         palette='Set2')
        g.map(plt.scatter, alpha=0.8)
        g.add_legend()
        # Add grid to each subplot
        for ax in g.axes.flatten():
            ax.grid(True)
            ax.set_axisbelow(True)
        plt.show()

    # def plot_radar(self):
    #     """
    #     Plot radar charts for each factor.
    #     """
    #     # Calculate factor means
    #     factor_means = self.scores_df.mean()
    #
    #     # Number of variables
    #     num_vars = len(factor_means)
    #
    #     # Create a radar chart for each factor
    #     for factor in self._factors_lbls:
    #         # Factor scores for the current factor
    #         factor_scores = self.scores_df[factor]
    #
    #         # Create a figure and a set of subplots
    #         fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    #
    #         # Variables for radar chart
    #         angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    #
    #         # The plot is circular, so we need to "complete the loop" and append the start to the end.
    #         factor_scores = np.concatenate((factor_scores,[factor_scores[0]]))
    #         angles += angles[:1]
    #
    #         # Plot data
    #         ax.plot(angles, factor_scores, linewidth=1, linestyle='solid', label=factor)
    #         ax.fill(angles, factor_scores, alpha=0.25)
    #
    #         # Add labels
    #         ax.set_yticklabels([])
    #         ax.set_xticks(angles[:-1])
    #         ax.set_xticklabels(self.data.columns)
    #         ax.set_title(factor)
    #
    #         # Show the plot
    #         plt.show()
    #
    # def plot_varimax_loadings_heatmap(self):
    #     """
    #     Plot Factor Loading Plot after Varimax rotation.
    #     """
    #     loadings = self.fa.loadings_
    #     loadings_df = pd.DataFrame(loadings, columns=self._factors_lbls, index=self.data.columns)
    #
    #     # Apply Varimax rotation to the loadings
    #     rotated_loadings = self.fa.loadings_
    #     # You may need to adjust the method to apply Varimax rotation correctly based on the library you are using.
    #
    #     # Create a figure and axes
    #     plt.figure(figsize=(10, 6))
    #
    #     # Plot loadings
    #     sns.heatmap(rotated_loadings, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    #
    #     # Customize plot
    #     plt.title('Factor Loading Plot after Varimax Rotation')
    #     plt.xlabel('Factors')
    #     plt.ylabel('Variables')
    #
    #     # Show plot
    #     plt.show()

    # def plot_euclidean_loadings(self):
    #     """
    #     Plot Factor Loading Plot in Euclidean space with magnitude and angle.
    #     """
    #     loadings = self.fa.loadings_
    #     loadings_df = pd.DataFrame(loadings, columns=self._factors_lbls, index=self.data.columns)
    #
    #     # Compute magnitude and angle
    #     magnitude = np.sqrt(np.sum(np.square(loadings), axis=1))
    #     angle = np.arctan2(loadings, loadings.max(axis=1, keepdims=True))
    #
    #     # Create a scatter plot
    #     plt.figure(figsize=(10, 6))
    #
    #     for i in range(loadings.shape[1]):
    #         plt.scatter(loadings[:, i], self._factors_lbls, s=100*magnitude, c=angle[:, i], cmap='hsv', alpha=0.6)
    #
    #     # Customize plot
    #     plt.title('Factor Loading Plot in Euclidean Space')
    #     plt.xlabel('Factor Loadings')
    #     plt.ylabel('Factors')
    #
    #     # Add colorbar
    #     cbar = plt.colorbar()
    #     cbar.set_label('Angle')
    #
    #     # Show plot
    #     plt.show()

    def plot_biplot(self,
                    folder_path:Optional[pathlib.Path] = None):
        """
        Create a biplot. This is the plot we see on SAS.

        In this biplot method, I first compute the factor scores for each observation using the transform method of
        FactorAnalyzer. Then, I create a scatter plot where each factor loading is represented by an arrow pointing
        in the direction of the factor. I also label each arrow with the corresponding factor name. Additionally,
        I plot each observation in the space defined by the first two factors, and label them with the observation
        names.

        https://statisticsglobe.com/biplot-pca-python
        """
        fontsize = {
            'lbls': 20,
            'text': 10,
        }
        loadings_df = self.factors_df.loc[self.factors_df.index.isin(self._features_lbls), :]

        # Compute factor scores for observations
        # factor_scores = self.fa.transform(self.data)
        factor_scores = self.scores_df

        # plot the factors 1 vs the rest
        for factor_ in self._factors_lbls[1::]:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot unitary circle
            circle = plt.Circle((0, 0), 1, color='gray', fill=False)
            ax.add_artist(circle)

            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)

            ax.axhline(0,
                       color='gray',
                       linestyle='--',
                       alpha=0.7)
            ax.axvline(0, color='gray',
                       linestyle='--',
                       alpha=0.7)

            scalePC1 = 1.0 / (factor_scores.loc[:, 'factor_1'].max() - factor_scores.loc[:, 'factor_1'].min())
            scalePCi = 1.0 / (factor_scores.loc[:, f'{factor_}'].max() - factor_scores.loc[:,f'{factor_}'].min())

            ax.scatter(factor_scores.loc[:, 'factor_1'] * scalePC1,
                       factor_scores.loc[:, f'{factor_}'] * scalePCi,
                       alpha=0.5
                       )

            for i, feature in enumerate(self._features_lbls, start=1):
                ax.arrow(x=0,
                         y=0,
                         dx=loadings_df.at[feature, 'factor_1'],  # length of the arrow along x and y direction.
                         dy=loadings_df.at[feature, f'{factor_}'],
                         # head_width=0.03,
                         # head_length=0.03,
                         )
                # Label arrows with feature names
                ax.annotate(feature,
                            xy=(loadings_df.at[feature, 'factor_1'], loadings_df.at[feature, f'{factor_}']),
                            xytext=(
                            loadings_df.at[feature, 'factor_1'] * 1.15, loadings_df.at[feature, f'{factor_}'] * 1.15),
                            fontsize=fontsize.get('text'))

            ax.set_xlabel('factor 1', fontsize=20)
            ax.set_xlabel(f'factor 1 ({self.factors_df.at["ss_loadings_perc", "factor_1"].round(2)} %)',
                          fontsize=fontsize.get('lbls'))
            lbl = factor_.replace('_', ' ')
            ax.set_ylabel(f'{lbl} ({self.factors_df.at["ss_loadings_perc", factor_].round(2)} %)',
                          fontsize=fontsize.get('lbls'))
            ax.set_title('Biplot', fontsize=20)
            plt.tight_layout()
            plt.grid(alpha=.9)
            if folder_path is not None:
                plt.savefig(folder_path.joinpath(f'fig_fca_biplot_f1_{factor_}.png'), dpi=300)
            plt.show()

    def plot_biplot_target_coded(self,
                                 target:pd.Series,
                                 folder_path:Optional[pathlib.Path]=None,
                                 target_lbls:Optional[dict]=None,
                                 txt_fontsize:int=0):
        """
        Generate a biplot for the factor analysis and save the figure in the folder path
        :param target: pd.Series, target to color code the scatter plot
        :param target_lbls: dict, labels of the targets if classifier
        :param folder_path:
        :param txt_fontsize: int, font for the features arrows, if zero, the text is not shown
        :return:
        """

        if target_lbls is None:
            target_lbls = {f'class_{i}': i for i in range(0, 6)}

        if target.nunique() > 6:
            # generate quartiles if the target is continuous
            target = np.digitize(target,np.quantile(target,[1 / 3, 2 / 3]))

        if len(np.unique(target)) > 10:
            print('Please use a smaller coding for the target')
            return -1

        loadings_df = self.factors_df.loc[self.factors_df.index.isin(self._features_lbls), :]
        factor_scores = self.scores_df

        # plot the factors 1 vs the rest
        for factor_ in self._factors_lbls[1::]:
            fig, ax = plt.subplots(figsize=(16, 12))
            scalePC1 = 1.0 / (factor_scores.loc[:, 'factor_1'].max() - factor_scores.loc[:, 'factor_1'].min())
            scalePCi = 1.0 / (factor_scores.loc[:, f'{factor_}'].max() - factor_scores.loc[:, f'{factor_}'].min())
            scatter = ax.scatter(factor_scores.loc[:, 'factor_1'] * scalePC1,
                       factor_scores.loc[:, f'{factor_}'] * scalePCi,
                       alpha=0.5,
                       c=target,
                       cmap='tab10')

            ax.axhline(0, color='gray')
            ax.axvline(0, color='gray')
            for i, feature in enumerate(self._features_lbls, start=1):
                ax.arrow(x=0,
                         y=0,
                         dx=loadings_df.loc[feature, 'factor_1'],  # length of the arrow along x and y direction.
                         dy=loadings_df.loc[feature, f'{factor_}'],
                         # head_width=0.03,
                         # head_length=0.03,
                         )
                if txt_fontsize > 0:
                    ax.text(loadings_df.loc[feature, 'factor_1'] * 1.15,
                            loadings_df.loc[feature, f'{factor_}'] * 1.15,
                            feature, fontsize=txt_fontsize)

                # if we want to plot the observation index, the un-comment this
                # for i, label in enumerate(factor_scores.index):
                #     ax.text(factor_scores.loc[:, 'factor_1'] * scalePC1,
                #             factor_scores.loc[:, f'{factor_}'] * scalePCi, str(label),
                #             fontsize=10)

            handles, _ = scatter.legend_elements()
            ax.legend(handles,
                      target_lbls,  # *scatter.legend_elements(),
                      loc="lower left",
                      title="Groups")
            ax.set_xlabel('factor_1', fontsize=20)
            ax.set_ylabel(f'{factor_}', fontsize=20)
            ax.set_title('Biplot', fontsize=20)
            plt.tight_layout()
            plt.grid(alpha=.7)
            if folder_path is not None:
                plt.savefig(folder_path.joinpath(f'biplot_f1_{factor_}.png'),
                            dpi=300)
            plt.show()

    def plot_heatmap(self, save_path:Optional[pathlib.Path] = None,
                     figsize:Optional[tuple]=(10, 27)):
        """
        https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
        Generate a heatmap to show the factor loadings
        :param save_path: path of the file
        :param figsize: figure size
        :return:
        """
        loadings_df = self.factors_df.loc[self.factors_df.index.isin(self._features_lbls), :]

        plt.figure(figsize=figsize)  # Adjust size according to your necessity
        # sns.set(font_scale=1.2)  # Increase the font size
        ax = sns.heatmap(loadings_df,
                         annot=True,
                         fmt=".2f",
                         cmap='coolwarm',
                         cbar_kws={'shrink': .5}, )  # Make colorbar 50% smaller
        y_to_num = {p[1]: p[0] for p in enumerate(loadings_df.index.to_list())}
        ax.set_yticks([y_to_num[v] for v in loadings_df.index.to_list()])
        ax.set_yticklabels(loadings_df.index.to_list())

        plt.title('Factor Loadings Heatmap')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path,
                        dpi=300)
        plt.show()

