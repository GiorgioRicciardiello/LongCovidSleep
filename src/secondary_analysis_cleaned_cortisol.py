"""
Would it be possible to re-analyze the data using the updated cohort N=87? The key demographic data (grouped into normal, high, low, and total) are:
- Median and SD Cortisol Level
- Median and SD Age
- Median and SD BMI
- Race
- Sex at Birth
- Diabetes
- Hypertension
- Prior use of Corticoids
- Current use of Corticoids
- Naltrexone
- Prednisone
- Functional Status (4-5)
 -Number of LC Symptoms
 -Median and SD ACTH

"""
import pandas as pd
import pathlib
from config.data_paths import data_paths
from library.table_one import MakeTableOne
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
from tabulate import tabulate
import re
from typing import Any, Union, Optional, Dict, List, Tuple
import statsmodels.api as sm
from scipy.stats import pearsonr


if __name__ == '__main__':
    df_cortisol_clean = pd.read_excel(data_paths.get('raw_data').get('cortisol_filtered'),
                                      sheet_name='sort df_cortisol.csv')

    # rows to drop because of time variation bias
    mrn_to_drop = [86184322, 83488239, 76811843]
    df_cortisol_clean = df_cortisol_clean.loc[~df_cortisol_clean['mrn'].isna(), :]
    df_cortisol_clean['mrn'] = df_cortisol_clean['mrn'].astype(int)
    df_cortisol_clean = df_cortisol_clean.loc[~df_cortisol_clean['mrn'].isin(mrn_to_drop), :]

    # From the ASQ we only need the following columns
    # col_asq = ['mrn']
    # col_asq = [col for col in df_asq.columns if col in [*alias.keys()] and not col in df_cortisol_clean.columns]
    # # Merge dataframes by 'MRN' and keep only rows in df_cortisol_clean
    # df_merged = df_asq.merge(df_cortisol_clean, on='mrn', how='inner')
    df_cortisol_clean.columns = df_cortisol_clean.columns.str.strip()
    # %% medication
    medication_series = (
        df_cortisol_clean['if so, what']
        .dropna()
        .str.split(',')
        .explode()
        .str.strip()
        .str.lower()  # Convert to lowercase for consistency
    )
    # Remove invalid entries (e.g., 'MEDIANS', 'High', 'Normal') if needed
    valid_medications = medication_series[~medication_series.isin(['medians', 'high', 'normal'])]

    # Count the occurrences of each medication
    medication_counts = valid_medications.value_counts()
    df_medication = medication_counts.to_frame().reset_index()
    df_medication.columns = ['Medication Usage', 'count']
    df_medication = df_medication[df_medication['Medication Usage'] != '']
    df_medication = df_medication.sort_values(by='count', ascending=False)
    df_medication = df_medication.sort_values(by='Medication Usage', ascending=True)

    # medication stratified by cortiol levels
    df_med_strata = df_cortisol_clean[['if so, what', 'cortisol_level_cat']].copy()
    exploded_medications = (
        df_med_strata.dropna(subset=['if so, what'])  # Drop NaN values in 'if so, what'
        .assign(**{'if so, what': df_med_strata['if so, what'].dropna().str.split(',')})  # Split by ','
        .explode('if so, what')  # Explode the list
        .assign(**{'if so, what': lambda df: df['if so, what'].str.strip().str.lower()})  # Clean strings
    )
    exploded_medications = exploded_medications.loc[(exploded_medications['if so, what'] != '') &
                                                    (~exploded_medications['cortisol_level_cat'].isna()), :]


    medication_counts_by_category = (
        exploded_medications.groupby('cortisol_level_cat')['if so, what']
        .value_counts(dropna=True)
        .reset_index(name='count')
    )

    medication_counts_pivot = (
        exploded_medications.groupby(['if so, what', 'cortisol_level_cat'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Rename columns for clarity
    medication_counts_pivot.columns.name = None
    medication_counts_pivot.rename(columns={'if so, what': 'Medication'}, inplace=True)


    # %% Table one
    var_continuous = ['dem_0110', 'cortisol_level']
    var_categorical = ['dem_1000', 'dem_0500',
                       'hypertension', 'diabetes',
                       'prior use of corticosteroids  (before abiliffy)',
                       'current use of corticosteroids', 'functional status',
                       'insomnia', 'lethargy', 'shortness_breath', 'unrefreshing_sleep','anx_depression',
                       'brain_fog', 'change_in_taste', 'change_in_smell'
                       ]

    make_table = MakeTableOne(df=df_cortisol_clean,
                              continuous_var=var_continuous,
                              categorical_var=var_categorical,
                              strata='cortisol_level_cat'
                              )
    tab_one = make_table.create_table()
    tab_one_clean = make_table.group_variables_table(tab_one)

    # %% correlation analysis
    df_corr = df_cortisol_clean.loc[~df_cortisol_clean['cortisol/ACTH'].isna(), ['cortisol_level',
                                                                                 'cortisol/ACTH',
                                                                                 # 'cortisol_level_cat'
                                                                                 ]]
    # Calculate correlation
    correlation, p_value = pearsonr(df_corr['cortisol_level'], df_corr['cortisol/ACTH'])

    # Perform simple linear regression
    X = sm.add_constant(df_corr['cortisol_level'])  # Add constant for intercept
    y = df_corr['cortisol/ACTH']
    model = sm.OLS(y, X).fit(cov_type='HC0')  # Fit the model
    regression_summary = model.summary()

    # Extract parameters and HC0 standard errors
    params_hc0 = model.params
    standard_errors_hc0 = model.bse

    # Display correlation and regression summary
    results = {
        "Correlation": correlation,
        "P-value": p_value,
        "Model (N)":  model.nobs,
        'Model (AIC)': model.aic,
        'Model (BIC)': model.bic,
        'Model (Rsquare)': model.rsquared_adj,
        "Params (coef): cortisol_level": params_hc0['cortisol_level'],
        "Params (coef): intercept": params_hc0['const'],
        "Params (std error): cortisol_level": standard_errors_hc0.values[1],
        "Params (std error): intercept": standard_errors_hc0.values[0],
    }
    df_results = pd.DataFrame(results, index=['values']).T
    df_results
    # Perform simple linear regression using the mock data
    X = sm.add_constant(df_corr['cortisol_level'])  # Add constant for intercept
    y = df_corr['cortisol/ACTH']
    model = sm.OLS(y, X).fit(cov_type='HC0')  # Fit the model

    # Extract regression parameters
    intercept = model.params['const']
    beta = model.params['cortisol_level']

    # Create scatter plot with regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(df_corr['cortisol_level'],
                df_corr['cortisol/ACTH'],
                label='Data points',
                alpha=0.7)
    plt.plot(
        df_corr['cortisol_level'],
        intercept + beta * df_corr['cortisol_level'],
        color='red',
        label=f'Regression line\nIntercept: {intercept:.2f}, Beta: {beta:.2f}'
    )
    plt.title('Scatter Plot with Regression Line')
    plt.xlabel('Cortisol Level')
    plt.ylabel('Cortisol/ACTH')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.grid(alpha=0.7)
    plt.show()
