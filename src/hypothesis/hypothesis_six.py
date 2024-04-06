"""
# Research Question: Risk Factors for Long COVID Sleep Disturbances

## Introduction
This document outlines the methodology and variables for a study investigating the risk factors associated with sleep disturbances in patients experiencing long COVID symptoms. The research aims to identify predictors contributing to insomnia and other sleep-related issues in individuals recovering from COVID-19.

## Data Source
The data for this study will be collected from the ASQ (Assessment of Sleep Questionnaire) and Electronic Medical Records (EMR).

## Predictor Variables
The following predictor variables will be analyzed:
- Hospitalization (yes/no)
- Infection Date
- Body Mass Index (BMI)
- Age
- Sex
- Race

### Other Considerations
Additional factors for consideration include:
- Vaccination Status
- Psychiatric Comorbidities
- Chronic Medical Diseases

## Outcome Variables
The study will assess sleep disturbances using scores from validated sleep scales and screens for other sleep symptoms,
including:
- Insomnia Severity Index (ISI)
- Epworth Sleepiness Scale (ESS)
- Restless Legs Syndrome (RLS) probability

### Variable Names in ASQ Dictionary
- ISI: ess_0900
- ESS: score
- RLS: rls_probability

#### Interpretation of Scores
- ISI: 0-7 = not clinically significant, 8-14 = subthreshold insomnia, 15-21 = moderate insomnia, 22-28 = severe insomnia
- ESS: 0-7 = unlikely that you are abnormally sleepy, 8-9 = average amount of daytime sleepiness, 10-15 = excessively
sleepy, 16-24 = excessively sleepy + seek medical attention
- RLS: Recoded as binary variable (0 = no RLS, 1 = yes RLS)

### Parasomnias
Questions of interest include:
- par_0205 (Sleepwalking)
- par_0305 (Sleep eating)
- par_0505 (Acting out dreams)
- par_0605 (Violent behavior during sleep)
- par_1005 (Sex with no memory)

#### Criteria for Parasomnia Identification
- If answer to all questions is "never" (-88) or "don't know" (-55), then no parasomnia
- If any other answer for parasomnia, the number represents frequency

### Sleep-related Breathing Disorders
Questions of interest include:
- Map_0100 (Loud snore)
- Map_0300 (Snorting/gasping)
- Map_0600 (Breathing stops)

#### Criteria for Identification
- If reports sometimes, frequently, or always (2-4) for any of the above, count as having some breathing symptoms

## Analysis
The analysis will involve multivariable regression to assess the relationship between predictor variables and sleep
disturbances among individuals with long COVID symptoms.
This markdown file serves as a comprehensive guide for conducting the research and analyzing the data on risk factors
for long COVID sleep disturbances.
"""
import pathlib
import pandas as pd
import statsmodels.api as sm
from config.data_paths import data_paths, multi_response_col
from library.utils import compute_sparse_encoding, FuzzySearch, compute_sparse_encoding
import seaborn as sns
import matplotlib.pyplot as plt
from config.columns_use import columns_interest, col_ehr
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS


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
    df_data = df_data[all_columns]
    df_data.dropna(inplace=True)
    df_data.reset_index(inplace=True,
                        drop=True)
    # Remove parentheses, commas, and equals signs from column names
    df_data.columns = [col.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '') for col
                       in df_data.columns]

    df_data.to_csv('data_multivariate_regression.csv', index=False)
    # %% select the column for this hypothesis
    columns_asq_ehr_interest = {**columns_interest, **col_ehr}
    df_x = df_data[col_predictors]
    df_y = df_data[outcome_vars]

    assert len(df_x) == len(df_y), "Number of rows in df_x and df_y are not equal"
    # %%
    model = _MultivariateOLS(endog=df_x, exog=df_y)
    result = model.fit()
    print(result.summary())













