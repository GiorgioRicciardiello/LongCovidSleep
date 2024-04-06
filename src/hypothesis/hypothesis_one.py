"""
### Research Question:
What is the prevalence of sleep disorders in the Long COVID Clinic, measured by validated scales?

#### Data Source:
ASQ

#### Variables of Interest:
- Scores from the validated sleep scales (ESS, ISI, RLS) and screens for other sleep symptoms

#### ASQ Dictionary Names for Variables of Interest:
- ESS: ess_0900
- ISI: score
- RLS: rls_probability

##### ISI Categories:
- 0-7: Not clinically significant
- 8-14: Subthreshold insomnia
- 15-21: Moderate insomnia
- 22-28: Severe insomnia

##### ESS Categories:
- 0-7: Unlikely that you are abnormally sleepy
- 8-9: Average amount of daytime sleepiness
- 10-15: Excessively sleepy
- 16-24: Excessively sleepy + seek medical attention

##### RLS Categories (Recode to Binary Variable):
- Unlikely, unlikely (possibly in past), possible: 0 (No RLS)
- Likely: 1 (Yes)

##### Parasomnias (Questions of Interest):
- par_0205 (Sleepwalking)
- par_0305 (Sleep eating)
- par_0505 (Acting out dreams)
- par_0605 (Violent behavior during sleep)
- par_1005 (Sex with no memory)

###### Parasomnia Criteria:
1. If answers to all questions above are "never" (-88) or "don't know" (-55), then no parasomnia.
2. If any other answer for parasomnia, then that number is the frequency.

##### Sleep-Related Breathing Disorders (Questions of Interest):
- Map_0100 (Loud snore)
- Map_0300 (Snorting/gasping)
- Map_0600 (Breathing stops)

###### Breathing Disorders Criteria:
1. If reports sometimes, frequently, or always (2-4) for any of the above, then count as having some breathing symptoms.

#### Analysis:
Frequencies, presented in a stacked bar or table.

"""
import pandas as pd
from config.data_paths import data_paths, multi_response_col
import numpy as np
from config.columns_use import columns_interest, col_ehr
import seaborn as sns
import matplotlib.pyplot as plt

def create_table_one(df:pd.DataFrame,
                     selected_columns:dict[str, str])-> pd.DataFrame:
    """
    Create the table 1 of a classic paper to present the data distribution and frequencies
    :param df:
    :return:
        pd.Dataframe, table one
    """
    col_tab_one = ['Variable', 'mean', 'std', 'Value', 'Frequency']
    table_one = pd.DataFrame(np.nan,
                             columns=col_tab_one,
                             index=range(0))
    for col, col_type in selected_columns.items():
        print(col)
        if isinstance(col_type, dict):
            df_value_counts = df[col].value_counts().reset_index()
            df_value_counts.columns = ['Value', 'Frequency']
            df_value_counts['Value'] = df_value_counts['Value'].map(col_type)
            df_value_counts['Variable'] = col
            table_one = pd.concat([table_one, df_value_counts])
        elif col_type == 'continuous':
            df_value_counts = {
                'mean': df[col].describe()['mean'],
                'std': df[col].describe()['std'],
                'Variable': col
            }
            df_value_counts = pd.DataFrame([df_value_counts], index=[0])
            table_one = pd.concat([table_one, df_value_counts])

        elif col_type == 'binary':
            df_value_counts = df[col].value_counts().reset_index()
            df_value_counts.columns = ['Value', 'Frequency']
            if col == 'gender':
                df_value_counts['Value'] = df_value_counts['Value'].map({0: 'Male',
                                                                         1: 'female'})
            else:
                df_value_counts['Value'] = df_value_counts['Value'].map({0: 'No',
                                                                         1: 'Yes'})
            df_value_counts['Variable'] = col
            table_one = pd.concat([table_one, df_value_counts])

    return table_one


if __name__ == "__main__":
    columns_hypothesis = [
        'age',
        'bmi',
        'ess_0900',
        'gender',
        'map_0100',
        'map_0300',
        'map_0600',
        'par_0205',
        'par_0305',
        'par_0505',
        'par_0605',
        'par_1005',
        'race',
        'rls_probability',
        'isi_score'
    ]
    # selected_columns = {col: columns_interest[col] for col in columns_hypothesis}
    # %% read the  data
    df_data = pd.read_csv(data_paths.get('pp_data').joinpath('pp_data.csv'))
    # %% select the column for this hypothesis
    col_features = list(columns_interest.keys())
    col_features.extend(list(col_ehr.keys()))
    df_features = df_data[col_features].copy()
    # combine the two feature sets
    columns_asq_ehr_interest = {**columns_interest, **col_ehr}
    # %% Table 1
    table_one = create_table_one(df=df_features, selected_columns=columns_asq_ehr_interest)
    table_one.to_excel(data_paths.get('results').joinpath('hyp_1_table_one_more_vars.xlsx'), index=False)
    # %% plots
    font_size = 12
    # Create a figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Plot histogram for age
    sns.histplot(df_features['age'], ax=axes[0])
    axes[0].set_title('Age Histogram', fontsize=font_size)
    axes[0].set_xlim(df_features['age'].min(), df_features['age'].max())
    axes[0].set_ylim(0, None)

    # Plot histogram for BMI
    sns.histplot(df_features['bmi'], ax=axes[1])
    axes[1].set_title('BMI Histogram', fontsize=font_size)
    axes[1].set_xlim(df_features['bmi'].min(), df_features['bmi'].max())
    axes[1].set_ylim(0, None)

    # Plot bar plot for gender
    sns.countplot(data=df_features, x='gender', ax=axes[2])
    axes[2].set_title('Gender Distribution', fontsize=font_size)
    axes[2].set_ylim(0, None)

    plt.tight_layout()
    plt.grid(.7)
    fig.savefig(data_paths.get('results').joinpath(f'fig_dem_distribution.png'), dpi=300)
    plt.show()












