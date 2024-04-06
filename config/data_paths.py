"""
Configuration file to keep the folder directories
"""

import pathlib
from pathlib import Path

root_path = pathlib.Path(__file__).resolve().parents[1]
raw_path = root_path.joinpath(pathlib.Path('data/raw_data'))
pp_path = root_path.joinpath(pathlib.Path('data/pp_data'))
results_path = root_path.joinpath(pathlib.Path('results'))


data_paths = {
    'raw_data': raw_path,
    'pp_data': pp_path,
    'results': results_path,
}


multi_response_col = list({
    # 'dem_1020',
    'bthbts_employment',
    'bthbts_sleep_disruption',

    'famhx_anxiety',
    'famhx_depression',
    'famhx_fibromyalgia',
    'famhx_insomnia',
    'famhx_narcolepsy',
    'famhx_other_sleep_disorder',
    'famhx_psych_illness',
    'famhx_psych_treatment',
    'famhx_rls',
    'famhx_sleep_apnea',
    'famhx_sleep_death',
    'famhx_sleepwalk',
    'mdhx_anxiety_problem',
    'mdhx_autoimmune_disease',
    'mdhx_cancer',
    'mdhx_cardio_problem',
    'mdhx_cardio_surgery',
    'mdhx_dental_work',
    'mdhx_eating_disorder',
    'mdhx_ent_problem',
    'mdhx_ent_surgery',
    'mdhx_gi_problem',
    'mdhx_headache_problem',
    'mdhx_hematological_disease',
    'mdhx_neurology_problem',
    'mdhx_orthodontics',
    'mdhx_other_problem',
    'mdhx_pain_fatigue_problem',
    'mdhx_pap_improvement',
    'mdhx_pap_problem',
    'mdhx_psych_problem',
    'mdhx_metabolic_endocrine_problem',
    'mdhx_pulmonary_problem',
    'mdhx_sleep_diagnosis',
    'mdhx_sleep_problem',
    'mdhx_sleep_treatment',
    'mdhx_urinary_kidney_problem',

    # 'med_0200',
    # 'med_0500'
    'soclhx_rec_drug',
    'soclhx_tobacco_use',

    'sched_alarm_clock_unused',
    'sched_alarm_clock_use',
    'sched_rotating_shift',
})