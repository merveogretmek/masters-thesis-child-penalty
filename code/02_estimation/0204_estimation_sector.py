# 0204 Estimation - Sector (by Gender Ratio)

## Libraries

import numpy as np
import pandas as pd
import statsmodels.api as sm

## Import the Data

data = pd.read_csv("../../data/cleaned/cleaned_data.csv", index_col=0)

# Remove unobserved sectors
data = data[(data['p_nace'] >= 0)] 

## Categorize into Sectors

# Determine if the sector is 'male' or 'female' dominated
sector_gender_counts = data.groupby(['p_nace', 'sex']).size().unstack(fill_value=0)
sector_gender_counts.columns = ['male_count', 'female_count']
sector_gender_counts['total'] = sector_gender_counts['male_count'] + sector_gender_counts['female_count']
sector_gender_counts['male_percentage'] = sector_gender_counts['male_count'] / sector_gender_counts['total'] * 100

# Categorize sectors based on the male percentage
def categorize_sector(row):
    if row['male_percentage'] > 60:
        return 'Male-Dominated'
    elif row['male_percentage'] < 40:
        return 'Female-Dominated'
    elif (row['male_percentage'] <= 60) & (row['male_percentage'] >= 40):
        return 'Equal Share'
    
sector_gender_counts['sector_gender_ratio'] = sector_gender_counts.apply(categorize_sector, axis=1)
sector_gender_counts = sector_gender_counts.reset_index()

# Merge the categorized sector data back to the original dataframe based on 'sector' column
data = data.merge(sector_gender_counts[['p_nace', 'sector_gender_ratio']], on='p_nace', how='left')

## Run the Analysis with Income Approach

# Remove unobserved rows
ia_data = data[(data['real_income'] >= 0)] 

### Male-Dominated Sectors

# Subset the data to keep individuals only in male-dominated sectors
male_sector_ia = ia_data[ia_data['sector_gender_ratio'] == 'Male-Dominated']
male_sector_ia = male_sector_ia.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data = male_sector_ia[male_sector_ia['sex'] == 2]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns = {eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(male_sector_ia[male_sector_ia['sex'] == 2]['age'].min())
max_age = int(male_sector_ia[male_sector_ia['sex'] == 2]['age'].max())
min_year = int(male_sector_ia[male_sector_ia['sex'] == 2]['syear'].min())
max_year = int(male_sector_ia[male_sector_ia['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        male_sector_ia.loc[male_sector_ia['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        male_sector_ia.loc[male_sector_ia['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

male_sector_ia['pred_year_w'] = male_sector_ia['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
male_sector_ia['pred_age_w'] = pd.to_numeric(male_sector_ia['pred_age_w'], errors='coerce')
male_sector_ia['pred_year_w'] = pd.to_numeric(male_sector_ia['pred_year_w'], errors='coerce')
male_sector_ia['pred_w'] = male_sector_ia['pred_age_w'] + male_sector_ia['pred_year_w']
male_sector_ia.loc[male_sector_ia['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = male_sector_ia.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data=male_sector_ia[male_sector_ia['sex'] == 1]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns={eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(male_sector_ia[male_sector_ia['sex'] == 1]['age'].min())
max_age = int(male_sector_ia[male_sector_ia['sex'] == 1]['age'].max())
min_year = int(male_sector_ia[male_sector_ia['sex'] == 1]['syear'].min())
max_year = int(male_sector_ia[male_sector_ia['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        male_sector_ia.loc[male_sector_ia['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        male_sector_ia.loc[male_sector_ia['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

male_sector_ia['pred_year_m'] = male_sector_ia['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
male_sector_ia['pred_age_m'] = pd.to_numeric(male_sector_ia['pred_age_m'], errors='coerce')
male_sector_ia['pred_year_m'] = pd.to_numeric(male_sector_ia['pred_year_m'], errors='coerce')
male_sector_ia['pred_m'] = male_sector_ia['pred_age_m'] + male_sector_ia['pred_year_m']
male_sector_ia.loc[male_sector_ia['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = male_sector_ia.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for men
results_m_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_m_df['coef'] = results_m_df['coef'].astype(float)
results_m_df['percentage_coef_m'] = results_m_df['coef']/results_m_df['pred_m'] 

#### Combine the Results

# Merge results for men and women
event_study_results = pd.merge(results_m_df[['event_time', 'percentage_coef_m']],
                               results_w_df[['event_time', 'percentage_coef_w']],
                               on = 'event_time')
event_study_results.loc[len(event_study_results)] = [-1, 0, 0]

# Calculate child penalty for each event time
event_study_results = event_study_results.sort_values('event_time', ascending=True)
event_study_results['child_penalty'] = event_study_results['percentage_coef_m'] - event_study_results['percentage_coef_w']

#### Export the Results

event_study_results.reset_index(inplace=True)
event_study_results.to_csv("../../data/processed/malesector_results_ia.csv", index=False)

### Female-Dominated Sectors

# Subset the data to keep individuals only in female-dominated sectors
female_sector_ia = ia_data[ia_data['sector_gender_ratio'] == 'Female-Dominated']
female_sector_ia = female_sector_ia.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data = female_sector_ia[female_sector_ia['sex'] == 2]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns = {eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(female_sector_ia[female_sector_ia['sex'] == 2]['age'].min())
max_age = int(female_sector_ia[female_sector_ia['sex'] == 2]['age'].max())
min_year = int(female_sector_ia[female_sector_ia['sex'] == 2]['syear'].min())
max_year = int(female_sector_ia[female_sector_ia['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        female_sector_ia.loc[female_sector_ia['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        female_sector_ia.loc[female_sector_ia['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

female_sector_ia['pred_year_w'] = female_sector_ia['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
female_sector_ia['pred_age_w'] = pd.to_numeric(female_sector_ia['pred_age_w'], errors='coerce')
female_sector_ia['pred_year_w'] = pd.to_numeric(female_sector_ia['pred_year_w'], errors='coerce')
female_sector_ia['pred_w'] = female_sector_ia['pred_age_w'] + female_sector_ia['pred_year_w']
female_sector_ia.loc[female_sector_ia['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = female_sector_ia.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data=female_sector_ia[female_sector_ia['sex'] == 1]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns={eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(female_sector_ia[female_sector_ia['sex'] == 1]['age'].min())
max_age = int(female_sector_ia[female_sector_ia['sex'] == 1]['age'].max())
min_year = int(female_sector_ia[female_sector_ia['sex'] == 1]['syear'].min())
max_year = int(female_sector_ia[female_sector_ia['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        female_sector_ia.loc[female_sector_ia['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        female_sector_ia.loc[female_sector_ia['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

female_sector_ia['pred_year_m'] = female_sector_ia['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
female_sector_ia['pred_age_m'] = pd.to_numeric(female_sector_ia['pred_age_m'], errors='coerce')
female_sector_ia['pred_year_m'] = pd.to_numeric(female_sector_ia['pred_year_m'], errors='coerce')
female_sector_ia['pred_m'] = female_sector_ia['pred_age_m'] + female_sector_ia['pred_year_m']
female_sector_ia.loc[female_sector_ia['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = female_sector_ia.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for men
results_m_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_m_df['coef'] = results_m_df['coef'].astype(float)
results_m_df['percentage_coef_m'] = results_m_df['coef']/results_m_df['pred_m'] 

#### Combine the Results

# Merge results for men and women
event_study_results = pd.merge(results_m_df[['event_time', 'percentage_coef_m']],
                               results_w_df[['event_time', 'percentage_coef_w']],
                               on = 'event_time')
event_study_results.loc[len(event_study_results)] = [-1, 0, 0]

# Calculate child penalty for each event time
event_study_results = event_study_results.sort_values('event_time', ascending=True)
event_study_results['child_penalty'] = event_study_results['percentage_coef_m'] - event_study_results['percentage_coef_w']

#### Export the Results

event_study_results.reset_index(inplace=True)
event_study_results.to_csv("../../data/processed/femalesector_results_ia.csv", index=False)

### Balanced Gender Sectors

# Subset the data to keep individuals only in equal-share sectors
balanced_sector_ia = ia_data[ia_data['sector_gender_ratio'] == 'Equal Share']
balanced_sector_ia = balanced_sector_ia.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data = balanced_sector_ia[balanced_sector_ia['sex'] == 2]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns = {eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(balanced_sector_ia[balanced_sector_ia['sex'] == 2]['age'].min())
max_age = int(balanced_sector_ia[balanced_sector_ia['sex'] == 2]['age'].max())
min_year = int(balanced_sector_ia[balanced_sector_ia['sex'] == 2]['syear'].min())
max_year = int(balanced_sector_ia[balanced_sector_ia['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        balanced_sector_ia.loc[balanced_sector_ia['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        balanced_sector_ia.loc[balanced_sector_ia['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

balanced_sector_ia['pred_year_w'] = balanced_sector_ia['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
balanced_sector_ia['pred_age_w'] = pd.to_numeric(balanced_sector_ia['pred_age_w'], errors='coerce')
balanced_sector_ia['pred_year_w'] = pd.to_numeric(balanced_sector_ia['pred_year_w'], errors='coerce')
balanced_sector_ia['pred_w'] = balanced_sector_ia['pred_age_w'] + balanced_sector_ia['pred_year_w']
balanced_sector_ia.loc[balanced_sector_ia['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = balanced_sector_ia.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data=balanced_sector_ia[balanced_sector_ia['sex'] == 1]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns={eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(balanced_sector_ia[balanced_sector_ia['sex'] == 1]['age'].min())
max_age = int(balanced_sector_ia[balanced_sector_ia['sex'] == 1]['age'].max())
min_year = int(balanced_sector_ia[balanced_sector_ia['sex'] == 1]['syear'].min())
max_year = int(balanced_sector_ia[balanced_sector_ia['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        balanced_sector_ia.loc[balanced_sector_ia['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        balanced_sector_ia.loc[balanced_sector_ia['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

balanced_sector_ia['pred_year_m'] = balanced_sector_ia['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
balanced_sector_ia['pred_age_m'] = pd.to_numeric(balanced_sector_ia['pred_age_m'], errors='coerce')
balanced_sector_ia['pred_year_m'] = pd.to_numeric(balanced_sector_ia['pred_year_m'], errors='coerce')
balanced_sector_ia['pred_m'] = balanced_sector_ia['pred_age_m'] + balanced_sector_ia['pred_year_m']
balanced_sector_ia.loc[balanced_sector_ia['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = balanced_sector_ia.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for men
results_m_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_m_df['coef'] = results_m_df['coef'].astype(float)
results_m_df['percentage_coef_m'] = results_m_df['coef']/results_m_df['pred_m'] 

#### Combine the Results

# Merge results for men and women
event_study_results = pd.merge(results_m_df[['event_time', 'percentage_coef_m']],
                               results_w_df[['event_time', 'percentage_coef_w']],
                               on = 'event_time')
event_study_results.loc[len(event_study_results)] = [-1, 0, 0]

# Calculate child penalty for each event time
event_study_results = event_study_results.sort_values('event_time', ascending=True)
event_study_results['child_penalty'] = event_study_results['percentage_coef_m'] - event_study_results['percentage_coef_w']

#### Export the Results

event_study_results.reset_index(inplace=True)
event_study_results.to_csv("../../data/processed/balanced_results_ia.csv", index=False)

## Run the Analysis with Working Hours Approach

# Remove unobserved rows
wha_data = data[(data['e11101'] >= 0)] 

### Male-Dominated Sectors

# Subset the data to keep individuals only in male-dominated sectors
male_sector_wha = wha_data[wha_data['sector_gender_ratio'] == 'Male-Dominated']
male_sector_wha = male_sector_wha.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data = male_sector_wha[male_sector_wha['sex'] == 2]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns = {eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(male_sector_wha[male_sector_wha['sex'] == 2]['age'].min())
max_age = int(male_sector_wha[male_sector_wha['sex'] == 2]['age'].max())
min_year = int(male_sector_wha[male_sector_wha['sex'] == 2]['syear'].min())
max_year = int(male_sector_wha[male_sector_wha['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        male_sector_wha.loc[male_sector_wha['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        male_sector_wha.loc[male_sector_wha['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

male_sector_wha['pred_year_w'] = male_sector_wha['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
male_sector_wha['pred_age_w'] = pd.to_numeric(male_sector_wha['pred_age_w'], errors='coerce')
male_sector_wha['pred_year_w'] = pd.to_numeric(male_sector_wha['pred_year_w'], errors='coerce')
male_sector_wha['pred_w'] = male_sector_wha['pred_age_w'] + male_sector_wha['pred_year_w']
male_sector_wha.loc[male_sector_wha['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = male_sector_wha.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data=male_sector_wha[male_sector_wha['sex'] == 1]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns={eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(male_sector_wha[male_sector_wha['sex'] == 1]['age'].min())
max_age = int(male_sector_wha[male_sector_wha['sex'] == 1]['age'].max())
min_year = int(male_sector_wha[male_sector_wha['sex'] == 1]['syear'].min())
max_year = int(male_sector_wha[male_sector_wha['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        male_sector_wha.loc[male_sector_wha['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        male_sector_wha.loc[male_sector_wha['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

male_sector_wha['pred_year_m'] = male_sector_wha['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
male_sector_wha['pred_age_m'] = pd.to_numeric(male_sector_wha['pred_age_m'], errors='coerce')
male_sector_wha['pred_year_m'] = pd.to_numeric(male_sector_wha['pred_year_m'], errors='coerce')
male_sector_wha['pred_m'] = male_sector_wha['pred_age_m'] + male_sector_wha['pred_year_m']
male_sector_wha.loc[male_sector_wha['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = male_sector_wha.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for men
results_m_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_m_df['coef'] = results_m_df['coef'].astype(float)
results_m_df['percentage_coef_m'] = results_m_df['coef']/results_m_df['pred_m'] 

#### Combine the Results

# Merge results for men and women
event_study_results = pd.merge(results_m_df[['event_time', 'percentage_coef_m']],
                               results_w_df[['event_time', 'percentage_coef_w']],
                               on = 'event_time')
event_study_results.loc[len(event_study_results)] = [-1, 0, 0]

# Calculate child penalty for each event time
event_study_results = event_study_results.sort_values('event_time', ascending=True)
event_study_results['child_penalty'] = event_study_results['percentage_coef_m'] - event_study_results['percentage_coef_w']

#### Export the Results

event_study_results.reset_index(inplace=True)
event_study_results.to_csv("../../data/processed/malesector_results_wha.csv", index=False)

### Female Dominated Sectors

# Subset the data to keep individuals only in female-dominated sectors
female_sector_wha = wha_data[wha_data['sector_gender_ratio'] == 'Female-Dominated']
female_sector_wha = female_sector_wha.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data = female_sector_wha[female_sector_wha['sex'] == 2]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns = {eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(female_sector_wha[female_sector_wha['sex'] == 2]['age'].min())
max_age = int(female_sector_wha[female_sector_wha['sex'] == 2]['age'].max())
min_year = int(female_sector_wha[female_sector_wha['sex'] == 2]['syear'].min())
max_year = int(female_sector_wha[female_sector_wha['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        female_sector_wha.loc[female_sector_wha['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        female_sector_wha.loc[female_sector_wha['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

female_sector_wha['pred_year_w'] = female_sector_wha['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
female_sector_wha['pred_age_w'] = pd.to_numeric(female_sector_wha['pred_age_w'], errors='coerce')
female_sector_wha['pred_year_w'] = pd.to_numeric(female_sector_wha['pred_year_w'], errors='coerce')
female_sector_wha['pred_w'] = female_sector_wha['pred_age_w'] + female_sector_wha['pred_year_w']
female_sector_wha.loc[female_sector_wha['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = female_sector_wha.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data=female_sector_wha[female_sector_wha['sex'] == 1]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns={eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(female_sector_wha[female_sector_wha['sex'] == 1]['age'].min())
max_age = int(female_sector_wha[female_sector_wha['sex'] == 1]['age'].max())
min_year = int(female_sector_wha[female_sector_wha['sex'] == 1]['syear'].min())
max_year = int(female_sector_wha[female_sector_wha['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        female_sector_wha.loc[female_sector_wha['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        female_sector_wha.loc[female_sector_wha['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

female_sector_wha['pred_year_m'] = female_sector_wha['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
female_sector_wha['pred_age_m'] = pd.to_numeric(female_sector_wha['pred_age_m'], errors='coerce')
female_sector_wha['pred_year_m'] = pd.to_numeric(female_sector_wha['pred_year_m'], errors='coerce')
female_sector_wha['pred_m'] = female_sector_wha['pred_age_m'] + female_sector_wha['pred_year_m']
female_sector_wha.loc[female_sector_wha['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = female_sector_wha.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for men
results_m_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_m_df['coef'] = results_m_df['coef'].astype(float)
results_m_df['percentage_coef_m'] = results_m_df['coef']/results_m_df['pred_m'] 

#### Combine the Results

# Merge results for men and women
event_study_results = pd.merge(results_m_df[['event_time', 'percentage_coef_m']],
                               results_w_df[['event_time', 'percentage_coef_w']],
                               on = 'event_time')
event_study_results.loc[len(event_study_results)] = [-1, 0, 0]

# Calculate child penalty for each event time
event_study_results = event_study_results.sort_values('event_time', ascending=True)
event_study_results['child_penalty'] = event_study_results['percentage_coef_m'] - event_study_results['percentage_coef_w']

#### Export the Results

event_study_results.reset_index(inplace=True)
event_study_results.to_csv("../../data/processed/femalesector_results_wha.csv", index=False)

### Balanced Gender Sectors

# Subset the data to keep individuals only in equal-share sectors
balanced_sector_wha = wha_data[wha_data['sector_gender_ratio'] == 'Equal Share']
balanced_sector_wha = balanced_sector_wha.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data = balanced_sector_wha[balanced_sector_wha['sex'] == 2]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns = {eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(balanced_sector_wha[balanced_sector_wha['sex'] == 2]['age'].min())
max_age = int(balanced_sector_wha[balanced_sector_wha['sex'] == 2]['age'].max())
min_year = int(balanced_sector_wha[balanced_sector_wha['sex'] == 2]['syear'].min())
max_year = int(balanced_sector_wha[balanced_sector_wha['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        balanced_sector_wha.loc[balanced_sector_wha['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        balanced_sector_wha.loc[balanced_sector_wha['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

balanced_sector_wha['pred_year_w'] = balanced_sector_wha['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
balanced_sector_wha['pred_age_w'] = pd.to_numeric(balanced_sector_wha['pred_age_w'], errors='coerce')
balanced_sector_wha['pred_year_w'] = pd.to_numeric(balanced_sector_wha['pred_year_w'], errors='coerce')
balanced_sector_wha['pred_w'] = balanced_sector_wha['pred_age_w'] + balanced_sector_wha['pred_year_w']
balanced_sector_wha.loc[balanced_sector_wha['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = balanced_sector_wha.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data=balanced_sector_wha[balanced_sector_wha['sex'] == 1]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns={eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(balanced_sector_wha[balanced_sector_wha['sex'] == 1]['age'].min())
max_age = int(balanced_sector_wha[balanced_sector_wha['sex'] == 1]['age'].max())
min_year = int(balanced_sector_wha[balanced_sector_wha['sex'] == 1]['syear'].min())
max_year = int(balanced_sector_wha[balanced_sector_wha['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        balanced_sector_wha.loc[balanced_sector_wha['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        balanced_sector_wha.loc[balanced_sector_wha['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

balanced_sector_wha['pred_year_m'] = balanced_sector_wha['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
balanced_sector_wha['pred_age_m'] = pd.to_numeric(balanced_sector_wha['pred_age_m'], errors='coerce')
balanced_sector_wha['pred_year_m'] = pd.to_numeric(balanced_sector_wha['pred_year_m'], errors='coerce')
balanced_sector_wha['pred_m'] = balanced_sector_wha['pred_age_m'] + balanced_sector_wha['pred_year_m']
balanced_sector_wha.loc[balanced_sector_wha['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = balanced_sector_wha.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for men
results_m_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_m_df['coef'] = results_m_df['coef'].astype(float)
results_m_df['percentage_coef_m'] = results_m_df['coef']/results_m_df['pred_m'] 

#### Combine the Results

# Merge results for men and women
event_study_results = pd.merge(results_m_df[['event_time', 'percentage_coef_m']],
                               results_w_df[['event_time', 'percentage_coef_w']],
                               on = 'event_time')
event_study_results.loc[len(event_study_results)] = [-1, 0, 0]

# Calculate child penalty for each event time
event_study_results = event_study_results.sort_values('event_time', ascending=True)
event_study_results['child_penalty'] = event_study_results['percentage_coef_m'] - event_study_results['percentage_coef_w']

#### Export the Results

event_study_results.reset_index(inplace=True)
event_study_results.to_csv("../../data/processed/balancedsector_results_wha.csv", index=False)

## Run the Analysis with Employment Status Approach

# Remove unobserved rows
esa_data = data[(data['e11102'] >= 0)] 

### Male-Dominated Sectors

# Subset the data to keep individuals only in male-dominated sectors
male_sector_esa = esa_data[esa_data['sector_gender_ratio'] == 'Male-Dominated']
male_sector_esa = male_sector_esa.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data = male_sector_esa[male_sector_esa['sex'] == 2]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns = {eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(male_sector_esa[male_sector_esa['sex'] == 2]['age'].min())
max_age = int(male_sector_esa[male_sector_esa['sex'] == 2]['age'].max())
min_year = int(male_sector_esa[male_sector_esa['sex'] == 2]['syear'].min())
max_year = int(male_sector_esa[male_sector_esa['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        male_sector_esa.loc[male_sector_esa['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        male_sector_esa.loc[male_sector_esa['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

male_sector_esa['pred_year_w'] = male_sector_esa['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
male_sector_esa['pred_age_w'] = pd.to_numeric(male_sector_esa['pred_age_w'], errors='coerce')
male_sector_esa['pred_year_w'] = pd.to_numeric(male_sector_esa['pred_year_w'], errors='coerce')
male_sector_esa['pred_w'] = male_sector_esa['pred_age_w'] + male_sector_esa['pred_year_w']
male_sector_esa.loc[male_sector_esa['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = male_sector_esa.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data=male_sector_esa[male_sector_esa['sex'] == 1]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns={eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(male_sector_esa[male_sector_esa['sex'] == 1]['age'].min())
max_age = int(male_sector_esa[male_sector_esa['sex'] == 1]['age'].max())
min_year = int(male_sector_esa[male_sector_esa['sex'] == 1]['syear'].min())
max_year = int(male_sector_esa[male_sector_esa['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        male_sector_esa.loc[male_sector_esa['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        male_sector_esa.loc[male_sector_esa['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

male_sector_esa['pred_year_m'] = male_sector_esa['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
male_sector_esa['pred_age_m'] = pd.to_numeric(male_sector_esa['pred_age_m'], errors='coerce')
male_sector_esa['pred_year_m'] = pd.to_numeric(male_sector_esa['pred_year_m'], errors='coerce')
male_sector_esa['pred_m'] = male_sector_esa['pred_age_m'] + male_sector_esa['pred_year_m']
male_sector_esa.loc[male_sector_esa['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = male_sector_esa.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for men
results_m_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_m_df['coef'] = results_m_df['coef'].astype(float)
results_m_df['percentage_coef_m'] = results_m_df['coef']/results_m_df['pred_m'] 

#### Combine the Results

# Merge results for men and women
event_study_results = pd.merge(results_m_df[['event_time', 'percentage_coef_m']],
                               results_w_df[['event_time', 'percentage_coef_w']],
                               on = 'event_time')
event_study_results.loc[len(event_study_results)] = [-1, 0, 0]

# Calculate child penalty for each event time
event_study_results = event_study_results.sort_values('event_time', ascending=True)
event_study_results['child_penalty'] = event_study_results['percentage_coef_m'] - event_study_results['percentage_coef_w']

#### Export the Results

event_study_results.reset_index(inplace=True)
event_study_results.to_csv("../../data/processed/malesector_results_esa.csv", index=False)

### Female-Dominated Sectors

# Subset the data to keep individuals only in male-dominated sectors
female_sector_esa = esa_data[esa_data['sector_gender_ratio'] == 'Female-Dominated']
female_sector_esa = female_sector_esa.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data = female_sector_esa[female_sector_esa['sex'] == 2]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns = {eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(female_sector_esa[female_sector_esa['sex'] == 2]['age'].min())
max_age = int(female_sector_esa[female_sector_esa['sex'] == 2]['age'].max())
min_year = int(female_sector_esa[female_sector_esa['sex'] == 2]['syear'].min())
max_year = int(female_sector_esa[female_sector_esa['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        female_sector_esa.loc[female_sector_esa['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        female_sector_esa.loc[female_sector_esa['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

female_sector_esa['pred_year_w'] = female_sector_esa['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
female_sector_esa['pred_age_w'] = pd.to_numeric(female_sector_esa['pred_age_w'], errors='coerce')
female_sector_esa['pred_year_w'] = pd.to_numeric(female_sector_esa['pred_year_w'], errors='coerce')
female_sector_esa['pred_w'] = female_sector_esa['pred_age_w'] + female_sector_esa['pred_year_w']
female_sector_esa.loc[female_sector_esa['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = female_sector_esa.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data=female_sector_esa[female_sector_esa['sex'] == 1]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns={eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(female_sector_esa[female_sector_esa['sex'] == 1]['age'].min())
max_age = int(female_sector_esa[female_sector_esa['sex'] == 1]['age'].max())
min_year = int(female_sector_esa[female_sector_esa['sex'] == 1]['syear'].min())
max_year = int(female_sector_esa[female_sector_esa['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        female_sector_esa.loc[female_sector_esa['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        female_sector_esa.loc[female_sector_esa['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

female_sector_esa['pred_year_m'] = female_sector_esa['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
female_sector_esa['pred_age_m'] = pd.to_numeric(female_sector_esa['pred_age_m'], errors='coerce')
female_sector_esa['pred_year_m'] = pd.to_numeric(female_sector_esa['pred_year_m'], errors='coerce')
female_sector_esa['pred_m'] = female_sector_esa['pred_age_m'] + female_sector_esa['pred_year_m']
female_sector_esa.loc[female_sector_esa['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = female_sector_esa.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for men
results_m_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_m_df['coef'] = results_m_df['coef'].astype(float)
results_m_df['percentage_coef_m'] = results_m_df['coef']/results_m_df['pred_m'] 

#### Combine the Results

# Merge results for men and women
event_study_results = pd.merge(results_m_df[['event_time', 'percentage_coef_m']],
                               results_w_df[['event_time', 'percentage_coef_w']],
                               on = 'event_time')
event_study_results.loc[len(event_study_results)] = [-1, 0, 0]

# Calculate child penalty for each event time
event_study_results = event_study_results.sort_values('event_time', ascending=True)
event_study_results['child_penalty'] = event_study_results['percentage_coef_m'] - event_study_results['percentage_coef_w']

#### Export the Results

event_study_results.reset_index(inplace=True)
event_study_results.to_csv("../../data/processed/femalesector_results_esa.csv", index=False)

### Balanced Gender Sectors

# Subset the data to keep individuals only in equal share sectors
balanced_sector_esa = esa_data[esa_data['sector_gender_ratio'] == 'Equal Share']
balanced_sector_esa = balanced_sector_esa.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data = balanced_sector_esa[balanced_sector_esa['sex'] == 2]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns = {eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(balanced_sector_esa[balanced_sector_esa['sex'] == 2]['age'].min())
max_age = int(balanced_sector_esa[balanced_sector_esa['sex'] == 2]['age'].max())
min_year = int(balanced_sector_esa[balanced_sector_esa['sex'] == 2]['syear'].min())
max_year = int(balanced_sector_esa[balanced_sector_esa['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        balanced_sector_esa.loc[balanced_sector_esa['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        balanced_sector_esa.loc[balanced_sector_esa['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

balanced_sector_esa['pred_year_w'] = balanced_sector_esa['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
balanced_sector_esa['pred_age_w'] = pd.to_numeric(balanced_sector_esa['pred_age_w'], errors='coerce')
balanced_sector_esa['pred_year_w'] = pd.to_numeric(balanced_sector_esa['pred_year_w'], errors='coerce')
balanced_sector_esa['pred_w'] = balanced_sector_esa['pred_age_w'] + balanced_sector_esa['pred_year_w']
balanced_sector_esa.loc[balanced_sector_esa['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = balanced_sector_esa.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data=balanced_sector_esa[balanced_sector_esa['sex'] == 1]).fit()
summary = model.summary()
summary_df = pd.DataFrame(summary.tables[1].data[1:], columns = summary.tables[1].data[0])

# Counterfactual Dataframe
counterfactual_df = summary_df[~summary_df.iloc[:, 0].str.startswith('C(event_time)')]
counterfactual_df = counterfactual_df.set_index(counterfactual_df.columns[0])

# Event-Time Dataframe
eventtime_df = summary_df[summary_df.iloc[:, 0].str.startswith('C(event_time)')]
eventtime_df.rename(columns={eventtime_df.columns[0]: 'variables'}, inplace=True)
eventtime_df['event_time'] = eventtime_df['variables'].str.extract(r'T\.(-?\d+)').astype(int)

# Counterfactual Prediction
# Range of age and year to run the loop
min_age = int(balanced_sector_esa[balanced_sector_esa['sex'] == 1]['age'].min())
max_age = int(balanced_sector_esa[balanced_sector_esa['sex'] == 1]['age'].max())
min_year = int(balanced_sector_esa[balanced_sector_esa['sex'] == 1]['syear'].min())
max_year = int(balanced_sector_esa[balanced_sector_esa['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        balanced_sector_esa.loc[balanced_sector_esa['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        balanced_sector_esa.loc[balanced_sector_esa['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

balanced_sector_esa['pred_year_m'] = balanced_sector_esa['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
balanced_sector_esa['pred_age_m'] = pd.to_numeric(balanced_sector_esa['pred_age_m'], errors='coerce')
balanced_sector_esa['pred_year_m'] = pd.to_numeric(balanced_sector_esa['pred_year_m'], errors='coerce')
balanced_sector_esa['pred_m'] = balanced_sector_esa['pred_age_m'] + balanced_sector_esa['pred_year_m']
balanced_sector_esa.loc[balanced_sector_esa['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = balanced_sector_esa.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for men
results_m_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_m_df['coef'] = results_m_df['coef'].astype(float)
results_m_df['percentage_coef_m'] = results_m_df['coef']/results_m_df['pred_m'] 

#### Combine the Results

# Merge results for men and women
event_study_results = pd.merge(results_m_df[['event_time', 'percentage_coef_m']],
                               results_w_df[['event_time', 'percentage_coef_w']],
                               on = 'event_time')
event_study_results.loc[len(event_study_results)] = [-1, 0, 0]

# Calculate child penalty for each event time
event_study_results = event_study_results.sort_values('event_time', ascending=True)
event_study_results['child_penalty'] = event_study_results['percentage_coef_m'] - event_study_results['percentage_coef_w']

#### Export the Results

event_study_results.reset_index(inplace=True)
event_study_results.to_csv("../../data/processed/balancedsector_results_esa.csv", index=False)