# 0201 Estimation - General

## Libraries

import numpy as np
import pandas as pd
import statsmodels.api as sm

## Import the Data

data = pd.read_csv("../../data/cleaned/cleaned_data.csv", index_col=0)

## Run the Analysis with Income Approach

# Remove unobserved rows
ia_data = data[(data['real_income'] >= 0)] 

### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data = ia_data[ia_data['sex'] == 2]).fit()
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
min_age = int(ia_data[ia_data['sex'] == 2]['age'].min())
max_age = int(ia_data[ia_data['sex'] == 2]['age'].max())
min_year = int(ia_data[ia_data['sex'] == 2]['syear'].min())
max_year = int(ia_data[ia_data['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        ia_data.loc[ia_data['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        ia_data.loc[ia_data['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

ia_data['pred_year_w'] = ia_data['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
ia_data['pred_age_w'] = pd.to_numeric(ia_data['pred_age_w'], errors='coerce')
ia_data['pred_year_w'] = pd.to_numeric(ia_data['pred_year_w'], errors='coerce')
ia_data['pred_w'] = ia_data['pred_age_w'] + ia_data['pred_year_w']
ia_data.loc[ia_data['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = ia_data.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

### Man

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data=ia_data[ia_data['sex'] == 1]).fit()
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
min_age = int(ia_data[ia_data['sex'] == 1]['age'].min())
max_age = int(ia_data[ia_data['sex'] == 1]['age'].max())
min_year = int(ia_data[ia_data['sex'] == 1]['syear'].min())
max_year = int(ia_data[ia_data['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        ia_data.loc[ia_data['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        ia_data.loc[ia_data['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

ia_data['pred_year_m'] = ia_data['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
ia_data['pred_age_m'] = pd.to_numeric(ia_data['pred_age_m'], errors='coerce')
ia_data['pred_year_m'] = pd.to_numeric(ia_data['pred_year_m'], errors='coerce')
ia_data['pred_m'] = ia_data['pred_age_m'] + ia_data['pred_year_m']
ia_data.loc[ia_data['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = ia_data.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for men
results_m_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_m_df['coef'] = results_m_df['coef'].astype(float)
results_m_df['percentage_coef_m'] = results_m_df['coef']/results_m_df['pred_m'] 

### Combine the Results

# Merge results for men and women
event_study_results = pd.merge(results_m_df[['event_time', 'percentage_coef_m']],
                               results_w_df[['event_time', 'percentage_coef_w']],
                               on = 'event_time')
event_study_results.loc[len(event_study_results)] = [-1, 0, 0]

# Calculate child penalty for each event time
event_study_results = event_study_results.sort_values('event_time', ascending=True)
event_study_results['child_penalty'] = event_study_results['percentage_coef_m'] - event_study_results['percentage_coef_w']

### Export the Results

event_study_results.reset_index(inplace=True)
event_study_results.to_csv("../../data/processed/general_results_ia.csv", index=False)

## Run the Analysis with Working Hours Approach

# Remove unobserved rows
wha_data = data[(data['e11101'] >= 0)] 

### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data = wha_data[wha_data['sex'] == 2]).fit()
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
min_age = int(wha_data[wha_data['sex'] == 2]['age'].min())
max_age = int(wha_data[wha_data['sex'] == 2]['age'].max())
min_year = int(wha_data[wha_data['sex'] == 2]['syear'].min())
max_year = int(wha_data[wha_data['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        wha_data.loc[wha_data['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        wha_data.loc[wha_data['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

wha_data['pred_year_w'] = wha_data['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
wha_data['pred_age_w'] = pd.to_numeric(wha_data['pred_age_w'], errors='coerce')
wha_data['pred_year_w'] = pd.to_numeric(wha_data['pred_year_w'], errors='coerce')
wha_data['pred_w'] = wha_data['pred_age_w'] + wha_data['pred_year_w']
wha_data.loc[wha_data['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = wha_data.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data=wha_data[wha_data['sex'] == 1]).fit()
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
min_age = int(wha_data[wha_data['sex'] == 1]['age'].min())
max_age = int(wha_data[wha_data['sex'] == 1]['age'].max())
min_year = int(wha_data[wha_data['sex'] == 1]['syear'].min())
max_year = int(wha_data[wha_data['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        wha_data.loc[wha_data['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        wha_data.loc[wha_data['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

wha_data['pred_year_m'] = wha_data['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
wha_data['pred_age_m'] = pd.to_numeric(wha_data['pred_age_m'], errors='coerce')
wha_data['pred_year_m'] = pd.to_numeric(wha_data['pred_year_m'], errors='coerce')
wha_data['pred_m'] = wha_data['pred_age_m'] + wha_data['pred_year_m']
wha_data.loc[wha_data['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = wha_data.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for men
results_m_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_m_df['coef'] = results_m_df['coef'].astype(float)
results_m_df['percentage_coef_m'] = results_m_df['coef']/results_m_df['pred_m'] 

### Combine the Results

# Merge results for men and women
event_study_results = pd.merge(results_m_df[['event_time', 'percentage_coef_m']],
                               results_w_df[['event_time', 'percentage_coef_w']],
                               on = 'event_time')
event_study_results.loc[len(event_study_results)] = [-1, 0, 0]

# Calculate child penalty for each event time
event_study_results = event_study_results.sort_values('event_time', ascending=True)
event_study_results['child_penalty'] = event_study_results['percentage_coef_m'] - event_study_results['percentage_coef_w']

### Export the Results

event_study_results.reset_index(inplace=True)
event_study_results.to_csv("../../data/processed/general_results_wha.csv", index=False)

## Run the Analysis with Employment Status Approach

# Remove unobserved rows
esa_data = data[(data['e11102'] >= 0)] 

### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data = esa_data[esa_data['sex'] == 2]).fit()
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
min_age = int(esa_data[esa_data['sex'] == 2]['age'].min())
max_age = int(esa_data[esa_data['sex'] == 2]['age'].max())
min_year = int(esa_data[esa_data['sex'] == 2]['syear'].min())
max_year = int(esa_data[esa_data['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        esa_data.loc[esa_data['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        esa_data.loc[esa_data['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

esa_data['pred_year_w'] = esa_data['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
esa_data['pred_age_w'] = pd.to_numeric(esa_data['pred_age_w'], errors='coerce')
esa_data['pred_year_w'] = pd.to_numeric(esa_data['pred_year_w'], errors='coerce')
esa_data['pred_w'] = esa_data['pred_age_w'] + esa_data['pred_year_w']
esa_data.loc[esa_data['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = esa_data.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data=esa_data[esa_data['sex'] == 1]).fit()
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
min_age = int(esa_data[esa_data['sex'] == 1]['age'].min())
max_age = int(esa_data[esa_data['sex'] == 1]['age'].max())
min_year = int(esa_data[esa_data['sex'] == 1]['syear'].min())
max_year = int(esa_data[esa_data['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        esa_data.loc[esa_data['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        esa_data.loc[esa_data['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

esa_data['pred_year_m'] = esa_data['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
esa_data['pred_age_m'] = pd.to_numeric(esa_data['pred_age_m'], errors='coerce')
esa_data['pred_year_m'] = pd.to_numeric(esa_data['pred_year_m'], errors='coerce')
esa_data['pred_m'] = esa_data['pred_age_m'] + esa_data['pred_year_m']
esa_data.loc[esa_data['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = esa_data.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for men
results_m_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_m_df['coef'] = results_m_df['coef'].astype(float)
results_m_df['percentage_coef_m'] = results_m_df['coef']/results_m_df['pred_m'] 

### Combine the Results

# Merge results for men and women
event_study_results = pd.merge(results_m_df[['event_time', 'percentage_coef_m']],
                               results_w_df[['event_time', 'percentage_coef_w']],
                               on = 'event_time')
event_study_results.loc[len(event_study_results)] = [-1, 0, 0]

# Calculate child penalty for each event time
event_study_results = event_study_results.sort_values('event_time', ascending=True)
event_study_results['child_penalty'] = event_study_results['percentage_coef_m'] - event_study_results['percentage_coef_w']

### Export the Results

event_study_results.reset_index(inplace=True)
event_study_results.to_csv("../../data/processed/general_results_esa.csv", index=False)
