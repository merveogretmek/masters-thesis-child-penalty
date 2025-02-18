# 0202 Estimation - Regions

## Libraries

import numpy as np
import pandas as pd
import statsmodels.api as sm

## Import the Data

data = pd.read_csv("../../data/cleaned/cleaned_data.csv", index_col=0)

## Categorize into Residential Areas (Rural vs Urban)

# Classify residential area as 'urban' or 'rural'
data['res_area'] = data['wum1'].apply(lambda x: 'rural' if x in [1, 2, 3] else 'urban')

## Run the Analysis with Income Approach

# Remove unobserved rows
ia_data = data[(data['real_income'] >= 0)] 

### Rural Residential Area

# Subset the data to keep only rural individuals
rural_ia = ia_data[ia_data['res_area'] == 'rural']
rural_ia = rural_ia.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data = rural_ia[rural_ia['sex'] == 2]).fit()
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
min_age = int(rural_ia[rural_ia['sex'] == 2]['age'].min())
max_age = int(rural_ia[rural_ia['sex'] == 2]['age'].max())
min_year = int(rural_ia[rural_ia['sex'] == 2]['syear'].min())
max_year = int(rural_ia[rural_ia['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        rural_ia.loc[rural_ia['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        rural_ia.loc[rural_ia['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

rural_ia['pred_year_w'] = rural_ia['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
rural_ia['pred_age_w'] = pd.to_numeric(rural_ia['pred_age_w'], errors='coerce')
rural_ia['pred_year_w'] = pd.to_numeric(rural_ia['pred_year_w'], errors='coerce')
rural_ia['pred_w'] = rural_ia['pred_age_w'] + rural_ia['pred_year_w']
rural_ia.loc[rural_ia['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = rural_ia.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data=rural_ia[rural_ia['sex'] == 1]).fit()
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
min_age = int(rural_ia[rural_ia['sex'] == 1]['age'].min())
max_age = int(rural_ia[rural_ia['sex'] == 1]['age'].max())
min_year = int(rural_ia[rural_ia['sex'] == 1]['syear'].min())
max_year = int(rural_ia[rural_ia['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        rural_ia.loc[rural_ia['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        rural_ia.loc[rural_ia['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

rural_ia['pred_year_m'] = rural_ia['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
rural_ia['pred_age_m'] = pd.to_numeric(rural_ia['pred_age_m'], errors='coerce')
rural_ia['pred_year_m'] = pd.to_numeric(rural_ia['pred_year_m'], errors='coerce')
rural_ia['pred_m'] = rural_ia['pred_age_m'] + rural_ia['pred_year_m']
rural_ia.loc[rural_ia['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = rural_ia.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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

#### Explore the Results

event_study_results.reset_index(inplace=True)
event_study_results.to_csv("../../data/processed/rural_results_ia.csv", index=False)

### Urban Residential Area

# Subset the data to keep only urban individuals
urban_ia = ia_data[ia_data['res_area'] == 'urban']
urban_ia = urban_ia.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data = urban_ia[urban_ia['sex'] == 2]).fit()
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
min_age = int(urban_ia[urban_ia['sex'] == 2]['age'].min())
max_age = int(urban_ia[urban_ia['sex'] == 2]['age'].max())
min_year = int(urban_ia[urban_ia['sex'] == 2]['syear'].min())
max_year = int(urban_ia[urban_ia['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        urban_ia.loc[urban_ia['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        urban_ia.loc[urban_ia['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

urban_ia['pred_year_w'] = urban_ia['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
urban_ia['pred_age_w'] = pd.to_numeric(urban_ia['pred_age_w'], errors='coerce')
urban_ia['pred_year_w'] = pd.to_numeric(urban_ia['pred_year_w'], errors='coerce')
urban_ia['pred_w'] = urban_ia['pred_age_w'] + urban_ia['pred_year_w']
urban_ia.loc[urban_ia['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = urban_ia.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data=urban_ia[urban_ia['sex'] == 1]).fit()
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
min_age = int(urban_ia[urban_ia['sex'] == 1]['age'].min())
max_age = int(urban_ia[urban_ia['sex'] == 1]['age'].max())
min_year = int(urban_ia[urban_ia['sex'] == 1]['syear'].min())
max_year = int(urban_ia[urban_ia['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        urban_ia.loc[urban_ia['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        urban_ia.loc[urban_ia['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

urban_ia['pred_year_m'] = urban_ia['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
urban_ia['pred_age_m'] = pd.to_numeric(urban_ia['pred_age_m'], errors='coerce')
urban_ia['pred_year_m'] = pd.to_numeric(urban_ia['pred_year_m'], errors='coerce')
urban_ia['pred_m'] = urban_ia['pred_age_m'] + urban_ia['pred_year_m']
urban_ia.loc[urban_ia['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = urban_ia.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/urban_results_ia.csv", index=False)

## Run the Analysis with Working Hours Approach

# Remove unobserved rows
wha_data = data[(data['e11101'] >= 0)] 

### Rural Residential Area

# Subset the data to keep only rural individuals
rural_wha = wha_data[wha_data['res_area'] == 'rural']
rural_wha = rural_wha.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data = rural_wha[rural_wha['sex'] == 2]).fit()
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
min_age = int(rural_wha[rural_wha['sex'] == 2]['age'].min())
max_age = int(rural_wha[rural_wha['sex'] == 2]['age'].max())
min_year = int(rural_wha[rural_wha['sex'] == 2]['syear'].min())
max_year = int(rural_wha[rural_wha['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        rural_wha.loc[rural_wha['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        rural_wha.loc[rural_wha['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

rural_wha['pred_year_w'] = rural_wha['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
rural_wha['pred_age_w'] = pd.to_numeric(rural_wha['pred_age_w'], errors='coerce')
rural_wha['pred_year_w'] = pd.to_numeric(rural_wha['pred_year_w'], errors='coerce')
rural_wha['pred_w'] = rural_wha['pred_age_w'] + rural_wha['pred_year_w']
rural_wha.loc[rural_wha['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = rural_wha.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data=rural_wha[rural_wha['sex'] == 1]).fit()
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
min_age = int(rural_wha[rural_wha['sex'] == 1]['age'].min())
max_age = int(rural_wha[rural_wha['sex'] == 1]['age'].max())
min_year = int(rural_wha[rural_wha['sex'] == 1]['syear'].min())
max_year = int(rural_wha[rural_wha['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        rural_wha.loc[rural_wha['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        rural_wha.loc[rural_wha['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

rural_wha['pred_year_m'] = rural_wha['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
rural_wha['pred_age_m'] = pd.to_numeric(rural_wha['pred_age_m'], errors='coerce')
rural_wha['pred_year_m'] = pd.to_numeric(rural_wha['pred_year_m'], errors='coerce')
rural_wha['pred_m'] = rural_wha['pred_age_m'] + rural_wha['pred_year_m']
rural_wha.loc[rural_wha['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = rural_wha.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/rural_results_wha.csv", index=False)

### Urban Residential Area

# Subset the data to keep only urban individuals
urban_wha = wha_data[wha_data['res_area'] == 'urban']
urban_wha = urban_wha.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data = urban_wha[urban_wha['sex'] == 2]).fit()
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
min_age = int(urban_wha[urban_wha['sex'] == 2]['age'].min())
max_age = int(urban_wha[urban_wha['sex'] == 2]['age'].max())
min_year = int(urban_wha[urban_wha['sex'] == 2]['syear'].min())
max_year = int(urban_wha[urban_wha['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        urban_wha.loc[urban_wha['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        urban_wha.loc[urban_wha['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

urban_wha['pred_year_w'] = urban_wha['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
urban_wha['pred_age_w'] = pd.to_numeric(urban_wha['pred_age_w'], errors='coerce')
urban_wha['pred_year_w'] = pd.to_numeric(urban_wha['pred_year_w'], errors='coerce')
urban_wha['pred_w'] = urban_wha['pred_age_w'] + urban_wha['pred_year_w']
urban_wha.loc[urban_wha['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = urban_wha.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data=urban_wha[urban_wha['sex'] == 1]).fit()
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
min_age = int(urban_wha[urban_wha['sex'] == 1]['age'].min())
max_age = int(urban_wha[urban_wha['sex'] == 1]['age'].max())
min_year = int(urban_wha[urban_wha['sex'] == 1]['syear'].min())
max_year = int(urban_wha[urban_wha['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        urban_wha.loc[urban_wha['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        urban_wha.loc[urban_wha['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

urban_wha['pred_year_m'] = urban_wha['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
urban_wha['pred_age_m'] = pd.to_numeric(urban_wha['pred_age_m'], errors='coerce')
urban_wha['pred_year_m'] = pd.to_numeric(urban_wha['pred_year_m'], errors='coerce')
urban_wha['pred_m'] = urban_wha['pred_age_m'] + urban_wha['pred_year_m']
urban_wha.loc[urban_wha['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = urban_wha.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/urban_results_wha.csv", index=False)

## Run the Anaysis with Employment Status Approach

# Remove unobserved rows
esa_data = data[(data['e11102'] >= 0)]  

### Rural Residential Area

# Subset the data to keep only rural individuals
rural_esa = esa_data[esa_data['res_area'] == 'rural']
rural_esa = rural_esa.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data = rural_esa[rural_esa['sex'] == 2]).fit()
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
min_age = int(rural_esa[rural_esa['sex'] == 2]['age'].min())
max_age = int(rural_esa[rural_esa['sex'] == 2]['age'].max())
min_year = int(rural_esa[rural_esa['sex'] == 2]['syear'].min())
max_year = int(rural_esa[rural_esa['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        rural_esa.loc[rural_esa['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        rural_esa.loc[rural_esa['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

rural_esa['pred_year_w'] = rural_esa['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
rural_esa['pred_age_w'] = pd.to_numeric(rural_esa['pred_age_w'], errors='coerce')
rural_esa['pred_year_w'] = pd.to_numeric(rural_esa['pred_year_w'], errors='coerce')
rural_esa['pred_w'] = rural_esa['pred_age_w'] + rural_esa['pred_year_w']
rural_esa.loc[rural_esa['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = rural_esa.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data=rural_esa[rural_esa['sex'] == 1]).fit()
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
min_age = int(rural_esa[rural_esa['sex'] == 1]['age'].min())
max_age = int(rural_esa[rural_esa['sex'] == 1]['age'].max())
min_year = int(rural_esa[rural_esa['sex'] == 1]['syear'].min())
max_year = int(rural_esa[rural_esa['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        rural_esa.loc[rural_esa['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        rural_esa.loc[rural_esa['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

rural_esa['pred_year_m'] = rural_esa['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
rural_esa['pred_age_m'] = pd.to_numeric(rural_esa['pred_age_m'], errors='coerce')
rural_esa['pred_year_m'] = pd.to_numeric(rural_esa['pred_year_m'], errors='coerce')
rural_esa['pred_m'] = rural_esa['pred_age_m'] + rural_esa['pred_year_m']
rural_esa.loc[rural_esa['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = rural_esa.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/rural_results_esa.csv", index=False)

### Urban Residential Area

# Subset the data to keep only urban individuals
urban_esa = esa_data[esa_data['res_area'] == 'urban']
urban_esa = urban_esa.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data = urban_esa[urban_esa['sex'] == 2]).fit()
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
min_age = int(urban_esa[urban_esa['sex'] == 2]['age'].min())
max_age = int(urban_esa[urban_esa['sex'] == 2]['age'].max())
min_year = int(urban_esa[urban_esa['sex'] == 2]['syear'].min())
max_year = int(urban_esa[urban_esa['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        urban_esa.loc[urban_esa['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        urban_esa.loc[urban_esa['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

urban_esa['pred_year_w'] = urban_esa['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
urban_esa['pred_age_w'] = pd.to_numeric(urban_esa['pred_age_w'], errors='coerce')
urban_esa['pred_year_w'] = pd.to_numeric(urban_esa['pred_year_w'], errors='coerce')
urban_esa['pred_w'] = urban_esa['pred_age_w'] + urban_esa['pred_year_w']
urban_esa.loc[urban_esa['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = urban_esa.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data=urban_esa[urban_esa['sex'] == 1]).fit()
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
min_age = int(urban_esa[urban_esa['sex'] == 1]['age'].min())
max_age = int(urban_esa[urban_esa['sex'] == 1]['age'].max())
min_year = int(urban_esa[urban_esa['sex'] == 1]['syear'].min())
max_year = int(urban_esa[urban_esa['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        urban_esa.loc[urban_esa['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        urban_esa.loc[urban_esa['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

urban_esa['pred_year_m'] = urban_esa['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
urban_esa['pred_age_m'] = pd.to_numeric(urban_esa['pred_age_m'], errors='coerce')
urban_esa['pred_year_m'] = pd.to_numeric(urban_esa['pred_year_m'], errors='coerce')
urban_esa['pred_m'] = urban_esa['pred_age_m'] + urban_esa['pred_year_m']
urban_esa.loc[urban_esa['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = urban_esa.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/urban_results_esa.csv", index=False)