# 0207 Estimation - Age at First Parenthood

## Libraries

import numpy as np
import pandas as pd
import statsmodels.api as sm

## Import the Data

data = pd.read_csv("../../data/cleaned/cleaned_data.csv", index_col=0)

## Categorize into Parenthood Age

# Define a function to map age of parenthood
def map_age_of_parenthood(age_at_first_birth):
    if age_at_first_birth < 25:
        return 'Early'
    elif (age_at_first_birth >= 25) & (age_at_first_birth < 35):
        return 'Middle'
    elif (age_at_first_birth >= 35):
        return 'Late'

# Apply the function to create the 'education' column
data['age_of_parenthood'] = data['age_at_first_birth'].apply(map_age_of_parenthood)

## Run the Analysis with Income Approach

# Remove unobserved rows
ia_data = data[(data['real_income'] >= 0)] 

### Early Parenthood

# Subset the data to keep early individuals only 
young_ia = ia_data[ia_data['age_of_parenthood'] == 'Early']
young_ia = young_ia.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data = young_ia[young_ia['sex'] == 2]).fit()
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
min_age = int(young_ia[young_ia['sex'] == 2]['age'].min())
max_age = int(young_ia[young_ia['sex'] == 2]['age'].max())
min_year = int(young_ia[young_ia['sex'] == 2]['syear'].min())
max_year = int(young_ia[young_ia['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        young_ia.loc[young_ia['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        young_ia.loc[young_ia['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

young_ia['pred_year_w'] = young_ia['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
young_ia['pred_age_w'] = pd.to_numeric(young_ia['pred_age_w'], errors='coerce')
young_ia['pred_year_w'] = pd.to_numeric(young_ia['pred_year_w'], errors='coerce')
young_ia['pred_w'] = young_ia['pred_age_w'] + young_ia['pred_year_w']
young_ia.loc[young_ia['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = young_ia.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data=young_ia[young_ia['sex'] == 1]).fit()
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
min_age = int(young_ia[young_ia['sex'] == 1]['age'].min())
max_age = int(young_ia[young_ia['sex'] == 1]['age'].max())
min_year = int(young_ia[young_ia['sex'] == 1]['syear'].min())
max_year = int(young_ia[young_ia['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        young_ia.loc[young_ia['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        young_ia.loc[young_ia['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

young_ia['pred_year_m'] = young_ia['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
young_ia['pred_age_m'] = pd.to_numeric(young_ia['pred_age_m'], errors='coerce')
young_ia['pred_year_m'] = pd.to_numeric(young_ia['pred_year_m'], errors='coerce')
young_ia['pred_m'] = young_ia['pred_age_m'] + young_ia['pred_year_m']
young_ia.loc[young_ia['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = young_ia.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/young_results_ia.csv", index=False)

### Median Parenthood

# Subset the data to keep early-mid parents only 
earlymid_ia = ia_data[ia_data['age_of_parenthood'] == 'Middle']
earlymid_ia = earlymid_ia.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data = earlymid_ia[earlymid_ia['sex'] == 2]).fit()
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
min_age = int(earlymid_ia[earlymid_ia['sex'] == 2]['age'].min())
max_age = int(earlymid_ia[earlymid_ia['sex'] == 2]['age'].max())
min_year = int(earlymid_ia[earlymid_ia['sex'] == 2]['syear'].min())
max_year = int(earlymid_ia[earlymid_ia['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        earlymid_ia.loc[earlymid_ia['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        earlymid_ia.loc[earlymid_ia['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

earlymid_ia['pred_year_w'] = earlymid_ia['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
earlymid_ia['pred_age_w'] = pd.to_numeric(earlymid_ia['pred_age_w'], errors='coerce')
earlymid_ia['pred_year_w'] = pd.to_numeric(earlymid_ia['pred_year_w'], errors='coerce')
earlymid_ia['pred_w'] = earlymid_ia['pred_age_w'] + earlymid_ia['pred_year_w']
earlymid_ia.loc[earlymid_ia['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = earlymid_ia.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data=earlymid_ia[earlymid_ia['sex'] == 1]).fit()
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
min_age = int(earlymid_ia[earlymid_ia['sex'] == 1]['age'].min())
max_age = int(earlymid_ia[earlymid_ia['sex'] == 1]['age'].max())
min_year = int(earlymid_ia[earlymid_ia['sex'] == 1]['syear'].min())
max_year = int(earlymid_ia[earlymid_ia['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        earlymid_ia.loc[earlymid_ia['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        earlymid_ia.loc[earlymid_ia['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

earlymid_ia['pred_year_m'] = earlymid_ia['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
earlymid_ia['pred_age_m'] = pd.to_numeric(earlymid_ia['pred_age_m'], errors='coerce')
earlymid_ia['pred_year_m'] = pd.to_numeric(earlymid_ia['pred_year_m'], errors='coerce')
earlymid_ia['pred_m'] = earlymid_ia['pred_age_m'] + earlymid_ia['pred_year_m']
earlymid_ia.loc[earlymid_ia['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = earlymid_ia.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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

### Late Parenthood

# Subset the data to keep late-mid parents only 
latemid_ia = ia_data[ia_data['age_of_parenthood'] == 'Late']
latemid_ia = latemid_ia.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data = latemid_ia[latemid_ia['sex'] == 2]).fit()
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
min_age = int(latemid_ia[latemid_ia['sex'] == 2]['age'].min())
max_age = int(latemid_ia[latemid_ia['sex'] == 2]['age'].max())
min_year = int(latemid_ia[latemid_ia['sex'] == 2]['syear'].min())
max_year = int(latemid_ia[latemid_ia['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        latemid_ia.loc[latemid_ia['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        latemid_ia.loc[latemid_ia['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

latemid_ia['pred_year_w'] = latemid_ia['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
latemid_ia['pred_age_w'] = pd.to_numeric(latemid_ia['pred_age_w'], errors='coerce')
latemid_ia['pred_year_w'] = pd.to_numeric(latemid_ia['pred_year_w'], errors='coerce')
latemid_ia['pred_w'] = latemid_ia['pred_age_w'] + latemid_ia['pred_year_w']
latemid_ia.loc[latemid_ia['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = latemid_ia.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data=latemid_ia[latemid_ia['sex'] == 1]).fit()
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
min_age = int(latemid_ia[latemid_ia['sex'] == 1]['age'].min())
max_age = int(latemid_ia[latemid_ia['sex'] == 1]['age'].max())
min_year = int(latemid_ia[latemid_ia['sex'] == 1]['syear'].min())
max_year = int(latemid_ia[latemid_ia['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        latemid_ia.loc[latemid_ia['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        latemid_ia.loc[latemid_ia['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

latemid_ia['pred_year_m'] = latemid_ia['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
latemid_ia['pred_age_m'] = pd.to_numeric(latemid_ia['pred_age_m'], errors='coerce')
latemid_ia['pred_year_m'] = pd.to_numeric(latemid_ia['pred_year_m'], errors='coerce')
latemid_ia['pred_m'] = latemid_ia['pred_age_m'] + latemid_ia['pred_year_m']
latemid_ia.loc[latemid_ia['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = latemid_ia.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/latemid_results_ia.csv", index=False)

## Run the Analysis with Working Hours Approach

# Remove unobserved rows
wha_data = data[(data['e11101'] >= 0)] 

### Early Parenthood

# Subset the data to keep young individuals only 
young_wha = wha_data[wha_data['age_of_parenthood'] == 'Early']
young_wha = young_wha.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data = young_wha[young_wha['sex'] == 2]).fit()
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
min_age = int(young_wha[young_wha['sex'] == 2]['age'].min())
max_age = int(young_wha[young_wha['sex'] == 2]['age'].max())
min_year = int(young_wha[young_wha['sex'] == 2]['syear'].min())
max_year = int(young_wha[young_wha['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        young_wha.loc[young_wha['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        young_wha.loc[young_wha['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

young_wha['pred_year_w'] = young_wha['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
young_wha['pred_age_w'] = pd.to_numeric(young_wha['pred_age_w'], errors='coerce')
young_wha['pred_year_w'] = pd.to_numeric(young_wha['pred_year_w'], errors='coerce')
young_wha['pred_w'] = young_wha['pred_age_w'] + young_wha['pred_year_w']
young_wha.loc[young_wha['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = young_wha.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data=young_wha[young_wha['sex'] == 1]).fit()
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
min_age = int(young_wha[young_wha['sex'] == 1]['age'].min())
max_age = int(young_wha[young_wha['sex'] == 1]['age'].max())
min_year = int(young_wha[young_wha['sex'] == 1]['syear'].min())
max_year = int(young_wha[young_wha['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        young_wha.loc[young_wha['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        young_wha.loc[young_wha['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

young_wha['pred_year_m'] = young_wha['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
young_wha['pred_age_m'] = pd.to_numeric(young_wha['pred_age_m'], errors='coerce')
young_wha['pred_year_m'] = pd.to_numeric(young_wha['pred_year_m'], errors='coerce')
young_wha['pred_m'] = young_wha['pred_age_m'] + young_wha['pred_year_m']
young_wha.loc[young_wha['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = young_wha.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/young_results_wha.csv", index=False)

### Middle Parenthood

# Subset the data to keep early-mid parents only 
earlymid_wha = wha_data[wha_data['age_of_parenthood'] == 'Middle']
earlymid_wha = earlymid_wha.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data = earlymid_wha[earlymid_wha['sex'] == 2]).fit()
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
min_age = int(earlymid_wha[earlymid_wha['sex'] == 2]['age'].min())
max_age = int(earlymid_wha[earlymid_wha['sex'] == 2]['age'].max())
min_year = int(earlymid_wha[earlymid_wha['sex'] == 2]['syear'].min())
max_year = int(earlymid_wha[earlymid_wha['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        earlymid_wha.loc[earlymid_wha['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        earlymid_wha.loc[earlymid_wha['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

earlymid_wha['pred_year_w'] = earlymid_wha['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
earlymid_wha['pred_age_w'] = pd.to_numeric(earlymid_wha['pred_age_w'], errors='coerce')
earlymid_wha['pred_year_w'] = pd.to_numeric(earlymid_wha['pred_year_w'], errors='coerce')
earlymid_wha['pred_w'] = earlymid_wha['pred_age_w'] + earlymid_wha['pred_year_w']
earlymid_wha.loc[earlymid_wha['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = earlymid_wha.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data=earlymid_wha[earlymid_wha['sex'] == 1]).fit()
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
min_age = int(earlymid_wha[earlymid_wha['sex'] == 1]['age'].min())
max_age = int(earlymid_wha[earlymid_wha['sex'] == 1]['age'].max())
min_year = int(earlymid_wha[earlymid_wha['sex'] == 1]['syear'].min())
max_year = int(earlymid_wha[earlymid_wha['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        earlymid_wha.loc[earlymid_wha['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        earlymid_wha.loc[earlymid_wha['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

earlymid_wha['pred_year_m'] = earlymid_wha['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
earlymid_wha['pred_age_m'] = pd.to_numeric(earlymid_wha['pred_age_m'], errors='coerce')
earlymid_wha['pred_year_m'] = pd.to_numeric(earlymid_wha['pred_year_m'], errors='coerce')
earlymid_wha['pred_m'] = earlymid_wha['pred_age_m'] + earlymid_wha['pred_year_m']
earlymid_wha.loc[earlymid_wha['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = earlymid_wha.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/earlymid_results_wha.csv", index=False)

### Late Parenthood

# Subset the data to keep late-mid parents only 
latemid_wha = wha_data[wha_data['age_of_parenthood'] == 'Late']
latemid_wha = latemid_wha.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data = latemid_wha[latemid_wha['sex'] == 2]).fit()
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
min_age = int(latemid_wha[latemid_wha['sex'] == 2]['age'].min())
max_age = int(latemid_wha[latemid_wha['sex'] == 2]['age'].max())
min_year = int(latemid_wha[latemid_wha['sex'] == 2]['syear'].min())
max_year = int(latemid_wha[latemid_wha['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        latemid_wha.loc[latemid_wha['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        latemid_wha.loc[latemid_wha['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

latemid_wha['pred_year_w'] = latemid_wha['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
latemid_wha['pred_age_w'] = pd.to_numeric(latemid_wha['pred_age_w'], errors='coerce')
latemid_wha['pred_year_w'] = pd.to_numeric(latemid_wha['pred_year_w'], errors='coerce')
latemid_wha['pred_w'] = latemid_wha['pred_age_w'] + latemid_wha['pred_year_w']
latemid_wha.loc[latemid_wha['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = latemid_wha.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data=latemid_wha[latemid_wha['sex'] == 1]).fit()
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
min_age = int(latemid_wha[latemid_wha['sex'] == 1]['age'].min())
max_age = int(latemid_wha[latemid_wha['sex'] == 1]['age'].max())
min_year = int(latemid_wha[latemid_wha['sex'] == 1]['syear'].min())
max_year = int(latemid_wha[latemid_wha['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        latemid_wha.loc[latemid_wha['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        latemid_wha.loc[latemid_wha['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

latemid_wha['pred_year_m'] = latemid_wha['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
latemid_wha['pred_age_m'] = pd.to_numeric(latemid_wha['pred_age_m'], errors='coerce')
latemid_wha['pred_year_m'] = pd.to_numeric(latemid_wha['pred_year_m'], errors='coerce')
latemid_wha['pred_m'] = latemid_wha['pred_age_m'] + latemid_wha['pred_year_m']
latemid_wha.loc[latemid_wha['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = latemid_wha.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/latemid_results_wha.csv", index=False)

## Run the Analysis with Employment Status Approach

# Remove unobserved rows
esa_data = data[(data['e11102'] >= 0)]  

### Early Parenthood

# Subset the data to keep young individuals only 
young_esa = esa_data[esa_data['age_of_parenthood'] == 'Early']
young_esa = young_esa.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data = young_esa[young_esa['sex'] == 2]).fit()
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
min_age = int(young_esa[young_esa['sex'] == 2]['age'].min())
max_age = int(young_esa[young_esa['sex'] == 2]['age'].max())
min_year = int(young_esa[young_esa['sex'] == 2]['syear'].min())
max_year = int(young_esa[young_esa['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        young_esa.loc[young_esa['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        young_esa.loc[young_esa['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

young_esa['pred_year_w'] = young_esa['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
young_esa['pred_age_w'] = pd.to_numeric(young_esa['pred_age_w'], errors='coerce')
young_esa['pred_year_w'] = pd.to_numeric(young_esa['pred_year_w'], errors='coerce')
young_esa['pred_w'] = young_esa['pred_age_w'] + young_esa['pred_year_w']
young_esa.loc[young_esa['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = young_esa.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data=young_esa[young_esa['sex'] == 1]).fit()
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
min_age = int(young_esa[young_esa['sex'] == 1]['age'].min())
max_age = int(young_esa[young_esa['sex'] == 1]['age'].max())
min_year = int(young_esa[young_esa['sex'] == 1]['syear'].min())
max_year = int(young_esa[young_esa['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        young_esa.loc[young_esa['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        young_esa.loc[young_esa['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

young_esa['pred_year_m'] = young_esa['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
young_esa['pred_age_m'] = pd.to_numeric(young_esa['pred_age_m'], errors='coerce')
young_esa['pred_year_m'] = pd.to_numeric(young_esa['pred_year_m'], errors='coerce')
young_esa['pred_m'] = young_esa['pred_age_m'] + young_esa['pred_year_m']
young_esa.loc[young_esa['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = young_esa.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/young_results_esa.csv", index=False)

### Middle Parenthood

# Subset the data to keep early-mid parents only 
earlymid_esa = esa_data[esa_data['age_of_parenthood'] == 'Middle']
earlymid_esa = earlymid_esa.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data = earlymid_esa[earlymid_esa['sex'] == 2]).fit()
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
min_age = int(earlymid_esa[earlymid_esa['sex'] == 2]['age'].min())
max_age = int(earlymid_esa[earlymid_esa['sex'] == 2]['age'].max())
min_year = int(earlymid_esa[earlymid_esa['sex'] == 2]['syear'].min())
max_year = int(earlymid_esa[earlymid_esa['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        earlymid_esa.loc[earlymid_esa['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        earlymid_esa.loc[earlymid_esa['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

earlymid_esa['pred_year_w'] = earlymid_esa['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
earlymid_esa['pred_age_w'] = pd.to_numeric(earlymid_esa['pred_age_w'], errors='coerce')
earlymid_esa['pred_year_w'] = pd.to_numeric(earlymid_esa['pred_year_w'], errors='coerce')
earlymid_esa['pred_w'] = earlymid_esa['pred_age_w'] + earlymid_esa['pred_year_w']
earlymid_esa.loc[earlymid_esa['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = earlymid_esa.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data=earlymid_esa[earlymid_esa['sex'] == 1]).fit()
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
min_age = int(earlymid_esa[earlymid_esa['sex'] == 1]['age'].min())
max_age = int(earlymid_esa[earlymid_esa['sex'] == 1]['age'].max())
min_year = int(earlymid_esa[earlymid_esa['sex'] == 1]['syear'].min())
max_year = int(earlymid_esa[earlymid_esa['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        earlymid_esa.loc[earlymid_esa['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        earlymid_esa.loc[earlymid_esa['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

earlymid_esa['pred_year_m'] = earlymid_esa['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
earlymid_esa['pred_age_m'] = pd.to_numeric(earlymid_esa['pred_age_m'], errors='coerce')
earlymid_esa['pred_year_m'] = pd.to_numeric(earlymid_esa['pred_year_m'], errors='coerce')
earlymid_esa['pred_m'] = earlymid_esa['pred_age_m'] + earlymid_esa['pred_year_m']
earlymid_esa.loc[earlymid_esa['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = earlymid_esa.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/earlymid_results_esa.csv", index=False)

### Late Parenthood

# Subset the data to keep late-mid parents only 
latemid_esa = esa_data[esa_data['age_of_parenthood'] == 'Late']
latemid_esa = latemid_esa.reset_index(drop=True)

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data = latemid_esa[latemid_esa['sex'] == 2]).fit()
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
min_age = int(latemid_esa[latemid_esa['sex'] == 2]['age'].min())
max_age = int(latemid_esa[latemid_esa['sex'] == 2]['age'].max())
min_year = int(latemid_esa[latemid_esa['sex'] == 2]['syear'].min())
max_year = int(latemid_esa[latemid_esa['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        latemid_esa.loc[latemid_esa['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        latemid_esa.loc[latemid_esa['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

latemid_esa['pred_year_w'] = latemid_esa['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
latemid_esa['pred_age_w'] = pd.to_numeric(latemid_esa['pred_age_w'], errors='coerce')
latemid_esa['pred_year_w'] = pd.to_numeric(latemid_esa['pred_year_w'], errors='coerce')
latemid_esa['pred_w'] = latemid_esa['pred_age_w'] + latemid_esa['pred_year_w']
latemid_esa.loc[latemid_esa['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = latemid_esa.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data=latemid_esa[latemid_esa['sex'] == 1]).fit()
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
min_age = int(latemid_esa[latemid_esa['sex'] == 1]['age'].min())
max_age = int(latemid_esa[latemid_esa['sex'] == 1]['age'].max())
min_year = int(latemid_esa[latemid_esa['sex'] == 1]['syear'].min())
max_year = int(latemid_esa[latemid_esa['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        latemid_esa.loc[latemid_esa['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        latemid_esa.loc[latemid_esa['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

latemid_esa['pred_year_m'] = latemid_esa['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
latemid_esa['pred_age_m'] = pd.to_numeric(latemid_esa['pred_age_m'], errors='coerce')
latemid_esa['pred_year_m'] = pd.to_numeric(latemid_esa['pred_year_m'], errors='coerce')
latemid_esa['pred_m'] = latemid_esa['pred_age_m'] + latemid_esa['pred_year_m']
latemid_esa.loc[latemid_esa['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = latemid_esa.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/latemid_results_esa.csv", index=False)
