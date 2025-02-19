# 0214 Reform Estimation - Education Level

## Libraries

import numpy as np
import pandas as pd
import statsmodels.api as sm

## Import the Data

data = pd.read_csv("../../data/cleaned/cleaned_data.csv", index_col=0)

# Remove unobserved education
data = data[(data['pgbilzeit'] >= 0)] 

## Categorize into Education Levels

# Define a function to map pgbilzeit to education levels
def map_education_level(pgbilzeit):
    if pgbilzeit < 12:
        return 'Low education'
    elif pgbilzeit >= 12:
        return 'High education'
    else:
        return None  # In case of missing or unexpected values

# Apply the function to create the 'education' column
data['education_level'] = data['pgbilzeit'].apply(map_education_level)

## Run the Analysis with Income Approach

# Remove unobserved rows
ia_data = data[(data['real_income'] >= 0)] 

### Low Education

# Subset the data to keep low-edu individuals only 
lowedu_ia = ia_data[ia_data['education_level'] == 'Low education']
lowedu_ia = lowedu_ia.reset_index(drop=True)

### Split the Sample Based on the Year of Childbirth

# Split the data
bef_2007_ia = lowedu_ia[lowedu_ia['kidgeb01'] < 2007]
aft_2007_ia = lowedu_ia[lowedu_ia['kidgeb01'] >= 2007]

#### Childbirth before 2007

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data = bef_2007_ia[bef_2007_ia['sex'] == 2]).fit()
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
min_age = int(bef_2007_ia[bef_2007_ia['sex'] == 2]['age'].min())
max_age = int(bef_2007_ia[bef_2007_ia['sex'] == 2]['age'].max())
min_year = int(bef_2007_ia[bef_2007_ia['sex'] == 2]['syear'].min())
max_year = int(bef_2007_ia[bef_2007_ia['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        bef_2007_ia.loc[bef_2007_ia['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        bef_2007_ia.loc[bef_2007_ia['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

bef_2007_ia['pred_year_w'] = bef_2007_ia['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
bef_2007_ia['pred_age_w'] = pd.to_numeric(bef_2007_ia['pred_age_w'], errors='coerce')
bef_2007_ia['pred_year_w'] = pd.to_numeric(bef_2007_ia['pred_year_w'], errors='coerce')
bef_2007_ia['pred_w'] = bef_2007_ia['pred_age_w'] + bef_2007_ia['pred_year_w']
bef_2007_ia.loc[bef_2007_ia['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = bef_2007_ia.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data=bef_2007_ia[bef_2007_ia['sex'] == 1]).fit()
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
min_age = int(bef_2007_ia[bef_2007_ia['sex'] == 1]['age'].min())
max_age = int(bef_2007_ia[bef_2007_ia['sex'] == 1]['age'].max())
min_year = int(bef_2007_ia[bef_2007_ia['sex'] == 1]['syear'].min())
max_year = int(bef_2007_ia[bef_2007_ia['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        bef_2007_ia.loc[bef_2007_ia['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        bef_2007_ia.loc[bef_2007_ia['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

bef_2007_ia['pred_year_m'] = bef_2007_ia['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
bef_2007_ia['pred_age_m'] = pd.to_numeric(bef_2007_ia['pred_age_m'], errors='coerce')
bef_2007_ia['pred_year_m'] = pd.to_numeric(bef_2007_ia['pred_year_m'], errors='coerce')
bef_2007_ia['pred_m'] = bef_2007_ia['pred_age_m'] + bef_2007_ia['pred_year_m']
bef_2007_ia.loc[bef_2007_ia['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = bef_2007_ia.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/bef2007_results_ia_lowedu.csv", index=False)

#### Childbirth after 2007

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data = aft_2007_ia[aft_2007_ia['sex'] == 2]).fit()
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
min_age = int(aft_2007_ia[aft_2007_ia['sex'] == 2]['age'].min())
max_age = int(aft_2007_ia[aft_2007_ia['sex'] == 2]['age'].max())
min_year = int(aft_2007_ia[aft_2007_ia['sex'] == 2]['syear'].min())
max_year = int(aft_2007_ia[aft_2007_ia['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        aft_2007_ia.loc[aft_2007_ia['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        aft_2007_ia.loc[aft_2007_ia['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

aft_2007_ia['pred_year_w'] = aft_2007_ia['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
aft_2007_ia['pred_age_w'] = pd.to_numeric(aft_2007_ia['pred_age_w'], errors='coerce')
aft_2007_ia['pred_year_w'] = pd.to_numeric(aft_2007_ia['pred_year_w'], errors='coerce')
aft_2007_ia['pred_w'] = aft_2007_ia['pred_age_w'] + aft_2007_ia['pred_year_w']
aft_2007_ia.loc[aft_2007_ia['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = aft_2007_ia.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data=aft_2007_ia[aft_2007_ia['sex'] == 1]).fit()
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
min_age = int(aft_2007_ia[aft_2007_ia['sex'] == 1]['age'].min())
max_age = int(aft_2007_ia[aft_2007_ia['sex'] == 1]['age'].max())
min_year = int(aft_2007_ia[aft_2007_ia['sex'] == 1]['syear'].min())
max_year = int(aft_2007_ia[aft_2007_ia['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        aft_2007_ia.loc[aft_2007_ia['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        aft_2007_ia.loc[aft_2007_ia['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

aft_2007_ia['pred_year_m'] = aft_2007_ia['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
aft_2007_ia['pred_age_m'] = pd.to_numeric(aft_2007_ia['pred_age_m'], errors='coerce')
aft_2007_ia['pred_year_m'] = pd.to_numeric(aft_2007_ia['pred_year_m'], errors='coerce')
aft_2007_ia['pred_m'] = aft_2007_ia['pred_age_m'] + aft_2007_ia['pred_year_m']
aft_2007_ia.loc[aft_2007_ia['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = aft_2007_ia.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/aft2007_results_ia_lowedu.csv", index=False)

### High Education

# Subset the data to keep high-edu individuals only 
highedu_ia = ia_data[ia_data['education_level'] == 'High education']
highedu_ia = highedu_ia.reset_index(drop=True)

### Split the Sample Based on the Year of Childbirth

# Split the data
bef_2007_ia = highedu_ia[highedu_ia['kidgeb01'] < 2007]
aft_2007_ia = highedu_ia[highedu_ia['kidgeb01'] >= 2007]

#### Childbirth before 2007

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data = bef_2007_ia[bef_2007_ia['sex'] == 2]).fit()
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
min_age = int(bef_2007_ia[bef_2007_ia['sex'] == 2]['age'].min())
max_age = int(bef_2007_ia[bef_2007_ia['sex'] == 2]['age'].max())
min_year = int(bef_2007_ia[bef_2007_ia['sex'] == 2]['syear'].min())
max_year = int(bef_2007_ia[bef_2007_ia['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        bef_2007_ia.loc[bef_2007_ia['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        bef_2007_ia.loc[bef_2007_ia['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

bef_2007_ia['pred_year_w'] = bef_2007_ia['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
bef_2007_ia['pred_age_w'] = pd.to_numeric(bef_2007_ia['pred_age_w'], errors='coerce')
bef_2007_ia['pred_year_w'] = pd.to_numeric(bef_2007_ia['pred_year_w'], errors='coerce')
bef_2007_ia['pred_w'] = bef_2007_ia['pred_age_w'] + bef_2007_ia['pred_year_w']
bef_2007_ia.loc[bef_2007_ia['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = bef_2007_ia.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data=bef_2007_ia[bef_2007_ia['sex'] == 1]).fit()
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
min_age = int(bef_2007_ia[bef_2007_ia['sex'] == 1]['age'].min())
max_age = int(bef_2007_ia[bef_2007_ia['sex'] == 1]['age'].max())
min_year = int(bef_2007_ia[bef_2007_ia['sex'] == 1]['syear'].min())
max_year = int(bef_2007_ia[bef_2007_ia['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        bef_2007_ia.loc[bef_2007_ia['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        bef_2007_ia.loc[bef_2007_ia['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

bef_2007_ia['pred_year_m'] = bef_2007_ia['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
bef_2007_ia['pred_age_m'] = pd.to_numeric(bef_2007_ia['pred_age_m'], errors='coerce')
bef_2007_ia['pred_year_m'] = pd.to_numeric(bef_2007_ia['pred_year_m'], errors='coerce')
bef_2007_ia['pred_m'] = bef_2007_ia['pred_age_m'] + bef_2007_ia['pred_year_m']
bef_2007_ia.loc[bef_2007_ia['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = bef_2007_ia.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/bef2007_results_ia_highedu.csv", index=False)

#### Childbirth after 2007

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data = aft_2007_ia[aft_2007_ia['sex'] == 2]).fit()
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
min_age = int(aft_2007_ia[aft_2007_ia['sex'] == 2]['age'].min())
max_age = int(aft_2007_ia[aft_2007_ia['sex'] == 2]['age'].max())
min_year = int(aft_2007_ia[aft_2007_ia['sex'] == 2]['syear'].min())
max_year = int(aft_2007_ia[aft_2007_ia['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        aft_2007_ia.loc[aft_2007_ia['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        aft_2007_ia.loc[aft_2007_ia['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

aft_2007_ia['pred_year_w'] = aft_2007_ia['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
aft_2007_ia['pred_age_w'] = pd.to_numeric(aft_2007_ia['pred_age_w'], errors='coerce')
aft_2007_ia['pred_year_w'] = pd.to_numeric(aft_2007_ia['pred_year_w'], errors='coerce')
aft_2007_ia['pred_w'] = aft_2007_ia['pred_age_w'] + aft_2007_ia['pred_year_w']
aft_2007_ia.loc[aft_2007_ia['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = aft_2007_ia.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "real_income ~ C(age) + C(syear) + C(event_time) - 1", data=aft_2007_ia[aft_2007_ia['sex'] == 1]).fit()
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
min_age = int(aft_2007_ia[aft_2007_ia['sex'] == 1]['age'].min())
max_age = int(aft_2007_ia[aft_2007_ia['sex'] == 1]['age'].max())
min_year = int(aft_2007_ia[aft_2007_ia['sex'] == 1]['syear'].min())
max_year = int(aft_2007_ia[aft_2007_ia['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        aft_2007_ia.loc[aft_2007_ia['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        aft_2007_ia.loc[aft_2007_ia['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

aft_2007_ia['pred_year_m'] = aft_2007_ia['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
aft_2007_ia['pred_age_m'] = pd.to_numeric(aft_2007_ia['pred_age_m'], errors='coerce')
aft_2007_ia['pred_year_m'] = pd.to_numeric(aft_2007_ia['pred_year_m'], errors='coerce')
aft_2007_ia['pred_m'] = aft_2007_ia['pred_age_m'] + aft_2007_ia['pred_year_m']
aft_2007_ia.loc[aft_2007_ia['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = aft_2007_ia.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/aft2007_results_ia_highedu.csv", index=False)

## Run the Analysis with Working Hours Approach

# Remove unobserved rows
wha_data = data[(data['e11101'] >= 0)] 

### Low Education

# Subset the data to keep low-edu individuals only 
lowedu_wha = wha_data[wha_data['education_level'] == 'Low education']
lowedu_wha = lowedu_wha.reset_index(drop=True)

### Split the Sample Based on the Year of Childbirth

# Split the data
bef_2007_wha = lowedu_wha[lowedu_wha['kidgeb01'] < 2007]
aft_2007_wha = lowedu_wha[lowedu_wha['kidgeb01'] >= 2007]

#### Childbirth before 2007

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data = bef_2007_wha[bef_2007_wha['sex'] == 2]).fit()
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
min_age = int(bef_2007_wha[bef_2007_wha['sex'] == 2]['age'].min())
max_age = int(bef_2007_wha[bef_2007_wha['sex'] == 2]['age'].max())
min_year = int(bef_2007_wha[bef_2007_wha['sex'] == 2]['syear'].min())
max_year = int(bef_2007_wha[bef_2007_wha['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        bef_2007_wha.loc[bef_2007_wha['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        bef_2007_wha.loc[bef_2007_wha['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

bef_2007_wha['pred_year_w'] = bef_2007_wha['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
bef_2007_wha['pred_age_w'] = pd.to_numeric(bef_2007_wha['pred_age_w'], errors='coerce')
bef_2007_wha['pred_year_w'] = pd.to_numeric(bef_2007_wha['pred_year_w'], errors='coerce')
bef_2007_wha['pred_w'] = bef_2007_wha['pred_age_w'] + bef_2007_wha['pred_year_w']
bef_2007_wha.loc[bef_2007_wha['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = bef_2007_wha.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data=bef_2007_wha[bef_2007_wha['sex'] == 1]).fit()
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
min_age = int(bef_2007_wha[bef_2007_wha['sex'] == 1]['age'].min())
max_age = int(bef_2007_wha[bef_2007_wha['sex'] == 1]['age'].max())
min_year = int(bef_2007_wha[bef_2007_wha['sex'] == 1]['syear'].min())
max_year = int(bef_2007_wha[bef_2007_wha['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        bef_2007_wha.loc[bef_2007_wha['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        bef_2007_wha.loc[bef_2007_wha['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

bef_2007_wha['pred_year_m'] = bef_2007_wha['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
bef_2007_wha['pred_age_m'] = pd.to_numeric(bef_2007_wha['pred_age_m'], errors='coerce')
bef_2007_wha['pred_year_m'] = pd.to_numeric(bef_2007_wha['pred_year_m'], errors='coerce')
bef_2007_wha['pred_m'] = bef_2007_wha['pred_age_m'] + bef_2007_wha['pred_year_m']
bef_2007_wha.loc[bef_2007_wha['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = bef_2007_wha.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/bef2007_results_wha_lowedu.csv", index=False)

#### Childbirth after 2007

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data = aft_2007_wha[aft_2007_wha['sex'] == 2]).fit()
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
min_age = int(aft_2007_wha[aft_2007_wha['sex'] == 2]['age'].min())
max_age = int(aft_2007_wha[aft_2007_wha['sex'] == 2]['age'].max())
min_year = int(aft_2007_wha[aft_2007_wha['sex'] == 2]['syear'].min())
max_year = int(aft_2007_wha[aft_2007_wha['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        aft_2007_wha.loc[aft_2007_wha['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        aft_2007_wha.loc[aft_2007_wha['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

aft_2007_wha['pred_year_w'] = aft_2007_wha['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
aft_2007_wha['pred_age_w'] = pd.to_numeric(aft_2007_wha['pred_age_w'], errors='coerce')
aft_2007_wha['pred_year_w'] = pd.to_numeric(aft_2007_wha['pred_year_w'], errors='coerce')
aft_2007_wha['pred_w'] = aft_2007_wha['pred_age_w'] + aft_2007_wha['pred_year_w']
aft_2007_wha.loc[aft_2007_wha['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = aft_2007_wha.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data=aft_2007_wha[aft_2007_wha['sex'] == 1]).fit()
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
min_age = int(aft_2007_wha[aft_2007_wha['sex'] == 1]['age'].min())
max_age = int(aft_2007_wha[aft_2007_wha['sex'] == 1]['age'].max())
min_year = int(aft_2007_wha[aft_2007_wha['sex'] == 1]['syear'].min())
max_year = int(aft_2007_wha[aft_2007_wha['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        aft_2007_wha.loc[aft_2007_wha['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        aft_2007_wha.loc[aft_2007_wha['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

aft_2007_wha['pred_year_m'] = aft_2007_wha['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
aft_2007_wha['pred_age_m'] = pd.to_numeric(aft_2007_wha['pred_age_m'], errors='coerce')
aft_2007_wha['pred_year_m'] = pd.to_numeric(aft_2007_wha['pred_year_m'], errors='coerce')
aft_2007_wha['pred_m'] = aft_2007_wha['pred_age_m'] + aft_2007_wha['pred_year_m']
aft_2007_wha.loc[aft_2007_wha['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = aft_2007_wha.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/aft2007_results_wha_lowedu.csv", index=False)

### High Education

# Subset the data to keep high-edu individuals only 
highedu_wha = wha_data[wha_data['education_level'] == 'High education']
highedu_wha = highedu_wha.reset_index(drop=True)

### Split the Sample Based on the Year of Childbirth

# Split the data
bef_2007_wha = highedu_wha[highedu_wha['kidgeb01'] < 2007]
aft_2007_wha = highedu_wha[highedu_wha['kidgeb01'] >= 2007]

#### Childbirth before 2007

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data = bef_2007_wha[bef_2007_wha['sex'] == 2]).fit()
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
min_age = int(bef_2007_wha[bef_2007_wha['sex'] == 2]['age'].min())
max_age = int(bef_2007_wha[bef_2007_wha['sex'] == 2]['age'].max())
min_year = int(bef_2007_wha[bef_2007_wha['sex'] == 2]['syear'].min())
max_year = int(bef_2007_wha[bef_2007_wha['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        bef_2007_wha.loc[bef_2007_wha['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        bef_2007_wha.loc[bef_2007_wha['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

bef_2007_wha['pred_year_w'] = bef_2007_wha['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
bef_2007_wha['pred_age_w'] = pd.to_numeric(bef_2007_wha['pred_age_w'], errors='coerce')
bef_2007_wha['pred_year_w'] = pd.to_numeric(bef_2007_wha['pred_year_w'], errors='coerce')
bef_2007_wha['pred_w'] = bef_2007_wha['pred_age_w'] + bef_2007_wha['pred_year_w']
bef_2007_wha.loc[bef_2007_wha['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = bef_2007_wha.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data=bef_2007_wha[bef_2007_wha['sex'] == 1]).fit()
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
min_age = int(bef_2007_wha[bef_2007_wha['sex'] == 1]['age'].min())
max_age = int(bef_2007_wha[bef_2007_wha['sex'] == 1]['age'].max())
min_year = int(bef_2007_wha[bef_2007_wha['sex'] == 1]['syear'].min())
max_year = int(bef_2007_wha[bef_2007_wha['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        bef_2007_wha.loc[bef_2007_wha['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        bef_2007_wha.loc[bef_2007_wha['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

bef_2007_wha['pred_year_m'] = bef_2007_wha['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
bef_2007_wha['pred_age_m'] = pd.to_numeric(bef_2007_wha['pred_age_m'], errors='coerce')
bef_2007_wha['pred_year_m'] = pd.to_numeric(bef_2007_wha['pred_year_m'], errors='coerce')
bef_2007_wha['pred_m'] = bef_2007_wha['pred_age_m'] + bef_2007_wha['pred_year_m']
bef_2007_wha.loc[bef_2007_wha['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = bef_2007_wha.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/bef2007_results_wha_highedu.csv", index=False)

#### Childbirth before 2007

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data = aft_2007_wha[aft_2007_wha['sex'] == 2]).fit()
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
min_age = int(aft_2007_wha[aft_2007_wha['sex'] == 2]['age'].min())
max_age = int(aft_2007_wha[aft_2007_wha['sex'] == 2]['age'].max())
min_year = int(aft_2007_wha[aft_2007_wha['sex'] == 2]['syear'].min())
max_year = int(aft_2007_wha[aft_2007_wha['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        aft_2007_wha.loc[aft_2007_wha['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        aft_2007_wha.loc[aft_2007_wha['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

aft_2007_wha['pred_year_w'] = aft_2007_wha['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
aft_2007_wha['pred_age_w'] = pd.to_numeric(aft_2007_wha['pred_age_w'], errors='coerce')
aft_2007_wha['pred_year_w'] = pd.to_numeric(aft_2007_wha['pred_year_w'], errors='coerce')
aft_2007_wha['pred_w'] = aft_2007_wha['pred_age_w'] + aft_2007_wha['pred_year_w']
aft_2007_wha.loc[aft_2007_wha['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = aft_2007_wha.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data=aft_2007_wha[aft_2007_wha['sex'] == 1]).fit()
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
min_age = int(aft_2007_wha[aft_2007_wha['sex'] == 1]['age'].min())
max_age = int(aft_2007_wha[aft_2007_wha['sex'] == 1]['age'].max())
min_year = int(aft_2007_wha[aft_2007_wha['sex'] == 1]['syear'].min())
max_year = int(aft_2007_wha[aft_2007_wha['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        aft_2007_wha.loc[aft_2007_wha['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        aft_2007_wha.loc[aft_2007_wha['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

aft_2007_wha['pred_year_m'] = aft_2007_wha['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
aft_2007_wha['pred_age_m'] = pd.to_numeric(aft_2007_wha['pred_age_m'], errors='coerce')
aft_2007_wha['pred_year_m'] = pd.to_numeric(aft_2007_wha['pred_year_m'], errors='coerce')
aft_2007_wha['pred_m'] = aft_2007_wha['pred_age_m'] + aft_2007_wha['pred_year_m']
aft_2007_wha.loc[aft_2007_wha['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = aft_2007_wha.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/aft2007_results_wha_highedu.csv", index=False)

## Run the Analysis with Employment Status Approach

# Remove unobserved rows
esa_data = data[(data['e11102'] >= 0)]  

### Low Education

# Subset the data to keep low-edu individuals only 
lowedu_esa = esa_data[esa_data['education_level'] == 'Low education']
lowedu_esa = lowedu_esa.reset_index(drop=True)

### Split the Sample Based on the Year of Childbirth

# Split the data
bef_2007_esa = lowedu_esa[lowedu_esa['kidgeb01'] < 2007]
aft_2007_esa = lowedu_esa[lowedu_esa['kidgeb01'] >= 2007]

#### Childbirth before 2007

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data = bef_2007_esa[bef_2007_esa['sex'] == 2]).fit()
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
min_age = int(bef_2007_esa[bef_2007_esa['sex'] == 2]['age'].min())
max_age = int(bef_2007_esa[bef_2007_esa['sex'] == 2]['age'].max())
min_year = int(bef_2007_esa[bef_2007_esa['sex'] == 2]['syear'].min())
max_year = int(bef_2007_esa[bef_2007_esa['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        bef_2007_esa.loc[bef_2007_esa['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        bef_2007_esa.loc[bef_2007_esa['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

bef_2007_esa['pred_year_w'] = bef_2007_esa['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
bef_2007_esa['pred_age_w'] = pd.to_numeric(bef_2007_esa['pred_age_w'], errors='coerce')
bef_2007_esa['pred_year_w'] = pd.to_numeric(bef_2007_esa['pred_year_w'], errors='coerce')
bef_2007_esa['pred_w'] = bef_2007_esa['pred_age_w'] + bef_2007_esa['pred_year_w']
bef_2007_esa.loc[bef_2007_esa['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = bef_2007_esa.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data=bef_2007_esa[bef_2007_esa['sex'] == 1]).fit()
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
min_age = int(bef_2007_esa[bef_2007_esa['sex'] == 1]['age'].min())
max_age = int(bef_2007_esa[bef_2007_esa['sex'] == 1]['age'].max())
min_year = int(bef_2007_esa[bef_2007_esa['sex'] == 1]['syear'].min())
max_year = int(bef_2007_esa[bef_2007_esa['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        bef_2007_esa.loc[bef_2007_esa['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        bef_2007_esa.loc[bef_2007_esa['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

bef_2007_esa['pred_year_m'] = bef_2007_esa['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
bef_2007_esa['pred_age_m'] = pd.to_numeric(bef_2007_esa['pred_age_m'], errors='coerce')
bef_2007_esa['pred_year_m'] = pd.to_numeric(bef_2007_esa['pred_year_m'], errors='coerce')
bef_2007_esa['pred_m'] = bef_2007_esa['pred_age_m'] + bef_2007_esa['pred_year_m']
bef_2007_esa.loc[bef_2007_esa['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = bef_2007_esa.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/bef2007_results_esa_lowedu.csv", index=False)

#### Childbirth after 2007

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data = aft_2007_esa[aft_2007_esa['sex'] == 2]).fit()
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
min_age = int(aft_2007_esa[aft_2007_esa['sex'] == 2]['age'].min())
max_age = int(aft_2007_esa[aft_2007_esa['sex'] == 2]['age'].max())
min_year = int(aft_2007_esa[aft_2007_esa['sex'] == 2]['syear'].min())
max_year = int(aft_2007_esa[aft_2007_esa['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        aft_2007_esa.loc[aft_2007_esa['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        aft_2007_esa.loc[aft_2007_esa['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

aft_2007_esa['pred_year_w'] = aft_2007_esa['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
aft_2007_esa['pred_age_w'] = pd.to_numeric(aft_2007_esa['pred_age_w'], errors='coerce')
aft_2007_esa['pred_year_w'] = pd.to_numeric(aft_2007_esa['pred_year_w'], errors='coerce')
aft_2007_esa['pred_w'] = aft_2007_esa['pred_age_w'] + aft_2007_esa['pred_year_w']
aft_2007_esa.loc[aft_2007_esa['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = aft_2007_esa.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data=aft_2007_esa[aft_2007_esa['sex'] == 1]).fit()
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
min_age = int(aft_2007_esa[aft_2007_esa['sex'] == 1]['age'].min())
max_age = int(aft_2007_esa[aft_2007_esa['sex'] == 1]['age'].max())
min_year = int(aft_2007_esa[aft_2007_esa['sex'] == 1]['syear'].min())
max_year = int(aft_2007_esa[aft_2007_esa['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        aft_2007_esa.loc[aft_2007_esa['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        aft_2007_esa.loc[aft_2007_esa['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

aft_2007_esa['pred_year_m'] = aft_2007_esa['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
aft_2007_esa['pred_age_m'] = pd.to_numeric(aft_2007_esa['pred_age_m'], errors='coerce')
aft_2007_esa['pred_year_m'] = pd.to_numeric(aft_2007_esa['pred_year_m'], errors='coerce')
aft_2007_esa['pred_m'] = aft_2007_esa['pred_age_m'] + aft_2007_esa['pred_year_m']
aft_2007_esa.loc[aft_2007_esa['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = aft_2007_esa.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/aft2007_results_esa_lowedu.csv", index=False)

### High Education

# Subset the data to keep high-edu individuals only 
highedu_esa = esa_data[esa_data['education_level'] == 'High education']
highedu_esa = highedu_esa.reset_index(drop=True)

### Split the Sample Based on the Year of Childbirth

# Split the data
bef_2007_esa = highedu_esa[highedu_esa['kidgeb01'] < 2007]
aft_2007_esa = highedu_esa[highedu_esa['kidgeb01'] >= 2007]

#### Childbirth before 2007

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data = bef_2007_esa[bef_2007_esa['sex'] == 2]).fit()
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
min_age = int(bef_2007_esa[bef_2007_esa['sex'] == 2]['age'].min())
max_age = int(bef_2007_esa[bef_2007_esa['sex'] == 2]['age'].max())
min_year = int(bef_2007_esa[bef_2007_esa['sex'] == 2]['syear'].min())
max_year = int(bef_2007_esa[bef_2007_esa['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        bef_2007_esa.loc[bef_2007_esa['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        bef_2007_esa.loc[bef_2007_esa['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

bef_2007_esa['pred_year_w'] = bef_2007_esa['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
bef_2007_esa['pred_age_w'] = pd.to_numeric(bef_2007_esa['pred_age_w'], errors='coerce')
bef_2007_esa['pred_year_w'] = pd.to_numeric(bef_2007_esa['pred_year_w'], errors='coerce')
bef_2007_esa['pred_w'] = bef_2007_esa['pred_age_w'] + bef_2007_esa['pred_year_w']
bef_2007_esa.loc[bef_2007_esa['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = bef_2007_esa.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data=bef_2007_esa[bef_2007_esa['sex'] == 1]).fit()
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
min_age = int(bef_2007_esa[bef_2007_esa['sex'] == 1]['age'].min())
max_age = int(bef_2007_esa[bef_2007_esa['sex'] == 1]['age'].max())
min_year = int(bef_2007_esa[bef_2007_esa['sex'] == 1]['syear'].min())
max_year = int(bef_2007_esa[bef_2007_esa['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        bef_2007_esa.loc[bef_2007_esa['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        bef_2007_esa.loc[bef_2007_esa['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

bef_2007_esa['pred_year_m'] = bef_2007_esa['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
bef_2007_esa['pred_age_m'] = pd.to_numeric(bef_2007_esa['pred_age_m'], errors='coerce')
bef_2007_esa['pred_year_m'] = pd.to_numeric(bef_2007_esa['pred_year_m'], errors='coerce')
bef_2007_esa['pred_m'] = bef_2007_esa['pred_age_m'] + bef_2007_esa['pred_year_m']
bef_2007_esa.loc[bef_2007_esa['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = bef_2007_esa.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/bef2007_results_esa_highedu.csv", index=False)

#### Childbirth after 2007

#### Woman

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data = aft_2007_esa[aft_2007_esa['sex'] == 2]).fit()
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
min_age = int(aft_2007_esa[aft_2007_esa['sex'] == 2]['age'].min())
max_age = int(aft_2007_esa[aft_2007_esa['sex'] == 2]['age'].max())
min_year = int(aft_2007_esa[aft_2007_esa['sex'] == 2]['syear'].min())
max_year = int(aft_2007_esa[aft_2007_esa['sex'] == 2]['syear'].max())
# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        aft_2007_esa.loc[aft_2007_esa['age'] == age, 'pred_age_w'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        aft_2007_esa.loc[aft_2007_esa['syear'] == year, 'pred_year_w'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

aft_2007_esa['pred_year_w'] = aft_2007_esa['pred_year_w'].apply(lambda x: '{:.6f}'.format(x))
aft_2007_esa['pred_age_w'] = pd.to_numeric(aft_2007_esa['pred_age_w'], errors='coerce')
aft_2007_esa['pred_year_w'] = pd.to_numeric(aft_2007_esa['pred_year_w'], errors='coerce')
aft_2007_esa['pred_w'] = aft_2007_esa['pred_age_w'] + aft_2007_esa['pred_year_w']
aft_2007_esa.loc[aft_2007_esa['sex'] == 1, 'pred_w'] = np.nan # Prediction for men will be calculated seperately
# Prediction of earning in the absence of childbirth event
prediction_df = aft_2007_esa.groupby(['event_time', 'sex'])['pred_w'].mean().reset_index()
prediction_df.dropna(inplace=True)
prediction_df = prediction_df.drop('sex', axis = 1)
# Result for women
results_w_df = pd.merge(prediction_df, eventtime_df[['event_time', 'coef']], on = 'event_time')
results_w_df['coef'] = results_w_df['coef'].astype(float)
results_w_df['percentage_coef_w'] = results_w_df['coef'] / results_w_df['pred_w']

#### Man

# OLS Model
model = sm.OLS.from_formula(formula = "e11102 ~ C(age) + C(syear) + C(event_time) - 1", data=aft_2007_esa[aft_2007_esa['sex'] == 1]).fit()
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
min_age = int(aft_2007_esa[aft_2007_esa['sex'] == 1]['age'].min())
max_age = int(aft_2007_esa[aft_2007_esa['sex'] == 1]['age'].max())
min_year = int(aft_2007_esa[aft_2007_esa['sex'] == 1]['syear'].min())
max_year = int(aft_2007_esa[aft_2007_esa['sex'] == 1]['syear'].max())

# Calculate the counterfactual
for age in range(min_age, max_age + 1):
    try:
        aft_2007_esa.loc[aft_2007_esa['age'] == age, 'pred_age_m'] = float(counterfactual_df.loc[f'C(age)[{age}]', 'coef'])
    except KeyError:
        continue

for year in range(min_year, max_year + 1):
    try:
        aft_2007_esa.loc[aft_2007_esa['syear'] == year, 'pred_year_m'] = float(counterfactual_df.loc[f'C(syear)[T.{year}]', 'coef'])
    except KeyError:
        continue

aft_2007_esa['pred_year_m'] = aft_2007_esa['pred_year_m'].apply(lambda x: '{:.6f}'.format(x))
aft_2007_esa['pred_age_m'] = pd.to_numeric(aft_2007_esa['pred_age_m'], errors='coerce')
aft_2007_esa['pred_year_m'] = pd.to_numeric(aft_2007_esa['pred_year_m'], errors='coerce')
aft_2007_esa['pred_m'] = aft_2007_esa['pred_age_m'] + aft_2007_esa['pred_year_m']
aft_2007_esa.loc[aft_2007_esa['sex'] == 2, 'pred_m'] = np.nan # Prediction for women is already calculated
# Prediction of earning in the absence of childbirth event
prediction_df = aft_2007_esa.groupby(['event_time', 'sex'])['pred_m'].mean().reset_index()
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
event_study_results.to_csv("../../data/processed/aft2007_results_esa_highedu.csv", index=False)
