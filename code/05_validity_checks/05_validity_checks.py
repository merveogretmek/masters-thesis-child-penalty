# 05 Validity

## Libraries

import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from scipy.stats import f
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

## Exogeneity of Childbirth Timing

### 1. Pre-Trend Analysis

#### Import the Results

event_study_results_ia = pd.read_csv("../../data/processed/general_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
event_study_results_wha = pd.read_csv("../../data/processed/general_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
event_study_results_esa = pd.read_csv("../../data/processed/general_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

# Income Approach
lr_mean_m_ia = event_study_results_ia.loc[event_study_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_ia = event_study_results_ia.loc[event_study_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_ia = lr_mean_m_ia - lr_mean_w_ia

# Working Hours Approach
lr_mean_m_wha = event_study_results_wha.loc[event_study_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_wha = event_study_results_wha.loc[event_study_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_wha = lr_mean_m_wha - lr_mean_w_wha

# Employment Status Approach
lr_mean_m_esa = event_study_results_esa.loc[event_study_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_esa = event_study_results_esa.loc[event_study_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_esa = lr_mean_m_esa - lr_mean_w_esa

#### Plot the Results

# Create a 1x3 subplot layout
fig = make_subplots(rows=1, cols=3, subplot_titles=("Labor Income", "Working Hours", "Employment Status"))

# ----- Labor Income -----
# Add the results to the first subplot (Male)
fig.add_trace(go.Scatter(
    x=event_study_results_ia['event_time'],
    y=event_study_results_ia['percentage_coef_m'],
    mode='lines+markers',
    name='Male',
    line=dict(color='black', dash='dash'),
    marker=dict(symbol='circle'),
    showlegend=False,
), row=1, col=1)

# Add the results to the first subplot (Female)
fig.add_trace(go.Scatter(
    x=event_study_results_ia['event_time'],
    y=event_study_results_ia['percentage_coef_w'],
    mode='lines+markers',
    name='Female',
    line=dict(color='black'),
    marker=dict(symbol='triangle-up', size=10),
    showlegend=False,
), row=1, col=1)

# ----- Working Hours -----
# Add the results to the second subplot (Male)
fig.add_trace(go.Scatter(
    x=event_study_results_wha['event_time'],
    y=event_study_results_wha['percentage_coef_m'],
    mode='lines+markers',
    name='Male',
    line=dict(color='black', dash='dash'),
    marker=dict(symbol='circle'),
    showlegend=False,
), row=1, col=2)

# Add the results to the second subplot (Female)
fig.add_trace(go.Scatter(
    x=event_study_results_wha['event_time'],
    y=event_study_results_wha['percentage_coef_w'],
    mode='lines+markers',
    name='Female',
    line=dict(color='black'),
    marker=dict(symbol='triangle-up', size=10),
    showlegend=False,
), row=1, col=2)

# ----- Employment Status -----
# Add the results to the third subplot (Male)
fig.add_trace(go.Scatter(
    x=event_study_results_esa['event_time'],
    y=event_study_results_esa['percentage_coef_m'],
    mode='lines+markers',
    name='Male',
    line=dict(color='black', dash='dash'),
    marker=dict(symbol='circle'),
    showlegend=False,
), row=1, col=3)

# Add the results to the third subplot (Female)
fig.add_trace(go.Scatter(
    x=event_study_results_esa['event_time'],
    y=event_study_results_esa['percentage_coef_w'],
    mode='lines+markers',
    name='Female',
    line=dict(color='black'),
    marker=dict(symbol='triangle-up', size=10),
    showlegend=False,
), row=1, col=3)

# Add dummy traces for the custom legend (only 2: one for male, one for female)
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='markers',
    name='Male',
    marker=dict(symbol='circle', color='black'),
    showlegend=True
))

fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='markers',
    name='Female',
    marker=dict(symbol='triangle-up', color='black'),
    showlegend=True
))

# Add a vertical line for the event-time (1st childbirth) to all subplots
for col in range(1, 4):
    fig.add_shape(type="line", x0=-1, x1=-1, y0=-1, y1=1, line=dict(color='gray'), row=1, col=col)

# Add annotations for long run child penalty in the top-left corner of each subplot
fig.add_annotation(text=f"LR Penalty: {lr_child_penalty_ia:.2f}",  # Low income penalty
                   xref="x domain", yref="y domain", x=0, y=1, showarrow=False,
                   row=1, col=1, font=dict(size=8, color="black"))

fig.add_annotation(text=f"LR Penalty: {lr_child_penalty_wha:.2f}",  # Medium income penalty
                   xref="x domain", yref="y domain", x=0, y=1, showarrow=False,
                   row=1, col=2, font=dict(size=8, color="black"))

fig.add_annotation(text=f"LR Penalty: {lr_child_penalty_esa:.2f}",  # High income penalty
                   xref="x domain", yref="y domain", x=0, y=1, showarrow=False,
                   row=1, col=3, font=dict(size=8, color="black"))

# Update the layout for all subplots
fig.update_layout(
    width=1200,  # Set the total width of the figure
    height=350,  # Adjust the height to make the figure shorter vertically
    showlegend=True,  # Enable the legend
)

# Set common y-axis range for all subplots
fig.update_yaxes(title_text="Relative to Event Time -1", range=[-1, 0.5], row=1, col=1)
fig.update_yaxes(range=[-1, 0.5], row=1, col=2)  # Ensure same range for medium income plot
fig.update_yaxes(range=[-1, 0.5], row=1, col=3)  # Ensure same range for high income plot

# Set x-axis title for each subplot
fig.update_xaxes(title_text="Event Time (Birth of the First Child)", row=1, col=1)
fig.update_xaxes(title_text="Event Time (Birth of the First Child)", row=1, col=2)
fig.update_xaxes(title_text="Event Time (Birth of the First Child)", row=1, col=3)

# Show the figure
fig.show()

## No Time-Varying Confounders

### Placebo Tests with Pseudo-Event Times

# Import the data
biobirth = pd.read_csv("../../data/raw/biobirth.csv", usecols=['pid', 'sex', 'sumkids', 'gebjahr'])
pequiv = pd.read_csv("../../data/raw/pequiv.csv", usecols=['hid', 'pid', 'syear', 'ijob1', 'y11101', 'e11101', 'e11102'])
hbrutto = pd.read_csv("../../data/raw/hbrutto.csv", usecols=['hid', 'syear', 'wum1', 'bula_ew'])
pl = pd.read_csv("../../data/raw/pl.csv", usecols=['pid', 'syear', 'p_nace', 'plj0014_v3', 'plj0071'])
pgen = pd.read_csv("../../data/raw/pgen.csv", usecols=['pid', 'syear', 'pgbilzeit', 'pgnation', 'pgfamstd', 'pgmonth'])

# Merge the data
data = pd.merge(pequiv, biobirth, on=['pid'], how='left')
data = pd.merge(data, hbrutto, on=['hid', 'syear'], how='left')
data = pd.merge(data, pl, on=['pid', 'syear'], how='left')
data = pd.merge(data, pgen, on=['pid', 'syear'], how='left')

# Filter people without children
data_no_children = data[data['sumkids'] == 0].copy()

# Generate pseudo childbirth year 'kidgeb01'
np.random.seed(42)  # For reproducibility

# Generate age at first birth based on the summary statistics
age_at_first_birth = np.random.normal(
    loc=29.37,  # mean
    scale=5.13,  # standard deviation
    size=len(data_no_children)
).round().astype(int)

# Ensure the age at first birth is within logical bounds
age_at_first_birth = np.clip(age_at_first_birth, 20, 45)

# Calculate 'kidgeb01' (pseudo childbirth year)
data_no_children['age_at_first_birth'] = age_at_first_birth
data_no_children['kidgeb01'] = data_no_children['gebjahr'] + data_no_children['age_at_first_birth']

# Ensure 'kidgeb01' is within the valid range (1984 to 2019)
data_no_children['kidgeb01'] = np.clip(data_no_children['kidgeb01'], 1984, 2019)

# Add the new columns back to the original dataset
data['kidgeb01'] = np.nan  # Initialize column in the original dataset
data['age_at_first_birth'] = np.nan  # Initialize column in the original dataset

# Assign values for people without children
data.loc[data['sumkids'] == 0, 'kidgeb01'] = data_no_children['kidgeb01']
data.loc[data['sumkids'] == 0, 'age_at_first_birth'] = data_no_children['age_at_first_birth']

# Create new variables
data['age'] = data['syear'] - data['gebjahr']
data['event_time'] = data['syear'] - data['kidgeb01']
data['cpi2015'] = data['y11101']/100
data['real_income'] = data['ijob1']*data['cpi2015']

data['real_income'] = data.groupby('pid')['real_income'].shift(-1)
data['e11101'] = data.groupby('pid')['e11101'].shift(-1)
data['e11102'] = data.groupby('pid')['e11102'].shift(-1)

data = data[(data['age_at_first_birth'] >= 20) & (data['age_at_first_birth'] <= 45)]
data = data[(data['event_time'] >= -6) & (data['event_time'] <= 10)]
data = data[data['event_time'] != -1]

# Observed at least once before and once after
data['before'] = data['event_time'].apply(lambda x: 1 if x <= 0 else 0)
data['after'] = data['event_time'].apply(lambda x: 1 if x > 0 else 0)

data['before_sum'] = data.groupby('pid')['before'].transform('sum')
data['after_sum'] = data.groupby('pid')['after'].transform('sum')

data = data[(data['before_sum'] > 0) & (data['after_sum'] > 0)]

# Individual must be observed at least 8 times in the window
data = data.sort_values(by = ['pid', 'syear'])

data['I'] = data.groupby('pid')['pid'].transform('size') # I = total number of obs. per pid
data['i'] = data.groupby('pid').cumcount() + 1 # i = current obs. number within each pid

data = data[data['I'] >= 8]

data.reset_index(drop=True, inplace=True)

# Filter the data to keep only people without children (sumkids == 0)
data = data[data['sumkids'] == 0]

#### Income Approach

# Remove unobserved rows
ia_data = data[(data['real_income'] >= 0)] 

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

# Merge results for men and women
event_study_results_ia = pd.merge(results_m_df[['event_time', 'percentage_coef_m']],
                               results_w_df[['event_time', 'percentage_coef_w']],
                               on = 'event_time')
event_study_results_ia.loc[len(event_study_results_ia)] = [-1, 0, 0]

# Calculate child penalty for each event time
event_study_results_ia = event_study_results_ia.sort_values('event_time', ascending=True)
event_study_results_ia['child_penalty'] = event_study_results_ia['percentage_coef_m'] - event_study_results_ia['percentage_coef_w']

# LR Child Penalty
lr_mean_m_ia = event_study_results_ia.loc[event_study_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_ia = event_study_results_ia.loc[event_study_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_ia = lr_mean_m_ia - lr_mean_w_ia

#### Working Hours Approach

# Remove unobserved rows
wha_data = data[(data['e11101'] >= 0)] 

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

# Merge results for men and women
event_study_results_wha = pd.merge(results_m_df[['event_time', 'percentage_coef_m']],
                               results_w_df[['event_time', 'percentage_coef_w']],
                               on = 'event_time')
event_study_results_wha.loc[len(event_study_results_wha)] = [-1, 0, 0]

# Calculate child penalty for each event time
event_study_results_wha = event_study_results_wha.sort_values('event_time', ascending=True)
event_study_results_wha['child_penalty'] = event_study_results_wha['percentage_coef_m'] - event_study_results_wha['percentage_coef_w']

# LR Child Penalty
lr_mean_m_wha = event_study_results_wha.loc[event_study_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_wha = event_study_results_wha.loc[event_study_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_wha = lr_mean_m_wha - lr_mean_w_wha

#### Employment Status Approach

# Remove unobserved rows
esa_data = data[(data['e11102'] >= 0)] 

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

# OLS Model
model = sm.OLS.from_formula(formula = "e11101 ~ C(age) + C(syear) + C(event_time) - 1", data=esa_data[esa_data['sex'] == 1]).fit()
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

# Merge results for men and women
event_study_results_esa = pd.merge(results_m_df[['event_time', 'percentage_coef_m']],
                               results_w_df[['event_time', 'percentage_coef_w']],
                               on = 'event_time')
event_study_results_esa.loc[len(event_study_results_esa)] = [-1, 0, 0]

# Calculate child penalty for each event time
event_study_results_esa = event_study_results_esa.sort_values('event_time', ascending=True)
event_study_results_esa['child_penalty'] = event_study_results_esa['percentage_coef_m'] - event_study_results_esa['percentage_coef_w']

# LR Child Penalty
lr_mean_m_esa = event_study_results_esa.loc[event_study_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_esa = event_study_results_esa.loc[event_study_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_esa = lr_mean_m_esa - lr_mean_w_esa

#### Plot

# Create a 1x3 subplot layout
fig, axes = plt.subplots(1, 3, figsize=(15, 3))

# Set font properties globally
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': ['Times New Roman']})

# Define the event times you want on the x-axis
event_times = list(range(-5, 11))  # [-5, -4, ..., 10]

# ----- Labor Income -----
axes[0].plot(event_study_results_ia['event_time'], event_study_results_ia['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0].plot(event_study_results_ia['event_time'], event_study_results_ia['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0].set_title('Labor Income', fontweight='bold', pad=20)
axes[0].annotate(f'Long-Run Penalty: {lr_child_penalty_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0].set_ylim(-1, 0.5)
axes[0].set_ylabel('Relative to Event Time -1')
axes[0].set_xlabel('Years Since Birt of First Child')
axes[0].set_xticks(event_times)
axes[0].legend().remove()

# ----- Working Hours -----
axes[1].plot(event_study_results_wha['event_time'], event_study_results_wha['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1].plot(event_study_results_wha['event_time'], event_study_results_wha['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1].set_title('Working Hours', fontweight = 'bold', pad=20)
axes[1].annotate(f'Long-Run Penalty: {lr_child_penalty_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1].set_ylim(-1, 0.5)
axes[1].set_xlabel('Years Since Birth of First Child')
axes[1].set_xticks(event_times)
axes[1].legend().remove()

# ----- Employment Status -----
axes[2].plot(event_study_results_esa['event_time'], event_study_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[2].plot(event_study_results_esa['event_time'], event_study_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[2].set_title('Employment Status', fontweight = 'bold', pad=20)
axes[2].annotate(f'Long-Run Penalty: {lr_child_penalty_esa:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[2].set_ylim(-1, 0.5)
axes[2].set_xlabel('Years Since Birth of First Child')
axes[2].set_xticks(event_times)
axes[2].legend().remove()

# Add a shared legend outside the subplots
fig.legend(labels=['Male', 'Female'], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12)

# Adjust layout to ensure there's no overlap
plt.tight_layout()

# Remove the top and right spines to only keep the left vertical and bottom horizontal lines
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', linestyle='--', color='gray', alpha=0.3) 

## Comparability of Pre- and Post-2007 Samples

data = pd.read_csv("../../data/cleaned/cleaned_data.csv", index_col=0)

# Define mappings for category relabeling
relabel_dict = {
    'rural': 'Rural',
    'urban': 'Urban',
    'east': 'East Germany',
    'west': 'West Germany',
    'immigrant': 'Immigrant',
    'Low education': 'Low Education',
    'High education': 'High Education',
    'no partner': 'Without Partner',
    'partnered': 'With Partner'
}

# Apply relabeling to relevant columns
data['res_area'] = data['res_area'].replace(relabel_dict)
data['region'] = data['region'].replace(relabel_dict)
data['origin'] = data['origin'].replace(relabel_dict)
data['education_level'] = data['education_level'].replace(relabel_dict)
data['partner_status'] = data['partner_status'].replace(relabel_dict)

# Ensure categorical order remains consistent
custom_order = ['Low Education', 'High Education']
data['education_level'] = pd.Categorical(
    data['education_level'], 
    categories=custom_order, 
    ordered=True
)

# Calculate relative frequencies for categorical columns
categorical_columns = ['res_area', 'region', 'origin', 'education_level', 'partner_status']

# Define custom titles for each subplot
custom_titles = [
    "Residential Area Distribution",
    "Regional Distribution",
    "Origin Distribution",
    "Education Level Distribution",
    "Partner Status Distribution"
]

# Create a subplot with 2 rows and 3 columns
fig = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=custom_titles  # Use custom titles here
)

# Add bar plots for each categorical column
colors = {'Before 2007': '#1f77b4', 'After 2007': '#ff7f0e'}  # Professional blue and orange
for i, col in enumerate(categorical_columns):
    # Calculate counts and normalize to get relative frequencies
    relative_freq = (
        data.groupby(['Group', col]).size()
        .div(data['Group'].value_counts(), level='Group')  # Divide by group size to get percentages
        .reset_index(name='Percentage')
    )
    
    # Add the bar plot
    for group in ['Before 2007', 'After 2007']:
        group_data = relative_freq[relative_freq['Group'] == group]
        fig.add_trace(
            go.Bar(
                x=group_data[col],
                y=group_data['Percentage'],
                name=f"{group}",
                legendgroup=group,
                marker_color=colors[group],
                showlegend=(i == 0)  # Only show legend once per group
            ),
            row=(i // 3) + 1,
            col=(i % 3) + 1
        )

# Update layout with Times New Roman font
fig.update_layout(
    title={
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 20, 'family': 'Times New Roman'}
    },
    height=800, width=1200,
    template='plotly_white',  # Apply clean theme
    showlegend=True,  # Enable legend
    legend=dict(
        title="Group",
        font=dict(size=12, family='Times New Roman'),
        orientation="h",
        x=0.5,
        xanchor="center",
        y=-0.1
    ),
    font=dict(size=12, family='Times New Roman'),  # Apply font globally
    xaxis_title=dict(font=dict(size=14, family='Times New Roman')),
    yaxis_title=dict(font=dict(size=14, family='Times New Roman')),
    margin=dict(l=40, r=40, t=60, b=60)
)