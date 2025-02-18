# 01 Cleaning

## Libraries

import pandas as pd

## Read and Merge the Datasets

biobirth = pd.read_csv("../../data/raw/biobirth.csv", usecols=['pid', 'sex', 'kidgeb01', 'kidmon01', 'gebjahr'])
pequiv = pd.read_csv("../../data/raw/pequiv.csv", usecols=['hid', 'pid', 'syear', 'ijob1', 'y11101', 'e11101', 'e11102'])
hbrutto = pd.read_csv("../../data/raw/hbrutto.csv", usecols=['hid', 'syear', 'wum1', 'bula_ew'])
pl = pd.read_csv("../../data/raw/pl.csv", usecols=['pid', 'syear', 'p_nace', 'plj0014_v3', 'plj0071'])
pgen = pd.read_csv("../../data/raw/pgen.csv", usecols=['pid', 'syear', 'pgbilzeit', 'pgnation', 'pgfamstd', 'pgmonth'])

data = pd.merge(pequiv, biobirth, on=['pid'], how='left')
data = pd.merge(data, hbrutto, on=['hid', 'syear'], how='left')
data = pd.merge(data, pl, on=['pid', 'syear'], how='left')
data = pd.merge(data, pgen, on=['pid', 'syear'], how='left')

## Create New Variables

data['age'] = data['syear'] - data['gebjahr']
data['age_at_first_birth'] = data['kidgeb01'] - data['gebjahr']
data['event_time'] = data['syear'] - data['kidgeb01']
data['cpi2015'] = data['y11101']/100
data['real_income'] = data['ijob1']*data['cpi2015']

## Shift Dependent Variables
# Note: Since these variables are observed a year lateri we should shift thme by one year.

data['real_income'] = data.groupby('pid')['real_income'].shift(-1)
data['e11101'] = data.groupby('pid')['e11101'].shift(-1)
data['e11102'] = data.groupby('pid')['e11102'].shift(-1)

## Subset the Sample

data = data[(data['age_at_first_birth'] >= 20) & (data['age_at_first_birth'] <= 45)]
data = data[(data['event_time'] >= -6) & (data['event_time'] <= 10)]
data = data[data['event_time'] != -1]

## Criteria for Analysis

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

## Export the Data
data.to_csv("../../data/cleaned/cleaned_data.csv")

