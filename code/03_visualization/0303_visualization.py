# 0301 Visualization

## Libraries

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.pyplot as plt

## General Analysis

### Import the Results

event_study_results_ia = pd.read_csv("../../data/processed/general_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
event_study_results_wha = pd.read_csv("../../data/processed/general_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
event_study_results_esa = pd.read_csv("../../data/processed/general_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

#### Child Penalty

# Income Approach
mean_m_ia = event_study_results_ia.loc[event_study_results_ia['event_time'] >= 0, 'percentage_coef_m'].mean() - event_study_results_ia.loc[event_study_results_ia['event_time'] < 0, 'percentage_coef_m'].mean()
mean_w_ia = event_study_results_ia.loc[event_study_results_ia['event_time'] >= 0, 'percentage_coef_w'].mean() - event_study_results_ia.loc[event_study_results_ia['event_time'] < 0, 'percentage_coef_w'].mean()
child_penalty_ia = mean_m_ia - mean_w_ia

# Working Hours Approach
mean_m_wha = event_study_results_wha.loc[event_study_results_wha['event_time'] >= 0, 'percentage_coef_m'].mean() - event_study_results_wha.loc[event_study_results_wha['event_time'] < 0, 'percentage_coef_m'].mean()
mean_w_wha = event_study_results_wha.loc[event_study_results_wha['event_time'] >= 0, 'percentage_coef_w'].mean() - event_study_results_wha.loc[event_study_results_wha['event_time'] < 0, 'percentage_coef_w'].mean()
child_penalty_wha = mean_m_wha - mean_w_wha

# Employment Status Approach
mean_m_esa = event_study_results_esa.loc[event_study_results_esa['event_time'] >= 0, 'percentage_coef_m'].mean() - event_study_results_esa.loc[event_study_results_esa['event_time'] < 0, 'percentage_coef_m'].mean()
mean_w_esa = event_study_results_esa.loc[event_study_results_esa['event_time'] >= 0, 'percentage_coef_w'].mean() - event_study_results_esa.loc[event_study_results_esa['event_time'] < 0, 'percentage_coef_w'].mean()
child_penalty_esa = mean_m_esa - mean_w_esa

### Long-Run Child Penalty

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

### Plot the Results

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
axes[0].annotate(f'Long-Run Penalty: {child_penalty_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
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
axes[1].annotate(f'Long-Run Penalty: {child_penalty_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1].set_ylim(-1, 0.5)
axes[1].set_xlabel('Years Since Birth of First Child')
axes[1].set_xticks(event_times)
axes[1].legend().remove()

# ----- Employment Status -----
axes[2].plot(event_study_results_esa['event_time'], event_study_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[2].plot(event_study_results_esa['event_time'], event_study_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[2].set_title('Employment Status', fontweight = 'bold', pad=20)
axes[2].annotate(f'Long-Run Penalty: {child_penalty_esa:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
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

# Save the plot
fig.savefig('../04_results/figures/general_figure.png', bbox_inches='tight', dpi=300)

## Residential Area Analysis

### Import the Results

# Rural Group
rural_results_ia = pd.read_csv("../../data/processed/rural_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
rural_results_wha = pd.read_csv("../../data/processed/rural_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
rural_results_esa = pd.read_csv("../../data/processed/rural_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

# Urban Group
urban_results_ia = pd.read_csv("../../data/processed/urban_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
urban_results_wha = pd.read_csv("../../data/processed/urban_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
urban_results_esa = pd.read_csv("../../data/processed/urban_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

### Long-Run Child Penalty

#### Rural

# Labor Income Appraoch
lr_mean_m_rural_ia = rural_results_ia.loc[rural_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_rural_ia = rural_results_ia.loc[rural_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_rural_ia = lr_mean_m_rural_ia - lr_mean_w_rural_ia

# Working Hours Approach
lr_mean_m_rural_wha = rural_results_wha.loc[rural_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_rural_wha = rural_results_wha.loc[rural_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_rural_wha = lr_mean_m_rural_wha - lr_mean_w_rural_wha

# Employment Status Approach
lr_mean_m_rural_esa = rural_results_esa.loc[rural_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_rural_esa = rural_results_esa.loc[rural_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_rural_esa = lr_mean_m_rural_esa - lr_mean_w_rural_esa

#### Urban

# Labor Income Appraoch
lr_mean_m_urban_ia = urban_results_ia.loc[urban_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_urban_ia = urban_results_ia.loc[urban_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_urban_ia = lr_mean_m_urban_ia - lr_mean_w_urban_ia

# Working Hours Approach
lr_mean_m_urban_wha = urban_results_wha.loc[urban_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_urban_wha = urban_results_wha.loc[urban_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_urban_wha = lr_mean_m_urban_wha - lr_mean_w_urban_wha

# Employment Status Approach
lr_mean_m_urban_esa = urban_results_esa.loc[urban_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_urban_esa = urban_results_esa.loc[urban_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_urban_esa = lr_mean_m_urban_esa - lr_mean_w_urban_esa

### Plot the Results

# Create 2x3 subplots
fig, axes = plt.subplots(2, 3, figsize=(13.5, 5))  # 2 rows, 3 columns

# Set font properties globally
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': ['Times New Roman']})

# Titles for each column (only for the top row)
column_titles = ['Labor Income', 'Working Hours', 'Employment Status']

# Loop through columns of the first row to assign titles
for j, ax in enumerate(axes[0, :]):  # Only for the top row (axes[0, :])
    ax.set_title(column_titles[j], fontsize=14, fontweight='bold', pad=20)

# Add outer y-axis labels for each row
fig.text(0.08, 0.7, 'Rural Residence', fontsize=14, fontweight='bold', ha='center', va='center', rotation='vertical')
fig.text(0.08, 0.25, 'Urban Residence', fontsize=14, fontweight='bold', ha='center', va='center', rotation='vertical')

# Define the event times you want on the x-axis
event_times = list(range(-5, 11))  # [-5, -4, ..., 10]

# ----- RURAL GROUP ------

# ----- Labor Income -----
axes[0, 0].plot(rural_results_ia['event_time'], rural_results_ia['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0, 0].plot(rural_results_ia['event_time'], rural_results_ia['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0, 0].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0, 0].annotate(f'Long-Run Penalty: {lr_child_penalty_rural_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0, 0].set_ylim(-1, 0.5)
axes[0, 0].set_ylabel('Relative to Event Time -1')
axes[0, 0].set_xlabel('Years Since Birth of First Child')
axes[0, 0].set_xticks(event_times)
axes[0, 0].legend().remove()

# ----- Working Hours -----
axes[0,1].plot(rural_results_wha['event_time'], rural_results_wha['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0,1].plot(rural_results_wha['event_time'], rural_results_wha['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0,1].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0,1].annotate(f'Long-Run Penalty: {lr_child_penalty_rural_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0,1].set_ylim(-1, 0.5)
axes[0,1].set_xlabel('Years Since Birth of First Child')
axes[0,1].set_xticks(event_times)
axes[0,1].legend().remove()

# ----- Employment Status -----
axes[0,2].plot(rural_results_esa['event_time'], rural_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0,2].plot(rural_results_esa['event_time'], rural_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0,2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0,2].annotate(f'Long-Run Penalty: {lr_child_penalty_rural_esa:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0,2].set_ylim(-1, 0.5)
axes[0,2].set_xlabel('Years Since Birth of First Child')
axes[0,2].set_xticks(event_times)
axes[0,2].legend().remove()

# ----- URBAN GROUP ------

# ----- Labor Income -----
axes[1, 0].plot(urban_results_ia['event_time'], urban_results_ia['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1, 0].plot(urban_results_ia['event_time'], urban_results_ia['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1, 0].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1, 0].annotate(f'Long-Run Penalty: {lr_child_penalty_urban_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1, 0].set_ylim(-1, 0.5)
axes[1, 0].set_ylabel('Relative to Event Time -1')
axes[1, 0].set_xlabel('Years Since Birth of First Child')
axes[1, 0].set_xticks(event_times)
axes[1, 0].legend().remove()

# ----- Working Hours -----
axes[1,1].plot(urban_results_wha['event_time'], urban_results_wha['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1,1].plot(urban_results_wha['event_time'], urban_results_wha['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1,1].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1,1].annotate(f'Long-Run Penalty: {lr_child_penalty_urban_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1,1].set_ylim(-1, 0.5)
axes[1,1].set_xlabel('Years Since Birth of First Child')
axes[1,1].set_xticks(event_times)
axes[1,1].legend().remove()

# ----- Employment Status -----
axes[1,2].plot(urban_results_esa['event_time'], urban_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1,2].plot(urban_results_esa['event_time'], urban_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1,2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1,2].annotate(f'Long-Run Penalty: {abs(lr_child_penalty_urban_esa):.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1,2].set_ylim(-1, 0.5)
axes[1,2].set_xlabel('Years Since Birth of First Child')
axes[1,2].set_xticks(event_times)
axes[1,2].legend().remove()

# Add a shared legend outside the subplots
fig.legend(labels=['Male', 'Female'], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12)

# Adjust layout to ensure there's no overlap
plt.tight_layout()

# Remove the top and right spines for each individual subplot
for row in axes:  # Loop through rows of subplots
    for ax in row:  # Loop through each subplot in the row
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', linestyle='--', color='gray', alpha=0.3)


# Adjust layout for better spacing
plt.tight_layout(rect=[0.1, 0, 1, 0.95])

# Save the plot
fig.savefig('../04_results/figures/region_figure.png', bbox_inches='tight', dpi=300)

## Region Analysis 

# West Group
west_results_ia = pd.read_csv("../../data/processed/west_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
west_results_wha = pd.read_csv("../../data/processed/west_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
west_results_esa = pd.read_csv("../../data/processed/west_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

# East Group
east_results_ia = pd.read_csv("../../data/processed/east_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
east_results_wha = pd.read_csv("../../data/processed/east_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
east_results_esa = pd.read_csv("../../data/processed/east_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

### Long-Run Child Penalty

#### West Germany

# Labor Income Appraoch
lr_mean_m_west_ia = west_results_ia.loc[west_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_west_ia = west_results_ia.loc[west_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_west_ia = lr_mean_m_west_ia - lr_mean_w_west_ia

# Working Hours Approach
lr_mean_m_west_wha = west_results_wha.loc[west_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_west_wha = west_results_wha.loc[west_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_west_wha = lr_mean_m_west_wha - lr_mean_w_west_wha

# Employment Status Approach
lr_mean_m_west_esa = west_results_esa.loc[west_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_west_esa = west_results_esa.loc[west_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_west_esa = lr_mean_m_west_esa - lr_mean_w_west_esa

#### East Germany

# Labor Income Appraoch
lr_mean_m_east_ia = east_results_ia.loc[east_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_east_ia = east_results_ia.loc[east_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_east_ia = lr_mean_m_east_ia - lr_mean_w_east_ia

# Working Hours Approach
lr_mean_m_east_wha = east_results_wha.loc[east_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_east_wha = east_results_wha.loc[east_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_east_wha = lr_mean_m_east_wha - lr_mean_w_east_wha

# Employment Status Approach
lr_mean_m_east_esa = east_results_esa.loc[east_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_east_esa = east_results_esa.loc[east_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_east_esa = lr_mean_m_east_esa - lr_mean_w_east_esa

### Plot the Results

# Create 2x3 subplots
fig, axes = plt.subplots(2, 3, figsize=(13.5, 5))  # 2 rows, 3 columns

# Set font properties globally
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': ['Times New Roman']})

# Titles for each column (only for the top row)
column_titles = ['Labor Income', 'Working Hours', 'Employment Status']

# Loop through columns of the first row to assign titles
for j, ax in enumerate(axes[0, :]):  # Only for the top row (axes[0, :])
    ax.set_title(column_titles[j], fontsize=14, fontweight='bold', pad=20)

# Add outer y-axis labels for each row
fig.text(0.08, 0.7, 'West Germany', fontsize=14, fontweight='bold', ha='center', va='center', rotation='vertical')
fig.text(0.08, 0.25, 'East Germany', fontsize=14, fontweight='bold', ha='center', va='center', rotation='vertical')

# Define the event times you want on the x-axis
event_times = list(range(-5, 11))  # [-5, -4, ..., 10]

# ----- WEST GROUP ------

# ----- Labor Income -----
axes[0, 0].plot(west_results_ia['event_time'], west_results_ia['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0, 0].plot(west_results_ia['event_time'], west_results_ia['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0, 0].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0, 0].annotate(f'Long-Run Penalty: {lr_child_penalty_west_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0, 0].set_ylim(-1, 0.5)
axes[0, 0].set_ylabel('Relative to Event Time -1')
axes[0, 0].set_xlabel('Years Since Birth of First Child')
axes[0, 0].set_xticks(event_times)
axes[0, 0].legend().remove()

# ----- Working Hours -----
axes[0,1].plot(west_results_wha['event_time'], west_results_wha['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0,1].plot(west_results_wha['event_time'], west_results_wha['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0,1].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0,1].annotate(f'Long-Run Penalty: {lr_child_penalty_west_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0,1].set_ylim(-1, 0.5)
axes[0,1].set_xlabel('Years Since Birth of First Child')
axes[0,1].set_xticks(event_times)
axes[0,1].legend().remove()

# ----- Employment Status -----
axes[0,2].plot(west_results_esa['event_time'], west_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0,2].plot(west_results_esa['event_time'], west_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0,2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0,2].annotate(f'Long-Run Penalty: {lr_child_penalty_west_esa:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0,2].set_ylim(-1, 0.5)
axes[0,2].set_xlabel('Years Since Birth of First Child')
axes[0,2].set_xticks(event_times)
axes[0,2].legend().remove()

# ----- EAST GROUP ------

# ----- Labor Income -----
axes[1, 0].plot(east_results_ia['event_time'], east_results_ia['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1, 0].plot(east_results_ia['event_time'], east_results_ia['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1, 0].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1, 0].annotate(f'Long-Run Penalty: {lr_child_penalty_east_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1, 0].set_ylim(-1, 0.5)
axes[1, 0].set_ylabel('Relative to Event Time -1')
axes[1, 0].set_xlabel('Years Since Birth of First Child')
axes[1, 0].set_xticks(event_times)
axes[1, 0].legend().remove()

# ----- Working Hours -----
axes[1,1].plot(east_results_wha['event_time'], east_results_wha['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1,1].plot(east_results_wha['event_time'], east_results_wha['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1,1].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1,1].annotate(f'Long-Run Penalty: {lr_child_penalty_east_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1,1].set_ylim(-1, 0.5)
axes[1,1].set_xlabel('Years Since Birth of First Child')
axes[1,1].set_xticks(event_times)
axes[1,1].legend().remove()

# ----- Employment Status -----
axes[1,2].plot(east_results_esa['event_time'], east_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1,2].plot(east_results_esa['event_time'], east_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1,2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1,2].annotate(f'Long-Run Penalty: {abs(lr_child_penalty_east_esa):.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1,2].set_ylim(-1, 0.5)
axes[1,2].set_xlabel('Years Since Birth of First Child')
axes[1,2].set_xticks(event_times)
axes[1,2].legend().remove()

# Add a shared legend outside the subplots
fig.legend(labels=['Male', 'Female'], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12)

# Adjust layout to ensure there's no overlap
plt.tight_layout()

# Remove the top and right spines for each individual subplot
for row in axes:  # Loop through rows of subplots
    for ax in row:  # Loop through each subplot in the row
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', linestyle='--', color='gray', alpha=0.3)


# Adjust layout for better spacing
plt.tight_layout(rect=[0.1, 0, 1, 0.95])

# Save the plot
fig.savefig('../04_results/figures/ew_region_figure.png', bbox_inches='tight', dpi=300)

## Sector Analysis

# Male-Dominated Sector
malesector_results_ia = pd.read_csv("../../data/processed/malesector_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
malesector_results_wha = pd.read_csv("../../data/processed/malesector_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
malesector_results_esa = pd.read_csv("../../data/processed/malesector_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

# Female-Dominated Sector
femalesector_results_ia = pd.read_csv("../../data/processed/femalesector_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
femalesector_results_wha = pd.read_csv("../../data/processed/femalesector_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
femalesector_results_esa = pd.read_csv("../../data/processed/femalesector_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

# Balanced Sector
balancedsector_results_ia = pd.read_csv("../../data/processed/balancedsector_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
balancedsector_results_wha = pd.read_csv("../../data/processed/balancedsector_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
balancedsector_results_esa = pd.read_csv("../../data/processed/balancedsector_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

### Long-Run Child Penalty

#### Male-Dominated Sector

# Labor Income Appraoch
lr_mean_m_malesector_ia = malesector_results_ia.loc[malesector_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_malesector_ia = malesector_results_ia.loc[malesector_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_malesector_ia = lr_mean_m_malesector_ia - lr_mean_w_malesector_ia

# Working Hours Approach
lr_mean_m_malesector_wha = malesector_results_wha.loc[malesector_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_malesector_wha = malesector_results_wha.loc[malesector_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_malesector_wha = lr_mean_m_malesector_wha - lr_mean_w_malesector_wha

# Employment Status Approach
lr_mean_m_malesector_esa = malesector_results_esa.loc[malesector_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_malesector_esa = malesector_results_esa.loc[malesector_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_malesector_esa = lr_mean_m_malesector_esa - lr_mean_w_malesector_esa

#### Female-Dominated Sector

# Labor Income Appraoch
lr_mean_m_femalesector_ia = femalesector_results_ia.loc[femalesector_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_femalesector_ia = femalesector_results_ia.loc[femalesector_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_femalesector_ia = lr_mean_m_femalesector_ia - lr_mean_w_femalesector_ia

# Working Hours Approach
lr_mean_m_femalesector_wha = femalesector_results_wha.loc[femalesector_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_femalesector_wha = femalesector_results_wha.loc[femalesector_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_femalesector_wha = lr_mean_m_femalesector_wha - lr_mean_w_femalesector_wha

# Employment Status Approach
lr_mean_m_femalesector_esa = femalesector_results_esa.loc[femalesector_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_femalesector_esa = femalesector_results_esa.loc[femalesector_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_femalesector_esa = lr_mean_m_femalesector_esa - lr_mean_w_femalesector_esa

#### Balanced Gender Sector

# Labor Income Appraoch
lr_mean_m_balancedsector_ia = balancedsector_results_ia.loc[balancedsector_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_balancedsector_ia = balancedsector_results_ia.loc[balancedsector_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_balancedsector_ia = lr_mean_m_balancedsector_ia - lr_mean_w_balancedsector_ia

# Working Hours Approach
lr_mean_m_balancedsector_wha = balancedsector_results_wha.loc[balancedsector_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_balancedsector_wha = balancedsector_results_wha.loc[balancedsector_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_balancedsector_wha = lr_mean_m_balancedsector_wha - lr_mean_w_balancedsector_wha

# Employment Status Approach
lr_mean_m_balancedsector_esa = balancedsector_results_esa.loc[balancedsector_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_balancedsector_esa = balancedsector_results_esa.loc[balancedsector_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_balancedsector_esa = lr_mean_m_balancedsector_esa - lr_mean_w_balancedsector_esa

### Plot the Results

# Create 3x3 subplots
fig, axes = plt.subplots(3, 3, figsize=(13.5, 7.5))  # 3 rows, 3 columns

# Set font properties globally
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': ['Times New Roman']})

# Titles for each column (only for the top row)
column_titles = ['Labor Income', 'Working Hours', 'Employment Status']

# Row titles for sector groups
row_titles = ['Male-Dominated', 'Female-Dominated', 'Balanced Gender']

# Loop through columns of the first row to assign titles
for j, ax in enumerate(axes[0, :]):  # Only for the top row (axes[0, :])
    ax.set_title(column_titles[j], fontsize=14, fontweight='bold', pad=20)

# Add outer y-axis labels for each row
fig.text(0.08, 0.79, 'Male-Dominated', fontsize=14, fontweight='bold', ha='center', va='center', rotation='vertical')
fig.text(0.08, 0.5, 'Female-Dominated', fontsize=14, fontweight='bold', ha='center', va='center', rotation='vertical')
fig.text(0.08, 0.21, 'Balanced Gender', fontsize=14, fontweight='bold', ha='center', va='center', rotation='vertical')

# Define the event times you want on the x-axis
event_times = list(range(-5, 11))  # [-5, -4, ..., 10]

# ----- MALE-DOMINATED SECTOR ------

# ----- Labor Income -----
axes[0, 0].plot(malesector_results_ia['event_time'], malesector_results_ia['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0, 0].plot(malesector_results_ia['event_time'], malesector_results_ia['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0, 0].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0, 0].annotate(f'Long-Run Penalty: {lr_child_penalty_malesector_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0, 0].set_ylim(-1, 0.5)
axes[0, 0].set_ylabel('Relative to Event Time -1')
axes[0, 0].set_xlabel('Years Since Birth of First Child')
axes[0, 0].set_xticks(event_times)
axes[0, 0].legend().remove()

# ----- Working Hours -----
axes[0,1].plot(malesector_results_wha['event_time'], malesector_results_wha['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0,1].plot(malesector_results_wha['event_time'], malesector_results_wha['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0,1].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0,1].annotate(f'Long-Run Penalty: {lr_child_penalty_malesector_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0,1].set_ylim(-1, 0.5)
axes[0,1].set_xlabel('Years Since Birth of First Child')
axes[0,1].set_xticks(event_times)
axes[0,1].legend().remove()

# ----- Employment Status -----
axes[0,2].plot(malesector_results_esa['event_time'], malesector_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0,2].plot(malesector_results_esa['event_time'], malesector_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0,2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0,2].annotate(f'Long-Run Penalty: {lr_child_penalty_malesector_esa:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0,2].set_ylim(-1, 0.5)
axes[0,2].set_xlabel('Years Since Birth of First Child')
axes[0,2].set_xticks(event_times)
axes[0,2].legend().remove()

# ----- FEMALE-DOMINATED SECTOR ------

# ----- Labor Income -----
axes[1, 0].plot(femalesector_results_ia['event_time'], femalesector_results_ia['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1, 0].plot(femalesector_results_ia['event_time'], femalesector_results_ia['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1, 0].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1, 0].annotate(f'Long-Run Penalty: {lr_child_penalty_femalesector_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1, 0].set_ylim(-1, 0.5)
axes[1, 0].set_ylabel('Relative to Event Time -1')
axes[1, 0].set_xlabel('Years Since Birth of First Child')
axes[1, 0].set_xticks(event_times)
axes[1, 0].legend().remove()

# ----- Working Hours -----
axes[1,1].plot(femalesector_results_wha['event_time'], femalesector_results_wha['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1,1].plot(femalesector_results_wha['event_time'], femalesector_results_wha['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1,1].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1,1].annotate(f'Long-Run Penalty: {lr_child_penalty_femalesector_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1,1].set_ylim(-1, 0.5)
axes[1,1].set_xlabel('Years Since Birth of First Child')
axes[1,1].set_xticks(event_times)
axes[1,1].legend().remove()

# ----- Employment Status -----
axes[1,2].plot(femalesector_results_esa['event_time'], femalesector_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1,2].plot(femalesector_results_esa['event_time'], femalesector_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1,2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1,2].annotate(f'Long-Run Penalty: {abs(lr_child_penalty_femalesector_esa):.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1,2].set_ylim(-1, 0.5)
axes[1,2].set_xlabel('Years Since Birth of First Child')
axes[1,2].set_xticks(event_times)
axes[1,2].legend().remove()

# ----- BALANCED GENDER SECTOR ------

# ----- Labor Income -----
axes[2, 0].plot(balancedsector_results_ia['event_time'], balancedsector_results_ia['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[2, 0].plot(balancedsector_results_ia['event_time'], balancedsector_results_ia['percentage_coef_w'], label='Female', marker='^', color='black')
axes[2, 0].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[2, 0].annotate(f'Long-Run Penalty: {lr_child_penalty_balancedsector_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[2, 0].set_ylim(-1, 0.5)
axes[2, 0].set_ylabel('Relative to Event Time -1')
axes[2, 0].set_xlabel('Years Since Birth of First Child')
axes[2, 0].set_xticks(event_times)
axes[2, 0].legend().remove()

# ----- Working Hours -----
axes[2,1].plot(balancedsector_results_wha['event_time'], balancedsector_results_wha['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[2,1].plot(balancedsector_results_wha['event_time'], balancedsector_results_wha['percentage_coef_w'], label='Female', marker='^', color='black')
axes[2,1].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[2,1].annotate(f'Long-Run Penalty: {lr_child_penalty_balancedsector_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[2,1].set_ylim(-1, 0.5)
axes[2,1].set_xlabel('Years Since Birth of First Child')
axes[2,1].set_xticks(event_times)
axes[2,1].legend().remove()

# ----- Employment Status -----
axes[2,2].plot(balancedsector_results_esa['event_time'], balancedsector_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[2,2].plot(balancedsector_results_esa['event_time'], balancedsector_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[2,2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[2,2].annotate(f'Long-Run Penalty: {abs(lr_child_penalty_balancedsector_esa):.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[2,2].set_ylim(-1, 0.5)
axes[2,2].set_xlabel('Years Since Birth of First Child')
axes[2,2].set_xticks(event_times)
axes[2,2].legend().remove()

# Add a shared legend outside the subplots
fig.legend(labels=['Male', 'Female'], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12)

# Adjust layout to ensure there's no overlap
plt.tight_layout()

# Remove the top and right spines for each individual subplot
for row in axes:  # Loop through rows of subplots
    for ax in row:  # Loop through each subplot in the row
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', linestyle='--', color='gray', alpha=0.3)


# Adjust layout for better spacing
plt.tight_layout(rect=[0.1, 0, 1, 0.95])

# Save the plot
fig.savefig('../04_results/figures/sectorgender_figure.png', bbox_inches='tight', dpi=300)

## Origin Analysis

# Native
native_results_ia = pd.read_csv("../../data/processed/native_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
native_results_wha = pd.read_csv("../../data/processed/native_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
native_results_esa = pd.read_csv("../../data/processed/native_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

# Immigrant
immigrant_results_ia = pd.read_csv("../../data/processed/immigrant_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
immigrant_results_wha = pd.read_csv("../../data/processed/immigrant_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
immigrant_results_esa = pd.read_csv("../../data/processed/immigrant_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

### Long-Run Child Penalty

#### Native

# Labor Income Appraoch
lr_mean_m_native_ia = native_results_ia.loc[native_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_native_ia = native_results_ia.loc[native_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_native_ia = lr_mean_m_native_ia - lr_mean_w_native_ia

# Working Hours Approach
lr_mean_m_native_wha = native_results_wha.loc[native_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_native_wha = native_results_wha.loc[native_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_native_wha = lr_mean_m_native_wha - lr_mean_w_native_wha

# Employment Status Approach
lr_mean_m_native_esa = native_results_esa.loc[native_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_native_esa = native_results_esa.loc[native_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_native_esa = lr_mean_m_native_esa - lr_mean_w_native_esa

#### Immigrant

# Labor Income Appraoch
lr_mean_m_immigrant_ia = immigrant_results_ia.loc[immigrant_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_immigrant_ia = immigrant_results_ia.loc[immigrant_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_immigrant_ia = lr_mean_m_immigrant_ia - lr_mean_w_immigrant_ia

# Working Hours Approach
lr_mean_m_immigrant_wha = immigrant_results_wha.loc[immigrant_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_immigrant_wha = immigrant_results_wha.loc[immigrant_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_immigrant_wha = lr_mean_m_immigrant_wha - lr_mean_w_immigrant_wha

# Employment Status Approach
lr_mean_m_immigrant_esa = immigrant_results_esa.loc[immigrant_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_immigrant_esa = immigrant_results_esa.loc[immigrant_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_immigrant_esa = lr_mean_m_immigrant_esa - lr_mean_w_immigrant_esa

### Plot the Results

# Create 2x3 subplots
fig, axes = plt.subplots(2, 3, figsize=(13.5, 5))  # 2 rows, 3 columns

# Set font properties globally
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': ['Times New Roman']})

# Titles for each column (only for the top row)
column_titles = ['Labor Income', 'Working Hours', 'Employment Status']

# Loop through columns of the first row to assign titles
for j, ax in enumerate(axes[0, :]):  # Only for the top row (axes[0, :])
    ax.set_title(column_titles[j], fontsize=14, fontweight='bold', pad=20)

# Add outer y-axis labels for each row
fig.text(0.08, 0.7, 'Native', fontsize=14, fontweight='bold', ha='center', va='center', rotation='vertical')
fig.text(0.08, 0.25, 'Immigrant', fontsize=14, fontweight='bold', ha='center', va='center', rotation='vertical')

# Define the event times you want on the x-axis
event_times = list(range(-5, 11))  # [-5, -4, ..., 10]

# ----- NATIVE GROUP ------

# ----- Labor Income -----
axes[0, 0].plot(native_results_ia['event_time'], native_results_ia['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0, 0].plot(native_results_ia['event_time'], native_results_ia['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0, 0].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0, 0].annotate(f'Long-Run Penalty: {lr_child_penalty_native_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0, 0].set_ylim(-1, 0.5)
axes[0, 0].set_ylabel('Relative to Event Time -1')
axes[0, 0].set_xlabel('Years Since Birth of First Child')
axes[0, 0].set_xticks(event_times)
axes[0, 0].legend().remove()

# ----- Working Hours -----
axes[0,1].plot(native_results_wha['event_time'], native_results_wha['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0,1].plot(native_results_wha['event_time'], native_results_wha['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0,1].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0,1].annotate(f'Long-Run Penalty: {lr_child_penalty_native_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0,1].set_ylim(-1, 0.5)
axes[0,1].set_xlabel('Years Since Birth of First Child')
axes[0,1].set_xticks(event_times)
axes[0,1].legend().remove()

# ----- Employment Status -----
axes[0,2].plot(native_results_esa['event_time'], native_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0,2].plot(native_results_esa['event_time'], native_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0,2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0,2].annotate(f'Long-Run Penalty: {lr_child_penalty_native_esa:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0,2].set_ylim(-1, 0.5)
axes[0,2].set_xlabel('Years Since Birth of First Child')
axes[0,2].set_xticks(event_times)
axes[0,2].legend().remove()

# ----- IMMIGRANT GROUP ------

# ----- Labor Income -----
axes[1, 0].plot(immigrant_results_ia['event_time'], immigrant_results_ia['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1, 0].plot(immigrant_results_ia['event_time'], immigrant_results_ia['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1, 0].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1, 0].annotate(f'Long-Run Penalty: {lr_child_penalty_immigrant_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1, 0].set_ylim(-1, 0.5)
axes[1, 0].set_ylabel('Relative to Event Time -1')
axes[1, 0].set_xlabel('Years Since Birth of First Child')
axes[1, 0].set_xticks(event_times)
axes[1, 0].legend().remove()

# ----- Working Hours -----
axes[1,1].plot(immigrant_results_wha['event_time'], immigrant_results_wha['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1,1].plot(immigrant_results_wha['event_time'], immigrant_results_wha['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1,1].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1,1].annotate(f'Long-Run Penalty: {lr_child_penalty_immigrant_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1,1].set_ylim(-1, 0.5)
axes[1,1].set_xlabel('Years Since Birth of First Child')
axes[1,1].set_xticks(event_times)
axes[1,1].legend().remove()

# ----- Employment Status -----
axes[1,2].plot(immigrant_results_esa['event_time'], immigrant_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1,2].plot(immigrant_results_esa['event_time'], immigrant_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1,2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1,2].annotate(f'Long-Run Penalty: {abs(lr_child_penalty_immigrant_esa):.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1,2].set_ylim(-1, 0.5)
axes[1,2].set_xlabel('Years Since Birth of First Child')
axes[1,2].set_xticks(event_times)
axes[1,2].legend().remove()

# Add a shared legend outside the subplots
fig.legend(labels=['Male', 'Female'], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12)

# Adjust layout to ensure there's no overlap
plt.tight_layout()

# Remove the top and right spines for each individual subplot
for row in axes:  # Loop through rows of subplots
    for ax in row:  # Loop through each subplot in the row
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', linestyle='--', color='gray', alpha=0.3)


# Adjust layout for better spacing
plt.tight_layout(rect=[0.1, 0, 1, 0.95])

# Save the plot
fig.savefig('../04_results/figures/origin_figure.png', bbox_inches='tight', dpi=300)

## Education Analysis

# Low-Education
lowedu_results_ia = pd.read_csv("../../data/processed/lowedu_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
lowedu_results_wha = pd.read_csv("../../data/processed/lowedu_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
lowedu_results_esa = pd.read_csv("../../data/processed/lowedu_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

# High-Education
highedu_results_ia = pd.read_csv("../../data/processed/highedu_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
highedu_results_wha = pd.read_csv("../../data/processed/highedu_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
highedu_results_esa = pd.read_csv("../../data/processed/highedu_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

### Long-Run Child Penalty

#### Low Education

# Labor Income Appraoch
lr_mean_m_lowedu_ia = lowedu_results_ia.loc[lowedu_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_lowedu_ia = lowedu_results_ia.loc[lowedu_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_lowedu_ia = lr_mean_m_lowedu_ia - lr_mean_w_lowedu_ia

# Working Hours Approach
lr_mean_m_lowedu_wha = lowedu_results_wha.loc[lowedu_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_lowedu_wha = lowedu_results_wha.loc[lowedu_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_lowedu_wha = lr_mean_m_lowedu_wha - lr_mean_w_lowedu_wha

# Employment Status Approach
lr_mean_m_lowedu_esa = lowedu_results_esa.loc[lowedu_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_lowedu_esa = lowedu_results_esa.loc[lowedu_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_lowedu_esa = lr_mean_m_lowedu_esa - lr_mean_w_lowedu_esa

#### High Education

# Labor Income Appraoch
lr_mean_m_highedu_ia = highedu_results_ia.loc[highedu_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_highedu_ia = highedu_results_ia.loc[highedu_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_highedu_ia = lr_mean_m_highedu_ia - lr_mean_w_highedu_ia

# Working Hours Approach
lr_mean_m_highedu_wha = highedu_results_wha.loc[highedu_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_highedu_wha = highedu_results_wha.loc[highedu_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_highedu_wha = lr_mean_m_highedu_wha - lr_mean_w_highedu_wha

# Employment Status Approach
lr_mean_m_highedu_esa = highedu_results_esa.loc[highedu_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_highedu_esa = highedu_results_esa.loc[highedu_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_highedu_esa = lr_mean_m_highedu_esa - lr_mean_w_highedu_esa

### Plot the Results

# Create 2x3 subplots
fig, axes = plt.subplots(2, 3, figsize=(13.5, 5))  # 2 rows, 3 columns

# Set font properties globally
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': ['Times New Roman']})

# Titles for each column (only for the top row)
column_titles = ['Labor Income', 'Working Hours', 'Employment Status']

# Loop through columns of the first row to assign titles
for j, ax in enumerate(axes[0, :]):  # Only for the top row (axes[0, :])
    ax.set_title(column_titles[j], fontsize=14, fontweight='bold', pad=20)

# Add outer y-axis labels for each row
fig.text(0.08, 0.7, 'Low-Education', fontsize=14, fontweight='bold', ha='center', va='center', rotation='vertical')
fig.text(0.08, 0.25, 'High-Education', fontsize=14, fontweight='bold', ha='center', va='center', rotation='vertical')

# Define the event times you want on the x-axis
event_times = list(range(-5, 11))  # [-5, -4, ..., 10]

# ----- LOW-EDUCATION GROUP ------

# ----- Labor Income -----
axes[0, 0].plot(lowedu_results_ia['event_time'], lowedu_results_ia['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0, 0].plot(lowedu_results_ia['event_time'], lowedu_results_ia['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0, 0].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0, 0].annotate(f'Long-Run Penalty: {lr_child_penalty_lowedu_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0, 0].set_ylim(-1, 0.5)
axes[0, 0].set_ylabel('Relative to Event Time -1')
axes[0, 0].set_xlabel('Years Since Birth of First Child')
axes[0, 0].set_xticks(event_times)
axes[0, 0].legend().remove()

# ----- Working Hours -----
axes[0,1].plot(lowedu_results_wha['event_time'], lowedu_results_wha['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0,1].plot(lowedu_results_wha['event_time'], lowedu_results_wha['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0,1].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0,1].annotate(f'Long-Run Penalty: {lr_child_penalty_lowedu_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0,1].set_ylim(-1, 0.5)
axes[0,1].set_xlabel('Years Since Birth of First Child')
axes[0,1].set_xticks(event_times)
axes[0,1].legend().remove()

# ----- Employment Status -----
axes[0,2].plot(lowedu_results_esa['event_time'], lowedu_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0,2].plot(lowedu_results_esa['event_time'], lowedu_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0,2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0,2].annotate(f'Long-Run Penalty: {lr_child_penalty_lowedu_esa:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0,2].set_ylim(-1, 0.5)
axes[0,2].set_xlabel('Years Since Birth of First Child')
axes[0,2].set_xticks(event_times)
axes[0,2].legend().remove()

# ----- HIGH-EDUCATION GROUP ------

# ----- Labor Income -----
axes[1, 0].plot(highedu_results_ia['event_time'], highedu_results_ia['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1, 0].plot(highedu_results_ia['event_time'], highedu_results_ia['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1, 0].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1, 0].annotate(f'Long-Run Penalty: {lr_child_penalty_highedu_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1, 0].set_ylim(-1, 0.5)
axes[1, 0].set_ylabel('Relative to Event Time -1')
axes[1, 0].set_xlabel('Years Since Birth of First Child')
axes[1, 0].set_xticks(event_times)
axes[1, 0].legend().remove()

# ----- Working Hours -----
axes[1,1].plot(highedu_results_wha['event_time'], highedu_results_wha['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1,1].plot(highedu_results_wha['event_time'], highedu_results_wha['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1,1].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1,1].annotate(f'Long-Run Penalty: {lr_child_penalty_highedu_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1,1].set_ylim(-1, 0.5)
axes[1,1].set_xlabel('Years Since Birth of First Child')
axes[1,1].set_xticks(event_times)
axes[1,1].legend().remove()

# ----- Employment Status -----
axes[1,2].plot(highedu_results_esa['event_time'], highedu_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1,2].plot(highedu_results_esa['event_time'], highedu_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1,2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1,2].annotate(f'Long-Run Penalty: {abs(lr_child_penalty_highedu_esa):.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1,2].set_ylim(-1, 0.5)
axes[1,2].set_xlabel('Years Since Birth of First Child')
axes[1,2].set_xticks(event_times)
axes[1,2].legend().remove()

# Add a shared legend outside the subplots
fig.legend(labels=['Male', 'Female'], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12)

# Adjust layout to ensure there's no overlap
plt.tight_layout()

# Remove the top and right spines for each individual subplot
for row in axes:  # Loop through rows of subplots
    for ax in row:  # Loop through each subplot in the row
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', linestyle='--', color='gray', alpha=0.3)


# Adjust layout for better spacing
plt.tight_layout(rect=[0.1, 0, 1, 0.95])

# Save the plot
fig.savefig('../04_results/figures/education_figure.png', bbox_inches='tight', dpi=300)

## Age at First Parenthood Analysis

# Young
young_results_ia = pd.read_csv("../../data/processed/young_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
young_results_wha = pd.read_csv("../../data/processed/young_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
young_results_esa = pd.read_csv("../../data/processed/young_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

# Late-Mid
latemid_results_ia = pd.read_csv("../../data/processed/median_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
latemid_results_wha = pd.read_csv("../../data/processed/median_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
latemid_results_esa = pd.read_csv("../../data/processed/median_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

# Old
old_results_ia = pd.read_csv("../../data/processed/old_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
old_results_wha = pd.read_csv("../../data/processed/old_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
old_results_esa = pd.read_csv("../../data/processed/old_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

### Long-Run Child Penalty

#### Young

# Labor Income Appraoch
lr_mean_m_young_ia = young_results_ia.loc[young_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_young_ia = young_results_ia.loc[young_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_young_ia = lr_mean_m_young_ia - lr_mean_w_young_ia

# Working Hours Approach
lr_mean_m_young_wha = young_results_wha.loc[young_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_young_wha = young_results_wha.loc[young_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_young_wha = lr_mean_m_young_wha - lr_mean_w_young_wha

# Employment Status Approach
lr_mean_m_young_esa = young_results_esa.loc[young_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_young_esa = young_results_esa.loc[young_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_young_esa = lr_mean_m_young_esa - lr_mean_w_young_esa

#### Median

# Labor Income Appraoch
lr_mean_m_earlymid_ia = latemid_results_ia.loc[latemid_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_earlymid_ia = latemid_results_ia.loc[latemid_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_earlymid_ia = lr_mean_m_earlymid_ia - lr_mean_w_earlymid_ia

# Working Hours Approach
lr_mean_m_earlymid_wha = latemid_results_wha.loc[latemid_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_earlymid_wha = latemid_results_wha.loc[latemid_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_earlymid_wha = lr_mean_m_earlymid_wha - lr_mean_w_earlymid_wha

# Employment Status Approach
lr_mean_m_earlymid_esa = latemid_results_esa.loc[latemid_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_earlymid_esa = latemid_results_esa.loc[latemid_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_earlymid_esa = lr_mean_m_earlymid_esa - lr_mean_w_earlymid_esa

#### Old

# Labor Income Appraoch
lr_mean_m_latemid_ia = old_results_ia.loc[old_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_latemid_ia = old_results_ia.loc[old_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_latemid_ia = lr_mean_m_latemid_ia - lr_mean_w_latemid_ia

# Working Hours Approach
lr_mean_m_latemid_wha = old_results_wha.loc[old_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_latemid_wha = old_results_wha.loc[old_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_latemid_wha = lr_mean_m_latemid_wha - lr_mean_w_latemid_wha

# Employment Status Approach
lr_mean_m_latemid_esa = old_results_esa.loc[old_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_latemid_esa = old_results_esa.loc[old_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_latemid_esa = lr_mean_m_latemid_esa - lr_mean_w_latemid_esa

### Plot the Results

# Create 4x3 subplots
fig, axes = plt.subplots(3, 3, figsize=(13.5, 7.5))  # 2 rows, 3 columns

# Set font properties globally
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': ['Times New Roman']})

# Titles for each column (only for the top row)
column_titles = ['Labor Income', 'Working Hours', 'Employment Status']

# Loop through columns of the first row to assign titles
for j, ax in enumerate(axes[0, :]):  # Only for the top row (axes[0, :])
    ax.set_title(column_titles[j], fontsize=14, fontweight='bold', pad=20)

# Add outer y-axis labels for each row
fig.text(0.08, 0.79, 'Early', fontsize=14, fontweight='bold', ha='center', va='center', rotation='vertical')
fig.text(0.08, 0.5, 'Median', fontsize=14, fontweight='bold', ha='center', va='center', rotation='vertical')
fig.text(0.08, 0.21, 'Late', fontsize=14, fontweight='bold', ha='center', va='center', rotation='vertical')

# Define the event times you want on the x-axis
event_times = list(range(-5, 11))  # [-5, -4, ..., 10]

# ----- EARLY PARENT GROUP ------

# ----- Labor Income -----
axes[0, 0].plot(young_results_ia['event_time'], young_results_ia['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0, 0].plot(young_results_ia['event_time'], young_results_ia['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0, 0].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0, 0].annotate(f'Long-Run Penalty: {lr_child_penalty_young_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0, 0].set_ylim(-1, 0.5)
axes[0, 0].set_ylabel('Relative to Event Time -1')
axes[0, 0].set_xlabel('Years Since Birth of First Child')
axes[0, 0].set_xticks(event_times)
axes[0, 0].legend().remove()

# ----- Working Hours -----
axes[0,1].plot(young_results_wha['event_time'], young_results_wha['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0,1].plot(young_results_wha['event_time'], young_results_wha['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0,1].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0,1].annotate(f'Long-Run Penalty: {lr_child_penalty_young_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0,1].set_ylim(-1, 0.5)
axes[0,1].set_xlabel('Years Since Birth of First Child')
axes[0,1].set_xticks(event_times)
axes[0,1].legend().remove()

# ----- Employment Status -----
axes[0,2].plot(young_results_esa['event_time'], young_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0,2].plot(young_results_esa['event_time'], young_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0,2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0,2].annotate(f'Long-Run Penalty: {lr_child_penalty_young_esa:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0,2].set_ylim(-1, 0.5)
axes[0,2].set_xlabel('Years Since Birth of First Child')
axes[0,2].set_xticks(event_times)
axes[0,2].legend().remove()

# ----- MIDDLE PARENT GROUP ------

# ----- Labor Income -----
axes[1, 0].plot(earlymid_results_ia['event_time'], earlymid_results_ia['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1, 0].plot(earlymid_results_ia['event_time'], earlymid_results_ia['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1, 0].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1, 0].annotate(f'Long-Run Penalty: {lr_child_penalty_earlymid_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1, 0].set_ylim(-1, 0.5)
axes[1, 0].set_ylabel('Relative to Event Time -1')
axes[1, 0].set_xlabel('Years Since Birth of First Child')
axes[1, 0].set_xticks(event_times)
axes[1, 0].legend().remove()

# ----- Working Hours -----
axes[1,1].plot(earlymid_results_wha['event_time'], earlymid_results_wha['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1,1].plot(earlymid_results_wha['event_time'], earlymid_results_wha['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1,1].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1,1].annotate(f'Long-Run Penalty: {lr_child_penalty_earlymid_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1,1].set_ylim(-1, 0.5)
axes[1,1].set_xlabel('Years Since Birth of First Child')
axes[1,1].set_xticks(event_times)
axes[1,1].legend().remove()

# ----- Employment Status -----
axes[1,2].plot(earlymid_results_esa['event_time'], earlymid_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1,2].plot(earlymid_results_esa['event_time'], earlymid_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1,2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1,2].annotate(f'Long-Run Penalty: {abs(lr_child_penalty_earlymid_esa):.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1,2].set_ylim(-1, 0.5)
axes[1,2].set_xlabel('Years Since Birth of First Child')
axes[1,2].set_xticks(event_times)
axes[1,2].legend().remove()

# ----- LATE PARENT GROUP ------

# ----- Labor Income -----
axes[2, 0].plot(latemid_results_ia['event_time'], latemid_results_ia['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[2, 0].plot(latemid_results_ia['event_time'], latemid_results_ia['percentage_coef_w'], label='Female', marker='^', color='black')
axes[2, 0].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[2, 0].annotate(f'Long-Run Penalty: {lr_child_penalty_latemid_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[2, 0].set_ylim(-1, 0.5)
axes[2, 0].set_ylabel('Relative to Event Time -1')
axes[2, 0].set_xlabel('Years Since Birth of First Child')
axes[2, 0].set_xticks(event_times)
axes[2, 0].legend().remove()

# ----- Working Hours -----
axes[2,1].plot(latemid_results_wha['event_time'], latemid_results_wha['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[2,1].plot(latemid_results_wha['event_time'], latemid_results_wha['percentage_coef_w'], label='Female', marker='^', color='black')
axes[2,1].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[2,1].annotate(f'Long-Run Penalty: {lr_child_penalty_latemid_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[2,1].set_ylim(-1, 0.5)
axes[2,1].set_xlabel('Years Since Birth of First Child')
axes[2,1].set_xticks(event_times)
axes[2,1].legend().remove()

# ----- Employment Status -----
axes[2,2].plot(latemid_results_esa['event_time'], latemid_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[2,2].plot(latemid_results_esa['event_time'], latemid_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[2,2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[2,2].annotate(f'Long-Run Penalty: {lr_child_penalty_latemid_esa:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[2,2].set_ylim(-1, 0.5)
axes[2,2].set_xlabel('Years Since Birth of First Child')
axes[2,2].set_xticks(event_times)
axes[2,2].legend().remove()

# Add a shared legend outside the subplots
fig.legend(labels=['Male', 'Female'], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12)

# Adjust layout to ensure there's no overlap
plt.tight_layout()

# Remove the top and right spines for each individual subplot
for row in axes:  # Loop through rows of subplots
    for ax in row:  # Loop through each subplot in the row
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', linestyle='--', color='gray', alpha=0.3)


# Adjust layout for better spacing
plt.tight_layout(rect=[0.1, 0, 1, 0.95])

plt.show()

## Partnertship Analysis

# No Partner
nopartner_results_ia = pd.read_csv("../../data/processed/nopartner_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
nopartner_results_wha = pd.read_csv("../../data/processed/nopartner_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
nopartner_results_esa = pd.read_csv("../../data/processed/nopartner_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

# Partnered
partnered_results_ia = pd.read_csv("../../data/processed/partnered_results_ia.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
partnered_results_wha = pd.read_csv("../../data/processed/partnered_results_wha.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])
partnered_results_esa = pd.read_csv("../../data/processed/partnered_results_esa.csv",
                                  usecols=['event_time', 'percentage_coef_m', 'percentage_coef_w', 'child_penalty'])

### Long-Run Child Penalty

#### Without Partner

# Labor Income Appraoch
lr_mean_m_nopartner_ia = nopartner_results_ia.loc[nopartner_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_nopartner_ia = nopartner_results_ia.loc[nopartner_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_nopartner_ia = lr_mean_m_nopartner_ia - lr_mean_w_nopartner_ia

# Working Hours Approach
lr_mean_m_nopartner_wha = nopartner_results_wha.loc[nopartner_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_nopartner_wha = nopartner_results_wha.loc[nopartner_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_nopartner_wha = lr_mean_m_nopartner_wha - lr_mean_w_nopartner_wha

# Employment Status Approach
lr_mean_m_nopartner_esa = nopartner_results_esa.loc[nopartner_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_nopartner_esa = nopartner_results_esa.loc[nopartner_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_nopartner_esa = lr_mean_m_nopartner_esa - lr_mean_w_nopartner_esa

#### With Partner

# Labor Income Appraoch
lr_mean_m_partnered_ia = partnered_results_ia.loc[partnered_results_ia['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_partnered_ia = partnered_results_ia.loc[partnered_results_ia['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_partnered_ia = lr_mean_m_partnered_ia - lr_mean_w_partnered_ia

# Working Hours Approach
lr_mean_m_partnered_wha = partnered_results_wha.loc[partnered_results_wha['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_partnered_wha = partnered_results_wha.loc[partnered_results_wha['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_partnered_wha = lr_mean_m_partnered_wha - lr_mean_w_partnered_wha

# Employment Status Approach
lr_mean_m_partnered_esa = partnered_results_esa.loc[partnered_results_esa['event_time'] >= 5, 'percentage_coef_m'].mean()
lr_mean_w_partnered_esa = partnered_results_esa.loc[partnered_results_esa['event_time'] >= 5, 'percentage_coef_w'].mean()
lr_child_penalty_partnered_esa = lr_mean_m_partnered_esa - lr_mean_w_partnered_esa

### Plot the Results

# Create 2x3 subplots
fig, axes = plt.subplots(2, 3, figsize=(13.5, 5))  # 2 rows, 3 columns

# Set font properties globally
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': ['Times New Roman']})

# Titles for each column (only for the top row)
column_titles = ['Labor Income', 'Working Hours', 'Employment Status']

# Loop through columns of the first row to assign titles
for j, ax in enumerate(axes[0, :]):  # Only for the top row (axes[0, :])
    ax.set_title(column_titles[j], fontsize=14, fontweight='bold', pad=20)

# Add outer y-axis labels for each row
fig.text(0.08, 0.7, 'No Partner', fontsize=14, fontweight='bold', ha='center', va='center', rotation='vertical')
fig.text(0.08, 0.25, 'Partnered', fontsize=14, fontweight='bold', ha='center', va='center', rotation='vertical')

# Define the event times you want on the x-axis
event_times = list(range(-5, 11))  # [-5, -4, ..., 10]

# ----- NO PARTNER GROUP ------

# ----- Labor Income -----
axes[0, 0].plot(nopartner_results_ia['event_time'], nopartner_results_ia['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0, 0].plot(nopartner_results_ia['event_time'], nopartner_results_ia['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0, 0].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0, 0].annotate(f'Long-Run Penalty: {lr_child_penalty_nopartner_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0, 0].set_ylim(-1, 0.5)
axes[0, 0].set_ylabel('Relative to Event Time -1')
axes[0, 0].set_xlabel('Years Since Birth of First Child')
axes[0, 0].set_xticks(event_times)
axes[0, 0].legend().remove()

# ----- Working Hours -----
axes[0,1].plot(nopartner_results_wha['event_time'], nopartner_results_wha['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0,1].plot(nopartner_results_wha['event_time'], nopartner_results_wha['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0,1].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0,1].annotate(f'Long-Run Penalty: {lr_child_penalty_nopartner_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0,1].set_ylim(-1, 0.5)
axes[0,1].set_xlabel('Years Since Birth of First Child')
axes[0,1].set_xticks(event_times)
axes[0,1].legend().remove()

# ----- Employment Status -----
axes[0,2].plot(nopartner_results_esa['event_time'], nopartner_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[0,2].plot(nopartner_results_esa['event_time'], nopartner_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[0,2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[0,2].annotate(f'Long-Run Penalty: {lr_child_penalty_nopartner_esa:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[0,2].set_ylim(-1, 0.5)
axes[0,2].set_xlabel('Years Since Birth of First Child')
axes[0,2].set_xticks(event_times)
axes[0,2].legend().remove()

# ----- PARTNERED GROUP ------

# ----- Labor Income -----
axes[1, 0].plot(partnered_results_ia['event_time'], partnered_results_ia['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1, 0].plot(partnered_results_ia['event_time'], partnered_results_ia['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1, 0].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1, 0].annotate(f'Long-Run Penalty: {lr_child_penalty_partnered_ia:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1, 0].set_ylim(-1, 0.5)
axes[1, 0].set_ylabel('Relative to Event Time -1')
axes[1, 0].set_xlabel('Years Since Birth of First Child')
axes[1, 0].set_xticks(event_times)
axes[1, 0].legend().remove()

# ----- Working Hours -----
axes[1,1].plot(partnered_results_wha['event_time'], partnered_results_wha['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1,1].plot(partnered_results_wha['event_time'], partnered_results_wha['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1,1].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1,1].annotate(f'Long-Run Penalty: {lr_child_penalty_partnered_wha:.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1,1].set_ylim(-1, 0.5)
axes[1,1].set_xlabel('Years Since Birth of First Child')
axes[1,1].set_xticks(event_times)
axes[1,1].legend().remove()

# ----- Employment Status -----
axes[1,2].plot(partnered_results_esa['event_time'], partnered_results_esa['percentage_coef_m'], label='Male', marker='o', color='black', linestyle='--')
axes[1,2].plot(partnered_results_esa['event_time'], partnered_results_esa['percentage_coef_w'], label='Female', marker='^', color='black')
axes[1,2].axvline(x=-0.5, color='gray', linestyle='-', linewidth=1)  # Vertical line for event time
axes[1,2].annotate(f'Long-Run Penalty: {abs(lr_child_penalty_partnered_esa):.2f}', xy=(0.99, 0.03), xycoords='axes fraction', fontsize=10, color='black', ha='right')
axes[1,2].set_ylim(-1, 0.5)
axes[1,2].set_xlabel('Years Since Birth of First Child')
axes[1,2].set_xticks(event_times)
axes[1,2].legend().remove()

# Add a shared legend outside the subplots
fig.legend(labels=['Male', 'Female'], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12)

# Adjust layout to ensure there's no overlap
plt.tight_layout()

# Remove the top and right spines for each individual subplot
for row in axes:  # Loop through rows of subplots
    for ax in row:  # Loop through each subplot in the row
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', linestyle='--', color='gray', alpha=0.3)


# Adjust layout for better spacing
plt.tight_layout(rect=[0.1, 0, 1, 0.95])

# Save the plot
fig.savefig('../04_results/figures/partnership_figure.png', bbox_inches='tight', dpi=300)
