# From Parenthood to Pay Gaps: A Heterogeneous Study of the Child Penalty in the German Labor Market (1984-2020) and the Role of Parental Leave Policy (Elterngeld)

This repository contains the code for my Master's thesis in Economics at LMU Munich. The thesis examines the effect of parenthood on pay gaps in the German labor market and investigates how parental leave policy (Elterngeld) influences these effects. The analysis spans data from 1984 to 2020 and utilizes heterogeneous analysis to uncover differences across various subgroups, such as region, sector, education, and more.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup & Dependencies](#setup--dependencies)
- [Data](#data)
- [Running the Code](#running-the-code)
- [Folder Details](#folder-details)
  - [01_cleaning](#01_cleaning)
  - [02_estimation](#02_estimation)
  - [03_visualization](#03_visualization)
  - [04_results](#04_results)
  - [05_validity_checks](#05_validity_checks)
  - [06_app](#06_app)
- [Reproducibility](#reproducibility)
- [Citations & Acknowledgements](#citations--acknowledgements)
- [License](#license)
- [Contact](#contact)

## Overview 

This thesis investigates the "child penalty"-how parenthood affects earnings- in the context of the German market between 1984 and 2020. A key focus of the study is understanding the role of parental leave policy (Elterngeld) in shaping these outcomes. The research uses event-study estimation, a series of robustness and validity checks, and visualization techniques to provide insights into the dynamics of pay gaps post-parenthood.

The project is structured to allow for a step-by-step reproduction of the analysis, from data cleaning and estimation to result generation and interactive application deployment.

## Repository Structure

```bash
/code
  ├── 01_cleaning          # Scripts for data cleaning and preprocessing.
  │    ├── 01_cleaning.py
  ├── 02_estimation        # Estimation routines, including heterogeneous analysis.
  │    ├── 0201_estimation_general.py
  │    ├── 0202_estimation_resarea.py
  │    ├── 0203_estimation_region.py
  │    ├── 0204_estimation_sector.py
  │    ├── 0205_estimation_origin.py
  │    ├── 0206_estimation_education.py
  │    ├── 0207_estimation_parenthoodage.py
  │    ├── 0208_estimation_partnership.py
  │    ├── 0209_reform_estimation_general.py
  │    ├── 0210_reform_estimation_resarea.py
  │    ├── 0211_reform_estimation_region.py
  │    ├── 0212_reform_estimation_sector.py
  │    ├── 0213_reform_estimation_origin.py
  │    ├── 0214_reform_estimation_education.py
  │    ├── 0215_reform_estimation_parenthoodage.py
  │    └── 0216_reform_estimation_partnership.py
  ├── 03_visualization     # Code for generating figures and visual summaries.
  │    ├── 03_visualization.py
  ├── 04_results           # Figures 
  ├── 05_validity_checks   # Script for performing robustness and validity checks.
  │    ├── 05_validity_checks.py
  └── 06_app               # Code for the interactive app component.
        ├── assets         
        ├── app.py         
        ├── requirements.txt  
        └── results        
  
/data                       # Data folder (not included on GitHub due to restrictions)
```

## Setup & Dependencies

Before running the code, ensure you have the following environment setup:
* **Python Version**: Python 3.7 or higher
* **Required Packages**
  * `pandas`
  * `numpy`
  * `statsmodels`
  * `plotly`
  * Additional packages as specified in the [requirements.txt](requirements.txt).

You can install the required packages with:

```bash
pip install -r requirements.txt
```


 
