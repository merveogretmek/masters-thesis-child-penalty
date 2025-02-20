# From Parenthood to Pay Gaps: A Heterogeneous Study of the Child Penalty in the German Labor Market (1984-2020) and the Role of Parental Leave Policy (Elterngeld)

This repository contains the code for my Master's thesis in Economics at LMU Munich. The thesis examines the effect of parenthood on pay gaps in the German labor market and investigates how parental leave policy (Elterngeld) influences these effects. The analysis spans data from 1984 to 2020 and utilizes heterogeneous analysis to uncover differences across various subgroups, such as region, sector, education, and more.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup & Dependencies](#setup--dependencies)
- [Data](#data)
- [Running the Code](#running-the-code)
- [Reproducibility](#reproducibility)
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

## Data

The repository references a data folder that contains the raw and processed data used for the analysis. Due to data privacy and licensing restrictions, the data is not included in this repository. Users interested in replicating the study will need to obtain the SOEP data by contacting the data owner directly.

## Running the Code

### Data Cleaning

* Navigate to the `01_cleaning` folder and run the cleaning script to preprocess the raw data.

```bash
python 01_cleaning.py
```

### Estimation

* The `02_estimation` folder contains multiple scripts to run various estimation models. For example, to run the general estimation:

```bash
python 0201_estimation_general.py
```

* For subgroup analyses (e.g., by residential area, region, sector, etc.), run the corresponding scripts (e.g., `0202_estimation_resarea.py`, `0203_estimation_region.py`, etc.).

### Reform Estimation

* To examine the impact of the reform (changes in parental leave policy), execute the reform-specific estimation scripts:

```bash
python 0209_reform_estimation_general.py
```

* Similarly, run scripts such as `0210_reform_estimation_resarea.py`, `0211_reform_estimation_region.py`, etc.

### Visualization

* Generate figures and graphical summaries by running the scripts in the `03_visualization` folder:

```bash
python 0303_visualization.py
```

### Validity Checks

* Run the following script to check for certain assumptions that strengthen the analysis:

```bash
python 05_validity_checks.py
```

### Interactive App

* The interactive app is located in the `06_app` folder. First, navigate to this folder and install dependencies specific to the app:

```bash
cd 06_app
pip install -r requirements.txt
```

* Then, launch the app by running:

```bash
python app.py
```

* The `assets` folder contains supplementary files (such as images and styles), and the results stores outputs used by the app.

## Reproducibility 

If you wish to replicate or extend this study, please ensure the following:

* Use the same data processing steps as detailed in the cleaning script.
* Follow the prescribed order of operations from data cleaning to result generation.
* Document any deviations from the original analysis and provide a rationale for such changes.

Most of the code is thoroughly commented to aid understanding and reproducibility.

## Contact

For any questions or further information, please contact:

* **Name:** Merve Ogretmek
* **Email:** Merve.Oegretmek@campus.lmu.de

Feedback and contributions are welcome.

  


 
