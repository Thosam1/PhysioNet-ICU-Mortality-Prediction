<div align="center">
  
# PhysioNet-ICU-Mortality-Prediction

[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=Thosam1_PhysioNet-ICU-Mortality-Prediction&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=Thosam1_PhysioNet-ICU-Mortality-Prediction)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=Thosam1_PhysioNet-ICU-Mortality-Prediction&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=Thosam1_PhysioNet-ICU-Mortality-Prediction)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=Thosam1_PhysioNet-ICU-Mortality-Prediction&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=Thosam1_PhysioNet-ICU-Mortality-Prediction)

</div>

## Overview
This project aims to predict ICU patient mortality using the **PhysioNet 2012 Challenge** dataset. We develop machine learning models based on **irregularly sampled multivariate time-series data**, capturing patient vitals and static attributes from the first **48 hours of ICU stay**. The goal is to predict whether a patient **survives or dies** in the ICU.

## Dataset
- **Source**: [PhysioNet 2012 Challenge](https://physionet.org/content/challenge-2012/)
- **Data**: First **48 hours** of ICU stay
- **37 dynamic variables**: Vital signs, lab test results, etc.
- **4 static variables**: Age, Gender, Height, Weight
- **Target**: Binary classification (Discharged Alive = 0, Deceased = 1)

## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Thosam1/PhysioNet-ICU-Mortality-Prediction.git
   cd PhysioNet-ICU-Mortality-Prediction
   ```

2. **Set Up the Environment**:
   - Have Python 3.8 or higher installed.
   - Create a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Prepare the Dataset**:
   - Download the dataset from [PhysioNet 2012 Challenge](https://physionet.org/content/challenge-2012/).
   - Unzip the dataset and place its contents in the `data/` folder.

4. **Run the Notebooks**:
   - Navigate to the `notebooks/` folder.
   - Execute the Jupyter notebooks in order:
     - `1_data_parsing.ipynb`
     - `2_data_preprocessing.ipynb`
     - `3_model_training.ipynb`
     - `4_evaluation_and_visualization.ipynb`

Some cells in the notebooks generate files required by subsequent notebooks. If you need to recompute the dataset for model training, uncomment the section in the first notebook labeled with the comment:

```
# Only need to run once
```

This will regenerate the `.parquet` files used in the later notebooks. Alternatively, you can use the pre-generated `.parquet` files provided in the repository. The same approach applies to the `.pkl` files containing the precomputed embeddings. For using these, please unzip the `outcomes.zip`, `set.zip` and `embeddings.zip`. 

Please notice that the `non_agg_{dataset}_embeddings.zip` were sent separately from the rest of the files in the submission as we were limited by the submission maximum file size. We included these as the non aggregated embedddings take a very long time to compute. To be able to use these files, unzip them into the root directory of the project. 

## Folder Structure

```
PhysioNet-ICU-Mortality-Prediction/
├── data/                   # Raw data from challenge .zip
├── data_parsing/           # Scripts for parsing raw data
├── data_preprocessing/     # Scripts for cleaning and preprocessing data
├── models/                 # Machine learning and deep learning models
├── utils/                  # Utility functions
├── visualization/          # Scripts for generating plots and visualizations
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
