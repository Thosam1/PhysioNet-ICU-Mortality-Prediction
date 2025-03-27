# PhysioNet-ICU-Mortality-Prediction

## Overview
This project aims to predict ICU patient mortality using the **PhysioNet 2012 Challenge** dataset. We develop machine learning models based on **irregularly sampled multivariate time-series data**, capturing patient vitals and static attributes from the first **48 hours of ICU stay**. The goal is to predict whether a patient **survives or dies** in the ICU.

## Dataset
- **Source**: [PhysioNet 2012 Challenge](https://physionet.org/content/challenge-2012/)
- **Data**: First **48 hours** of ICU stay
- **37 dynamic variables**: Vital signs, lab test results, etc.
- **4 static variables**: Age, Gender, Height, Weight
- **Target**: Binary classification (Discharged Alive = 0, Deceased = 1)

## Approach
We experiment with multiple models:
- **Baseline:** Logistic Regression, Random Forest, XGBoost
- **Deep Learning:** LSTM for time-series modeling
- **Feature Engineering:** Handling irregular time-series, missing data, and temporal dependencies

## Get started
Download the zip file from:
unzip and put the content in in the `data/` folder

## Folder Structure
