# Home Credit Default Risk Categorical Data Cleanining Using Machine Learning. 

## Project Summary
The purpose of the project is to make accurate predictions for categorical data missing values using machine learning model. Accuracy of predictions also determined.

## Table of Contents
- [Data Source](#data-source)
- [Data Sets](#data-sets)
- [Data Cleaning](#data-cleaning)
  - [Data Loading](#data-loading)
  - [Data Segregation](#data-segregation)
  - [Data Processing](#data-processing)
  - [Data Prediction](#data-prediction)
  - [Data Export](#data-export)
  - [Results and Conclusions](#results-and-conclusions)
- [Dependencies](#dependencies)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)
- [Author](#author)
- [References](#references)

## Data Source

## Data Sets
Provided data has seven tables that are used in the project for cleaning and further analysis and machine learning models for prediction various features.
Provided data is in csv forms. Since data size is too big. it is not uploaded into github. it can be downloaded from kaggle directly.
- [Application Train]
- [Bureau]
- [Bureau Balance]
- [Previous Application]
- [Credit Card Balance]
- [Installments payments]
- [POS CASH Balance]

## Data Cleaning

### Data Loading

### Data Segregation

### Data Processing

### Data Prediction

### Data Export


## Results and Conclusions



  
## Dependencies

This project involves using various dependences listed below for data cleaning and predictions.

```bash
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

```

## How to Run


## Future Work
Data cleaning is very important aspect for performing analysis and making predictions. If data is not cleaned properly our models can make wrong prediction and vital relations can not be understood.

## Acknowledgments
I would like to thank Home credit for providing Data set on Kaggle.com for analysis.

## Author

Muhammad Kashif 

## References
[1] [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)
