# Home Credit Default Risk Categorical Data Cleanining Using Machine Learning. 

## Project Summary
The purpose of the project is to make accurate predictions for categorical data missing values using machine learning model. Accuracy of predictions also determined.
# Process Diagram
![Alt Text](process_diagram.PNG)
## Table of Contents
- [Extract](#extract)
- [Data Source](#data-source)
- [Data Sets](#data-sets)
- [Transform](#transform)
- [Data Cleaning](#data-cleaning)
  - [Data Loading](#data-loading)
  - [Data Segregation](#data-segregation)
  - [Data Processing](#data-processing)
  - [Data Prediction](#data-prediction)
-[Load](#load)
  - [Data Export](#data-export)
- [Results and Conclusions](#results-and-conclusions)
- [Dependencies](#dependencies)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)
- [Author](#author)
- [References](#references)

# Extract
The process of obtaining data from the source. 
## Data Source
In this project data is obtained from Home credit default risk at Kaggle.com 
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

# Transform
Transformation is the process where data is renamed, adjusted or manipulated in a way to give missing data fields either meaningful data or removing the empty fields from dataset.In this project our aim is study the process where we lose minimal data and fill empty data fields by using Machine learning models especially categorial data. Commonly transformation process is refered to as cleaning data as well so we will dig deeper into cleaning process.

## Data Cleaning
The process of data cleaning is divided into three stages below as per techniquie followed to better understand segregate , process the numerial data and then make the predictions for categorical data and put the predicted values into missing fields.

### Data Loading
Data is loaded from csv files in this project using absolute path method. since Data files are big size which can not be accomodated in the github. so for sake of project they were kept in different folder to avoid large size file errors during commit stage. Code for loading data using absolute path is below.

``` bash
# Dictionary to hold file names and their paths
file_paths = {
    'application_train': '../Resources/application_train.csv',
    'bureau': '../Resources/bureau.csv',
    'bureau_balance': '../Resources/bureau_balance.csv',
    'credit_card_balance': '../Resources/credit_card_balance.csv',
    'POS_CASH_balance': '../Resources/POS_CASH_balance.csv',
    'previous_application': '../Resources/previous_application.csv',
    'installments_payments': '../Resources/installments_payments.csv'
}

# Dictionary to hold the loaded data
data_frames = {}

# Loop through the file_paths dictionary to load each file
for file_name, relative_path in file_paths.items():
    # Construct the absolute path
    file_path = os.path.abspath(relative_path)
    
    # Attempt to load the CSV file into a DataFrame
    try:
        data_frames[file_name] = pd.read_csv(file_path)
        print(f"{file_name} data loaded successfully!")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while loading {file_name}: {e}")

```
The data is present is dictionary  data_frames each data set can be accessed using code below. 
``` bash
# view application_train data
application_train_df = data_frames['application_train']
application_train_df
```
### Data Segregation
After accessing data. Data is segregated by data type e.g Numerical and Categorical Data. This will help to better understand data and treat each column with care so each column can be filled with right values. In this project our focus is to use categorical data for cleaning and predicting values using Machine Learning models. In this project, SK_ID_CURR is considered unique ID for table application_train, previous_application and bureau, while SK_ID_BUREAU is taken as unqiue in the Table bureau_balance. SK_ID_PREV is taken as unique ID in tables POS_CASH_balance, installments_payments and credit_card_balance.
This unique ID consideration is taken as measure clean data and then merge different tables.
code for segregating categorical data and numerical data is shown below.
''' bash
# Separate SK_ID_CURR along with numerical columns
numerical_and_id = application_train_df.select_dtypes(include=['number']).copy()

# Include SK_ID_CURR in numerical set
numerical_and_id['SK_ID_CURR'] = application_train_df['SK_ID_CURR']

# Separate SK_ID_CURR along with categorical columns
categorical_and_id = application_train_df.select_dtypes(include=['object', 'category']).copy()

# Include SK_ID_CURR in categorical set
categorical_and_id['SK_ID_CURR'] = application_train_df['SK_ID_CURR']

# Print the selected numerical data
print("SK_ID_CURR along with Numerical Columns:")
print(numerical_and_id)

# print selected categorical data
print("\nSK_ID_CURR along with Categorical Columns:")
print(categorical_and_id)
```
### Data Processing

### Data Prediction

# Load

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
draw.io for generating Process diagram.
Lovepreet singh, Audrey Nkrumah and Vinay for data search during class project.

## Author

Muhammad Kashif 

## References
[1] [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)
