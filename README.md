# Holiday Package Prediction

###### Author: Michaen Ho

#

### Project Overview

"Trips & Travel.Com" company aims to expand its customer base by introducing a new Wellness Tourism Package. Wellness Tourism is defined as travel that allows the traveler to maintain, enhance, or kick-start a healthy lifestyle, and support or increase one's sense of well-being. The company plans to utilize existing customer data to efficiently target potential customers, reducing marketing costs and increasing the likelihood of package purchases.

### Table of Contents

### 1. [Problem Statement](#1-problem-statement)
### 2. [Data Collection](#2-data-collection)
### 3. [Data Cleaning](#3-data-cleaning)
### 4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
### 5. [Feature Engineering](#5-feature-engineering)
### 6. [Model Training and Evaluation](#6-model-training-and-evaluation)
### 7. [Hyperparameter Tuning](#7-hyperparameter-tuning)
### 8. [Conclusion](#8-conclusion)


## 1. Problem Statement

"Trips & Travel.Com" observed that 18% of customers purchased packages last year. However, high marketing costs were incurred due to random customer contacts. The goal is to predict which customers are likely to purchase the new Wellness Tourism Package using available data, thereby optimizing marketing expenditures.


## 2. Data Collection

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction). It contains 20 columns and 4888 rows, providing information on customer demographics, past interactions, and package preferences.


## 3. Data Cleaning

### 1. Imputation of missing values

By plotting the distribution of features that contain missing values, we could determine how to impute the missing values.

```
plt.figure(figsize = (15,12))
plt.subplot(2,2,1)
sns.kdeplot(df.Age, shade = True)
plt.title('Age Distribution')
plt.subplot(2,2,2)
sns.kdeplot(df.DurationOfPitch, shade = True)
plt.title('Duration Of Pitch Distribution')
plt.subplot(2,2,3)
sns.kdeplot(df.MonthlyIncome, shade = True)
plt.title('Monthly Income Distribution')
plt.subplot(2,2,4)
sns.kdeplot(df.NumberOfTrips, shade = True)
plt.title('Number Of Trips Distribution')
```
![Plot](/Plots/NormalDistribution.png)

We could tell from the plots above that those features have a normal distribution, hence it is appropriate to use their mean values to replace the missing values. For the categorical features with missing values, we use mode to replace them.

```
cols_with_na = ['Age', 'DurationOfPitch', 'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting', 'MonthlyIncome', 'TypeOfContact']
for col in cols_with_na:
    if col in ['Age', 'DurationOfPitch', 'MonthlyIncome', 'NumberOfTrips']:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])
```

### 2. Ensuring data consistency

Firstly, we change those column types from float to integer of those columns that only contain integer values to save memory space. Also, we replace values and column names to ensure data consistency.

```
df.rename(columns = {'TypeofContact': 'TypeOfContact'}, inplace = True)
df.Gender = df.Gender.str.replace("Fe Male", "Female")

float_cols = df.select_dtypes(include = ['float64']).columns
for col in float_cols:
    df[col] = df[col].astype(int)
```

## 4. Exploratory Data Analysis

## 5. Feature Engineering

## 6. Model Training and Evaluation

## 7. Hyperparameter Tuning

## 8. Conclusion

