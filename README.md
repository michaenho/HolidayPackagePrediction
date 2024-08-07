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

"Trips & Travel.Com" observed that 18% of customers purchased packages last year. However, high marketing costs were incurred due to random customer contacts. The goal is to provide recommendations to the marketing team and also build a model to predict the potential customer who is going to purchase the newly introduced Wellness Tourism Package.


## 2. Data Collection

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction). It contains 20 columns and 4888 rows, providing information on customer demographics, past interactions, and package preferences. Some of the key features include designation, passport, city tier, martial status, occupation and product taken.


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

For our data analysis, we need to find out the most significant variables in impacting whether a customer will purchase a package, so that we could optimise the sales focus. I have created a function below that extracts the top n% of features that are positively correlated with the ProdTaken variable.

```
def get_top_positive_correlated_features(df,percentage):

    df_corr_analysis = pd.get_dummies(df, columns = ['TypeOfContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation'])
    corr_sorted = df_corr_analysis.corr()['ProdTaken'].sort_values(ascending = True)
  
    num_of_cols_to_drop = int(percentage * len(corr_sorted))
    cols_to_drop = corr_sorted.iloc[:num_of_cols_to_drop].index
    df_corr_analysis.drop(columns = cols_to_drop, inplace = True)
    final_corr = df_corr_analysis.corr()['ProdTaken'].drop('ProdTaken').sort_values(ascending = False)

    
    sns.barplot(x = final_corr.index, y = final_corr.values)
    plt.xticks(rotation = 30, size = 6)
    plt.xlabel('Features')
    plt.ylabel('Correlation')
    plt.title(f'Top {(1-percentage)*100:.1f}% features positively correlated with ProdTaken')
    plt.show()
    return final_corr
get_top_positive_correlated_features(df, 0.8)
```
The top 20% of features that are positively correlated with ProdTaken are shown below. As we can see, those who are single, own a passport or are executives are more likely to take up the package.

![Plot](/Plots/CorrelatedFeatures.png)


Knowing those are the features that are highly correlated with taking up the package, we would like to find out what are the % of customers belonging to those categories actually took up the package.

```
top_features = pd.DataFrame(
    {'Feature': ['Passport', 'ProductBasic', 'Executive', 'SingleStatus'],
    'Percentage of Product Taken': [round((df[(df.ProdTaken == 1) & (df.Passport == 1)]['ProdTaken'].count() / df[df.Passport == 1]['ProdTaken'].count())*100, 2),
    round((df[(df.ProdTaken == 1) & (df.ProductPitched == 'Basic')]['ProdTaken'].count() / df[df.ProductPitched == 'Basic']['ProdTaken'].count())*100, 2),
    round((df[(df.ProdTaken == 1) & (df.Designation == 'Executive')]['ProdTaken'].count() / df[df.Designation == 'Executive']['ProdTaken'].count())*100, 2),
    round((df[(df.ProdTaken == 1) & (df.MaritalStatus == 'Single')]['ProdTaken'].count() / df[df.MaritalStatus == 'Single']['ProdTaken'].count())*100, 2)]})

top_features
```

Based on the results obtained, we could see that about 35% of customers who own a passport or are single, and 30% of customers who are executives, have purchased the package. Those are significant proportion and hence, it is wise to target these groups of customers to increase the likelihood of package purchase and reduce marketing costs.


## 5. Feature Engineering

Before we perform machine learning, we need to ensure the features are suitable for training by the models.

### 1. Resampling

As we can observe from the plot, there is imbalance in the dependent variable 'ProdTaken' where the number of no is greater than the number of yes. This could cause the machine learning models to be bias towards the negative class.

```
df.ProdTaken.value_counts().plot(kind = 'bar', xlabel='Product Taken', ylabel='Count')
plt.xticks(rotation = 0, ticks = [0,1], labels = ['No', 'Yes'])
```
![Plot](/Plots/Imbalance.png)

We could resolve this by performing oversampling of the 'Yes' class to ensure the number of 'No' and 'Yes' is the same.

```
df_majority = df[df.ProdTaken == 0]
df_minority = df[df.ProdTaken == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state = 42)
df_resampled = pd.concat([df_majority, df_minority_upsampled])
```

### 2. Encoding of categorical features

As the machine learning models are less able to understand those categories, it is better to transform these categories into binary values so that they are more appropriate for model training. Before performing such transformation, it is essential to do train-test-split to ensure there is no data leakage where the test data will be seen by the encoder, which can introduce information about the distribution of categories that should not be known to the model during training.

```
# Perform train test split to split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 42)

# For features with 2 category values, we can use LabelEncoder to encode them, whereas for those with more than 2 category values, we can use OneHotEncoder to encode them
toc = LabelEncoder()
sex = LabelEncoder()
train_x.TypeOfContact = toc.fit_transform(train_x.TypeOfContact)
test_x.TypeOfContact = toc.transform(test_x.TypeOfContact)
train_x.Gender = sex.fit_transform(train_x.Gender)
test_x.Gender = sex.transform(test_x.Gender)

ohe = OneHotEncoder()
ct = ColumnTransformer(transformers = [('encoder', ohe, [5,9,11,17])], remainder = 'passthrough')
train_x_onehot = ct.fit_transform(train_x)
test_x_onehot = ct.transform(test_x)

train_x_onehot_df = pd.DataFrame(train_x_onehot, columns = final_columns)
test_x_onehot_df = pd.DataFrame(test_x_onehot, columns = final_columns)
```


## 6. Model Training and Evaluation

### 1. Classical machine learning models

We start off the model training by using some classical machine learning models. As this is a classification problem, the appropriate models to use are Logistic Regression, Naive Bayes, Decision Tree and Support Vector Machine models.

```
models = {
    'LogisticRegression': LogisticRegression(),
    'NaiveBayes': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(),
    'SupportVectorMachine': SVC(),

}
def model_fit(models):
    for model_name in models:
        model = models[model_name]
        model.fit(train_x_onehot_df, train_y)
        y_pred = model.predict(test_x_onehot_df)
        print('------------------------------------')
        print(model_name)
        print(f'Accuracy: {accuracy_score(test_y,y_pred):.2f}')
        print(f'Recall: {recall_score(test_y,y_pred):.2f}')
        print('Classification Report:')
        print(classification_report(test_y,y_pred))
        ConfusionMatrixDisplay(confusion_matrix(test_y,y_pred)).plot()
        plt.title("Confusion Matrix")
        plt.show()
        print('------------------------------------')
    
model_fit(models)
```
For our evaluation, we will focus more on the recall score as it is more important to reduce the number of false negatives (it is more costly to misclassify a customer who would have taken the product). After training and evaluating the models, we realised that the decision tree is the best performing model with a recall score of 0.99 and accuracy of 0.96 using the test set.

### 2. Ensemble models

We will be trying out a few ensemble techniques (bagging and boosting) and see if there are any improvement, though the recall score for decision tree is already very high. Those models include the Random Forest, AdaBoost, Gradient Boost and XGBoost.

```
ensemble_models = {
    'RandomForest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'GradientBoost': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier()
}

model_fit(ensemble_models)
```

After training and evaluating the models, we observed that the Random Forest is the best performing model with recall and accuracy to be 0.99. This is to be expected as the ensemble models are more robust than a simple decision tree.

## 7. Hyperparameter Tuning

Finally, we do hyperparameter tuning for the top 3 perfoming models (Decision Tree, Random Forest and XGBoost) from previous trainings to get their best parameters and boost their performances.

```
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

xgboost_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, 20],
    'learning_rate': [0.05, 0.1, 0.15, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'colsample_bylevel': [0.7, 0.8, 0.9],
    'colsample_bynode': [0.7, 0.8, 0.9]
}

models_for_tuning = [('DecisionTreeClassifier', DecisionTreeClassifier(), dt_param_grid),
                    ('RandomForestClassifier', RandomForestClassifier(), rf_param_grid),
                    ('XGBClassifier', XGBClassifier(), xgboost_param_grid)
]

def hyperparameter_tuning(model_with_params):
    for i in model_with_params:
        name, model, param = i[0], i[1], i[2]
        randomized_search = RandomizedSearchCV(estimator = model, param_distributions = param, n_iter = 100, cv = 5, n_jobs = -1)
        randomized_search.fit(train_x_onehot_df, train_y)
        y_pred = randomized_search.predict(test_x_onehot_df)
        print('------------------------------------')
        print(f'{name} with hyperparameter tuning')
        print(f'Accuracy: {accuracy_score(test_y,y_pred):.2f}')
        print(f'Recall: {recall_score(test_y,y_pred):.2f}')
        print('Classification Report:')
        print(classification_report(test_y,y_pred))
        ConfusionMatrixDisplay(confusion_matrix(test_y,y_pred)).plot()
        plt.title("Confusion Matrix")
        plt.show()
        print(f'Best parameters for the {name} model:')
        print(randomized_search.best_params_)
        print('------------------------------------')
hyperparameter_tuning(models_for_tuning)
```
Random Forest still perform the best with both recall and accuracy score to be 0.99. Hence, we will stick to Random Forest for predicting if potential customers want to purchase the package.


## 8. Conclusion

- In order to reduce marketing costs and sell the new package effectively, we should target customers who fulfill at least one of the following criterias: Own a passport, single, executives, have purchased the basic package previously.

- Random Forest model is the best model to predict the potential customer who is going to purchase the newly introduced travel package going forward.
