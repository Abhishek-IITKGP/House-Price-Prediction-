# -*- coding: utf-8 -*-


Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JvRI_BKHY_rP4AmXoMf20IZ-YlMFCttQ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()

dataset = pd.concat([train, test], axis = 0)

dataset.shape

dataset.head()

"""# Exploratory Data Analysis

# Missing Values
"""

features_with_na = [features for features in dataset.columns if dataset[features].isnull().sum() > 1]

for features in features_with_na:
  print(features,np.round(dataset[features].isnull().mean() * 100,3),'% missing values')

for features in features_with_na:
  data = dataset.copy()
  data[features] = np.where(data[features].isnull(),1,0)

  data.groupby(features)['SalePrice'].median().plot.bar()
  plt.title('features')
  plt.show()

#check the columns which are not object
numerical_features = [features for features in dataset if dataset[features].dtype != 'O']
print('number of numerical features: ',len(numerical_features))
dataset[numerical_features].head()

#getting the year features
year_features = [features for features in numerical_features if 'Yr' in features or 'Year' in features]
year_features

dataset.groupby('YrSold')['SalePrice'].median().plot()

for features in year_features:
  if features != 'YrSold':
    data = dataset.copy()
    data[features] = data['YrSold'] - data[features]

    plt.scatter(data[features], data['SalePrice'])
    plt.xlabel(features)
    plt.ylabel('SalePrice')
    plt.show()

#selecting the discrete features
discrete_features = [features for features in numerical_features if (len(dataset[features].unique()) < 30) and features not in year_features+['Id']]
len(discrete_features)

discrete_features

dataset[discrete_features].head()

for features in discrete_features:
  data = dataset.copy()
  data.groupby(features)['SalePrice'].median().plot.bar()
  plt.xlabel(features)
  plt.ylabel('Sale Price')
  plt.show()

"""Continous features"""

continous_features = [features for features in numerical_features if features not in discrete_features + year_features + ['Id']]
len(continous_features)

continous_features

for features in continous_features:
  data = dataset.copy()
  data[features].hist(bins = 30)
  plt.title(features)
  plt.ylabel('count')
  plt.show()

"""Converting the non gaussian features into gaussian features by log transformation"""

for features in continous_features:
  data = dataset.copy()
  if 0 in data[features].unique():
    pass
  else:
    data[features] = np.log(data[features])
    data['SalePrice'] = np.log(data['SalePrice'])
    plt.scatter(data[features], data['SalePrice'])
    plt.xlabel(features)
    plt.ylabel('Sale Price')
    plt.show()

"""# Outliers detection with boxplot"""

for features in continous_features:
  data = dataset.copy()
  if 0 in data[features].unique():
    pass
  else:
    data[features] = np.log(data[features])
    data.boxplot(column = features)
    plt.ylabel(features)
    plt.show()

"""# Categorical features"""

#categorical features are object data type
categorical_features = [features for features in dataset if dataset[features].dtype == 'O']
len(categorical_features)

categorical_features

dataset[categorical_features].head()

for features in categorical_features:
  print('number of unique categories in',features , 'is: ', len(dataset[features].unique()))

#realtionship between categorical features and output label
for features in categorical_features:
  data = dataset.copy()
  data.groupby(features)['SalePrice'].median().plot.bar()
  plt.xlabel(features)
  plt.ylabel('Sale Price')
  plt.show()

"""# Feature Engineering

# Missing values

Missing values in categorical features
"""

cat_feature_na = [features for features in dataset if dataset[features].isnull().sum() > 1 and dataset[features].dtype == 'O']
print(features, dataset[cat_feature_na].isnull().mean(), "% NaN values")

"""Handelling the missing values, replaing the nan value with 'missing values'"""

def replace_nan(dataset, features):
  data = dataset.copy()
  data[features] = data[features].fillna('missing')
  return data
dataset = replace_nan(dataset, cat_feature_na)
dataset[cat_feature_na].isnull().sum()

dataset.head()

numerical_feature_na = [features for features in dataset if dataset[features].isnull().sum() > 1 and dataset[features].dtype != 'O']
numerical_feature_na

"""Replacing the nan values in the numerical column with median as it has many outliers"""

for features in numerical_feature_na:

  median_value = dataset[features].median()

  #creating the new features that tells us whether it contains the nan value or not
  dataset[features + 'nan'] = np.where(dataset[features].isnull(),1,0)
  dataset[features].fillna(median_value,inplace = True)
print(dataset[numerical_feature_na].isnull().sum())

dataset.head(10)

"""As with the year sold the sale price is decreasing which is not true so we will find the correlation between year built, year modified, and garage year built with year sold column"""

#temporal variable (data time type)
for features in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
  dataset[features] = dataset['YrSold'] - dataset[features]

dataset.head(10)

dataset[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()

"""# Log transformation"""

skewed_num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

for features in skewed_num_features:
  dataset[features] = np.log(dataset[features])

dataset[skewed_num_features].head()

for features in skewed_num_features:
  data = dataset.copy()
  data[features].hist(bins = 50)
  plt.title(features)
  plt.show()

"""Now these numerical features looks more normally distributed

# Rare categorical variables
remove those categories that present less than 1% labels
"""

categorical_features

for features in categorical_features:
  percentage = dataset.groupby(features)['SalePrice'].count()/len(dataset)
  rare_index = percentage[percentage < 0.01].index
  dataset[features] = np.where(dataset[features].isin(rare_index),'Rare var', dataset[features])

dataset.head(50)

"""# Feature Scaling

encoding the categorical variables as the string value cannot be converted into the numerical variable by the scaler
"""

for feature in categorical_features:
    labels_ordered=dataset.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    dataset[feature]=dataset[feature].map(labels_ordered)

dataset.head(20)

#standerdising the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_to_scale = [features for features in dataset.columns if features not in ['SalePrice', 'Id']]

#conatinating the Id and SalePrice column with the scaled dataset
data = pd.concat([dataset[['Id', 'SalePrice']].reset_index(drop=True),
                    pd.DataFrame(scaler.fit_transform(dataset[data_to_scale]), columns=data_to_scale)],
                    axis=1)

data.head(20)

data.to_csv('train(1).csv')

"""# Feature Selection"""

from sklearn.feature_selection import SelectFromModel
#will use regularisation technique to select the important features
from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split
x = dataset.drop(columns = 'SalePrice', axis = 1)
y = dataset['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)

cols_with_nan = []
for cols in x_train.columns:
  if x_train[cols].isnull().sum() > 0:
    cols_with_nan.append(cols)

cols_with_nan
for cols in cols_with_nan:
  print(cols,x_train[cols].isnull().sum())

x_train.fillna(x_train.median(),inplace = True)

"""Applying regularisation to select from all the features."""

# apply regularistaion Lasso and SelectFromModel to identify the model with only important features
feature_selection_model = SelectFromModel(Lasso(alpha = 0.001, random_state = 0))
feature_selection_model.fit(x_train,y_train)

# it gives a list of boolean type with selected features as True and ignored eatures as False
feature_selection_model.get_support()

selected_features = x_train.columns[feature_selection_model.get_support()]
print(selected_features)
print('number of features selected is: ',len(selected_features))

#this is the new dataset with less number of features after applying regularisation.
xtrain = x_train[selected_features]
xtrain.head()

"""# Hyperparameter Tunning using GridSearchCV"""

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()

from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators':[50,100,150],
    'max_depth':[5,10,20],
    'min_samples_split':[2,5,10]
}
grid_search = GridSearchCV(regressor,param_grid = param_grid,cv = 5, scoring = 'r2')
grid_search.fit(x_train,y_train)

print('best parametrs for randomforest regressor are: ',grid_search.best_params_)
print('best score for the randomforest regressor is: ',grid_search.best_score_)

regressor = RandomForestRegressor(n_estimators=200, max_depth = 20, min_samples_split = 5)

"""# Model Training"""

regressor.fit(x_train,y_train)

"""# Model Evaluation"""

y_pred = regressor.predict(x_test)

from sklearn.metrics import r2_score
r_squared = r2_score(y_test,y_pred)
print('r_square: ',r_squared)
n = len(y_test)
p = x.shape[1]
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
print('adjusted r_square: ', adjusted_r_squared)

"""Since the model is doing well with cross validation data during the hyperparameter tunning using GridSearchCV, also with the unseen x_test dataset, so we can say that our model is not overfittd."""

