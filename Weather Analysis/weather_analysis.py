# -*- coding: utf-8 -*-
"""Weather Analysis.ipynb

# **IMPORT MODULES**
"""

#step 1: import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from numpy import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, SGDRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNetCV
import xgboost as xgb
warnings.filterwarnings('ignore')

"""# **LOAD THE DATASET**"""

df = pd.read_csv("/content/drive/MyDrive/climate_data.csv", delimiter=',', quoting = 3)

"""The delimiter parameter specifies the character used to separate fields in the CSV file.
If not specified, the default delimiter is a comma (,), which is why CSV files are often referred to as comma-separated values files.
The quoting parameter controls the handling of quotes inside fields. It tells pandas how to deal with quotes (such as double quotes) surrounding certain fields in the CSV file.
In this case, quoting=3 indicates that pandas should ignore quotes. The value 3 corresponds to csv.QUOTE_NONE, meaning quotes are not recognized and should be ignored.
If not specified, the default behavior is quoting=0, which means quotes are recognized, and fields surrounded by quotes are parsed accordingly.
"""

df.head()

df.info()

"""**Our Dataset has total of 22 columns, 3902 entries (rows).**"""

df.describe()

df.isnull().sum()

"""# **Hence here there are no missing values.**

# **EDA - EXPLORATORY DATA ANALYSIS**

# **What is KDE?**
KDE Plot (Kernel Density Estimate Plot):

A KDE plot is a smoothed representation of the distribution of a continuous variable.
It provides a non-parametric estimate of the probability density function (PDF) of the data.

It is created by placing a kernel (a smooth, symmetric function) at each data point and summing up the contributions from all kernels to generate a smooth curve.

# **What is Box Plot (Box-and-Whisker Plot)?**
A box plot is a graphical representation of the distribution of a continuous variable through five summary statistics: minimum, first quartile (Q1), median (Q2), third quartile (Q3), and maximum.

It consists of a box (or rectangle) that represents the interquartile range (IQR) of the data (Q1 to Q3), with a line inside the box indicating the median.

# **Let's plot a distribution graph, kde plot, box plot for:**

Average temperature (°F)    

Average humidity (%)        

Average dewpoint (°F)       

Average barometer (in)      

Average windspeed (mph)     

Rainfall for month (in)     

Rainfall for year (in)
"""

# Set figure and axes
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5)) #it creates a 1x3 grid of subplots, resulting in a 1-dimensional array of axes.
#Therefore, you should use single indices to access each subplot, not two indices as if accessing elements of a 2D array.
# Plot distribution curve for Average temperature (°F)
sns.histplot(df['Average temperature (°F)'], bins=20, alpha=0.7, kde=True, color='pink', ax=axes[0])
axes[0].set_title('Distribution of Avg temp (°F)')
axes[0].set_xlabel('Avg temp (°F)')
axes[0].set_ylabel('Count/ Frequency')

# Plot KDE curve for Average temperature (°F)
sns.kdeplot(df['Average temperature (°F)'], fill='True', color='pink', ax=axes[1])
axes[1].set_title('KDE plot for Avg temp (°F)')
axes[1].set_xlabel('Avg temp (°F)')
axes[1].set_ylabel('Density')

# Box plot for Average temperature (°F)
sns.boxplot(df['Average temperature (°F)'], color='lightgray', width=0.3, linewidth=2, fliersize=5, ax=axes[2])
axes[2].set_title('Box Plot for Avg temp (°F)')
axes[2].set_xlabel('Avg temp (°F)')
axes[2].set_xlabel('value')

##set figure and axes
fig, axes= plt.subplots(nrows=4, ncols=3, figsize=(18,20))
#let's plot distribution curve for Average humidity (%)
sns.histplot(df['Average humidity (%)'], bins=20, color='pink', kde= True, ax=axes[0, 0])
axes[0,0].set_title('Distribution of Average humidity (%)')
axes[0,0].set_xlabel('Average humidity (%)')
axes[0,0].set_ylabel('Counts')

#let's plot kde curve for Average humidity (%)
sns.kdeplot(df['Average humidity (%)'], fill=True, color='pink', ax=axes[0,1])
axes[0,1].set_title('KDE Plot for Average humidity (%)')
axes[0,1].set_xlabel('Average humidity (%)')
axes[0,1].set_ylabel('density')


#box plot for Average humidity (%)
sns.boxplot(df['Average humidity (%)'], color='lightgray', width=0.3, linewidth=2, fliersize=5, ax=axes[0, 2])
axes[0,2].set_title('BOX Plot for Average humidity (%)')
axes[0,2].set_xlabel('Average humidity (%)')
axes[0,2].set_ylabel('value')

#let's plot distribution curve for dewpoint (°F)
sns.histplot(df['Average dewpoint (°F)'], bins=20, color='pink', kde= True, ax=axes[1, 0])
axes[1,0].set_title('Distribution of Avg dewpoint (°F)')
axes[1,0].set_xlabel('Average dewpoint (°F)')
axes[1,0].set_ylabel('Counts')

#let's plot kde curve for Average dewpoint (°F)
sns.kdeplot(df['Average dewpoint (°F)'], fill=True, color='pink', ax=axes[1,1])
axes[1,1].set_title('KDE Plot for Average humidity (%)')
axes[1,1].set_xlabel('Averagedewpoint (°F)')
axes[1,1].set_ylabel('density')


#box plot for Average dewpoint (°F)
sns.boxplot(df['Average dewpoint (°F)'], color='lightgray', width=0.3, linewidth=2, fliersize=5, ax=axes[1,2])
axes[1,2].set_title('BOX Plot for Average dewpoint (°F)')
axes[1,2].set_xlabel('Average dewpoint (°F)')
axes[1,2].set_ylabel('value')

#let's plot distribution curve for Average barometer (in)
sns.histplot(df['Average barometer (in)'], bins=20, color='pink', kde= True, ax=axes[2, 0])
axes[2,0].set_title('Distribution of Average barometer (in)')
axes[2,0].set_xlabel('Average barometer (in)')
axes[2,0].set_ylabel('Counts')

#let's plot kde curve for Average barometer (in)
sns.kdeplot(df['Average barometer (in)'], fill=True, color='pink', ax=axes[2,1])
axes[2,1].set_title('KDE Plot for Average barometer (in)')
axes[2,1].set_xlabel('Average barometer (in)')
axes[2,1].set_ylabel('density')


#box plot for Average barometer (in)
sns.boxplot(df['Average barometer (in)'], color='lightgray', width=0.3, linewidth=2, fliersize=5, ax=axes[2,2])
axes[2,2].set_title('BOX Plot for Average barometer (in))')
axes[2,2].set_xlabel('Average barometer (in)')
axes[2,2].set_ylabel('value')

#let's plot distribution curve for Average windspeed (mph)
sns.histplot(df['Average windspeed (mph)'], bins=20, color='pink', kde= True, ax=axes[3, 0])
axes[3,0].set_title('Distribution of Average windspeed (mph)')
axes[3,0].set_xlabel('Average windspeed (mph)')
axes[3,0].set_ylabel('Counts')

#let's plot kde curve for Average windspeed (mph)
sns.kdeplot(df['Average windspeed (mph)'], fill=True, color='pink', ax=axes[3,1])
axes[3,1].set_title('KDE Plot for Average windspeed (mph)')
axes[3,1].set_xlabel('Average windspeed (mph)')
axes[3,1].set_ylabel('density')


#box plot for Average windspeed (mph)
sns.boxplot(df['Average windspeed (mph)'],  color='lightgray', width=0.3, linewidth=2, fliersize=5, ax=axes[3,2])
axes[3,2].set_title('BOX Plot for Average windspeed (mph)')
axes[3,2].set_xlabel('Average windspeed (mph))')
axes[3,2].set_ylabel('value')

##set figure and axes
fig, axes= plt.subplots(nrows=2, ncols=3, figsize=(18,10))
#let's plot distribution curve for Average windspeed (mph)
sns.histplot(df['Average windspeed (mph)'], bins=20, color='pink', kde= True, ax=axes[0, 0])
axes[0,0].set_title('Distribution of Average windspeed (mph)')
axes[0,0].set_xlabel('Average windspeed (mph)')
axes[0,0].set_ylabel('Counts')

#let's plot kde curve for Average windspeed (mph)
sns.kdeplot(df['Average windspeed (mph)'], fill=True, color='pink', ax=axes[0,1])
axes[0,1].set_title('KDE Plot for Average windspeed (mph)')
axes[0,1].set_xlabel('Average windspeed (mph)')
axes[0,1].set_ylabel('density')

#box plot for Average windspeed (mph)
sns.boxplot(df['Average windspeed (mph)'],  color='lightgray', width=0.3, linewidth=2, fliersize=5, ax=axes[0, 2])
axes[0,2].set_title('Box plot of Average windspeed (mph)')
axes[0,2].set_xlabel('Average windspeed (mph)')
axes[0,2].set_ylabel('value')

#let's plot distribution curve for Rainfall for year (in)
sns.histplot(df['Rainfall for year (in)'], bins=20, color='pink', kde= True, ax=axes[1, 0])
axes[1,0].set_title('Distribution of Rainfall for year (in)')
axes[1,0].set_xlabel('Rainfall for year (in)')
axes[1,0].set_ylabel('Counts')

#let's plot kde curve for Rainfall for year (in)
sns.kdeplot(df['Rainfall for year (in)'], fill=True, color='pink', ax=axes[1,1])
axes[1,1].set_title('KDE Plot for Rainfall for year (in)')
axes[1,1].set_xlabel('Rainfall for year (in)')
axes[1,1].set_ylabel('density')

#box plot for Rainfall for year (in)
sns.boxplot(df['Rainfall for year (in)'],  color='lightgray', width=0.3, linewidth=2, fliersize=5, ax=axes[1, 2])
axes[1,2].set_title('Box plot for Rainfall for year (in)')
axes[1,2].set_xlabel('Rainfall for year (in)')
axes[1,2].set_ylabel('value')

"""In the context of **sns.histplot**, kde, alpha and bins are parameters used to customize the appearance of the histogram:

**kde:** This parameter stands for Kernel Density Estimation. When set to True, it overlays a kernel density estimate (KDE) plot on top of the histogram bars. KDE is a non-parametric way to estimate the probability density function of a continuous random variable. It provides a smooth, continuous representation of the distribution.

**alpha** is a parameter used to control the transparency of the histogram bars plotted by sns.histplot(). It takes a value between 0 and 1, where:

alpha=0 makes the histogram bars fully transparent (invisible).
alpha=1 makes the histogram bars fully opaque (completely solid).
By default, alpha is set to 1, resulting in solid histogram bars.

**bins:** This parameter determines the number of bins or intervals into which the data is divided in the histogram. Each bin represents a range of values, and the height of each bar in the histogram corresponds to the frequency or count of data points falling within that bin. Specifying the bins parameter allows you to control the granularity of the histogram. Increasing the number of bins can provide more detailed insight into the distribution of the data, while decreasing it can result in a more generalized view.

In a **KDE plot** created with sns.kdeplot, there are **no bins** because KDE plots do not use bins like histograms. Instead, KDE plots estimate the probability density function (PDF) of the data using a kernel smoothing technique. The smoothness of the resulting curve is controlled by parameters such as bandwidth.

The **fill** parameter in **sns.kdeplot** determines whether the area under the KDE curve is filled with color. When fill=True, the area under the curve is filled with the specified color

# **Regression Plot**
Regression plots can help visualize the linear relationship between each feature and the target variable, while correlation plots can provide a numerical measure of the strength and direction of the linear relationship.

Let our model predicts rainfall for given month, hence our target variable is 'Rainfall for month (in)'. Let's plot regression plot for various X_features and target variable (y).
"""

# Define the target variable
target_variable = 'Rainfall for month (in)'

# List of features (X)
features = ['Average temperature (°F)', 'Average humidity (%)', 'Average dewpoint (°F)',
            'Average barometer (in)', 'Average windspeed (mph)', 'Average gustspeed (mph)'
            ]

# Plot EDA for each feature
for feature in features:
    plt.figure(figsize=(6, 4))
                # Scatter plot with regression line
    sns.lmplot(x=feature, y=target_variable, data=df, scatter='True', line_kws={'color': 'orange'})
    plt.title(f'Regression Plot: {feature} vs. {target_variable}')
    plt.xlabel(feature)
    plt.ylabel(target_variable)

# Define the target variable
target_variable = 'Rainfall for month (in)'

# List of features (X)
features = ['Average direction (°deg)', 'Rainfall for year (in)', 'Month',
            'Maximum temperature (°F)', 'Minimum temperature (°F)', 'Maximum humidity (%)'
            ]

# Plot EDA for each feature
for feature in features:
    plt.figure(figsize=(6, 4))
                # Scatter plot with regression line
    sns.lmplot(x=feature, y=target_variable, data=df, scatter='True', line_kws={'color': 'orange'})
    plt.title(f'Regression Plot: {feature} vs. {target_variable}')
    plt.xlabel(feature)
    plt.ylabel(target_variable)

"""# **MODEL DEVELOPMENT**

# **LINEAR REGRESSION MODEL**
"""

y = df['Rainfall for month (in)']
X = df.drop(columns =['diff_pressure', 'Date','Rainfall for month (in)', 'Date1'] )
print(X.info())
print(y.head(10))
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train.info())
print(X_train.info())
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse_test)
print("R-squared:", r2_test)

y_train_pred = model.predict(X_train)

# Evaluate the model
mae = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse_train)
print("R-squared:", r2_train)

results = pd.DataFrame(columns=['Model', 'MSE_train', 'R2_train', 'MSE_test', 'R2_test'])
results.loc[len(results)] = ['Linear Regression', mse_train, r2_train, mse_test, r2_test]

"""# **Random Forest Regression Model**"""

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse_test)
print("R-squared:", r2_test)

y_train_pred = model.predict(X_train)

# Evaluate the model
mae = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse_train)
print("R-squared:", r2_train)

results.loc[len(results)] = ['Random Forest Regression', mse_train, r2_train, mse_test, r2_test]

"""# **Nueral Network Model**"""

model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse_test)
print("R-squared:", r2_test)

y_train_pred = model.predict(X_train)

# Evaluate the model
mae = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse_train)
print("R-squared:", r2_train)

results.loc[len(results)] = ['Neural Network Regression', mse_train, r2_train, mse_test, r2_test]

"""# **Decision Tree Algorithm**"""

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse_test)
print("R-squared:", r2_test)

y_train_pred = model.predict(X_train)

# Evaluate the model
mae = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse_train)
print("R-squared:", r2_train)

results.loc[len(results)] = ['Decision Tree', mse_train, r2_train, mse_test, r2_test]

"""# **Ridge Regression Model**"""

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse_test)
print("R-squared:", r2_test)

y_train_pred = model.predict(X_train)

# Evaluate the model
mae = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse_train)
print("R-squared:", r2_train)

results.loc[len(results)] = ['Riddge Regression', mse_train, r2_train, mse_test, r2_test]

"""# **Elastic Net Regressor Model**"""

model = ElasticNet()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse_test)
print("R-squared:", r2_test)

y_train_pred = model.predict(X_train)

# Evaluate the model
mae = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse_train)
print("R-squared:", r2_train)

results.loc[len(results)] = ['Elastic Net Regression', mse_train, r2_train, mse_test, r2_test]

"""# **XGBoost Algorithm**"""

import xgboost as xgb
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse_test)
print("R-squared:", r2_test)

y_train_pred = model.predict(X_train)

# Evaluate the model
mae = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse_train)
print("R-squared:", r2_train)

results.loc[len(results)] = ['XGBoost Regression', mse_train, r2_train, mse_test, r2_test]

"""# **HYPER PARAMETER TUNING FOR RANDOM FOREST REGRESSION**"""

model = RandomForestRegressor(random_state=42)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Select important features using feature importance
selector = SelectFromModel(best_model)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Fit the model on the selected features
best_model.fit(X_train_selected, y_train)

# Make predictions
y_pred = best_model.predict(X_test_selected)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)
print("Best Parameters:", grid_search.best_params_)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse_test)
print("R-squared:", r2_test)

# Make predictions
y_pred = best_model.predict(X_train_selected)

# Evaluate the model
mae = mean_absolute_error(y_train, y_pred)
mse_train = mean_squared_error(y_train, y_pred)
r2_train = r2_score(y_train, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse_train)
print("R-squared:", r2_train)

"""Hence we got hiighest value of R-squared for testing data using Random Forest Regression model which is 0.773. Let's update this value to our results dataframe."""

results.iloc[1] = ['Random Forest Regression', mse_train, r2_train, mse_test, r2_test]

"""# **HYPER PARAMETER TUNING FOR XGBOOST REGRESSION**"""

model = xgb.XGBRegressor()

# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'gamma': [0, 0.1, 0.3],
    'reg_alpha': [0, 0.1, 0.3],
    'reg_lambda': [0, 0.1, 0.3]
}

# Perform grid search with cross-validation to find the best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Fit the best model on the training data
best_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_test = best_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
print("Best Parameters:", grid_search.best_params_)
print("Test R-squared:", r2_test)
print("Mean Absolute Error:", mae)
print("Test R-squared:", r2_test)

# Make predictions
y_pred = best_model.predict(X_train)

# Evaluate the model
mae = mean_absolute_error(y_train, y_pred)
mse_train = mean_squared_error(y_train, y_pred)
r2_train = r2_score(y_train, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse_train)
print("R-squared:", r2_train)

"""Hence we got hiighest value of R-squared for testing data using XGBoost Regression model which is 0.755. Let's update this value to our results dataframe."""

results.iloc[6] = ['XGBoost Regression', mse_train, r2_train, mse_test, r2_test]

"""# **Let's analyse the R square score and MSE value for all different algorithms after Hyper parameter Tuning, cross validation.**"""

results = results.sort_values(by='MSE_test', ascending=True)
results

