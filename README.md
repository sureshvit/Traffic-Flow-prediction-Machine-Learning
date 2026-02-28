# Traffic-Flow-prediction-Machine-Learning
Traffic Flow Prediction Using Machine Learning, Ensemble & Deep Learning Models
Project Overview
This project focuses on predicting traffic volume using various machine learning, ensemble learning, and deep learning models. The goal is to analyze historical traffic data, preprocess it, and apply different predictive algorithms to forecast future traffic flow.

Dataset
The dataset used in this project is Traffic_data.csv, which contains features related to weather conditions, time, and holidays, along with the traffic_volume as the target variable.

Features:
holiday: Categorical feature indicating if the day is a holiday.
temp: Temperature in Celsius.
rain_1h: Amount of rain in one hour (mm).
snow_1h: Amount of snow in one hour (mm).
clouds_all: Percentage of clouds.
weather_main: Main weather condition (e.g., Clouds, Clear, Rain).
weather_description: Detailed weather description.
date_time: Timestamp of the record.
traffic_volume: Target variable, representing the traffic volume.
Data Preprocessing & Feature Engineering
Libraries Import: Essential libraries like pandas, numpy, matplotlib, and seaborn are imported.
Data Loading: The Traffic_data.csv file is loaded into a pandas DataFrame.
Initial Exploration: head(), tail(), shape, info(), dtypes, describe(), and isnull().sum() are used to understand the data structure, types, and missing values.
Unique Value Analysis: Unique values for holiday, temp, snow_1h, weather_main, and weather_description are displayed.
Data Visualization: Various plots (countplot, distplot, boxplot, barplot, lineplot, heatmap) are used to visualize data distributions, relationships between features, and the impact of categorical variables on traffic volume.
Temperature Conversion: temp (Kelvin) is converted to Celsius.
Date-Time Features: The date_time column is converted to datetime objects, and new features like weekday, date, hour, month, and year are extracted.
Hour Categorization: hour is categorized into 'Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', and 'Late_Night'.
Holiday Feature Modification: The holiday column is transformed into a boolean (True/False) indicating the presence of a holiday.
Outlier Handling for rain_1h: Outliers in rain_1h are addressed by filtering values less than 2000. rain_1h is then categorized into 'light', 'moderate', 'heavy', and 'no_rain' using pd.qcut.
Snow, Fog, Haze, Mist, Thunderstorm Categorization: snow_1h, fog, haze, mist, and thunderstorm columns are modified to 'snow'/'no_snow', 'fog'/'no_fog', etc.
One-Hot Encoding: weather_description is one-hot encoded and then simplified into broader categories like 'fog', 'haze', 'mist', 'thunderstorm', and 'other'. Irrelevant columns like weather_description_other and weather_main are dropped.
Label Encoding: Categorical features including holiday, snow_1h, hour, fog, haze, mist, thunderstorm, weekday, rain_1h, and month are label encoded.
Feature Selection: The date_time column is dropped as its information has been extracted into other features.
Data Splitting: The dataset is split into independent variables (x) and the dependent variable (y - traffic_volume). The data is further divided into training and testing sets (80:20 ratio) using train_test_split.
Machine Learning Models
The following regression models were implemented and evaluated:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
Support Vector Regressor (SVR) with different kernels (RBF, Sigmoid, Polynomial, Linear)
Ensemble Learning Models
Bagging Regressor
Base Estimator: Decision Tree Regressor
Base Estimator: SVR (Linear Kernel)
AdaBoost Regressor
Hyperparameter Tuning with GridSearchCV
XGBoost Regressor
Hyperparameter Tuning with GridSearchCV
Deep Learning Model
Long Short-Term Memory (LSTM) Neural Network
A sequential LSTM model with multiple LSTM layers and Dropout layers.
Compiled with 'adam' optimizer and 'mean_absolute_error' loss function.
Model Evaluation
For each model, the following metrics were calculated:

Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Summary of Key Results (RMSE, MAE, MSE):
Model	RMSE	MAE	MSE
Linear Regression	1839.21	1548.86	3382698.18
Decision Tree Regressor	1231.95	705.61	1517694.05
Random Forest Regressor	975.92	628.74	952412.35
SVR (RBF Kernel)	1987.51	1739.44	3950177.57
SVR (Sigmoid Kernel)	1987.65	1739.64	3950770.47
SVR (Polynomial Kernel)	1987.29	1739.27	3949320.39
SVR (Linear Kernel)	1870.97	1532.08	3500528.56
Bagging (Decision Tree)	1014.56	646.45	1029338.81
Bagging (SVR Linear)	1869.38	1532.02	3494583.04
AdaBoost Regressor	1177.79	878.80	1387179.06
Tuned AdaBoost Regressor	1194.29	892.98	1426333.99
XGBoost Regressor	1042.92	734.03	1087684.47
Tuned Random Forest Regressor	1041.29	730.35	1084289.65
Tuned Decision Tree Regressor	1256.23	947.09	1578110.43
Tuned SVR (Linear Kernel)	1868.09	1529.42	3489747.78
LSTM Regressor	1987.51	1739.69	3950215.08
Random Forest Regressor (both untuned and tuned) demonstrates superior performance among the tested models, exhibiting the lowest RMSE, MAE, and MSE values.

Usage
To predict traffic volume with new input data, you can use the interactive input script provided in the notebook. This script will prompt you to enter values for various features, preprocess them, and then use the best performing model (Random Forest in this case) to make a prediction.
