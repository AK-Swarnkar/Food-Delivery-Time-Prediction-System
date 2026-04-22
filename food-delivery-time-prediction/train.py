# Food Delivery Time Prediction using XGBoost

# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pickle
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load and explore the dataset
df=pd.read_csv('Food_Delivery_Times.csv')
df.head()
df.info()
df.describe()
for col in ['Weather', 'Traffic_Level', 'Time_of_Day']:
    df[col] = df[col].fillna(df[col].mode()[0])

df['Courier_Experience_yrs'] = df['Courier_Experience_yrs'].fillna(df['Courier_Experience_yrs'].median())

print(df.isnull().sum())

#EDA
plt.figure(figsize=(10, 6))
plt.scatter(df["Distance_km"], df["Delivery_Time_min"], alpha=0.6, edgecolor='k')
plt.title("Distance_km vs Delivery_Time_min", fontsize=14)
plt.xlabel("Distance_km", fontsize=12)
plt.ylabel("Delivery_Time_min", fontsize=12)
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(9, 6))
sns.barplot(x='Traffic_Level', y='Delivery_Time_min', data=df, errorbar=None)
plt.title("Average Delivery Time by Traffic Level")
plt.xlabel("Traffic Level")
plt.ylabel("Average Delivery Time (minutes)")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Courier_Experience_yrs', y='Delivery_Time_min', data=df)
plt.title("Delivery Time by Courier Experience")
plt.xlabel("Courier Experience (Years)")
plt.ylabel("Delivery Time (minutes)")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(df["Courier_Experience_yrs"], df["Delivery_Time_min"], alpha=0.6, edgecolor='k')
plt.title("Courier_Experience vs Delivery_Time_min", fontsize=14)
plt.xlabel("Courier_Experience_yrs", fontsize=12)
plt.ylabel("Delivery_Time_min", fontsize=12)
plt.grid(alpha=0.3)
plt.show()

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

plt.figure(figsize=(14, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(len(numeric_cols) // 3 + 1, 3, i)
    sns.histplot(df[col], kde=True, bins=20, color='firebrick')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

sns.boxplot(x=df['Distance_km'], ax=axes[0, 0])
axes[0, 0].set_title('Distance_km')

sns.boxplot(x=df['Preparation_Time_min'], ax=axes[0, 1])
axes[0, 1].set_title('Preparation_Time_min')

sns.boxplot(x=df['Courier_Experience_yrs'], ax=axes[1, 0])
axes[1, 0].set_title('Courier_Experience_yrs')

sns.boxplot(x=df['Delivery_Time_min'], ax=axes[1, 1])
axes[1, 1].set_title('Delivery_Time_min')

plt.tight_layout()
plt.show()

numeric_df = df.select_dtypes(include=['number'])
spearman_corr = numeric_df.corr(method='spearman')

# Выводим матрицу корреляции
print(spearman_corr)

# Построение тепловой карты для визуализации матрицы корреляции
plt.figure(figsize=(8, 7))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Spearman Correlation Matrix')
plt.show()

# Encoding categorical variables
df = pd.get_dummies(df, columns=['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type'], drop_first=True)
df.head()
df.shape

# Define features and target variable
X = df.drop(['Order_ID', 'Delivery_Time_min'], axis = 1)
y = df['Delivery_Time_min']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=250)

# Train the XGBoost model
xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='mae')

# Запишем необходимые нам параметры для дальнейшего перебора.
param_xgb = {
    'n_estimators': [50, 100, 200],  
    'learning_rate': [0.1, 0.2],  
    'max_depth': [3, 5, 10],
    'subsample': [0.8, 1.0],
    'min_child_weight': [1, 3, 5]    
}

# Сделаем перебор заданных выше параметров, при этом разделив выборку данных на 5 частей.
grid_search__xgb = GridSearchCV(xgb_regressor, param_xgb, cv=5)

# Обучим модель на тренировочных данных
grid_search__xgb.fit(X_train, y_train)

grid_search__xgb.best_params_

best_gs_xgb_two = grid_search__xgb.best_estimator_

y_test_pred1 = best_gs_xgb_two.predict(X_test)
mae_test = mean_absolute_error(y_test, y_test_pred1)

print('Score on train data = ', round(best_gs_xgb_two.score(X_train, y_train), 4))
print('Score on test data = ', round(best_gs_xgb_two.score(X_test, y_test), 4))
print('MAE on test data =', round(mae_test, 4))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred1, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--') 
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Extreme Gradient Boosting')
plt.show()

# Save the XGBoost model to a .pkl file
filename_xgb = 'xgboost_model.pkl'
pickle.dump(best_gs_xgb_two, open(filename_xgb, 'wb'))

print(f"XGBoost model saved as {filename_xgb}")