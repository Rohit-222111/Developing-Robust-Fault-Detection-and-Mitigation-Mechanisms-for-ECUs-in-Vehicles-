import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

data = pd.read_csv('oxygen_sensor_final_with_faults.csv')
df = pd.DataFrame(data)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X = pd.DataFrame(X)

corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_cols = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
X = X.drop(columns=drop_cols)
column_names = X.columns

Q1 = np.percentile(X, 25, axis = 0)
Q3 = np.percentile(X, 75, axis = 0)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
mask = (X >= lower_bound) & (X <= upper_bound)
filtered_indices = np.all(mask, axis=1)
X_filtered = X[filtered_indices]
y_filtered = y[filtered_indices]

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_filtered)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_filtered, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

xgb_model = XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=6, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    random_state=42)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

def predict(x):
    return xgb_model.predict(x)