import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(df.head())

x = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

models = {
    "Linear Regression" : LinearRegression(),
    "Ridge Regression" : Ridge(alpha=1.0),
    "Lasso Regression" : Lasso(alpha=0.1),
    "Elastic Net Regression" : ElasticNet(alpha=0.1, l1_ratio=0.5)
}

for name, model in models.items():
    model.fit(x_train, y_train)
    
    pred = model.predict(x_test)
    
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    print("=" * 40)
    print(name)
    print(f"mae: {mae:.3f}")
    print(f"mse: {mse:.3f}")
    print(f"r2: {r2:.3f}")

