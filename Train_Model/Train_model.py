# ======================================
# Import Libraries
# ======================================
import pandas as pd
import numpy as np
from datetime import datetime
import pandera as pa
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
import xgboost as xgb
import shap
import joblib

# ======================================
# Data Loading & Validation
# ======================================
# Load dataset
df = pd.read_csv('airline_profit_data.csv')

# Update schema to match your dataset columns
schema = pa.DataFrameSchema({
    "Profit (USD)": pa.Column(float),  # Allow both positive and negative values
    "Fuel Efficiency (ASK)": pa.Column(float, nullable=True),
    "Delay (Minutes)": pa.Column(float, nullable=True, coerce=True),
    # Add all other columns...
})
schema.validate(df)  # Enforce data quality

# ======================================
# Preprocessing Pipeline
# ======================================
# Drop redundant columns (adjust based on your business logic)
df = df.drop(columns=[
    'Flight Number',  # Likely a unique identifier (not useful for modeling)
    'Revenue (USD)', 'Operating Cost (USD)',  # Direct proxies for profit
    'Net Profit Margin (%)',  # Derived from Profit (USD), redundant
    'Revenue per ASK', 'Cost per ASK'  # Optional: Remove if redundant
])

# Extract temporal features from departure times
def extract_time_features(df):
    # Convert to datetime
    df['Scheduled Departure Time'] = pd.to_datetime(df['Scheduled Departure Time'])
    df['Actual Departure Time'] = pd.to_datetime(df['Actual Departure Time'])
    
    # Extract features
    for col in ['Scheduled Departure Time', 'Actual Departure Time']:
        df[f'{col}_Month'] = df[col].dt.month
        df[f'{col}_DayOfWeek'] = df[col].dt.dayofweek
        df[f'{col}_Hour'] = df[col].dt.hour
        df = df.drop(columns=[col])  # Drop original datetime columns
    return df

df = extract_time_features(df)

# Auto-detect categorical columns (if any remain after preprocessing)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Encode categorical columns (if applicable)
if categorical_cols:
    encoder = TargetEncoder(cols=categorical_cols)
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols], df['Profit (USD)'])

# Impute missing numerical data
num_imputer = SimpleImputer(strategy='median')
numerical_cols = df.select_dtypes(include=[np.number]).columns.drop('Profit (USD)')
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

# ======================================
# Train-Test Split
# ======================================
X = df.drop(columns=['Profit (USD)'])
y = df['Profit (USD)']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================
# Model Training & Tuning
# ======================================
model = xgb.XGBRegressor(objective='reg:squarederror')
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# ======================================
# Evaluation
# ======================================
y_pred = best_model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# ======================================
# Explainability (SHAP)
# ======================================
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

