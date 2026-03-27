#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# Load data
train = pd.read_csv("Loan_Default.csv")

X = train.drop(columns=['Status'])
y = train['Status']

# Feature Engineering Function
def add_features(df):
    df = df.copy()
    
    # Ratios
    df['loan_to_income'] = df['loan_amount'] / df['income'].replace(0, np.nan)
    df['property_to_income'] = df['property_value'] / df['income'].replace(0, np.nan)
    
    # Effective interest
    df['effective_rate'] = df['rate_of_interest'] + df['Interest_rate_spread']
    
    # Binary indicator for high DTI
    df['high_dti'] = (df['dtir1'] > 50).astype(int)
    
    # Missing indicators
    for col in ['Upfront_charges', 'Interest_rate_spread', 'rate_of_interest',
                'dtir1', 'LTV', 'property_value']:
        df[col + '_missing'] = df[col].isnull().astype(int)
    
    return df

# Train-test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply feature engineering
X_train_fe = add_features(X_train_raw)
X_test_fe = add_features(X_test_raw)

# Numeric imputation
num_cols_fe = X_train_fe.select_dtypes(include=['float64','int64']).columns.tolist()

num_imputer = SimpleImputer(strategy='median')
X_train_fe[num_cols_fe] = num_imputer.fit_transform(X_train_fe[num_cols_fe])
X_test_fe[num_cols_fe] = num_imputer.transform(X_test_fe[num_cols_fe])

# Identify columns
binary_features = ['high_dti'] + [c for c in X_train_fe.columns if c.endswith("_missing")]
num_features = [c for c in num_cols_fe if c not in binary_features]
cat_features = X_train_fe.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipelines
numeric_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

binary_pipeline = Pipeline([
    ('passthrough', 'passthrough')
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_features),
    ('cat', categorical_pipeline, cat_features),
    ('bin', binary_pipeline, binary_features)
])

# XGBoost Model
xgb_model = XGBClassifier(
    n_estimators=250,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=4,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
)

# Full pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', xgb_model)
])

print("Training XGBoost model...")
pipeline.fit(X_train_fe, y_train)

# Save the trained pipeline
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Save feature engineering function and imputer separately
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump({
        'add_features': add_features,
        'num_imputer': num_imputer
    }, f)

print("Model saved successfully!")
print(f"Test accuracy: {pipeline.score(X_test_fe, y_test):.4f}")