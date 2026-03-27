# feature_engineering.py
import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
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
