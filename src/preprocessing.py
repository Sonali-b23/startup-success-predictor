import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    """
    Cleans and prepares data for ML. 
    Drops leaky columns, extracts target variables, handles missing values, and scales variables.
    """
    # 1. Drop useless and leaky columns
    cols_to_drop = ['Unnamed: 0', 'Unnamed: 6', 'id', 'object_id', 'state_code.1', 'labels', 'name', 'zip_code']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # 2. Extract targets
    y_class = df['status'].apply(lambda x: 1 if x == 'acquired' else 0)
    
    if 'funding_total_usd' in df.columns:
        y_reg = df.pop('funding_total_usd')
    else:
        y_reg = pd.Series([0] * len(df))
        
    df = df.drop(columns=['status'])
    
    # 3. Organize column types (Drop raw date strings since Age metrics already exist)
    date_cols = [c for c in df.columns if c.endswith('_at') or 'date' in c.lower()]
    df = df.drop(columns=date_cols, errors='ignore')

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 4. Handle Missings (Imputation)
    num_imputer = SimpleImputer(strategy='median')
    if num_cols:
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
        
    cat_imputer = SimpleImputer(strategy='most_frequent')
    if cat_cols:
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    # 5. Label Encode categorical data
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    return df, y_class, y_reg