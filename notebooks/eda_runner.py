import pandas as pd
import numpy as np
import io

def run_eda():
    df = pd.read_csv('data/Data.csv')
    print('--- DATASET SHAPE ---')
    print(df.shape)
    print('\n--- COLUMNS AND MISSING VALUES ---')
    missing = df.isnull().sum()
    print(missing[missing > 0].sort_values(ascending=False))
    
    print('\n--- TARGET VARIABLE (status) DISTRIBUTION ---')
    if 'status' in df.columns:
        print(df['status'].value_counts(normalize=True))
    
    print('\n--- CORRELATIONS WITH TARGET ---')
    # Convert status to binary temporarily for correlation
    temp_df = df.copy()
    if 'status' in temp_df.columns:
        temp_df['status_binary'] = temp_df['status'].apply(lambda x: 1 if x == 'acquired' else 0)
        numeric_cols = temp_df.select_dtypes(include=[np.number])
        corr = numeric_cols.corr()['status_binary'].sort_values(ascending=False)
        print(corr.head(10))
        print('...')
        print(corr.tail(5))
        
    print('\n--- DATES (first 5 rows) ---')
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'at' in c.lower()]
    print(df[date_cols].head())

if __name__ == '__main__':
    run_eda()
