import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess():
    # Load data
    df = pd.read_excel('data/raw/BusinessDS.xlsx')
    
    # Handle missing values
    df = df.dropna()
    
    # Normalize
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Save processed data
    df.to_csv('data/processed/cleaned_data.csv', index=False)

if __name__ == "__main__":
    preprocess()