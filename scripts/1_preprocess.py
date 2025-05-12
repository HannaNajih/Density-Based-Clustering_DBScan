import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories if they don't exist
os.makedirs('../outputs/plots', exist_ok=True)
os.makedirs('../outputs/clusters', exist_ok=True)

def load_data():
    """Load and return the dataset."""
    try:
        df = pd.read_excel('../data/BusinessDS.xlsx')
        print("✅ Data loaded successfully. Shape:", df.shape)
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def preprocess_data(df):
    """Handle missing values and scale numeric features."""
    # Drop or impute missing values (adjust as needed)
    df_cleaned = df.dropna()  # or df.fillna(df.mean())
    
    # Select numeric columns only
    numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    X = df_cleaned[numeric_cols]
    
    # Scale features (crucial for DBSCAN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaled data
    pd.DataFrame(X_scaled, columns=numeric_cols).to_csv('../outputs/scaled_data.csv', index=False)
    print("🔄 Scaled data saved to '../outputs/scaled_data.csv'")
    
    return X_scaled, numeric_cols

def eda_visualizations(df, numeric_cols):
    """Generate EDA plots."""
    # Histograms
    df[numeric_cols].hist(figsize=(12, 8), color='purple', alpha=0.7)
    plt.suptitle('Numeric Features Distribution', y=1.02)
    plt.tight_layout()
    plt.savefig('../outputs/plots/histograms.png')
    plt.close()
    
    # Correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='PuRd', fmt=".2f")
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.savefig('../outputs/plots/correlation_matrix.png')
    plt.close()

if __name__ == "__main__":
    # Pipeline
    df = load_data()
    X_scaled, numeric_cols = preprocess_data(df)
    eda_visualizations(df, numeric_cols)