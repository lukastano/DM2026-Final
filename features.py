import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_advanced_features(df, is_train=True):
    """
    Create advanced feature engineering
    """
    print("Creating advanced features...")
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Sort by region and date
    df = df.sort_values(['region_id', 'date']).reset_index(drop=True)
    
    # ===== TIME FEATURES =====
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['day_of_month'] = df['date'].dt.day
    
    # Cyclical encoding for seasonal patterns
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # ===== WEATHER INTERACTIONS =====
    # Temperature-based
    df['tmp_humidity_ratio'] = df['tmp'] / (df['humidity'] + 1)
    df['tmp_prec_ratio'] = df['tmp'] / (df['prec'] + 0.1)
    df['tmp_squared'] = df['tmp'] ** 2
    
    # Drought indicators
    df['heat_dryness_index'] = (df['tmp'] * (100 - df['humidity'])) / 100
    df['evaporation_potential'] = df['tmp'] * (1 - df['humidity']/100)
    df['moisture_deficit'] = df['tmp_max'] - df['dp_tmp']
    
    # Wind effects
    df['wind_humidity_interaction'] = df['wind'] * df['humidity']
    df['wind_temp_interaction'] = df['wind'] * df['tmp']
    
    # Pressure patterns
    df['pressure_temp_ratio'] = df['surf_pre'] / (df['tmp'] + 273.15)  # Convert to Kelvin
    
    # ===== ROLLING STATISTICS (TEMPORAL PATTERNS) =====
    print("Computing rolling features...")
    
    rolling_windows = [7, 14, 28]  # 1 week, 2 weeks, 4 weeks
    rolling_features = ['tmp', 'humidity', 'prec', 'wind', 'tmp_range']
    
    for window in rolling_windows:
        for feature in rolling_features:
            # Rolling mean
            df[f'{feature}_roll_mean_{window}d'] = df.groupby('region_id')[feature].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            # Rolling std
            df[f'{feature}_roll_std_{window}d'] = df.groupby('region_id')[feature].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            # Rolling max
            df[f'{feature}_roll_max_{window}d'] = df.groupby('region_id')[feature].transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )
            # Rolling min
            df[f'{feature}_roll_min_{window}d'] = df.groupby('region_id')[feature].transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )
    
    # ===== LAG FEATURES =====
    print("Computing lag features...")
    
    lag_features = ['tmp', 'humidity', 'prec', 'wind', 'tmp_range', 'surf_pre']
    lag_periods = [7, 14, 21, 28]  # Weekly lags
    
    for feature in lag_features:
        for lag in lag_periods:
            df[f'{feature}_lag_{lag}d'] = df.groupby('region_id')[feature].shift(lag)
    
    # ===== TREND FEATURES =====
    print("Computing trend features...")
    
    trend_features = ['tmp', 'humidity', 'prec']
    for feature in trend_features:
        # 7-day vs 28-day comparison (recent vs long-term)
        df[f'{feature}_trend_7_28'] = (
            df[f'{feature}_roll_mean_7d'] - df[f'{feature}_roll_mean_28d']
        )
        
        # Difference from previous week
        df[f'{feature}_diff_7d'] = df.groupby('region_id')[feature].diff(7)
    
    # ===== CUMULATIVE FEATURES (DROUGHT ACCUMULATION) =====
    print("Computing cumulative features...")
    
    # Cumulative precipitation deficit
    df['prec_cumsum_28d'] = df.groupby('region_id')['prec'].transform(
        lambda x: x.rolling(28, min_periods=1).sum()
    )
    
    # Consecutive dry days (prec < threshold)
    df['is_dry_day'] = (df['prec'] < 0.1).astype(int)
    df['consecutive_dry_days'] = df.groupby('region_id')['is_dry_day'].transform(
        lambda x: x.rolling(28, min_periods=1).sum()
    )
    
    # Heat wave indicator
    df['is_hot_day'] = (df['tmp_max'] > df.groupby('region_id')['tmp_max'].transform('quantile', 0.75)).astype(int)
    df['consecutive_hot_days'] = df.groupby('region_id')['is_hot_day'].transform(
        lambda x: x.rolling(14, min_periods=1).sum()
    )
    
    # ===== STATISTICAL FEATURES =====
    # Variability indicators
    df['tmp_variability_14d'] = df['tmp_roll_std_14d'] / (df['tmp_roll_mean_14d'] + 1)
    df['prec_variability_14d'] = df['prec_roll_std_14d'] / (df['prec_roll_mean_14d'] + 0.1)
    
    print(f"✓ Created {len(df.columns)} total features")
    
    return df


if __name__ == "__main__":
    # Test the feature engineering
    print("Testing feature engineering...")
    
    train = pd.read_csv('data/train.csv')
    print(f"Original columns: {len(train.columns)}")
    
    train_features = create_advanced_features(train, is_train=True)
    print(f"After feature engineering: {len(train_features.columns)}")
    
    # Save
    train_features.to_csv('data/train_features.csv', index=False)
    print("✓ Saved to data/train_features.csv")