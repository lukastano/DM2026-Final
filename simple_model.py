import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

print("="*60)
print("SIMPLE MODEL TRAINING")
print("="*60)

# Load data
print("\nLoading training data...")
train = pd.read_csv('data/train.csv')

print(f"Total rows: {len(train):,}")
train = train[train['score'].notna()]
print(f"Rows with scores: {len(train):,}")

# Define features
features = ['wind', 'wind_min', 'wind_max', 'wind_range', 
            'humidity', 'tmp', 'tmp_range', 'tmp_max', 'tmp_min',
            'surf_tmp', 'surf_pre', 'dp_tmp', 'wb_tmp', 'prec']

# Prepare X and y
X = train[features].fillna(train[features].mean())
y = train['score']

print(f"\nFeatures: {len(features)}")
print(f"Training samples: {len(X):,}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train set: {len(X_train):,}")
print(f"Validation set: {len(X_val):,}")

# Train model
print("\nTraining Random Forest model...")
model = RandomForestRegressor(
    n_estimators=200, 
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)

# Evaluate
print("\nEvaluating model...")
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

train_mae = mean_absolute_error(y_train, train_pred)
val_mae = mean_absolute_error(y_val, val_pred)

print(f"\n{'='*50}")
print(f"Training MAE:   {train_mae:.4f}")
print(f"Validation MAE: {val_mae:.4f}")
print(f"{'='*50}")

# Feature importance
print("\nTop 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# Save model
print("\nSaving model...")
with open('simple_model.pkl', 'wb') as f:
    pickle.dump((model, features), f)
print("✓ Model saved to simple_model.pkl")

print("\n✓ Training complete!")