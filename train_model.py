import pandas as pd
import numpy as np
import os 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ADVANCED MODEL TRAINING")
print("="*60)

# Load feature-engineered data
print("\nLoading data...")
if not os.path.exists('data/train_features.csv'):
    print("Creating features first...")
    import features
    train = pd.read_csv('data/train.csv')
    train = features.create_advanced_features(train, is_train=True)
    train.to_csv('data/train_features.csv', index=False)
else:
    train = pd.read_csv('data/train_features.csv')

print(f"Total rows: {len(train):,}")
print(f"Total columns: {len(train.columns)}")

# Filter to rows with scores
train = train[train['score'].notna()].copy()
print(f"Rows with scores: {len(train):,}")

# Select features (exclude non-feature columns)
exclude_cols = ['region_id', 'date', 'score', 'is_dry_day', 'is_hot_day']
feature_cols = [col for col in train.columns if col not in exclude_cols]

# Remove any remaining NaN/inf
X = train[feature_cols].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())
y = train['score']

print(f"\n✓ Using {len(feature_cols)} features")

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42
)

print(f"Train set: {len(X_train):,}")
print(f"Validation set: {len(X_val):,}")

# ===== MODEL 1: XGBOOST =====
print("\n" + "="*60)
print("Training XGBoost Model...")
print("="*60)

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=50
)

xgb_pred_train = xgb_model.predict(X_train)
xgb_pred_val = xgb_model.predict(X_val)
xgb_train_mae = mean_absolute_error(y_train, xgb_pred_train)
xgb_val_mae = mean_absolute_error(y_val, xgb_pred_val)

print(f"\nXGBoost Train MAE: {xgb_train_mae:.4f}")
print(f"XGBoost Val MAE: {xgb_val_mae:.4f}")

# ===== MODEL 2: LIGHTGBM =====
print("\n" + "="*60)
print("Training LightGBM Model...")
print("="*60)

lgb_model = LGBMRegressor(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
)

lgb_pred_train = lgb_model.predict(X_train)
lgb_pred_val = lgb_model.predict(X_val)
lgb_train_mae = mean_absolute_error(y_train, lgb_pred_train)
lgb_val_mae = mean_absolute_error(y_val, lgb_pred_val)

print(f"\nLightGBM Train MAE: {lgb_train_mae:.4f}")
print(f"LightGBM Val MAE: {lgb_val_mae:.4f}")

# ===== MODEL 3: RANDOM FOREST =====
print("\n" + "="*60)
print("Training Random Forest Model...")
print("="*60)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)

rf_pred_train = rf_model.predict(X_train)
rf_pred_val = rf_model.predict(X_val)
rf_train_mae = mean_absolute_error(y_train, rf_pred_train)
rf_val_mae = mean_absolute_error(y_val, rf_pred_val)

print(f"\nRandom Forest Train MAE: {rf_train_mae:.4f}")
print(f"Random Forest Val MAE: {rf_val_mae:.4f}")

# ===== MODEL 4: EXTRA TREES =====
print("\n" + "="*60)
print("Training Extra Trees Model...")
print("="*60)

et_model = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

et_model.fit(X_train, y_train)

et_pred_train = et_model.predict(X_train)
et_pred_val = et_model.predict(X_val)
et_train_mae = mean_absolute_error(y_train, et_pred_train)
et_val_mae = mean_absolute_error(y_val, et_pred_val)

print(f"\nExtra Trees Train MAE: {et_train_mae:.4f}")
print(f"Extra Trees Val MAE: {et_val_mae:.4f}")

# ===== ENSEMBLE =====
print("\n" + "="*60)
print("Creating Weighted Ensemble...")
print("="*60)

# Weighted average based on validation performance
weights = {
    'xgb': 1 / xgb_val_mae,
    'lgb': 1 / lgb_val_mae,
    'rf': 1 / rf_val_mae,
    'et': 1 / et_val_mae
}

total_weight = sum(weights.values())
weights = {k: v/total_weight for k, v in weights.items()}

print(f"\nEnsemble Weights:")
for model_name, weight in weights.items():
    print(f"  {model_name}: {weight:.4f}")

ensemble_pred_val = (
    weights['xgb'] * xgb_pred_val +
    weights['lgb'] * lgb_pred_val +
    weights['rf'] * rf_pred_val +
    weights['et'] * et_pred_val
)

ensemble_mae = mean_absolute_error(y_val, ensemble_pred_val)

print(f"\n{'='*60}")
print(f"FINAL ENSEMBLE VALIDATION MAE: {ensemble_mae:.4f}")
print(f"{'='*60}")

# Save all models
print("\nSaving models...")
models = {
    'xgb': xgb_model,
    'lgb': lgb_model,
    'rf': rf_model,
    'et': et_model,
    'weights': weights,
    'features': feature_cols
}

with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(models, f)

print("✓ All models saved to ensemble_model.pkl")

# Feature importance from XGBoost
print("\nTop 20 Most Important Features (XGBoost):")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20).to_string(index=False))

print("\n✓ Training complete!")