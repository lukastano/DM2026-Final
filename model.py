import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print(" MODEL TRAINING")
print("=" * 60)

# ── 1. Load data ──────────────────────────────────────────────
print("\nLoading training data...")
train = pd.read_csv('data/train.csv')
train['date'] = pd.to_datetime(train['date'], errors='coerce')  # FIX: robust parsing
train = train.sort_values(['region_id', 'date']).reset_index(drop=True)
print(f"Total rows: {len(train):,}")

# ── 2. Feature engineering ────────────────────────────────────
print("\nEngineering features...")

train['heat_dryness_index'] = (train['tmp'] * (100 - train['humidity'])) / 100
train['moisture_deficit']   = train['tmp_max'] - train['dp_tmp']
train['is_dry_day']         = (train['prec'] < 0.1).astype(int)

ROLL_FEATS = ['tmp', 'humidity', 'prec', 'wind']
WINDOWS    = [7, 14, 28]
LAGS       = [7, 14]

for feat in ROLL_FEATS:
    g = train.groupby('region_id')[feat]
    for w in WINDOWS:
        train[f'{feat}_roll_mean_{w}d'] = g.transform(lambda x: x.rolling(w, min_periods=1).mean())
        train[f'{feat}_roll_std_{w}d']  = g.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
    for lag in LAGS:
        train[f'{feat}_lag_{lag}d'] = g.shift(lag)

for feat in ['tmp', 'humidity', 'prec']:
    train[f'{feat}_trend'] = train[f'{feat}_roll_mean_7d'] - train[f'{feat}_roll_mean_28d']

train['consec_dry_days'] = train.groupby('region_id')['is_dry_day'].transform(
    lambda x: x.rolling(28, min_periods=1).sum())

train['month']     = train['date'].dt.month.fillna(0).astype(int)
train['month_sin'] = np.sin(2 * np.pi * train['month'] / 12)
train['month_cos'] = np.cos(2 * np.pi * train['month'] / 12)

# ── 3. Keep scored rows only ──────────────────────────────────
train = train[train['score'].notna()].copy()
print(f"Rows with scores: {len(train):,}")

EXCLUDE      = {'region_id', 'date', 'score', 'is_dry_day', 'month'}
feature_cols = [c for c in train.columns if c not in EXCLUDE]

X        = train[feature_cols].replace([np.inf, -np.inf], np.nan)
col_means = X.mean()
X        = X.fillna(col_means)
y        = train['score'].values
print(f"Features used: {len(feature_cols)}")

# ── 4. Temporal split ─────────────────────────────────────────
cutoff = train['date'].quantile(0.85)
mask   = train['date'] >= cutoff
X_train, y_train = X[~mask], y[~mask]
X_val,   y_val   = X[mask],  y[mask]
print(f"Train: {len(X_train):,}  Val: {len(X_val):,}")

# ── 5. Train ──────────────────────────────────────────────────
print("\nTraining Random Forest...")
model = RandomForestRegressor(
    n_estimators=300, max_depth=20, min_samples_split=5,
    min_samples_leaf=2, max_features='sqrt',
    random_state=42, n_jobs=-1, verbose=1,
)
model.fit(X_train, y_train)

print(f"\nTrain MAE : {mean_absolute_error(y_train, model.predict(X_train)):.4f}")
print(f"Val MAE   : {mean_absolute_error(y_val,   model.predict(X_val)):.4f}")

# ── 6. Save ───────────────────────────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': feature_cols, 'col_means': col_means}, f)
print("\n✓ Saved model.pkl")