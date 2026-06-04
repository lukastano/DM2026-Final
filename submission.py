import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SUBMISSION")
print("=" * 60)

# ── Load model ────────────────────────────────────────────────
print("\nLoading model...")
with open('model.pkl', 'rb') as f:
    payload = pickle.load(f)
model        = payload['model']
feature_cols = payload['features']
col_means    = payload['col_means']
print(f"✓ Loaded ({len(feature_cols)} features)")

# ── Load & engineer test features ────────────────────────────
print("\nLoading test data...")
test = pd.read_csv('data/test.csv')
test['date'] = pd.to_datetime(test['date'], errors='coerce')  # FIX: robust parsing
test = test.sort_values(['region_id', 'date']).reset_index(drop=True)
print(f"Test shape: {test.shape}  |  Regions: {test['region_id'].nunique()}")

# Mirror same feature engineering
test['heat_dryness_index'] = (test['tmp'] * (100 - test['humidity'])) / 100
test['moisture_deficit']   = test['tmp_max'] - test['dp_tmp']
test['is_dry_day']         = (test['prec'] < 0.1).astype(int)

ROLL_FEATS = ['tmp', 'humidity', 'prec', 'wind']
WINDOWS    = [7, 14, 28]
LAGS       = [7, 14]

for feat in ROLL_FEATS:
    g = test.groupby('region_id')[feat]
    for w in WINDOWS:
        test[f'{feat}_roll_mean_{w}d'] = g.transform(lambda x: x.rolling(w, min_periods=1).mean())
        test[f'{feat}_roll_std_{w}d']  = g.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
    for lag in LAGS:
        test[f'{feat}_lag_{lag}d'] = g.shift(lag)

for feat in ['tmp', 'humidity', 'prec']:
    test[f'{feat}_trend'] = test[f'{feat}_roll_mean_7d'] - test[f'{feat}_roll_mean_28d']

test['consec_dry_days'] = test.groupby('region_id')['is_dry_day'].transform(
    lambda x: x.rolling(28, min_periods=1).sum())

test['month']     = test['date'].dt.month.fillna(0).astype(int)
test['month_sin'] = np.sin(2 * np.pi * test['month'] / 12)
test['month_cos'] = np.cos(2 * np.pi * test['month'] / 12)

# ── Predict ───────────────────────────────────────────────────
print("\nGenerating predictions...")
predictions = []
regions = test['region_id'].unique()

for i, region in enumerate(regions, 1):
    if i % 300 == 0:
        print(f"  {i}/{len(regions)}...")

    rdf      = test[test['region_id'] == region]
    last_row = rdf.iloc[[-1]][feature_cols].replace([np.inf, -np.inf], np.nan)
    last_row = last_row.fillna(col_means)

    base_pred  = model.predict(last_row)[0]
    prec_trend = float(rdf.iloc[-1].get('prec_trend', 0))
    tmp_trend  = float(rdf.iloc[-1].get('tmp_trend',  0))
    weekly_drift = np.clip(-prec_trend * 0.05 + tmp_trend * 0.02, -0.1, 0.15)

    preds = [float(np.clip(base_pred + weekly_drift * w, 0, 5)) for w in range(5)]
    predictions.append({
        'region_id': region,
        'pred_week1': preds[0], 'pred_week2': preds[1],
        'pred_week3': preds[2], 'pred_week4': preds[3],
        'pred_week5': preds[4],
    })

# ── Save ──────────────────────────────────────────────────────
submission = pd.DataFrame(predictions)
for col in [f'pred_week{w}' for w in range(1, 6)]:
    submission[col] = submission[col].clip(0, 5)

submission.to_csv('submission.csv', index=False)
print(f"\n✓ Saved submission.csv  ({len(submission)} rows)")
print("\nStats:")
for col in [f'pred_week{w}' for w in range(1, 6)]:
    print(f"  {col}: min={submission[col].min():.3f}  mean={submission[col].mean():.3f}  max={submission[col].max():.3f}")
print("\nUpload 'submission.csv' to Kaggle!")