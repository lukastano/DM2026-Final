import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
import features

print("="*60)
print("CREATING ADVANCED SUBMISSION")
print("="*60)

# Load models
print("\nLoading ensemble models...")
with open('ensemble_model.pkl', 'rb') as f:
    models = pickle.load(f)

xgb_model = models['xgb']
lgb_model = models['lgb']
rf_model = models['rf']
et_model = models['et']
weights = models['weights']
feature_cols = models['features']

print(f"✓ Loaded 4 models with {len(feature_cols)} features")

# Load and process test data
print("\nLoading test data...")
test = pd.read_csv('data/test.csv')
print(f"Test data shape: {test.shape}")

# Create features
print("\nCreating features for test data...")
test_features = features.create_advanced_features(test, is_train=False)

# Generate predictions for each region
print("\nGenerating predictions...")
predictions = []

for i, region in enumerate(test_features['region_id'].unique(), 1):
    if i % 200 == 0:
        print(f"  Processing region {i}/{test_features['region_id'].nunique()}...")
    
    region_data = test_features[test_features['region_id'] == region].copy()
    
    # Prepare features
    X_region = region_data[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_region = X_region.fillna(X_region.mean())
    
    # Strategy: Use last 4 weeks (28 days) for prediction
    if len(X_region) >= 28:
        X_recent = X_region.tail(28)
    else:
        X_recent = X_region
    
    # Average recent features
    X_pred = X_recent.mean().values.reshape(1, -1)
    
    # Get predictions from all models
    xgb_pred = xgb_model.predict(X_pred)[0]
    lgb_pred = lgb_model.predict(X_pred)[0]
    rf_pred = rf_model.predict(X_pred)[0]
    et_pred = et_model.predict(X_pred)[0]
    
    # Ensemble prediction
    base_pred = (
        weights['xgb'] * xgb_pred +
        weights['lgb'] * lgb_pred +
        weights['rf'] * rf_pred +
        weights['et'] * et_pred
    )
    
    # Trend analysis: is drought getting worse or better?
    if len(region_data) >= 28:
        recent_tmp = region_data['tmp_roll_mean_28d'].iloc[-1]
        recent_prec = region_data['prec_roll_mean_28d'].iloc[-1]
        recent_humidity = region_data['humidity_roll_mean_28d'].iloc[-1]
        
        # Drought worsening indicators
        trend_factor = 0
        if recent_prec < region_data['prec_roll_mean_28d'].quantile(0.3):
            trend_factor += 0.05  # Low precipitation
        if recent_tmp > region_data['tmp_roll_mean_28d'].quantile(0.7):
            trend_factor += 0.03  # High temperature
        if recent_humidity < region_data['humidity_roll_mean_28d'].quantile(0.3):
            trend_factor += 0.02  # Low humidity
    else:
        trend_factor = 0
    
    # Create 5-week predictions with trend
    preds = []
    for week in range(5):
        week_pred = base_pred + (trend_factor * week * 0.5)
        week_pred = np.clip(week_pred, 0, 5)
        preds.append(week_pred)
    
    predictions.append({
        'region_id': region,
        'pred_week1': preds[0],
        'pred_week2': preds[1],
        'pred_week3': preds[2],
        'pred_week4': preds[3],
        'pred_week5': preds[4],
    })

# Create submission
submission = pd.DataFrame(predictions)

# Final clipping
for col in ['pred_week1', 'pred_week2', 'pred_week3', 'pred_week4', 'pred_week5']:
    submission[col] = submission[col].clip(0, 5)

# Save
submission.to_csv('submission_advanced.csv', index=False)

print("\n✓ Advanced submission created: submission_advanced.csv")
print(f"\nFirst 5 rows:")
print(submission.head(10))

print(f"\nPrediction statistics:")
for col in ['pred_week1', 'pred_week2', 'pred_week3', 'pred_week4', 'pred_week5']:
    print(f"{col}: min={submission[col].min():.2f}, "
          f"mean={submission[col].mean():.2f}, "
          f"max={submission[col].max():.2f}")

print("\n" + "="*60)
print("UPLOAD 'submission_advanced.csv' TO KAGGLE!")
print("="*60)