import pandas as pd
import pickle
import numpy as np

# Load model
print("Loading trained model...")
with open('simple_model.pkl', 'rb') as f:
    model, features = pickle.load(f)

# Load test data
print("Loading test data...")
test = pd.read_csv('data/test.csv')

print(f"Test data shape: {test.shape}")
print(f"Unique regions: {test['region_id'].nunique()}")

# Make predictions for each region
print("\nGenerating predictions...")
predictions = []

for i, region in enumerate(test['region_id'].unique(), 1):
    if i % 100 == 0:
        print(f"  Processing region {i}/{test['region_id'].nunique()}...")
    
    region_data = test[test['region_id'] == region].copy()
    
    # Prepare features
    X_test = region_data[features].fillna(region_data[features].mean())
    
    # Use last 28 days average
    recent_features = X_test.tail(28).mean().values.reshape(1, -1)
    
    # Make prediction
    base_pred = model.predict(recent_features)[0]
    
    # Create 5 predictions with slight trend
    preds = []
    for week in range(5):
        pred = base_pred + (week * 0.02)
        pred = np.clip(pred, 0, 5)
        preds.append(pred)
    
    predictions.append({
        'region_id': region,
        'pred_week1': preds[0],
        'pred_week2': preds[1],
        'pred_week3': preds[2],
        'pred_week4': preds[3],
        'pred_week5': preds[4],
    })

# Create submission dataframe
submission = pd.DataFrame(predictions)

# Ensure all predictions are in valid range [0, 5]
for col in ['pred_week1', 'pred_week2', 'pred_week3', 'pred_week4', 'pred_week5']:
    submission[col] = submission[col].clip(0, 5)

# Save submission
submission.to_csv('submission_simple.csv', index=False)

print("\n✓ Submission created: submission_simple.csv")
print(f"\nFirst 5 rows:")
print(submission.head())

print(f"\nPrediction statistics:")
for col in ['pred_week1', 'pred_week2', 'pred_week3', 'pred_week4', 'pred_week5']:
    print(f"{col}: min={submission[col].min():.2f}, "
          f"mean={submission[col].mean():.2f}, "
          f"max={submission[col].max():.2f}")

print("\n" + "="*60)
print("UPLOAD 'submission_simple.csv' TO KAGGLE!")
print("="*60)