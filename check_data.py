import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print("="*60)
print("TRAIN DATA ANALYSIS")
print("="*60)

# Check one region in detail
r1_train = train[train['region_id'] == 'R1'].copy()
r1_train['date'] = pd.to_datetime(r1_train['date'])

print(f"\nRegion R1 in training data:")
print(f"Total rows: {len(r1_train)}")
print(f"Date range: {r1_train['date'].min()} to {r1_train['date'].max()}")
print(f"Days span: {(r1_train['date'].max() - r1_train['date'].min()).days + 1} days")

# Check score pattern
r1_with_scores = r1_train[r1_train['score'].notna()]
print(f"\nRows with scores: {len(r1_with_scores)}")
print(f"Expected (5480/7): ~{5480/7:.0f}")

# Check if scores are exactly weekly
r1_train_sorted = r1_train.sort_values('date')
score_dates = r1_train_sorted[r1_train_sorted['score'].notna()]['date']
if len(score_dates) > 1:
    date_diffs = score_dates.diff().dt.days.dropna()
    print(f"Days between scores: {date_diffs.unique()}")

print("\n" + "="*60)
print("TEST DATA ANALYSIS")
print("="*60)

r1_test = test[test['region_id'] == 'R1'].copy()
r1_test['date'] = pd.to_datetime(r1_test['date'])

print(f"\nRegion R1 in test data:")
print(f"Total rows: {len(r1_test)}")
print(f"Date range: {r1_test['date'].min()} to {r1_test['date'].max()}")
print(f"Days span: {(r1_test['date'].max() - r1_test['date'].min()).days + 1} days")

# Check if test comes after train
print(f"\nGap between train and test:")
print(f"Train ends: {r1_train['date'].max()}")
print(f"Test starts: {r1_test['date'].min()}")

# Are they consecutive?
gap_days = (r1_test['date'].min() - r1_train['date'].max()).days
print(f"Gap: {gap_days} days")

print("\n" + "="*60)
print("PREDICTION TARGET")
print("="*60)

# What weeks do we need to predict?
test_end = r1_test['date'].max()
week1_start = test_end + pd.Timedelta(days=1)
week5_end = week1_start + pd.Timedelta(days=34)  # 5 weeks = 35 days

print(f"\nTest data ends: {test_end}")
print(f"Week 1 should be: {week1_start} to {week1_start + pd.Timedelta(days=6)}")
print(f"Week 5 should be: {week5_end - pd.Timedelta(days=6)} to {week5_end}")

print("\n" + "="*60)
print("SAMPLE SUBMISSION CHECK")
print("="*60)

sample = pd.read_csv('data/sample_submission.csv')
print(f"\nSample submission shape: {sample.shape}")
print(f"Unique regions: {sample['region_id'].nunique()}")
print(f"\nFirst few rows:")
print(sample.head())