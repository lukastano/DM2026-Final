import pandas as pd
import matplotlib.pyplot as plt

# Load data
print("Loading data...")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Basic info
print("\n=== TRAIN DATA ===")
print(f"Shape: {train.shape}")
print(f"Unique regions: {train['region_id'].nunique()}")
print(f"\nColumns:\n{train.columns.tolist()}")

print("\n=== TEST DATA ===")
print(f"Shape: {test.shape}")
print(f"Unique regions: {test['region_id'].nunique()}")

# Check scores
print("\n=== SCORES (TARGET VARIABLE) ===")
print(train['score'].value_counts().sort_index())
print(f"\nNon-null scores: {train['score'].notna().sum():,}")
print(f"Total rows: {len(train):,}")
print(f"Percentage with scores: {train['score'].notna().sum() / len(train) * 100:.1f}%")

# Missing values
print("\n=== MISSING VALUES ===")
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

# Plot score distribution
plt.figure(figsize=(10, 6))
train['score'].value_counts().sort_index().plot(kind='bar', color='steelblue')
plt.title('Drought Severity Score Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Score (0-5)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('score_distribution.png', dpi=150)
print("\n✓ Saved plot to score_distribution.png")

# Feature statistics
print("\n=== FEATURE STATISTICS ===")
feature_cols = ['wind', 'humidity', 'tmp', 'prec', 'surf_tmp', 'surf_pre']
print(train[feature_cols].describe())

# Check data by region
print("\n=== DATA BY REGION ===")
print(f"Rows per region in train: {len(train) // train['region_id'].nunique()}")
print(f"Rows per region in test: {len(test) // test['region_id'].nunique()}")

print("\n✓ Exploration complete!")