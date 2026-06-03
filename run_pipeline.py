import subprocess

print("="*70)
print(" ADVANCED DROUGHT PREDICTION PIPELINE")
print("="*70)

steps = [
    ("1. Feature Engineering", "python features.py"),
    ("2. Train Advanced Models", "python train_model.py"),
    ("3. Create Submission", "python create_submission.py"),
]

for step_name, command in steps:
    print(f"\n{'='*70}")
    print(f"STEP: {step_name}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR in {step_name}")
        break
else:
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("✓ Upload 'submission_advanced.csv' to Kaggle!")
    print("="*70)