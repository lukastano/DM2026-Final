import os
import kagglehub
import shutil
import glob
from dotenv import load_dotenv

load_dotenv()

# Set the token
print("Downloading competition data...")
path = kagglehub.competition_download('data-mining-2026-final-project')
print(f"Data downloaded to: {path}")

# Create local data folder and copy files there
os.makedirs('data', exist_ok=True)

# Find and copy files
for filename in ['train.csv', 'test.csv', 'sample_submission.csv']:
    found_files = glob.glob(os.path.join(path, '**', filename), recursive=True)
    
    if found_files:
        src = found_files[0]
        dst = os.path.join('data', filename)
        shutil.copy(src, dst)
        print(f"Copied {filename} to data/")
    else:
        print(f" Could not find {filename}")

print("\n All files are now in ./data/ folder for easy access!")