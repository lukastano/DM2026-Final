#  Setup & Run Instructions

## Step 1: Install Packages
```bash
pip install -r requirements.txt
```

## Step 2: Setup Kaggle Token
Create a file named `.env` in your project folder with:
KAGGLE_API_TOKEN=your_token_here
Replace `your_token_here` with your actual Kaggle API token.
Add this line to `.gitignore`:
.env

## Step 3: Download Data
```bash
python download_data.py
```

## Step 4: Run Pipeline
```bash
python run_pipeline.py
```

**Wait ~30 minutes for training to complete.**

## Expected Score

- Target: 0.76-0.82 (beat Baseline 3: 0.8056)