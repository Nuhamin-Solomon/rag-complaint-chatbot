"""
Task 1: Exploratory Data Analysis & Preprocessing
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path

RAW_PATH = Path("data/raw/cfpb_complaints.csv")
OUTPUT_PATH = Path("data/processed/filtered_complaints.csv")

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(RAW_PATH)
print(f"Total records: {len(df)}")

# ---------------------------
# EDA
# ---------------------------
print("\nProduct distribution:")
print(df["Product"].value_counts())

df["narrative_length"] = (
    df["Consumer complaint narrative"]
    .fillna("")
    .apply(lambda x: len(x.split()))
)

print("\nNarrative length summary:")
print(df["narrative_length"].describe())

print("\nComplaints with narratives:",
      (df["Consumer complaint narrative"].notna()).sum())

# ---------------------------
# Filtering
# ---------------------------
TARGET_PRODUCTS = [
    "Credit card",
    "Personal loan",
    "Savings account",
    "Money transfer"
]

df = df[df["Product"].isin(TARGET_PRODUCTS)]
df = df[df["Consumer complaint narrative"].notna()]

print(f"\nRecords after filtering: {len(df)}")

# ---------------------------
# Cleaning
# ---------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["cleaned_narrative"] = df["Consumer complaint narrative"].apply(clean_text)

# ---------------------------
# Export
# ---------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved cleaned dataset to {OUTPUT_PATH}")
