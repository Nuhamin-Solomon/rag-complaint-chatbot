import pandas as pd
import re
from pathlib import Path

# -------------------
# Paths
# -------------------
RAW_DATA = Path("data/raw/complaints.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "filtered_complaints.csv"

# -------------------
# Text cleaning
# -------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------
# Main
# -------------------
def main():
    print("📥 Loading raw dataset...")
    df = pd.read_csv(RAW_DATA)

    print("🔎 Columns found:")
    print(df.columns)

    products = [
        "Credit card",
        "Personal loan",
        "Savings account",
        "Money transfer"
    ]

    print("🎯 Filtering products...")
    df = df[df["Product"].isin(products)]

    print("🧹 Removing empty narratives...")
    df = df.dropna(subset=["Consumer complaint narrative"])

    print("✍️ Cleaning narratives...")
    df["cleaned_narrative"] = df["Consumer complaint narrative"].apply(clean_text)

    print("💾 Saving processed dataset...")
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Task 1 complete: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
