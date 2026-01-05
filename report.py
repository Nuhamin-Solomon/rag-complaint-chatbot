# repot.py
import os
import pandas as pd

# Paths
DATA_PATH = "data/processed/filtered_complaints.csv"

def generate_report(df):
    print("===== Dataset Report =====\n")
    
    # General info
    print("Total complaints:", len(df))
    print("\nColumns:", df.columns.tolist())
    
    # Missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Complaints by product
    if "Product" in df.columns:
        print("\nComplaints per Product:")
        print(df["Product"].value_counts())
    
    # Narrative length statistics
    if "clean_narrative" in df.columns:
        df["narrative_length"] = df["clean_narrative"].apply(lambda x: len(str(x).split()))
        print("\nNarrative length statistics (words):")
        print(df["narrative_length"].describe())
    
    print("\n===== End of Report =====")

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    # Remove rows without narratives
    df = df.dropna(subset=["clean_narrative"]).reset_index(drop=True)
    
    # Generate the report
    generate_report(df)

if __name__ == "__main__":
    main()
