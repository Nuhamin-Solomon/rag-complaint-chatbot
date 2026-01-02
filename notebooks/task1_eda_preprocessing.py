import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Load dataset
df = pd.read_csv("data/raw/complaints.csv")
print("Dataset shape:", df.shape)

# Product distribution
product_counts = df['Product'].value_counts()
plt.figure(figsize=(10,5))
sns.barplot(x=product_counts.index, y=product_counts.values)
plt.xticks(rotation=45)
plt.title("Complaint Distribution by Product")
plt.tight_layout()
plt.show()

# Narrative length
df['narrative_length'] = df['Consumer complaint narrative'].astype(str).apply(lambda x: len(x.split()))
plt.figure(figsize=(8,4))
df['narrative_length'].hist(bins=50)
plt.title("Complaint Narrative Length Distribution")
plt.show()

# Filter products
products = ["Credit card", "Personal loan", "Savings account", "Money transfer"]
df = df[df['Product'].isin(products)]

# Remove empty narratives
df = df[df['Consumer complaint narrative'].notna()]

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df['cleaned_narrative'] = df['Consumer complaint narrative'].apply(clean_text)

# Save cleaned data
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/filtered_complaints.csv", index=False)
print("Filtered dataset saved at data/processed/filtered_complaints.csv")
