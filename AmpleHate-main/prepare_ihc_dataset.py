import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset (STG2 already contains labels)
df = pd.read_csv(
    "data/implicit-hate-corpus-nov-2021/implicit-hate-corpus/implicit_hate_v1_stg2_posts.tsv",
    sep="\t"
)

# Keep required columns
df = df[["post", "implicit_class"]]

# Rename columns to match expected format
df.columns = ["text", "label"]

# Split dataset
train, temp = train_test_split(df, test_size=0.2, random_state=42)
dev, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save files
train.to_csv("data/ihc/train.tsv", sep="\t", index=False)
dev.to_csv("data/ihc/dev.tsv", sep="\t", index=False)
test.to_csv("data/ihc/test.tsv", sep="\t", index=False)

print("Dataset conversion complete!")
