import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "./data/implicit-hate-corpus-nov-2021/implicit-hate-corpus/implicit_hate_v1_stg1_posts.tsv",
    sep="\t"
)

# Optional: keep only the labels you want
df = df[df["class"].isin(["not_hate", "implicit_hate"])]

train, temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df["class"])
dev, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["class"])

train.to_csv("data/ihc/train.tsv", sep="\t", index=False)
dev.to_csv("data/ihc/valid.tsv", sep="\t", index=False)
test.to_csv("data/ihc/test.tsv", sep="\t", index=False)