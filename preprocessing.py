import pandas as pd

columns = ["Score", "Summary", "Review"]

df = pd.read_csv(
    "data/Reviews.csv",
    header=None,
    names=columns,
    engine="python",
    encoding="latin-1"
)

df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

print("Score distribution:\n", df["Score"].value_counts())

df["clean_review"] = df["Review"].fillna("").astype(str).str.lower()

# IMPORTANT FIX ⬇️
df["sentiment"] = (df["Score"] == 2).astype(int)

print("Sentiment distribution:\n", df["sentiment"].value_counts())

df[["clean_review", "sentiment"]].to_csv("data/cleaned_reviews.csv", index=False)
