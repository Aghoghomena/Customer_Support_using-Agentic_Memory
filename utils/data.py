# data.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import os
from utils.config import rows_per_category, DATA_PATH



def sample_per_category(path: str, n_per_category) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Remove groups with fewer rows than n to avoid errors
    valid = df.groupby("category").filter(lambda g: len(g) >= n_per_category)

    # .sample() on groupby avoids the column-drop bug in .apply() (pandas 2.x)
    sampled = (
        valid.groupby("category")
             .sample(n=n_per_category)   # no random_state = different every run
             .reset_index(drop=True)
    )
    return sampled


def train_test_split_df(df: pd.DataFrame, train_ratio: float = 0.7, seed: int = 42):
    train = df.sample(frac=train_ratio, random_state=seed)
    test = df.drop(train.index).reset_index(drop=True)
    train = train.reset_index(drop=True)
    return train, test


def save_sample(df: pd.DataFrame, path: str = "memory/sample.parquet"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

     # Show each row's full content before saving
    for i, row in df.iterrows():
        print(f"\n--- Row {i} ---")
        for col, val in row.items():
            print(f"  {col}: {val}")
    
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} rows to {path}")


def load_stored_sample(path: str = "memory/sample.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)

if __name__ == "__main__":
    STORE_PATH = "memory/sample.parquet"
    df = pd.read_csv("data/customer-dataset.csv")  # only load 100 rows
    print(df.columns.tolist())           # see all column names
    print(df["category"].unique())       # see all category values
    print(df["category"].nunique())      # how many categories

    # Inspect first
    print("Categories:", pd.read_csv(DATA_PATH)["category"].value_counts().to_string())

    df = sample_per_category(DATA_PATH, rows_per_category)
    print(f"Total sampled: {len(df)} rows across {df['category'].nunique()} categories")
    print(df[["category", "intent"]].to_string())
    save_sample(df, STORE_PATH)   # ← add this

    train, test = train_test_split_df(df)
    print(f"\nTrain: {len(train)} rows")
    print(f"Test: {len(test)} rows")
    print(f"\nTrain categories:\n{train['category'].value_counts().to_string()}")
    print(f"\nTest categories:\n{test['category'].value_counts().to_string()}")
