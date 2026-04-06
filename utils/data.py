# data.py
import pandas as pd
import os



def sample_per_category(path: str, n_per_category: int = 2) -> pd.DataFrame:
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
    DATA_PATH = "data/customer-dataset.csv"
    STORE_PATH = "memory/sample.parquet"
    df = pd.read_csv("data/customer-dataset.csv")  # only load 100 rows
    print(df.columns.tolist())           # see all column names
    print(df["category"].unique())       # see all category values
    print(df["category"].nunique())      # how many categories

    # Inspect first
    print("Categories:", pd.read_csv(DATA_PATH)["category"].value_counts().to_string())

    df = sample_per_category(DATA_PATH, n_per_category=2)
    print(df[["category", "intent"]].to_string())

    save_sample(df, STORE_PATH)

    loaded = load_stored_sample(STORE_PATH)
    print(f"\nLoaded back: {len(loaded)} rows, {loaded['category'].nunique()} categories")


