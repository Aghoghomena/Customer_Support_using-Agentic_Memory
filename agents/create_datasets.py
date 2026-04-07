# evaluation/create_dataset.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
from langsmith import Client
from utils.data import load_stored_sample, train_test_split_df
from utils.config import TRAIN_DATASET_NAME, EVAL_DATASET_NAME


def get_or_recreate_dataset(client: Client, name: str, description: str):
    datasets = list(client.list_datasets(dataset_name=name))
    if datasets:
        print(f"Dataset '{name}' already exists — deleting and recreating")
        client.delete_dataset(dataset_id=datasets[0].id)
    return client.create_dataset(dataset_name=name, description=description)

def create_dataset():
    client = Client()
    # Load the already saved sample and split it
    df = load_stored_sample("memory/sample.parquet")
    print(f"Loaded {len(df)} rows from parquet")
    train, test = train_test_split_df(df, train_ratio=0.7)

    print(f"Total: {len(df)} | Train: {len(train)} | Test: {len(test)}")

    # Create train dataset
    print(f"\nCreating train dataset '{TRAIN_DATASET_NAME}'")
    train_dataset = get_or_recreate_dataset(
        client,
        TRAIN_DATASET_NAME,
        "Customer support training set",
    )
    for _, row in train.iterrows():
        client.create_example(
            inputs={
                "query": row["instruction"],
                "category": row["category"],
                "intent": row.get("intent", "unknown"),
            },
            outputs={"model_answer": row["response"]},
            dataset_id=train_dataset.id,
        )
    print(f"Train dataset created with {len(train)} examples | ID: {train_dataset.id}")

    # Create eval dataset
    print(f"\nCreating eval dataset '{EVAL_DATASET_NAME}'")
    eval_dataset = get_or_recreate_dataset(
        client,
        EVAL_DATASET_NAME,
        "Customer support eval set — never seen during training",
    )
    for _, row in test.iterrows():
        client.create_example(
            inputs={
                "query": row["instruction"],
                "category": row["category"],
                "intent": row.get("intent", "unknown"),
            },
            outputs={"model_answer": row["response"]},
            dataset_id=eval_dataset.id,
        )
    print(f"Eval dataset created with {len(test)} examples | ID: {eval_dataset.id}")

if __name__ == "__main__":
    load_dotenv()
    #print(os.getenv("LANGCHAIN_API_KEY"))
    create_dataset()


