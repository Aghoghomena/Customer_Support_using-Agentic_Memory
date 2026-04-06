# evaluation/create_dataset.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
from langsmith import Client
from utils.data import load_stored_sample

DATASET_NAME = "customer-support-eval"

def create_dataset():
    client = Client()
    df = load_stored_sample()

     # Use a fixed split — last 30% as test set
    test_df = df.sample(frac=0.3, random_state=42).reset_index(drop=True)

    print(f"Creating dataset '{DATASET_NAME}' with {len(test_df)} examples")

    # Check if dataset already exists
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if datasets:
        print(f"Dataset '{DATASET_NAME}' already exists with ID: {datasets[0].id}")
        client.delete_dataset(dataset_id=datasets[0].id)

    
    # Create new dataset
    dataset = client.create_dataset(dataset_name=DATASET_NAME, description="Evaluation dataset for customer support")
    # Add examples
    for _, row in test_df.iterrows():
        client.create_example(
            dataset_id=dataset.id,
            inputs={"query": row["instruction"], "category": row["category"], "intent": row["intent"]},
            outputs={"reference_answer": row["response"]},
        )

    print(f"Dataset created with {len(test_df)} examples")
    return dataset

if __name__ == "__main__":
    load_dotenv()
    #print(os.getenv("LANGCHAIN_API_KEY"))
    create_dataset()

