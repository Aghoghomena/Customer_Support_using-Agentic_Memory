# run_training.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import mygraph
from utils.data import load_stored_sample
from utils.config import TRAINING_MODE


def run_training():
    #df = load_stored_sample()
    df = load_stored_sample().sample(n=1)
    print(f"Running on {len(df)} examples | training_mode={TRAINING_MODE}\n")

    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] Category: {row['category']} | Intent: {row['intent']}")
        print(f"  Query: {row['instruction'][:80]}...")


        state = {
            "query": row["instruction"],
            "category": row["category"],
            "intent": row["intent"],
            "model_answer": row["response"] if TRAINING_MODE else None,
            "skills_context": None,
            "response": None,
            "used_model_answer": False,
            "training_mode": TRAINING_MODE,
            "confident": False,
            "next": "",              # ← add this
            "agent_response": None,
            "extracted_skill": None,
            "mode": "retrieve",
        }
        result = mygraph.invoke(state)

        print(f"  Agent response: {result['response'][:120]}...")
        print(f"  Used model answer: {result['used_model_answer']}\n")

if __name__ == "__main__":
    run_training()