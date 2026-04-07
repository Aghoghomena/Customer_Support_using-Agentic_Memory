import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from langsmith import Client
from utils.data import load_stored_sample
from langchain_core.messages import SystemMessage, HumanMessage
from agents.agent import mygraph
from utils.config import TRAINING_MODE, llm,EVAL_DATASET_NAME
from langsmith import evaluate



# --- Target function --- 
# This is what LangSmith calls for each example

def run_agent(inputs: dict) -> dict:
    """Runs the agent on a single example and returns its response."""
    state = {
        "query": inputs["query"],
        "category": inputs["category"],
        "intent": inputs["intent"],
        "model_answer": None,       # never provide model answer during eval
        "skills_context": None,
        "response": None,
        "used_model_answer": False,
        "training_mode": False,     # always False during eval
        "extracted_skill": None,
        "guidelines_updated": False,
        "next": "",
    }

    result = mygraph.invoke(state)
    return {"response": result["response"]}


# --- LLM as a Judge ---
def llm_judge(run, example) -> dict:
    """
    Scores the agent response against the model answer.
    Returns a numeric score between 0 and 3.
    """

    agent_response = run.outputs.get("response", "")
    model_answer = example.outputs.get("reference_answer", "")
    query = example.inputs.get("query", "")

    response = llm.invoke([
        SystemMessage(content="""You are an expert and strict customer service quality evaluator.
            You are comparing an AI agent response to a gold standard model answer.
            You must be critical and score harshly most responses should score 1 or 2, not 3.
                      
            You must evaluate ALL of these dimensions:

            1. APPROACH — does the agent follow the same strategy as the model answer?
            (e.g. if model asks for more info first, agent should too)
            2. TONE — is the tone similar? (formal/empathetic/professional)
            3. CONTENT — does it include the same key information?
            4. COMPLETENESS — does it fully address the query?
            5. ASSUMPTIONS — does the agent assume information it does not have, inventing details it doesn't have?
            (e.g. making up URLs, email addresses, phone numbers, portal names, specific policies, or any placeholder details like [support email])

            Strict Scoring:
            0 — Wrong approach, fabricates details, or completely unhelpful
            1 — Partially correct but different approach OR missing key information OR wrong tone
            2 — Same approach and mostly correct but noticeable differences in tone or content
            3 — Near perfect match in approach, tone, content and completeness — rare, very high bar
            
            IMPORTANT:
            - If the model answer asks for clarification first but agent gives steps directly, max score is 1
            - If the model answer gives steps but agent asks for info instead, max score is 1
            - Penalise heavily for wrong approach even if the content seems helpful
                      
            AUTOMATIC score cap at 1 if ANY of these are true:
            - Model asks for info first but agent gives steps directly
            - Agent uses placeholders like [email], [URL], [contact form]
            - Agent invents specific details like phone numbers, emails, URLs
            - Agent assumes access to account or order information it does not have
            - Agent gives generic advice that does not match the model answer strategy
                      
            AUTOMATIC score of 0 if:
            - Agent response is completely off topic
            - Agent refuses to help
            - Agent fabricates specific company details

            Be harsh. A score of 3 should be rare. A score of 3 means the agent response is nearly indistinguishable from the model answer in terms of approach, tone, content, and completeness. Most good responses will score 2 or 1 due to minor differences in tone or missing some details. Always provide a reason for your score in one sentence.

            Return ONLY a JSON object in this exact format:
            {"score": <0-3>, "reason": "<one sentence explanation>"}"""),
                    HumanMessage(content=f"""Customer query: {query}

            Model answer: {model_answer}

            Agent response: {agent_response}

            Score the agent response."""), 
    ])

    try:
        result = json.loads(response.content.strip())
        # Normalise to 0-1
        normalised_score = result["score"] / 3.0
        return {
            "key": "response_quality",
            "score": normalised_score,
            "comment": result.get("reason", ""),
        }
    except Exception as e:
        print(f"  [Evaluator] Failed to parse score: {e}")
        return {
            "key": "response_quality",
            "score": 0.0,
            "comment": "Failed to parse score",
        }
    

# --- Run evaluation ---
def run_evaluation(experiment_prefix: str = "baseline"):
    """
    Runs evaluation on the dataset. 
    experiment_prefix: label for this run e.g. 'baseline', 'after-training-1'
    """
    print(f"\nRunning evaluation: {experiment_prefix}")

    results = evaluate(
        run_agent,
        data=EVAL_DATASET_NAME,
        evaluators=[llm_judge],
        experiment_prefix=experiment_prefix,
        metadata={"training_mode": str(TRAINING_MODE)},
    )

    # Print summary
    scores = [r["evaluation_results"]["results"][0].score
              for r in results._results
              if r["evaluation_results"]["results"]]

    if scores:
        avg = sum(scores) / len(scores)
        print(f"\n--- {experiment_prefix} Results ---")
        print(f"Examples evaluated: {len(scores)}")
        print(f"Average score: {avg:.3f}")
        print(f"Min score: {min(scores):.3f}")
        print(f"Max score: {max(scores):.3f}")

    return results

#--- DataSet creation ---
def create_dataset():
    client = Client()
    df = load_stored_sample()

     # Use a fixed split — last 30% as test set
    #test_df = df.sample(frac=0.3, random_state=42).reset_index(drop=True)
    test_df = df.sample(n=2, random_state=42).reset_index(drop=True)

    print(f"Creating dataset '{EVAL_DATASET_NAME}' with {len(test_df)} examples")

    # Check if dataset already exists
    datasets = list(client.list_datasets(dataset_name=EVAL_DATASET_NAME))
    if datasets:
        print(f"Dataset '{EVAL_DATASET_NAME}' already exists with ID: {datasets[0].id}")
        client.delete_dataset(dataset_id=datasets[0].id)

    
    # Create new dataset
    dataset = client.create_dataset(name=EVAL_DATASET_NAME, description="Evaluation dataset for customer support")
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
    run_evaluation(experiment_prefix="baseline")
