# config.py
from dotenv import load_dotenv

load_dotenv()

from utils.lms import get_model
llm = get_model("deepseek", temperature=0.7)


# Paths
DATA_PATH = "data/customer-dataset.csv"
SKILLS_DIR = "memory/skills"          # approach 2 — file-based
GUIDELINES_PATH = "memory/guidelines.txt"
CHROMA_DIR = "memory/chroma_db"       # approach 1 — vector store

# Training flag — flip this to False for inference/testing
TRAINING_MODE = False

# How many top skills to retrieve (approach 1)
TOP_K_SKILLS = 3

# Which skill ingestion approach to use (1 or 2)
SKILL_APPROACH = 2   # 1 = ChromaDB vector, 2 = file-based list

# For simplicity, we keep guidelines in a text file and limit to 7 key guidelines.
MAX_GUIDELINES = 7

# Number of rows to sample per category during training to speed up development. Set to None to use all.
rows_per_category = 4

#dataset names
TRAIN_DATASET_NAME = "customer-support-train"
EVAL_DATASET_NAME = "customer-support-eval"