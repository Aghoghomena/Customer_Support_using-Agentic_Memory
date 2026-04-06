# config.py
from dotenv import load_dotenv

load_dotenv()

from utils.lms import get_model
llm = get_model("deepseek", temperature=0.7)


# Paths
DATA_PATH = "data/customer_support.csv"
SKILLS_DIR = "memory/skills"          # approach 2 — file-based
GUIDELINES_PATH = "memory/guidelines.txt"
CHROMA_DIR = "memory/chroma_db"       # approach 1 — vector store

# Training flag — flip this to False for inference/testing
TRAINING_MODE = True

# How many top skills to retrieve (approach 1)
TOP_K_SKILLS = 3

SKILL_APPROACH = 2    # 1 = ChromaDB vector, 2 = file-based list

MAX_GUIDELINES = 5