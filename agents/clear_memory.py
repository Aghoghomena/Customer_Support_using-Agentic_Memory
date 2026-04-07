# clear_memory.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import shutil
from utils.config import CHROMA_DIR, SKILLS_DIR, GUIDELINES_PATH


def clear_memory():
    # Clear ChromaDB
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        os.makedirs(CHROMA_DIR)
        print("Cleared ChromaDB")

    # Clear skill files
    if os.path.exists(SKILLS_DIR):
        for item in os.listdir(SKILLS_DIR):
            item_path = os.path.join(SKILLS_DIR, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print("Cleared skill files")

    # Clear guidelines
    with open(GUIDELINES_PATH, "w") as f:
        f.write("")
    print("Cleared guidelines")

    print("\nMemory cleared — ready for fresh training run")


if __name__ == "__main__":
    clear_memory()