import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.messages import SystemMessage, HumanMessage
from utils.config import CHROMA_DIR, TOP_K_SKILLS
import uuid
import re
from langchain_core.runnables import RunnableLambda



from utils.config import llm

def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"   # small, fast, good enough
    )
    collection = chroma_client.get_or_create_collection(
        name="customer_support_skills",
        embedding_function=embedding_fn,
        metadata={"description": "Skills extracted from conversations about customer support to guide future interactions."},
    )
    return collection

def add_skill(query: str, skill_text: str, category: str, intent: str):
    """
    Stores a skill in ChromaDB.
    Before ingesting, checks if a semantically similar skill already exists
    to avoid storing duplicates.
    """
    collection = get_collection()

    # Only check for duplicates if collection has items
    if collection.count() > 0:
        results = collection.query(
            query_texts=[skill_text],   # search by skill content, not query
            n_results=1,
            include=["metadatas", "distances"],
        )

        if results["distances"] and results["distances"][0]:
            similarity = 1 - results["distances"][0][0]
            if similarity > 0.85:       # very similar skill already exists
                print(f"  [ChromaDB] Similar skill already exists (similarity={similarity:.2f}), skipping.")
                return

    # No duplicate found — generate a unique ID and store
    skill_id = str(uuid.uuid4())

    collection.add(
        ids=[skill_id],
        documents=[skill_text],         # embed the skill text itself
        metadatas=[{
            "skill": skill_text,
            "query": query,
            "category": category,
            "intent": intent,
        }]
    )
    print(f"  [ChromaDB] Skill ingested (id={skill_id})")


def retrieve_skills(query: str, top_k: int = TOP_K_SKILLS) -> str:
    """
    Returns the top-k most relevant skills as a formatted string
    ready to inject into the system prompt.
    """
    collection = get_collection()

    count = collection.count()
    if count == 0:
        return ""

    # Don't request more than what's stored
    k = min(top_k, count)

    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["metadatas", "distances"],
    )

    skills = []
    for i, metadata in enumerate(results["metadatas"][0]):
        distance = results["distances"][0][i]
        similarity = 1 - distance  # cosine distance → similarity
        if similarity > 0.3:       # ignore weak matches
            skills.append(
                f"[Skill {i+1} | {metadata['intent']} | similarity={similarity:.2f}]\n"
                f"{metadata['skill']}"
            )

    if not skills:
        return ""

    return "\n\n".join(skills)


def list_all_skills():
    collection = get_collection()
    results = collection.get(include=["documents", "metadatas"])
    print(f"\nTotal skills in ChromaDB: {collection.count()}")
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        print(f"\n[{i+1}] Intent: {meta['intent']} | Category: {meta['category']}")
        print(f"  Query: {doc[:80]}...")
        print(f"  Skill: {meta['skill'][:120]}...")

if __name__ == "__main__":
    list_all_skills()