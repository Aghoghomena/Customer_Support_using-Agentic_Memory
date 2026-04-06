# Agentic Memory — Customer Support Agent

A LangGraph-based multi-agent system that learns from customer support interactions over time using agentic memory. The agent improves with each training run by extracting reusable skills and guidelines from examples.

---

## What It Does

The system processes customer support queries through a pipeline of specialised agents:

```
Supervisor → Skill Retrieval → Service Agent → Skill Ingestion → Guideline Update
```

1. **Supervisor** — orchestrates the flow, decides which agent to call next based on current state
2. **Skill Retrieval** — fetches relevant past skills from memory before answering
3. **Service Agent** — answers the customer query; first attempts alone, falls back to model answer if not confident
4. **Skill Ingestion** — extracts reusable skills from the interaction and saves them to memory
5. **Guideline Agent** — reviews interactions and updates `memory/guidelines.txt` with improved rules

### Memory Approaches

The system supports two approaches for skill storage (configured via `SKILL_APPROACH` in `config.py`):

| Approach | Storage | Agent |
|----------|---------|-------|
| 1 | ChromaDB (vector search) | `skill_agent.py` |
| 2 | Markdown files (file-based) | `skill_file_agent.py` |

---

## Project Structure

```
agentic-memory/
├── agents/
│   ├── agent.py               # Main supervisor graph — entry point
│   ├── service_agent.py       # Answers customer queries (with fallback to model answer)
│   ├── skill_agent.py         # ChromaDB skill retrieval and ingestion (approach 1)
│   ├── skill_file_agent.py    # File-based skill retrieval and ingestion (approach 2)
│   ├── skill_function.py      # ChromaDB read/write functions
│   ├── skills_file_functions.py # Markdown skill file read/write functions
│   ├── guideline_agent.py     # Updates guidelines.txt from interactions
│   └── training.py            # Training loop entry point
├── utils/
│   ├── config.py              # LLM setup, paths, flags (TRAINING_MODE, SKILL_APPROACH)
│   ├── data.py                # CSV loading and parquet sample storage
│   ├── lms.py                 # LLM factory (Groq, DeepSeek, LM Studio, Ollama, etc.)
│   └── myclasses.py           # AgentState TypedDict
├── memory/
│   ├── chroma_db/             # Persistent ChromaDB vector store (approach 1)
│   ├── skills/                # Markdown skill files (approach 2)
│   └── guidelines.txt         # Accumulated guidelines updated during training
├── data/
│   └── customer-dataset.csv   # 26k+ labelled customer support examples
└── pyproject.toml
```

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

Copy the `.env` template and fill in your keys:

```bash
cp .env .env.local  # or edit .env directly
```

Required keys in `.env`:

```env
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=your_project_name

GROQ_API_KEY=your_key
DEEPSEEK_API_KEY=your_key       # optional
MISTRAL_API_KEY=your_key        # optional
```

### 3. Configure the agent

In `utils/config.py`:

```python
TRAINING_MODE = True       # True = use model answers as training signal
SKILL_APPROACH = 1         # 1 = ChromaDB, 2 = file-based markdown
TOP_K_SKILLS = 3           # how many skills to retrieve per query
```

---

## Running

### Prepare training data (run once)

Samples 2 rows per category from the CSV and saves to `memory/sample.parquet`:

```bash
uv run python utils/data.py
```

### Run a training session

Picks random examples from the sample, runs the full agent pipeline:

```bash
uv run python agents/training.py
```

This will print each step the supervisor takes:

```
[1/1] Category: ORDER | Intent: track_refund
  Query: I want to know the status of my refund...

[Supervisor] No skills yet — retrieve first
  [Skill Agent] No relevant skills found.
[Supervisor] No response yet — send to service agent
  [Service Agent] First attempt...
  [Service Agent] Not confident — consulting model answer
  [Service Agent] Answered: Here is how to track your refund...
[Supervisor] Agent needed help — delegating ingestion to skill agent
  [Skill Agent] Extracted skill: ...
  [ChromaDB] Skill ingested
[Supervisor] Delegate guideline development
  [Guideline Agent] Guidelines updated
```

### Inspect stored skills (ChromaDB approach)

```bash
uv run python -c "from agents.skill_function import list_all_skills; list_all_skills()"
```

### View the agent graph

The graph image is saved automatically on startup:

```
supervisor_graph.png
```

---

## How Training Works

1. A customer query + ground truth `response` from the CSV is loaded
2. The supervisor checks memory for relevant past skills
3. The service agent tries to answer from its own knowledge and skills
4. If not confident (`NEED_MORE_INFO`), the model answer is injected as a reference
5. After answering, if the model answer was used, a skill is extracted and saved
6. The guideline agent reviews the interaction and may update `guidelines.txt`
7. On the next run, retrieved skills and guidelines are injected into the prompt — the agent improves

> The LLM weights are not updated. Learning happens through the memory files (skills + guidelines) that accumulate across runs.
