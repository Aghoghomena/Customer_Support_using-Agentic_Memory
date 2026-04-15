# Agentic Memory — Customer Support Agent

> A multi-agent AI system that gets smarter with every conversation — without retraining the model. Built with LangGraph, LangSmith, ChromaDB, Python.

---

## What It Does

This project is an AI-powered customer support agent that learns from experience over time. Every time it handles a support query it can't fully answer on its own, it extracts a reusable "skill" from the correct answer and stores it — so the next time a similar question comes in, it already knows what to do. Unlike typical AI systems that stay static after deployment, this one improves continuously through accumulated memory across sessions.

---

## Why It's Interesting

Most AI chatbots are frozen — they give the same quality of answer on day one as they do on day one hundred. This project solves that by separating *knowledge* from *model weights*. The LLM itself never changes, but its behaviour improves because it gains access to a growing library of skills and guidelines written from real interactions.

This is meaningful for any business running repetitive customer support at scale: instead of prompt-engineering a static system, you get a system that teaches itself what good answers look like, and bakes those lessons into its future behaviour.

---

## If You're New to This — Key Concepts

You don't need a technical background to understand what makes this project interesting. Here are the four ideas it builds on.

**Large Language Model (LLM)**
An LLM is the AI "brain" — a model like GPT or DeepSeek that has read vast amounts of text and can generate human-like responses. Think of it as a very well-read new hire who joined the company with broad general knowledge but no experience with *your* specific customers, policies, or common issues.

**The problem: standard AI doesn't learn on the job**
Once an LLM is trained, its knowledge is fixed. It can't remember yesterday's conversation, pick up on patterns from past tickets, or get better at your specific workflows over time. Every interaction starts from scratch. This is fine for general tasks, but limiting for specialised, repetitive work like customer support — where the most valuable knowledge comes from *experience*, not textbooks.

**Agentic memory: a workaround that actually works**
Instead of retraining the model (which is expensive and slow), agentic memory stores what the AI learns *outside* the model — in plain files. After each interaction, the system writes down what worked: the right tone, the right steps, the key information to include. The next time a similar question comes in, those notes are handed to the AI before it answers. The model didn't change — but it now has better notes to work from.

> Think of it like a new hire who reads the team's best-practices wiki before every call. The wiki gets updated after each tricky case. Over time, responses improve — not because anyone got smarter, but because the knowledge is written down and shared.

**Multi-agent system**
Rather than one AI trying to do everything, this project splits the work across several specialised agents — one that looks up relevant past solutions, one that answers the query, one that extracts new skills from each interaction, and one that updates the team's rules. A supervisor coordinates them all. This mirrors how a well-run support team works: different people handle different parts of the process, and a team lead decides who does what next.

---

## Architecture

```
                    ┌──────────────────────────────────────────────────────┐
                    │                   TRAINING LOOP                      │
                    │        (customer-support-train dataset)              │
                    └───────────────────────┬──────────────────────────────┘
                                            │
                                            ▼
                               ┌────────────────────┐
                               │     Supervisor      │  ← orchestrates all agents
                               └────────┬───────────┘
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
         ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐
         │  Skill Retrieval │  │  Service Agent  │  │  Skill Ingestion │
         │  (ChromaDB or   │  │  (answers query │  │  (extracts &     │
         │   Markdown)     │  │   + fallback)   │  │   stores skill)  │
         └────────┬────────┘  └────────┬────────┘  └────────┬─────────┘
                  │                    │                     │
                  └────────────────────┼─────────────────────┘
                                       ▼
                              ┌─────────────────┐
                              │ Guideline Agent │
                              │ (updates rules  │
                              │  in guidelines) │
                              └────────┬────────┘
                                       │
                          ┌────────────▼─────────────┐
                          │         MEMORY            │
                          │  ┌────────────────────┐  │
                          │  │ skills/ (Markdown) │  │
                          │  │ chroma_db/ (vector)│  │
                          │  │ guidelines.txt     │  │
                          │  └────────────────────┘  │
                          └───────────────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   Evaluator     │  ← LLM-as-judge scoring
                              │  (LangSmith)    │     0–1 across 5 dimensions
                              └─────────────────┘
```

**How the loop works:**

1. The **Supervisor** checks state and routes to the right agent
2. **Skill Retrieval** searches memory for relevant past skills before answering
3. The **Service Agent** attempts to answer; if not confident, falls back to the ground-truth model answer
4. If the fallback was used, **Skill Ingestion** extracts a reusable skill and saves it to memory
5. The **Guideline Agent** reviews the interaction and updates `guidelines.txt` with improved rules
6. On the next run, injected skills + guidelines make the agent meaningfully better

> The LLM weights never change. All learning is stored in memory files that persist across runs.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) — stateful multi-agent graph |
| LLM providers | DeepSeek (default), Groq, Mistral, LM Studio, Ollama |
| Vector memory (approach 1) | [ChromaDB](https://www.trychroma.com/) — persistent local vector store |
| File memory (approach 2) | Markdown skill files — human-readable, git-friendly |
| Observability & datasets | [LangSmith](https://smith.langchain.com/) — tracing, datasets, experiments |
| Evaluation | LLM-as-judge (5 dimensions: approach, tone, content, completeness, fabrication) |
| Data | 26k+ labelled customer support examples (CSV → Parquet) |
| Package manager | [uv](https://github.com/astral-sh/uv) |
| Language | Python 3.11 |

---

## What I Learned / Challenges

**Memory without fine-tuning is viable — but tricky to get right.**
The core insight was that you can improve agent behaviour without touching model weights at all. Skills and guidelines injected into the prompt at inference time act as a lightweight, interpretable "memory layer." The challenge was figuring out *when* to ingest a skill (only when the agent genuinely needed help, not for every example) to avoid noisy or redundant entries.

**State routing in LangGraph requires careful design.**
Building the supervisor's conditional routing logic — deciding whether to retrieve, answer, ingest, or update guidelines — forced me to think clearly about agent state at each step. Getting the ordering right (retrieve before answer, ingest only after fallback) was non-trivial and easy to break silently.

**Two memory backends surface a real trade-off.**
ChromaDB gives fast semantic search at scale; Markdown files are human-readable and easy to debug. Implementing both (switchable via a single config flag) gave me direct experience with the practical difference: vector search found more semantically related skills, but file-based memory was easier to inspect and iterate on during development.

**Evaluation is harder than building.**
Writing a strict LLM-as-judge that scores harshly (and consistently) took several iterations. The biggest risk was a judge that was too lenient, masking whether training was actually helping. Scoring across five explicit dimensions — rather than a single quality score — gave much more actionable signal.

---

## Demo

> Coming soon — a short screen recording of a full training session and before/after LangSmith evaluation scores.

---

## Running It

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

```bash
cp .env .env.local   # or edit .env directly
```

Required keys:

```env
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=your_project_name

GROQ_API_KEY=your_key
DEEPSEEK_API_KEY=your_key    # default model
MISTRAL_API_KEY=your_key     # optional
```

### 3. Configure the agent

In `utils/config.py`:

```python
TRAINING_MODE = True     # True = use model answers as training signal
SKILL_APPROACH = 2       # 1 = ChromaDB, 2 = Markdown files (default)
TOP_K_SKILLS = 3         # skills retrieved per query
MAX_GUIDELINES = 7       # max rules kept in guidelines.txt
rows_per_category = 4    # rows sampled per category for the parquet sample
```

### 4. Full workflow

```bash
# Step 1 — build the parquet sample from the CSV (run once)
uv run python utils/data.py

# Step 2 — push train/eval datasets to LangSmith (run once)
uv run python agents/create_datasets.py

# Step 3 — baseline evaluation (before any training)
uv run python agents/evaluator.py

# Step 4 — run training sessions (repeat as many times as you like)
uv run python agents/training.py

# Step 5 — re-evaluate to measure improvement
uv run python agents/evaluator.py

# Reset memory to start a new experiment
uv run python agents/clear_memory.py
```

### Training output (example)

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
[Supervisor] Delegate guideline development
  [Guideline Agent] Guidelines updated
```

### Inspect memory

```bash
# View stored skills (ChromaDB approach)
uv run python -c "from agents.skill_function import list_all_skills; list_all_skills()"

# View the agent graph (saved on startup)
open supervisor_graph.png
```

---

## Project Structure

```
agentic-memory/
├── agents/
│   ├── agent.py               # Supervisor graph — main entry point
│   ├── service_agent.py       # Answers queries (with fallback to model answer)
│   ├── skill_agent.py         # ChromaDB skill retrieval + ingestion (approach 1)
│   ├── skill_file_agent.py    # Markdown skill retrieval + ingestion (approach 2)
│   ├── skill_function.py      # ChromaDB read/write helpers
│   ├── skills_file_functions.py  # Markdown skill file read/write helpers
│   ├── guideline_agent.py     # Updates guidelines.txt from interactions
│   ├── training.py            # Training loop — local parquet or LangSmith dataset
│   ├── create_datasets.py     # Creates train/eval datasets in LangSmith
│   ├── evaluator.py           # LLM-as-judge evaluation via LangSmith
│   └── clear_memory.py        # Wipes ChromaDB, skill files, and guidelines
├── utils/
│   ├── config.py              # LLM setup, paths, flags
│   ├── data.py                # CSV loading, parquet sample, train/test split
│   ├── lms.py                 # LLM factory (Groq, DeepSeek, LM Studio, Ollama)
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

Built by Aghogho Joy Olokpa — connect with me on [LinkedIn](https://linkedin.com).
