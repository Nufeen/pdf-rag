# pdf-rag (🌵 a.k.a `pedro`)

Local RAG (Retrieval-Augmented Generation) CLI for searching a folder of PDF books or papers.

Key ideas:

- Base scenario is fully local usage — no cloud services. Ollama for embeddings and LLM inference, ChromaDB for vector storage.
- Keeping things as simple as it can be

## Getting Started

**How it works:**

```
your PDFs → pedro index → vector DB → pedro ask → answer
```

1. You have a folder of PDF books
2. `pedro index` reads them, splits into chunks, embeds each chunk, stores in a local vector DB
3. `pedro ask` embeds your question, finds the most relevant chunks, sends them to a local LLM, streams the answer back

Nothing leaves your machine.

After playing with parameters and models, when you get `pedro ask` to work quickly
you can start using `pedro research` for multi-step reasoning.

**Quickstart (in ideal world with python 3.13 and Ollama running on local network):**

```bash
# 1. Clone and install
git clone https://github.com/Nufeen/pdf-rag.git && cd pdf-rag
uv venv && source .venv/bin/activate && uv pip install -e .

# 2. Point to your Ollama host and configure
cp .env.example .env
# edit .env — set OLLAMA_BASE_URL=http://<your-ollama-host>:11434

# 3. Index your books
pedro index ~/Books/

# 4. Ask a question
pedro ask "What is backpropagation?"

# 5. Deep research (multi-step reasoning)
pedro research "Compare symbolic and connectionist approaches to AI"
```

That's it. Re-run `pedro index ~/Books/` whenever you add new PDFs — only new files are processed.

## Stack

| Component      | Choice                              | Reason                                                   |
| -------------- | ----------------------------------- | -------------------------------------------------------- |
| PDF extraction | PyMuPDF (`fitz`)                    | Fast, page-level metadata, handles most encodings        |
| Chunking       | Custom recursive splitter           | Split on `\n\n` → `\n` → `.` → ` ` to preserve semantics |
| Embeddings     | `nomic-embed-text` via Ollama       | See model recomendations below                           |
| Vector DB      | ChromaDB (embedded/persistent)      | No server, persists to disk, metadata filtering built-in |
| LLM            | any via Ollama                      | See recomendations below                                 |
| Ollama host    | Remote (local network)              | Set via `OLLAMA_BASE_URL=http://<host-ip>:11434`         |
| CLI            | Click + Textual (adr 1 for details) | https://click.palletsprojects.com/en/stable/             |

## Testing

See [tests/README.md](tests/README.md) for setup and how to run tests.

## Project Structure

```
pdf-rag/
├── pdf_rag/
│   ├── cli.py              # Click entry point — all commands
│   ├── config.py           # Constants + env var overrides
│   ├── researcher.py       # Pipeline logic: run_ask(), research()
│   ├── chunker.py          # Recursive text splitter
│   ├── indexer.py          # PDF extraction, chunking, embedding, ChromaDB writes
│   ├── retriever.py        # Query embedding + ChromaDB search
│   ├── llm.py              # Prompt loading + Ollama streaming chat
│   ├── session_log.py      # JSONL session logging
│   ├── tui/                # Textual TUI app
│   │   ├── app.py          # PedroApp — layout, bindings, worker dispatch
│   │   └── stream_client.py # SSE client for server mode
│   └── server/             # FastAPI HTTP server
│       ├── app.py          # make_app() factory, /v1/* endpoints
│       └── README.md       # Server docs, SSE format, OpenAI usage
├── prompts/                # Prompt templates (editable, no reinstall needed)
├── tests/                  # pytest suite (30 tests)
├── adr/                    # Architecture Decision Records
└── pyproject.toml
```

ChromaDB is stored at `~/.pdf-rag/chroma_db` by default (overridable via `--db-path` or `DB_PATH` env var).

## Prerequisites

On the machine running Ollama:

```bash
# Allow network access (add to ~/.bashrc or systemd service)
export OLLAMA_HOST=0.0.0.0

ollama pull mxbai-embed-large
ollama pull command-r:35b    # or see model recommendations above
```

On machine with project running, point to the remote Ollama host:

```bash
export OLLAMA_BASE_URL=http://192.168.1.X:11434   # replace with actual IP
```

## Local Model Recommendations (actual for apr 2026)

### Embedding model

| Model               | Dims | Notes                                                 |
| ------------------- | ---- | ----------------------------------------------------- |
| `mxbai-embed-large` | 1024 | **Recommended** — better retrieval quality than nomic |
| `nomic-embed-text`  | 768  | Default, fast, good baseline                          |
| `bge-m3`            | 1024 | Best quality, multilingual, slightly slower           |

### LLM

| Model           | Size    | Context | Notes                                                                      |
| --------------- | ------- | ------- | -------------------------------------------------------------------------- |
| `command-r:35b` | 35B     | 128k    | **Recommended** — fine-tuned specifically for RAG, native citation support |
| `mixtral:8x7b`  | 47B MoE | 32k     | Excellent quality, fast due to MoE architecture                            |
| `llama3.1:70b`  | 70B     | 128k    | Best reasoning if VRAM allows                                              |
| `qwen2.5:32b`   | 32B     | 128k    | Strong choice for non-English books                                        |
| `mistral:7b`    | 7B      | 8k      | Minimum viable, good for low-memory hosts                                  |

`command-r:35b` is the best fit for RAG specifically — it's trained to ground answers in retrieved context
and produce accurate citations rather than hallucinate beyond the provided excerpts.

## macOS Setup

The recommended way to manage Python on macOS is [uv](https://docs.astral.sh/uv/) — it handles Python installation, virtual environments, and dependencies in one tool.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.13 and pin it for this project
uv python install 3.13
uv python pin 3.13

# Create venv and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .
```

Then proceed with the Installation steps below.

> [!NOTE]
> Do not forget to allow local network usage for terminal sessions in mac os!
> If you get "No route to host" error with local network ollama probably thats it

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

**Copy and edit the env file:**

```bash
cp .env.example .env
# edit .env — set OLLAMA_BASE_URL to your Ollama host IP
```

## Indexing

Scan a folder and index all PDFs:

```bash
pedro index ~/Books/
```

Output:

```
Embedding: deep_learning.pdf (1847 chunks)...
Indexed: deep_learning.pdf (1847 chunks)
Embedding: pattern_recognition.pdf (2103 chunks)...
Indexed: pattern_recognition.pdf (2103 chunks)
```

Indexing is incremental — each file's SHA-256 hash is stored. Re-running the command only processes new or changed files:

```bash
pedro index ~/Books/
# Skipping (already indexed): deep_learning.pdf
# Skipping (already indexed): pattern_recognition.pdf
```

## Adding New Books

Drop the new PDF into the folder and re-run the index command:

```bash
cp new_book.pdf ~/Books/
pedro index ~/Books/
# Skipping (already indexed): deep_learning.pdf
# Skipping (already indexed): pattern_recognition.pdf
# Embedding: new_book.pdf (1523 chunks)...
# Indexed: new_book.pdf (1523 chunks)
```

Only the new file is processed. Existing books are skipped instantly.

> [!NOTE]
> Take care of library size, ChromaDB has RAM limitations
> If index size will be greater than machine RAM it can trigger errors:
> https://github.com/chroma-core/chroma/issues/1323
> For now it will work only with reasonable local library sizes

## Querying

```bash
pedro ask "What is the vanishing gradient problem?"
```

Output:

```
Retrieved sources:
  - deep_learning.pdf (page 289, score: 0.912)
  - deep_learning.pdf (page 291, score: 0.887)
  - pattern_recognition.pdf (page 144, score: 0.743)

Answer:

The vanishing gradient problem occurs when... [Book: deep_learning.pdf, Page: 289]
Residual connections solve this by... [Book: deep_learning.pdf, Page: 291]
```

Hide the source list:

```bash
pedro ask "Explain attention mechanisms" --no-sources
```

Retrieve more context chunks:

```bash
pedro ask "Compare LSTM and GRU" --top-k 8
```

Override model or embedding per-run:

```bash
pedro ask "What is entropy?" --model llama3.1:70b
pedro ask "What is entropy?" --embed-model bge-m3 --top-k 8
```

Available flags: `--model`, `--embed-model`, `--ollama-url`, `--top-k`, `--no-sources`, `--db-path`

## 🧠 Deep Research

For complex questions that need multi-angle reasoning, use `pedro research`. It decomposes the question into sub-questions, answers each via RAG, then synthesizes and iteratively refines the result.

```bash
pedro research "What are the fundamental differences between symbolic and connectionist AI?"
```

Output:

```
🪅 Planning 3 sub-questions...
  1. What is symbolic AI and what are its core assumptions?
  2. What is connectionist AI and how do neural networks differ from symbolic systems?
  3. What are the practical tradeoffs between the two approaches?

🪅 Executing 3 sub-question(s)...
  [1/3] What is symbolic AI...
  [2/3] What is connectionist AI...
  [3/3] What are the practical tradeoffs...

🪅 Reflecting (pass 1/1)...
  → 1 follow-up sub-question(s) identified

🪅 Synthesizing final answer...

The fundamental differences between symbolic and connectionist AI...
```

**Pipeline steps:**

| Step | What happens | Model |
| ---- | ------------ | ----- |
| Plan | Decomposes the question into N focused sub-questions | `FAST_MODEL` |
| Execute | For each sub-question: retrieves chunks from vector DB, generates a partial answer | `FAST_MODEL` |
| Reflect | Evaluates completeness; identifies gaps or follow-up questions. Repeats Execute if needed, up to `--depth` passes | `FAST_MODEL` |
| Synthesize | Combines all findings into a final answer with citations | `LLM_MODEL` |
| Sources | Lists PDF files and page numbers from all retrieved chunks | — |

Control depth and breadth:

```bash
pedro research "Explain attention mechanisms" --depth 1   # single pass, no reflection
pedro research "Compare LSTM, GRU and Transformer" --depth 3 --sub-questions 5
```

Override models per-run:

```bash
pedro research "..." --model llama3.1:70b --fast-model mistral:7b
```

Available flags: `--model`, `--fast-model`, `--embed-model`, `--ollama-url`, `--depth`, `--sub-questions`, `--top-k`, `--languages`, `--translate-model`, `--db-path`

Configure via `.env`:

```
RESEARCH_DEPTH=2
RESEARCH_N_SUBQUESTIONS=3
```

[How it works](adr/0000-deep-research-mode.md)

## Multilingual Libraries

If your PDF collection contains books in multiple languages, retrieval quality drops when the query language differs from the document language. There are two ways to handle this.

### Option 1 — Multilingual embedding model (recommended)

Switch to an embedding model that maps all languages into the same vector space. No translation step, no extra latency per query — but requires a full re-index.

```bash
# Pull a multilingual embedding model
ollama pull bge-m3   # already listed in the embedding table above

# Set it in .env
EMBED_MODEL=bge-m3

# Re-index everything
pedro index ~/Books/ --force
```

`bge-m3` handles 100+ languages and is the cleanest long-term solution.

### Option 2 — Query translation at search time

If you want to keep the existing index, enable query translation. For each sub-question in `pedro research`, pedro will translate the query into each configured language and merge the results before generating an answer.

```bash
# In .env
SEARCH_LANGUAGES=Russian,French
TRANSLATE_MODEL=qwen2.5:3b   # any small model works; defaults to FAST_MODEL
```

Or pass per-run via CLI:

```bash
pedro research "What is entropy?" --languages Russian,French
pedro research "What is entropy?" --languages Russian --translate-model qwen2.5:3b
```

During research, translated queries are shown inline in the log:

```
🪅 Executing 3 sub-question(s)...
  [1/3] What is entropy?
  (→ Russian: Что такое энтропия?)
  (→ French: Qu'est-ce que l'entropie ?)
```

**Notes:**

- Translation only runs in `pedro research`, not in `pedro ask`
- Each language adds one extra embedding + retrieval call per sub-question
- The translation model needs to be pulled: `ollama pull qwen2.5:3b`
- Chunks retrieved across languages are deduplicated before answer generation

### Which option to choose

|                           | Multilingual embeddings | Query translation                        |
| ------------------------- | ----------------------- | ---------------------------------------- |
| Re-index required         | Yes                     | No                                       |
| Extra latency per query   | None                    | 1 LLM call × N languages per subquestion |
| Works in `pedro ask`      | Yes                     | No                                       |
| Works in `pedro research` | Yes                     | Yes                                      |

If you are starting fresh or can afford a re-index, use `bge-m3`. If you have an existing index and want to extend coverage without re-indexing, use `SEARCH_LANGUAGES`.

## PDF Export

In TUI mode, type `/pdf` and press Enter to export the last answer to a PDF file.

```
> /pdf
✓ Saved to /Users/you/.pedro/exports/pedro_20260406_143022.pdf
```

The output directory is configurable via env var:

```bash
export PEDRO_PDF_PATH=~/Documents/pedro-exports
```

| Variable          | Default              | Description                  |
|-------------------|----------------------|------------------------------|
| `PEDRO_PDF_PATH`  | `~/.pedro/exports`   | Directory for exported PDFs  |

Files are named `pedro_<timestamp>.pdf` and contain the question and the full answer text. The directory is created automatically if it doesn't exist.

## Server Mode

Pedro can run as an HTTP server with SSE streaming and OpenAI-compatible endpoints.

```bash
pedro serve                        # binds to 127.0.0.1:8000
pedro serve --host 0.0.0.0 --port 9000
```

See **[pdf_rag/server/README.md](pdf_rag/server/README.md)** for endpoints, SSE format, OpenAI usage, and how to connect the TUI to the server.

| Variable | Default | Description |
|----------|---------|-------------|
| `PEDRO_SERVER_URL` | `` (standalone) | If set, TUI connects to this server instead of running locally |

## Re-indexing Everything

Use `--force` to re-index all files, for example after changing chunk size:

```bash
pedro index ~/Books/ --force
```

## Environment Variables

| Variable                  | Default                  | Description                                                                                 |
| ------------------------- | ------------------------ | ------------------------------------------------------------------------------------------- |
| `OLLAMA_BASE_URL`         | `http://localhost:11434` | Ollama host URL                                                                             |
| `DB_PATH`                 | `~/.pdf-rag/chroma_db`   | ChromaDB storage path                                                                       |
| `EMBED_MODEL`             | `nomic-embed-text`       | Ollama embedding model (recommend `mxbai-embed-large`)                                      |
| `LLM_MODEL`               | `mistral:7b`             | Quality model — `ask` and final research synthesis (recommend `command-r:35b`)              |
| `FAST_MODEL`              | `LLM_MODEL`              | Intermediate model — sub-questions, planning, reflection (3B–7B recommended)                |
| `CHUNK_SIZE`              | `800`                    | Characters per chunk                                                                        |
| `CHUNK_OVERLAP`           | `150`                    | Overlap between chunks                                                                      |
| `TOP_K`                   | `5`                      | Chunks retrieved per query                                                                  |
| `RESEARCH_DEPTH`          | `2`                      | Max reflection iterations for `pedro research`                                              |
| `RESEARCH_N_SUBQUESTIONS` | `3`                      | Sub-questions per iteration for `pedro research`                                            |
| `SEARCH_LANGUAGES`        | `` (disabled)            | Comma-separated languages for query translation in `pedro research` (e.g. `Russian,French`) |
| `TRANSLATE_MODEL`         | `FAST_MODEL`             | Model used to translate sub-questions when `SEARCH_LANGUAGES` is set                        |
| `PEDRO_SERVER_URL`        | `` (standalone)          | If set, TUI connects to this server instead of running the pipeline in-process               |
| `PEDRO_PDF_PATH`          | `~/.pedro/exports`       | Directory where `/pdf` TUI command writes exported PDF files                                 |

All variables can also be passed as CLI flags — run `pedro index --help`, `pedro ask --help`, or `pedro research --help` for details.

### Optimal Chunk Size Guidelines (google claim)

| Size   | Tokens   | Best for                                                      |
| ------ | -------- | ------------------------------------------------------------- |
| Small  | 128–256  | Specific, fact-based questions (FAQ, short answer)            |
| Medium | 256–512  | Semantic search, general documentation, RAG chatbot           |
| Large  | 512–1024 | Summarizing, relationships in content, long-document analysis |

## Customizing Prompts

All prompts live in the `prompts/` folder. Edit any file directly — changes take effect on the next command, no reinstall needed.

| File                             | Used by                | Model        | Placeholders              | Purpose                                                                                 |
| -------------------------------- | ---------------------- | ------------ | ------------------------- | --------------------------------------------------------------------------------------- |
| `prompts/answer.txt`             | `pedro ask`            | `LLM_MODEL`  | `{question}`, `{context}` | System prompt for answer generation — controls tone, citation format, grounding rules   |
| `prompts/plan_subquestions.txt`  | `pedro research`       | `FAST_MODEL` | `{question}`, `{n}`       | Instructs the model to decompose the question into N sub-questions                      |
| `prompts/reflect.txt`            | `pedro research`       | `FAST_MODEL` | `{question}`, `{answer}`  | Asks the model to evaluate completeness and identify gaps in the current answer         |
| `prompts/synthesize.txt`         | `pedro research`       | `LLM_MODEL`  | `{question}`, `{context}` | Instructs the model to combine all research findings into a final answer                |
| `prompts/translate_question.txt` | `pedro research`       | `FAST_MODEL` | `{text}`, `{lang}`        | Translates a sub-question into a target language (used when `SEARCH_LANGUAGES` is set)  |

Prompt files support `{placeholders}` filled at runtime. Do not remove placeholders — the tool will fail if they are missing.

## RAG Evaluation

Pedro includes a custom evaluation framework to measure answer quality across different configurations (models, top_k, etc.). See **[ADR-0007](adr/0007-rag-evaluation-framework.md)** for design rationale.

### Setup

```bash
# Copy the example dataset and edit with your Q&A pairs
cp eval/dataset.jsonl.example eval/dataset.jsonl
```

Edit `eval/dataset.jsonl` — one JSON object per line:

```jsonl
{"question": "...", "ground_truth": "...", "tags": ["topic"]}
```

### Run Evaluation

```bash
# Single model, default top_k=5
uv run python -m eval.evaluate

# Full matrix: multiple models and top_k values
uv run python -m eval.evaluate --models mistral:7b,llama3:8b --top-k 3,5,10 --judge mistral:7b
```

**CLI options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--models` | `LLM_MODEL` | Comma-separated model names to evaluate |
| `--top-k` | `TOP_K` | Comma-separated top_k values to test |
| `--judge` | first `--models` entry | Model used for factual scoring |
| `--db-path` | `DB_PATH` | Path to ChromaDB |
| `--ollama-url` | `OLLAMA_BASE_URL` | Ollama host URL |
| `--embed-model` | `EMBED_MODEL` | Embedding model for semantic similarity |
| `--dataset` | `eval/dataset.jsonl` | Path to your Q&A dataset |
| `--output-dir` | `eval/results/` | Where to save CSV results |

### Output

Results are saved to `eval/results/<timestamp>.csv` and a pivot table is printed showing:

| Column | Description |
|--------|-------------|
| `Avg Score` | Weighted average of factual (0.75) and semantic (0.25) scores |
| `Avg Time` | Average time per question (retrieval + generation + scoring) |
| `Q/min` | Questions processed per minute |

## Notes

Tools and projects to look at in the context of the problem:

- https://github.com/zylon-ai/private-gpt

- https://github.com/assafelovic/gpt-researcher

- https://developers.llamaindex.ai/python/examples/

- https://github.com/langchain-ai/langchain

- https://github.com/Mintplex-Labs/anything-llm

I tried some but surprisingly ended with generating own code since in all cases
it turned out to be not that easy to get fully local stack working out of the box
