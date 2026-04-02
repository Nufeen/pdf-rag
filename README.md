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
make setup
source .venv/bin/activate

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

| Component      | Choice                                   | Reason                                                   |
| -------------- | ---------------------------------------- | -------------------------------------------------------- |
| PDF extraction | PyMuPDF (`fitz`)                         | Fast, page-level metadata, handles most encodings        |
| Chunking       | Custom recursive splitter                | Split on `\n\n` → `\n` → `.` → ` ` to preserve semantics |
| Embeddings     | `nomic-embed-text` via Ollama            | No extra deps, runs locally, good retrieval quality      |
| Vector DB      | ChromaDB (embedded/persistent)           | No server, persists to disk, metadata filtering built-in |
| LLM            | `mistral:7b` or `llama3.2:3b` via Ollama | Streaming, 8k context, instruction-following             |
| Ollama host    | Remote (local network)                   | Set via `OLLAMA_BASE_URL=http://<host-ip>:11434`         |
| CLI            | Click (command group)                    | Cleaner subcommands than argparse                        |
| Framework      | None (raw components)                    | RAG pipeline is simple; no LlamaIndex/LangChain overhead |

## Project Structure

```
pdf-rag/
├── pdf_rag/
│   ├── __init__.py
│   ├── cli.py          # Click entry point: `index` and `ask` commands
│   ├── config.py       # Constants + env var overrides
│   ├── chunker.py      # Recursive text splitter
│   ├── indexer.py      # PDF extraction, chunking, embedding, ChromaDB writes
│   ├── retriever.py    # Query embedding + ChromaDB search
│   └── llm.py          # Prompt construction + Ollama streaming chat
├── requirements.txt
├── pyproject.toml      # entry_points: `pdf-rag = pdf_rag.cli:cli`
└── README.md           # setup, usage, reindexing workflow, env vars
```

ChromaDB is stored at `~/.pdf-rag/chroma_db` by default (overridable via `--db-path` or `RAG_DB_PATH` env var).

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

macOS ships with an outdated system Python. Install a current version first:

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.13
brew install python@3.13

# Verify
python3.13 --version   # Python 3.13.x
```

Then proceed with `make setup` — it will pick up `python3.13` automatically.

> [!NOTE]
> Do not forget to allow local network usage for terminal sessions in mac os!
> If you get "No route to host" error with local network ollama probably thats it

## Installation

**With Make (recommended):**

```bash
make setup
source .venv/bin/activate
```

`make setup` creates a `.venv`, installs all dependencies, and prints next steps.

**Manually:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Copy and edit the env file:**

```bash
cp .env.example .env
# edit .env — set OLLAMA_BASE_URL to your Ollama host IP
```

Then load it before running commands:

```bash
export $(grep -v '^#' .env | xargs)
```

**Available Make targets:**

| Target         | Description                          |
| -------------- | ------------------------------------ |
| `make setup`   | Create venv and install dependencies |
| `make venv`    | Create venv only                     |
| `make install` | Install into existing venv           |
| `make clean`   | Remove venv and build artifacts      |

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

Control depth and breadth:

```bash
pedro research "Explain attention mechanisms" --depth 1   # single pass, no reflection
pedro research "Compare LSTM, GRU and Transformer" --depth 3 --sub-questions 5
```

Configure via `.env`:

```
RESEARCH_DEPTH=2
RESEARCH_N_SUBQUESTIONS=3
```

[How it works](adr/0000-deep-research-mode.md)

## Re-indexing Everything

Use `--force` to re-index all files, for example after changing chunk size:

```bash
pedro index ~/Books/ --force
```

## Environment Variables

| Variable                  | Default                  | Description                                            |
| ------------------------- | ------------------------ | ------------------------------------------------------ |
| `OLLAMA_BASE_URL`         | `http://localhost:11434` | Ollama host URL                                        |
| `RAG_DB_PATH`             | `~/.pdf-rag/chroma_db`   | ChromaDB storage path                                  |
| `RAG_EMBED_MODEL`         | `nomic-embed-text`       | Ollama embedding model (recommend `mxbai-embed-large`) |
| `RAG_DEEP_MODEL`           | `mistral:7b`             | Quality model — `ask` and final research synthesis (recommend `command-r:35b`) |
| `RAG_FAST_MODEL`          | `RAG_DEEP_MODEL`          | Medium model — per-sub-question answers and intermediate synthesis |
| `RAG_TINY_MODEL`          | `RAG_FAST_MODEL`         | Fast model — planning and reflection (3B recommended, e.g. `qwen2.5:3b`) |
| `RAG_CHUNK_SIZE`          | `800`                    | Characters per chunk                                   |
| `RAG_CHUNK_OVERLAP`       | `150`                    | Overlap between chunks                                 |
| `RAG_TOP_K`               | `5`                      | Chunks retrieved per query                             |
| `RESEARCH_DEPTH`          | `2`                      | Max reflection iterations for `pedro research`         |
| `RESEARCH_N_SUBQUESTIONS` | `3`                      | Sub-questions per iteration for `pedro research`       |

All variables can also be passed as CLI flags — run `pedro index --help`, `pedro ask --help`, or `pedro research --help` for details.

### Optimal Chunk Size Guidelines (google claim)

| Size   | Tokens   | Best for                                                      |
| ------ | -------- | ------------------------------------------------------------- |
| Small  | 128–256  | Specific, fact-based questions (FAQ, short answer)            |
| Medium | 256–512  | Semantic search, general documentation, RAG chatbot           |
| Large  | 512–1024 | Summarizing, relationships in content, long-document analysis |

## Customizing Prompts

All prompts live in the `prompts/` folder. Edit any file directly — changes take effect on the next command, no reinstall needed.

| File                            | Used by          | Purpose                                                                               |
| ------------------------------- | ---------------- | ------------------------------------------------------------------------------------- |
| `prompts/answer.txt`            | `pedro ask`      | System prompt for answer generation — controls tone, citation format, grounding rules |
| `prompts/plan_subquestions.txt` | `pedro research` | Instructs the model to decompose the question into N sub-questions                    |
| `prompts/reflect.txt`           | `pedro research` | Asks the model to evaluate completeness and identify gaps in the current answer       |
| `prompts/synthesize.txt`        | `pedro research` | Instructs the model to combine all research findings into a final answer              |

Prompt files support `{placeholders}` that are filled at runtime (e.g. `{question}`, `{n}`, `{answer}`, `{context}`). Do not remove placeholders — the tool will fail if they are missing.

## Notes

Tools and projects to look at in the context of the problem:

- https://github.com/zylon-ai/private-gpt

- https://github.com/assafelovic/gpt-researcher

- https://developers.llamaindex.ai/python/examples/

- https://github.com/langchain-ai/langchain

- https://github.com/Mintplex-Labs/anything-llm

I tried some but surprisingly ended with generating own code since in all cases
it turned out to be not that easy to get fully local stack working out of the box
