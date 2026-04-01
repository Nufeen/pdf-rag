# pdf-rag (🌵 a.k.a `pedro`)

Local RAG (Retrieval-Augmented Generation) CLI for searching a folder of PDF books or papers.

Key ideas:

- Base scenario is fully local usage — no cloud services. Ollama for embeddings and LLM inference, ChromaDB for vector storage.
- Keeping things as simple as it can be

## 🪅 Stack

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

## 🌮 Project Structure

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

## 🌶️ Prerequisites

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

## 🎺 Local Model Recommendations (actual for apr 2026)

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

## 🦅 macOS Setup

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

## 🪇 Installation

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

## 🌵 Indexing

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

## 📚 Adding New Books

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

## 🔍 Querying

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

## 🌯 Re-indexing Everything

Use `--force` to re-index all files, for example after changing chunk size:

```bash
pedro index ~/Books/ --force
```

## 🎸 Environment Variables

| Variable            | Default                  | Description                                            |
| ------------------- | ------------------------ | ------------------------------------------------------ |
| `OLLAMA_BASE_URL`   | `http://localhost:11434` | Ollama host URL                                        |
| `RAG_DB_PATH`       | `~/.pdf-rag/chroma_db`   | ChromaDB storage path                                  |
| `RAG_EMBED_MODEL`   | `nomic-embed-text`       | Ollama embedding model (recommend `mxbai-embed-large`) |
| `RAG_LLM_MODEL`     | `mistral:7b`             | Ollama LLM model (recommend `command-r:35b`)           |
| `RAG_CHUNK_SIZE`    | `800`                    | Characters per chunk                                   |
| `RAG_CHUNK_OVERLAP` | `150`                    | Overlap between chunks                                 |
| `RAG_TOP_K`         | `5`                      | Chunks retrieved per query                             |

All variables can also be passed as CLI flags — run `pedro index --help` or `pedro ask --help` for details.

### Optimal Chunk Size Guidelines (google claim)

| Size | Tokens | Best for |
|---|---|---|
| Small | 128–256 | Specific, fact-based questions (FAQ, short answer) |
| Medium | 256–512 | Semantic search, general documentation, RAG chatbot |
| Large | 512–1024 | Summarizing, relationships in content, long-document analysis |

## 🪄 Customizing the System Prompt

The LLM system prompt lives in `prompt.txt` at the project root. Edit it directly to change how the model answers.

Changes take effect immediately on the next `pedro ask` — no reinstall needed.
