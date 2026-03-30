# pdf-rag

Local RAG (Retrieval-Augmented Generation) CLI for searching a folder of PDF books. Fully local — no cloud services. Uses Ollama for embeddings and LLM inference, ChromaDB for vector storage.

## Model Recommendations

### Embedding model

| Model | Dims | Notes |
|---|---|---|
| `mxbai-embed-large` | 1024 | **Recommended** — better retrieval quality than nomic |
| `nomic-embed-text` | 768 | Default, fast, good baseline |
| `bge-m3` | 1024 | Best quality, multilingual, slightly slower |

### LLM

| Model | Size | Context | Notes |
|---|---|---|---|
| `command-r:35b` | 35B | 128k | **Recommended** — fine-tuned specifically for RAG, native citation support |
| `mixtral:8x7b` | 47B MoE | 32k | Excellent quality, fast due to MoE architecture |
| `llama3.1:70b` | 70B | 128k | Best reasoning if VRAM allows |
| `qwen2.5:32b` | 32B | 128k | Strong choice for non-English books |
| `mistral:7b` | 7B | 8k | Minimum viable, good for low-memory hosts |

`command-r:35b` is the best fit for RAG specifically — it's trained to ground answers in retrieved context and produce accurate citations rather than hallucinate beyond the provided excerpts.

---

## Prerequisites

On the machine running Ollama:

```bash
# Allow network access (add to ~/.bashrc or systemd service)
export OLLAMA_HOST=0.0.0.0

ollama pull mxbai-embed-large
ollama pull command-r:35b    # or see model recommendations above
```

On this machine, point to the remote Ollama host:

```bash
export OLLAMA_BASE_URL=http://192.168.1.X:11434   # replace with actual IP
```

## Installation

```bash
pip install -e .
```

## Indexing

Scan a folder and index all PDFs:

```bash
pdf-rag index ~/Books/
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
pdf-rag index ~/Books/
# Skipping (already indexed): deep_learning.pdf
# Skipping (already indexed): pattern_recognition.pdf
```

## Adding New Books

Drop the new PDF into the folder and re-run the index command:

```bash
cp new_book.pdf ~/Books/
pdf-rag index ~/Books/
# Skipping (already indexed): deep_learning.pdf
# Skipping (already indexed): pattern_recognition.pdf
# Embedding: new_book.pdf (1523 chunks)...
# Indexed: new_book.pdf (1523 chunks)
```

Only the new file is processed. Existing books are skipped instantly.

## Querying

```bash
pdf-rag ask "What is the vanishing gradient problem?"
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
pdf-rag ask "Explain attention mechanisms" --no-sources
```

Retrieve more context chunks:

```bash
pdf-rag ask "Compare LSTM and GRU" --top-k 8
```

## Re-indexing Everything

Use `--force` to re-index all files, for example after changing chunk size:

```bash
pdf-rag index ~/Books/ --force
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama host URL |
| `RAG_DB_PATH` | `~/.pdf-rag/chroma_db` | ChromaDB storage path |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model (recommend `mxbai-embed-large`) |
| `RAG_LLM_MODEL` | `mistral:7b` | Ollama LLM model (recommend `command-r:35b`) |
| `RAG_CHUNK_SIZE` | `800` | Characters per chunk |
| `RAG_CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `RAG_TOP_K` | `5` | Chunks retrieved per query |

All variables can also be passed as CLI flags — run `pdf-rag index --help` or `pdf-rag ask --help` for details.
