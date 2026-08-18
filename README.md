# simple-rag-bot

An educational RAG chatbot built on LangChain 1.x and LangGraph. It indexes local
documents (and optionally a Confluence space) into a local Chroma vector store, then
answers questions over them with inline citations.

Everything except the LLM call runs locally: embeddings are ONNX models via
[fastembed](https://github.com/qdrant/fastembed) (no torch, no GPU), and the vector
store is Chroma on disk.

## Features

- **Citations that mean something** — the model cites `[1]`, `[2]`, and only the
  sources it actually referenced are printed, with the file and character offset.
- **Incremental indexing** — chunk ids are a hash of `source:offset:text`, so
  re-running the indexer upserts changed chunks and deletes removed ones instead of
  duplicating everything.
- **Conversational memory that survives restarts** — LangGraph checkpoints the graph
  state to SQLite, keyed by a session name.
- **Answers in the question's language** — detected per question, independent of the
  language the documents are written in.
- **Pluggable sources** — a local folder (`.md`, `.txt`, `.pdf`) and Confluence.

## Requirements

- Python 3.12+
- An API key for one of: OpenAI, Anthropic, or a Langdock gateway

On WSL, keep the project and its virtualenv on the native filesystem (`/home/...`),
not under `/mnt/c` — running Python across the 9p mount is dramatically slower.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # installs the project itself (editable)

cp .env.example .env                   # then fill in your key
```

The first indexing run downloads the embedding model (~220 MB) into `~/.cache`.

## Usage

```bash
ragbot-ingest        # index docs/ into Chroma — re-run after changing documents
ragbot               # start chatting
```

Both are also available as modules if you prefer: `python -m ragbot.ingest`,
`python -m ragbot.cli`.

### Web UI

```bash
pip install -e ".[gui]"
chainlit run app.py -w
```

A ChatGPT-style chat with streaming answers and a side panel: clicking a `[1]` in an
answer opens the exact chunk it was based on, with its file and character offset.

### Terminal

Conversations are keyed by session name, and they persist:

```bash
ragbot                 # session "default"
ragbot stoicism        # a separate conversation, resumable later
```

Example:

```
You: What does Stoicism say about things outside our control?

Bot:
Stoicism draws a sharp line between what is "up to us" and what is not [1].
...

Sources:
[1] stoicism.md (offset 0)
```

## Configuration

All settings live in `.env` (see `.env.example`); defaults are declared in
`src/ragbot/config.py`.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai`, or `langdock` |
| `LLM_MODEL` | `claude-sonnet-4-5` | model id for that provider |
| `DOCS_DIR` | `./docs` | folder to index |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | chunking, in characters |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | fastembed model (384-dim, multilingual) |
| `CHROMA_DIR` | `./data/chroma` | vector store location |
| `RETRIEVER_K` | `8` | chunks retrieved per question |
| `DETECT_LANGUAGES` | `English,Russian` | languages the bot may answer in |
| `CHECKPOINT_DB` | `./data/checkpoints.sqlite` | conversation memory |

Changing `EMBEDDING_MODEL` requires deleting `data/chroma` — vector dimensions
will not match the existing collection.

Confluence is optional and only runs when `CONFLUENCE_URL` and `CONFLUENCE_SPACE_KEY`
are set. If it fails, indexing continues with the local documents alone.

## How it works

Indexing (`ragbot-ingest`):

```
connectors/ -> chunking (RecursiveCharacterTextSplitter, keeps start_index)
            -> ids (sha256 of source:offset:text)
            -> Chroma upsert + delete of removed chunks
```

Answering — a three-node LangGraph:

```
START -> condense -> retrieve -> generate -> END
```

- `condense` rewrites a follow-up ("what about the second one?") into a standalone
  question using the history. On the first turn it is a no-op and costs no LLM call.
- `retrieve` embeds that question and pulls the top `RETRIEVER_K` chunks from Chroma.
- `generate` numbers the chunks, puts them in the prompt, and requires the model to
  cite those numbers.

Memory is the LangGraph checkpointer plus a `thread_id` — the CLI never assembles a
message list by hand.

## Development

```bash
pip install -e ".[dev]"
pytest                          # ~4s, no network, no API key
RAGBOT_LLM_TESTS=1 pytest       # also runs the handful that call the model
```

Tests cover the invariants the pipeline depends on — deterministic chunk ids, citation
parsing, surrogate handling — plus a retrieval quality evaluation.

### Retrieval evaluation

`tests/eval_questions.json` holds 32 questions over the demo corpus in `docs/`, each
labelled with the document that must be retrieved. It is deliberately bilingual: Russian
questions against English documents is the demanding case, and the one an English-only
embedding model fails silently. The evaluation builds its own Chroma index from `docs/`
in a temp directory, so it is reproducible in a fresh clone and independent of whatever
is in your real store.

Metrics (`src/ragbot/evaluation.py`), ranked over retrieved *chunks*:

- **recall@k** — did a chunk of the right document reach the top k, i.e. would it be in
  the model's context at that `RETRIEVER_K`
- **MRR** — how high it ranked; distinguishes "first result" from "barely scraped in"

Measured on the current configuration versus the English-only model shipped before:

| | recall@1 | recall@8 | MRR |
|---|---|---|---|
| multilingual (current) | 0.812 | 0.969 | 0.862 |
| bge-small-en | 0.438 | 0.938 | 0.617 |
| bge-small-en, Russian questions only | 0.190 | 0.905 | 0.440 |

Note how little recall@8 moves: eight chunks out of a 24-chunk corpus is a third of
everything, so the right document turns up almost by accident. recall@1 and MRR are what
discriminate, and the thresholds lean on them. Swapping `EMBEDDING_MODEL` back to an
English-only model fails four tests with a per-question breakdown of what was missed.

Out-of-domain questions are checked too: they must land measurably further away than
real ones, and with `RAGBOT_LLM_TESTS=1` the bot must refuse them rather than improvise.

See [CLAUDE.md](CLAUDE.md) for architecture notes and known rough edges.

## License

MIT — see [LICENSE](LICENSE).
