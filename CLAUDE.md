# CLAUDE.md

Educational RAG bot on LangChain 1.x + LangGraph. Local embeddings (fastembed/ONNX),
local Chroma, LLM via a provider (OpenAI / Anthropic / Langdock).

## Commands

```bash
source .venv/bin/activate
export PYTHONPATH=src            # the package lives under src/ and is not pip-installed

python -m ragbot.ingest          # index docs/ (+ Confluence if configured) into Chroma
python -m ragbot.cli             # interactive terminal chat
```

Dependencies: `pip install -r requirements.txt`. Config lives in `.env` (template in
`.env.example`), read through pydantic-settings; every field is declared in
`src/ragbot/config.py`.

## Architecture

Two independent layers: **indexing** (offline) and **answering** (online).

Indexing — `ingest.py:index()`:
`connectors/` → `chunking.py` → `ids.py` → `vectorstore.py`

- `connectors/local_folder.py` — `.md/.txt` via TextLoader, `.pdf` via PyPDFLoader.
  Normalizes `metadata["source"]` to a path relative to `docs_dir` — that string is what
  ends up in the citation.
- `connectors/confluence.py` — optional; only runs when `CONFLUENCE_*` is set, and its
  errors are swallowed so ingestion never dies on it. Also writes a human-readable `source`.
- `chunking.py` — RecursiveCharacterTextSplitter with `add_start_index=True`;
  `start_index` is the precise anchor inside the document, used in citations.
- `ids.py` — deterministic chunk id = sha256(`source:start_index:text`).
  This is the key invariant: same chunk → same id (upsert), edited text → new id.
  That is what makes `index()` a real sync — it adds new ids and deletes from Chroma
  the ones no longer present in the source. Re-running it is idempotent.

Answering — `graph.py`, a LangGraph graph `condense → retrieve → generate`:

- `condense` — rewrites the question into a standalone one using history.
  On the first turn (empty history) the LLM is not called at all.
- `retrieve` — Chroma `as_retriever(k=retriever_k)` over the rewritten question.
- `generate` — numbers the chunks `[1]`, `[2]`… in the context; the model must cite
  those numbers.
- Memory *is* the checkpointer (`MemorySaver`) plus `thread_id`, not a hand-managed
  message list. `cli.py` passes only `{"question": ...}`; history is restored by the
  checkpointer. MemorySaver is process-local — a restart starts an empty conversation.
- `sources_for_answer()` parses `[N]` **out of the answer text** and prints only the
  sources actually cited, not every retrieved chunk.

Answer language: `lang.py` (lingua) detects the language of the **question**, and it is
hard-substituted into the system prompt. The answer is always in the question's language,
regardless of the context's language.

## Things to know before editing

- The model and retriever are constructed at module level in `graph.py` (`_model`,
  `_retriever`) — importing `ragbot.graph` already reads `.env` and initializes Chroma.
- Embeddings are `FastEmbedEmbeddings` from `langchain_community` (ONNX on CPU, no torch).
  Default model `BAAI/bge-small-en-v1.5`, 384 dimensions, downloaded once into `~/.cache`.
  If you change `embedding_model`, delete the existing Chroma collection — the
  dimensions will not match.
- `.venv` must live on the native WSL filesystem (`/home/...`), never under `/mnt/c` —
  otherwise everything is dramatically slower.
- `data/` (Chroma) and `.env` are not tracked in git.

## Pinned dependencies

- `atlassian-python-api<5` — 5.x renamed `get_all_pages_from_space(space=)` to
  `(space_key=)`, and langchain-community's `ConfluenceLoader` still passes `space=`.
  Unpinning it makes the Confluence connector fail with a `TypeError` that
  `ingest.py` swallows, so the failure is silent. Re-check when langchain-community updates.

## Known tech debt

- A connector that fails is indistinguishable from a source that went empty:
  `ingest.index()` computes `stale = existing - current` and deletes those ids.
  So when Confluence errors out, its chunks are silently dropped from Chroma and only
  come back on the next successful run. Deleting per-source (or skipping the delete
  phase when a connector raised) would be safer.
- `requirements.txt` is otherwise unpinned, so a fresh install can drift into
  incompatible majors — that is exactly how the atlassian break above appeared.
