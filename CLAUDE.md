# CLAUDE.md

Educational RAG bot on LangChain 1.x + LangGraph. Local embeddings (fastembed/ONNX),
local Chroma, LLM via a provider (OpenAI / Anthropic / Langdock).

## Commands

```bash
source .venv/bin/activate

ragbot-ingest                    # index docs/ (+ Confluence if configured) into Chroma
ragbot [session-name]            # interactive terminal chat, default session "default"
pytest                           # ~4s, no network, no API key needed
RAGBOT_LLM_TESTS=1 pytest        # plus the three tests that call the model
```

The package is installed editable (`pip install -r requirements.txt` resolves to `-e .`),
so no `PYTHONPATH=src` and the two console scripts come from `[project.scripts]`.
`python -m ragbot.ingest` / `python -m ragbot.cli` still work.

Dependencies live in `pyproject.toml`, pinned by major — `requirements.txt` is just a
pointer to the project. Dev extras: `pip install -e ".[dev]"`.

Config lives in `.env` (template in `.env.example`), read through pydantic-settings;
every field is declared in `src/ragbot/config.py`.

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
  Each chunk is then prefixed with its `source` (file name / Confluence page title):
  the title otherwise lives only in metadata and is invisible to the embedding, so
  "what projects do we have" had nothing to match against names like `l1ve-audio-sender`.
  Prefixing happens *after* splitting, so every chunk carries it and `start_index`
  still refers to the original document.
- `ids.py` — deterministic chunk id = sha256(`source:start_index:text`).
  This is the key invariant: same chunk → same id (upsert), edited text → new id.
  That is what makes `index()` a real sync — it adds new ids and deletes from Chroma
  the ones no longer present in the source. Re-running it is idempotent.
- Deletion is scoped by origin. Every document carries `metadata["origin"]`
  (`local` / `confluence`), and `index()` only deletes orphaned chunks whose origin
  loaded successfully this run — a connector that raised must not wipe its own content
  out of the index. Rows predating this metadata are treated as `local`.

Answering — `graph.py`, a LangGraph graph `condense → retrieve → generate`:

- `condense` — rewrites the question into a standalone one using history.
  On the first turn (empty history) the LLM is not called at all.
- `retrieve` — Chroma `as_retriever(k=retriever_k)` over the rewritten question.
- `generate` — numbers the chunks `[1]`, `[2]`… in the context; the model must cite
  those numbers.
- Memory *is* the checkpointer (`SqliteSaver` over `checkpoint_db`) plus `thread_id`,
  not a hand-managed message list. `cli.py` passes only `{"question": ...}`; history is
  restored by the checkpointer. The CLI's positional argument *is* the `thread_id`, so
  `ragbot stoicism` resumes that conversation across restarts.
- `graph.warmup()` loads the embedding model by hitting the retriever only. It must not
  invoke the graph: that would spend an LLM call and write a checkpoint on every startup.
- `citations.py:sources_for_answer()` parses `[N]` **out of the answer text** and prints
  only the sources actually cited, not every retrieved chunk. It lives in its own module
  (re-exported from `graph`) so tests can import it without booting the LLM and Chroma.

Answer language: `lang.py` (lingua) detects the language of the **question**, and it is
hard-substituted into the system prompt. The answer is always in the question's language,
regardless of the context's language.

## Things to know before editing

- The model and retriever are constructed at module level in `graph.py` (`_model`,
  `_retriever`) — importing `ragbot.graph` already reads `.env` and initializes Chroma.
- Embeddings are `FastEmbedEmbeddings` from `langchain_community` (ONNX on CPU, no torch).
  Default `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions,
  downloaded once into `~/.cache`. **Keep it multilingual.** Questions are asked in
  Russian about English pages; the previous `BAAI/bge-small-en-v1.5` scored
  cos(амбисоник, ambisonics)=0.61 against cos(амбисоник, banana)=0.57 — a 0.04 margin,
  i.e. noise, and Russian questions retrieved essentially random chunks. The current
  model scores 0.86 vs 0.31 on the same pair.
  If you change `embedding_model`, delete `data/chroma` — dimensions will not match.
  fastembed warns that this model now uses mean pooling instead of CLS; that is expected,
  do not pin fastembed back to 0.5.1 over it.
- `.venv` must live on the native WSL filesystem (`/home/...`), never under `/mnt/c` —
  otherwise everything is dramatically slower.
- `data/` holds both the Chroma collection and `checkpoints.sqlite`; it is gitignored
  and disposable, but deleting it also drops every saved conversation.

## Retrieval evaluation

`tests/test_retrieval_quality.py` scores 32 labelled questions (`tests/eval_questions.json`)
against an index built from `docs/` alone, in a temp directory — never the project's own
store, which also holds Confluence content and drifts as people edit pages.

- Metrics live in `src/ragbot/evaluation.py` and are unit-tested themselves: a wrong
  metric does not fail loudly, it reports a plausible number and protects nothing.
- Scoring ranks **chunks**, not documents. With four documents in the corpus, collapsing
  to distinct documents would make recall@4 identically 1.0.
- **recall@8 is nearly decorative here** and passes even with a broken embedding model —
  eight of 24 chunks is a third of the corpus. recall@1 and MRR carry the signal.
  Thresholds sit between the current model (0.812 / 0.862) and the English-only one
  that was shipped before (0.438 / 0.617; 0.190 / 0.440 on Russian questions alone).
- Keep the question set bilingual. The English-only regression looked acceptable on the
  aggregate and was catastrophic on Russian questions specifically.
- Tests that call the model are skipped unless `RAGBOT_LLM_TESTS=1`, and they redirect
  `settings.checkpoint_db` to a temp file so they never write into real conversations.

## Pinned dependencies

- `atlassian-python-api<5` — 5.x renamed `get_all_pages_from_space(space=)` to
  `(space_key=)`, and langchain-community's `ConfluenceLoader` still passes `space=`.
  Unpinning it makes the Confluence connector fail with a `TypeError` that
  `ingest.py` swallows, so the failure is silent. Re-check when langchain-community updates.

## Known tech debt

- **A partial Confluence load would still delete.** Origin-scoped deletion covers a
  connector that *raises*, but a connector that returns an incomplete page set reads as
  success, and the missing pages' chunks get purged. Not observed so far, and there is
  no reason to suspect the loader of it — but a mid-pagination network error would look
  exactly like a shrinking space.
- Confluence is a live source that other people and agents write to, so page counts
  legitimately change between runs. Do not read a jump in chunk count as a bug without
  checking the space first.
- `confluence.py` sets `metadata["source"]` to the page title, so two pages with the
  same title are indistinguishable in a citation, and a retitled page re-indexes as new.
- No test covers the graph itself — the nodes need an LLM. Injecting a fake chat model
  would make `condense`/`generate` testable.
- Confluence space-template pages get indexed as content. The page titled "Ambisonics"
  is just the space homepage ("In a sentence or two, describe the purpose of this
  space", "Filter by Label") and is pure retrieval noise.
- Inventory questions ("list everything we have") are a poor fit for top-k retrieval —
  it returns `retriever_k` chunks, not a complete enumeration. Prefixing chunks with
  their titles made these answerable, but the answer is still bounded by k.
