# CLAUDE.md

Educational RAG bot on LangChain 1.x + LangGraph. Local embeddings (fastembed/ONNX),
local Chroma, LLM via a provider (OpenAI / Anthropic / Langdock).

## Commands

```bash
source .venv/bin/activate

ragbot-ingest                    # index docs/ (+ Confluence if configured) into Chroma
ragbot [session-name]            # interactive terminal chat, default session "default"
pytest                           # ~4s, no API key needed
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

Frontends talk to `service.py`, never to `graph.py` directly. It owns the compiled
graph (built once — `build_graph()` opens a fresh SQLite connection per call), input
sanitising, and session listing. `cli.py` is a thin consumer of it and any GUI should be
too; the moment a frontend assembles the graph itself, the two drift apart.

- `ask()` blocks and returns a whole `Answer`; `stream_answer()` / `astream_answer()`
  yield tokens. The async variant exists because Chainlit-style servers stall their
  event loop on a blocking `invoke`.
- **There are two compiled graphs, and they are not interchangeable.** `graph()` uses
  `SqliteSaver`, `agraph()` uses `AsyncSqliteSaver`; the sync saver raises
  NotImplementedError on *every* async method, so an async server driving the sync
  graph dies on its first turn. Both write the same file and schema, so history is
  shared. Sync callers must not touch `agraph()` and vice versa.
- `Answer.citations` gives structured `Citation` objects (including the chunk text, for
  a GUI side panel); `Answer.sources_text` gives the terminal's plain rendering. One
  parser in `citations.py`, two renderings.
- Streaming yields text only. Citations need the retrieved documents, so a frontend
  streams tokens and then calls `last_answer(session)` — the graph already saved them.
- `sessions()` is the sidebar list of a chat UI: it *is* the checkpointer's set of
  thread ids. Do not add a second store of conversation history beside it.

**`stream_mode="messages"` needs two filters, and both are load-bearing** (see
`service._answer_tokens`):

1. by node — `condense` also calls the LLM, so an unfiltered stream prints the
   rewritten question as if it were the answer;
2. by type — the stream carries the model's `AIMessageChunk` tokens *and* the finished
   `HumanMessage`/`AIMessage` the node appends to state, both tagged `generate`.
   Filtering by node alone emits the answer twice with the question wedged in between.

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

The detector chooses only between the languages in `DETECT_LANGUAGES` (default
`English,Russian`), not all 75 lingua knows. That is not tuning, it is a fix: across the
full set, "кто основал стоицизм?" was detected as Serbian and "а я кто?" as Bulgarian,
and the bot answered a Russian user in those languages. Widening the list brings the
problem back — add a language only when it is genuinely in use.

## Frontends

`app.py` is a Chainlit UI (`chainlit run app.py -w`, install with `pip install -e ".[gui]"`).
It is thin on purpose and talks only to `service.py`.

- Chainlit's thread id *is* the LangGraph thread id (`app.session_id()`). Keeping them
  equal is what stops the UI's conversation list and the graph's memory from diverging.
  Do not give the graph its own id.
- Citations become side-panel `cl.Text` elements named `[1]`, `[2]` — matching the
  markers in the answer, which is what makes Chainlit render them as clickable
  references. Renaming them breaks the link silently.
- The element strips the source prefix that `chunking.py` adds, or the panel repeats
  the heading shown directly above it.
- Progress steps come from `service.astream_events`, which streams graph updates
  alongside tokens. Five to six seconds pass before the first token on a cold turn
  (condense, then retrieval), so without them the UI looks frozen.

### Look (`public/`)

- `logo_light.svg` / `logo_dark.svg` / `favicon.svg` — Chainlit picks these up by
  filename from `public/`; there is no config entry for them. The favicon is a
  deliberately reduced version of the mark (brackets plus one bar): the full three-line
  logo turns to mush below ~20px.
- **`custom_css` is linked BEFORE Chainlit's own stylesheet**, not after. So an
  equal-specificity rule there loses the cascade — which is why `theme.css` carries no
  colours and prefixes its class selectors with `:root` to lift them from (0,1,0) to
  (0,2,0). Id selectors are already strong enough. Symptom of getting this wrong: the
  UI keeps Chainlit's stock magenta while the logo picks up correctly.
- **Colours go in `public/theme.json`**, not CSS. Chainlit injects
  `variables.light` / `variables.dark` as `window.theme`, and the frontend writes them
  as inline styles on `<html>` — which no stylesheet can outrank. Values are shadcn HSL
  triplets (`"--primary": "35 65% 47%"`), not hex.
- `theme.css` only targets ids and classes present in Chainlit 2.11's bundle —
  `#message-composer`, `#welcome-screen`, `#starters`, `#thread-history`,
  `#side-view-content`, `.message-content`, `.step`. Tailwind utility combos are avoided
  on purpose: they churn between releases, and a selector that stops matching fails
  silently rather than loudly.
- `cot = "tool_call"` keeps the progress step visible while it runs and collapses it
  afterwards. `"hidden"` would hide the very thing that fills the pre-token pause.
- `confirm_new_chat = false` removes the "This will clear your current chat history"
  prompt. That warning is accurate for stock Chainlit — every persistence call is
  guarded by `if data_layer:`, so by default the browser holds the only copy and
  "New Chat" really does destroy it. The dialog is rendered unconditionally, with no
  check for whether a data layer exists, so with `chainlit_store.py` wired up it warns
  about something that cannot happen. Re-enable it if persistence is ever removed.
  (Clicking it mid-generation does abandon that one turn — `clear()` drops the socket,
  and LangGraph only checkpoints on completion.)

### Docker (`Dockerfile`, `docker-entrypoint.sh`)

Built for "download and run": `docker run --env-file .env -p 8000:8000 ragbot`.

- The embedding model **and** an index built from `docs/` are baked in at build time.
  Both steps must work without any secrets — verified: `ragbot.ingest` needs no API key
  (only embeddings), and Confluence is skipped when unconfigured. Do not add a build step
  that needs a key.
- Confluence is deliberately *not* indexed at build time: those credentials belong to
  whoever runs the container. `docker run ... ragbot ingest` pulls it in at run time.
- `VOLUME /app/data` — Docker copies the baked index into a fresh named volume, so it
  survives; a **bind** mount does not, which is why the entrypoint rebuilds when
  `/app/data/chroma` is missing.
- The entrypoint validates `CHAINLIT_AUTH_SECRET`, the login pair and the provider's API
  key before starting, because the failure modes are otherwise opaque (Chainlit's JWT
  error is a raw traceback). Its subcommands are `serve` (default), `ingest`, `secret`,
  `shell`.
- `--host 0.0.0.0`, or the published port maps to nothing listening.

### CI (`.github/workflows/`)

- `tests.yml` runs the suite on push/PR and is also `workflow_call`-able, which is how
  `release.yml` gates publishing on a green run.
- It pins `FASTEMBED_CACHE_PATH` into the workspace so `actions/cache` can keep the
  ~240 MB model; the default `/tmp/fastembed_cache` is not reliably cacheable. Cache key
  is a hash of `config.py`, where `embedding_model` is declared.
- Installs `[dev,gui]`: without chainlit the web-UI store tests skip silently, and a
  skipped test protects nothing.
- `RAGBOT_LLM_TESTS` stays unset in CI — those tests need a paid key. Everything else
  runs, including the retrieval evaluation (local embeddings, no key).
- `release.yml` publishes `linux/amd64` only. arm64 would run the baked-index build step
  (ONNX inference) under QEMU emulation.

### Sidebar history (`chainlit_store.py`)

- **Login is mandatory, not a nicety.** Chainlit's thread endpoints read
  `current_user.identifier`, and `get_current_user()` returns None when no auth callback
  is registered — so the sidebar raises rather than degrading. Credentials come from
  `CHAINLIT_USER` / `CHAINLIT_PASSWORD`; unset means nobody can log in, deliberately.
  `CHAINLIT_AUTH_SECRET` is also required (`chainlit create-secret`).
- **`SQLAlchemyDataLayer` never creates its schema** — `ensure_schema()` does. It builds
  INSERTs from whatever keys a `ThreadDict`/`StepDict`/`ElementDict` carries, so a
  missing column fails mid-conversation rather than at startup.
- Chainlit stores threads separately from the LangGraph checkpointer. That duplication
  is fine *because the ids match* — verified: the same uuid appears in both stores for
  one conversation. `@cl.on_chat_resume` therefore restores nothing by hand; the graph
  already remembers the thread it is handed.
- Chainlit ships storage clients for S3/GCS/Azure only. Without one it logs
  "No blob_storage_client is configured" and silently drops every element, so a reopened
  chat would have dead `[1]` links. `LocalStorageClient` writes them to `data/elements`.
- The static mount for those files is inserted at the **front** of the route table:
  Chainlit already registered a catch-all serving the SPA, and `app.mount()` appends —
  the catch-all wins and returns index.html instead of the file.
- Stopping generation mid-answer abandons the turn: LangGraph writes its checkpoint when
  the run completes, so a cancelled turn leaves no trace in the graph's memory.

## Things to know before editing

- The model and retriever are built lazily via `graph.model()` / `graph.retriever()`.
  They used to be module-level, which made importing `ragbot.graph` read `.env`, open
  Chroma and load the ONNX model as a side effect — that alone tripled the test suite's
  runtime once `service.py` started importing it.
- Embeddings are `FastEmbedEmbeddings` from `langchain_community` (ONNX on CPU, no torch).
  Default `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions,
  downloaded once into `FASTEMBED_CACHE_PATH` (default `/tmp/fastembed_cache`, *not*
  `~/.cache` — and /tmp is wiped on reboot on many systems). **Keep it multilingual.** Questions are asked in
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
