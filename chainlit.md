# RAG Bot

Ask questions about the indexed documents — the demo corpus in `docs/`, plus a
Confluence space if one is configured.

Answers are built only from retrieved text. If the answer is not in the documents, the
bot says it does not know rather than improvising.

Every claim is cited. Click a `[1]` to open the exact chunk it came from, with its file
name and character offset — so you can check the answer instead of trusting it.

Ask in Russian or English; the reply comes back in the language you asked in.
