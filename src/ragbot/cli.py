import sys

from ragbot.graph import build_graph, sources_for_answer, warmup
from ragbot.textio import strip_surrogates


def _flush_stdin() -> None:
    """Drop whatever was typed while the model was loading.

    Only safe to call when nothing is buffered mid-character: tcflush clears the
    kernel queue but not Python's own buffer, so calling it between turns can cut
    a multibyte character in half and leave a lone surrogate behind.
    """
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def run_chat() -> None:
    # The session name IS the LangGraph thread_id, and the checkpointer is on
    # disk — so reusing a name resumes that conversation after a restart.
    args = sys.argv[1:]
    if args and args[0].startswith("-"):
        # otherwise `ragbot --help` silently opens a session literally named "--help"
        print(f"usage: ragbot [session-name]   (default: 'default')\n"
              f"Reusing a session name resumes that conversation.")
        return
    session = args[0] if args else "default"

    # Under LANG=C.UTF-8 stdin defaults to surrogateescape; make undecodable bytes
    # visible as U+FFFD here instead of becoming surrogates that blow up later.
    try:
        sys.stdin.reconfigure(errors="replace")
    except Exception:
        pass

    print("Loading model, please wait...", flush=True)

    graph = build_graph()
    config = {"configurable": {"thread_id": session}}

    warmup()

    _flush_stdin()          # once, after the slow load — never inside the loop
    print(f"Ready (session: {session}). Ask a question, or 'exit' / Ctrl-C to quit.\n",
          flush=True)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        question = strip_surrogates(question)

        try:
            # messages omitted here — the checkpointer restores prior history
            result = graph.invoke({"question": question}, config=config)
        except KeyboardInterrupt:
            print("\n(cancelled)\n")
            continue
        except Exception as e:
            # one bad turn must not end the session — the history is already saved
            print(f"\n[error] {type(e).__name__}: {e}\n")
            continue

        answer = result["answer"]
        docs = result["docs"]

        print("\nBot:")
        print(answer)
        print("\nSources:")
        print(sources_for_answer(answer, docs))
        print("-" * 50)


if __name__ == "__main__":
    run_chat()
