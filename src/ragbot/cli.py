import sys

from ragbot import service


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
        print("usage: ragbot [session-name]   (default: 'default')\n"
              "Reusing a session name resumes that conversation.")
        return
    session = args[0] if args else service.DEFAULT_SESSION

    # Under LANG=C.UTF-8 stdin defaults to surrogateescape; make undecodable bytes
    # visible as U+FFFD here instead of becoming surrogates that blow up later.
    try:
        sys.stdin.reconfigure(errors="replace")
    except Exception:
        pass

    print("Loading model, please wait...", flush=True)

    service.warmup()

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

        print("\nBot:")
        try:
            for token in service.stream_answer(question, session):
                print(token, end="", flush=True)
        except KeyboardInterrupt:
            print("\n\n(cancelled)\n")
            continue
        except Exception as e:
            # one bad turn must not end the session — the history is already saved
            print(f"\n\n[error] {type(e).__name__}: {e}\n")
            continue

        # citations come from the retrieved documents, which the stream does not
        # carry; the graph already saved them, so read them back
        answer = service.last_answer(session)
        print("\n\nSources:")
        print(answer.sources_text if answer else "(no sources cited)")
        print("-" * 50)


if __name__ == "__main__":
    run_chat()
