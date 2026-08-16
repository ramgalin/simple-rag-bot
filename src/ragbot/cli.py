import sys
import uuid

from ragbot.graph import build_graph, sources_for_answer


def _flush_stdin() -> None:
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def run_chat() -> None:
    print("Loading model, please wait...", flush=True)

    graph = build_graph()
    thread_id = str(uuid.uuid4())               # one conversation thread
    config = {"configurable": {"thread_id": thread_id}}

    # warm up embedder on a throwaway thread
    graph.invoke(
        {"question": "warmup", "messages": []},
        config={"configurable": {"thread_id": "warmup"}},
    )

    _flush_stdin()
    print("Ready. Ask a question, or 'exit' / Ctrl-C to quit.\n", flush=True)

    while True:
        try:
            _flush_stdin()
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        # messages omitted here — the checkpointer restores prior history
        result = graph.invoke({"question": question}, config=config)

        answer = result["answer"]
        docs = result["docs"]

        print("\nBot:")
        print(answer)
        print("\nSources:")
        print(sources_for_answer(answer, docs))
        print("-" * 50)


if __name__ == "__main__":
    run_chat()