"""Chainlit frontend.

    chainlit run app.py -w

Deliberately thin: everything it knows about the bot comes from `ragbot.service`.
If logic starts appearing here that the CLI also needs, it belongs in the service
layer instead — two frontends reimplementing the same thing is how they drift.

Chainlit is async and the graph's nodes are synchronous, so blocking calls are
either awaited through `astream_answer` (LangGraph runs sync nodes in a worker
thread) or wrapped in `cl.make_async`. A plain blocking call here would stall the
event loop for every connected client, including the stop button.
"""

import chainlit as cl

from ragbot import service

SIDE_PANEL = "side"


@cl.set_starters
async def starters() -> list[cl.Starter]:
    """The cards shown on an empty chat, as in ChatGPT and Claude."""
    return [
        cl.Starter(
            label="Что не в нашей власти",
            message="Что стоики говорят о вещах, которые не в нашей власти?",
        ),
        cl.Starter(
            label="Существование и сущность",
            message="Что означает тезис «существование предшествует сущности»?",
        ),
        cl.Starter(
            label="Principle of utility",
            message="What is the principle of utility, and who formulated it?",
        ),
        cl.Starter(
            label="Юм о причинности",
            message="К какому выводу Юм пришёл о причинно-следственной связи?",
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    # loading the embedding model takes a couple of seconds; do it before the
    # first question rather than inside it
    await cl.make_async(service.warmup)()


def session_id() -> str:
    """Chainlit's thread id doubles as the LangGraph thread id.

    Using the same id for both means the conversation list in the UI and the
    memory the graph actually has are the same thing. Giving the graph its own
    id would create a second, silently diverging history.
    """
    return cl.context.session.thread_id


def source_elements(answer: service.Answer) -> list[cl.Text]:
    """One side-panel element per citation, named after its `[N]` marker.

    Chainlit links an element whose name occurs in the message text, so naming
    them `[1]`, `[2]` turns the model's own citation markers into clickable
    references to the retrieved chunk.
    """
    elements = []
    for c in answer.citations:
        body = c.text
        # chunking prepends the source to every chunk so the title is searchable;
        # in the panel it would just repeat the heading right above it
        prefix = f"{c.source}\n\n"
        if body.startswith(prefix):
            body = body[len(prefix):]

        elements.append(
            cl.Text(
                name=f"[{c.number}]",
                content=f"**{c.source}** — offset {c.start_index}\n\n{body}",
                display=SIDE_PANEL,
            )
        )
    return elements


@cl.on_message
async def on_message(message: cl.Message) -> None:
    session = session_id()

    reply = cl.Message(content="")
    await reply.send()

    try:
        async for token in service.astream_answer(message.content, session):
            await reply.stream_token(token)
    except Exception as e:
        # one failed turn must not kill the chat; the history is already saved
        reply.content = f"⚠️ {type(e).__name__}: {e}"
        await reply.update()
        return

    # the token stream carries text only — citations need the retrieved
    # documents, which the graph saved when the turn finished
    answer = await service.alast_answer(session)
    if answer:
        reply.elements = source_elements(answer)

    await reply.update()
