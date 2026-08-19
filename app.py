"""Chainlit frontend.

    chainlit run app.py -w

Deliberately thin: everything it knows about the bot comes from `ragbot.service`.
If logic starts appearing here that the CLI also needs, it belongs in the service
layer instead — two frontends reimplementing the same thing is how they drift.

Chainlit is async and the graph's nodes are synchronous, so blocking calls are
either awaited through `astream_events` (LangGraph runs sync nodes in a worker
thread) or wrapped in `cl.make_async`. A plain blocking call here would stall the
event loop for every connected client, including the stop button.
"""

import chainlit as cl
from chainlit.server import app as fastapi_app
from chainlit.types import ThreadDict
from fastapi.staticfiles import StaticFiles
from starlette.routing import Mount

import chainlit_store
from ragbot import service
from ragbot.config import settings

SIDE_PANEL = "side"

# Serve stored element payloads, so citations still open when an old chat is
# reopened from the sidebar. Inserted at the FRONT of the route table on purpose:
# Chainlit has already registered a catch-all that serves the SPA for every
# unmatched path, and `app.mount()` appends — the catch-all would win and hand
# back index.html instead of the file.
fastapi_app.router.routes.insert(
    0,
    Mount(
        chainlit_store.ELEMENTS_ROUTE,
        app=StaticFiles(directory=chainlit_store.elements_dir()),
        name="element-files",
    ),
)


# --------------------------------------------------------------- persistence


@cl.data_layer
def data_layer():
    """Storing threads is what puts past conversations in the sidebar."""
    return chainlit_store.data_layer()


@cl.password_auth_callback
def auth(username: str, password: str):
    """Chainlit's thread endpoints read `current_user.identifier`, and without an
    auth callback that user is None — so the sidebar needs a login even for a
    local single-user app.

    Credentials come from .env (CHAINLIT_USER / CHAINLIT_PASSWORD). If they are
    unset nobody can log in: a default password would be worse than a locked door.
    """
    if not settings.chainlit_user or not settings.chainlit_password:
        return None
    if username == settings.chainlit_user and password == settings.chainlit_password:
        return cl.User(identifier=username)
    return None


# ------------------------------------------------------------------ chat flow


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


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict) -> None:
    """Reopening a conversation from the sidebar.

    Nothing to restore by hand: the session id below is the thread id, and the
    graph's memory is keyed by exactly that — so the bot already remembers this
    conversation. Only the embedding model needs warming again.
    """
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

    try:
        # Several seconds pass before the first token — condensing and retrieval
        # come first. Showing them turns dead air into visible progress.
        async with cl.Step(name="Поиск по документам", type="retrieval") as step:
            step.output = "Формулирую запрос…"
            streaming = False

            async for event in service.astream_events(message.content, session):
                if isinstance(event, service.Rewritten):
                    step.output = (
                        f"Ищу: «{event.question}»" if event.changed
                        else "Ищу по исходному вопросу"
                    )
                    await step.update()

                elif isinstance(event, service.Retrieved):
                    found = ", ".join(event.sources) or "ничего"
                    step.output = f"Найдено фрагментов: {event.chunks} — {found}"
                    await step.update()

                elif isinstance(event, service.Token):
                    if not streaming:
                        # send the answer bubble only once there is something to
                        # put in it, so it does not sit empty under the step
                        streaming = True
                    await reply.stream_token(event.text)

    except Exception as e:
        # one failed turn must not kill the chat; the history is already saved
        reply.content = f"⚠️ {type(e).__name__}: {e}"
        await reply.send()
        return

    # the token stream carries text only — citations need the retrieved
    # documents, which the graph saved when the turn finished
    answer = await service.alast_answer(session)
    if answer:
        reply.elements = source_elements(answer)

    await reply.send()
