#!/bin/sh
# Fail early and legibly. Without these checks Chainlit dies deep in its own
# stack ("You must provide a JWT secret...") and the user has to read a
# traceback to learn they forgot a line in their .env.
set -e

die() {
    echo "" >&2
    echo "ragbot: $1" >&2
    echo "" >&2
    echo "Pass your settings with --env-file, for example:" >&2
    echo "    cp .env.example .env      # then fill it in" >&2
    echo "    docker run --env-file .env -p 8000:8000 ragbot" >&2
    echo "" >&2
    exit 1
}

require_config() {
    [ -n "$CHAINLIT_AUTH_SECRET" ] || die "CHAINLIT_AUTH_SECRET is not set.
Generate one with:  docker run --rm ragbot secret"

    { [ -n "$CHAINLIT_USER" ] && [ -n "$CHAINLIT_PASSWORD" ]; } || die \
"CHAINLIT_USER and CHAINLIT_PASSWORD are not set.
The web UI has no default login on purpose, so nobody can sign in until you set them."

    case "$LLM_PROVIDER" in
        anthropic|"") key="$ANTHROPIC_API_KEY"; name="ANTHROPIC_API_KEY" ;;
        openai)       key="$OPENAI_API_KEY";    name="OPENAI_API_KEY" ;;
        langdock)     key="$LANGDOCK_API_KEY";  name="LANGDOCK_API_KEY" ;;
        *) die "LLM_PROVIDER=$LLM_PROVIDER is not one of: anthropic, openai, langdock" ;;
    esac
    [ -n "$key" ] || die "LLM_PROVIDER=${LLM_PROVIDER:-anthropic} needs $name to be set."
}

case "$1" in
    serve)
        require_config
        # A mounted volume starts empty, which would shadow the index baked into
        # the image. Rebuild it once rather than answering "I don't know" to
        # everything with no explanation.
        if [ ! -d /app/data/chroma ]; then
            echo "ragbot: no index in /app/data — building one from docs/"
            python -m ragbot.ingest
        fi
        # 0.0.0.0, or the port publish maps to something nothing is listening on.
        exec chainlit run app.py --host 0.0.0.0 --port 8000 --headless
        ;;
    ingest)
        # Re-index after changing mounted docs, or to pull in Confluence.
        require_config
        exec python -m ragbot.ingest
        ;;
    secret)
        exec chainlit create-secret
        ;;
    env-template)
        # `docker run --rm IMAGE env-template > .env` — no checkout needed.
        # Written to stdout alone so the redirect produces a usable file.
        exec cat /app/.env.example
        ;;
    shell)
        exec /bin/sh
        ;;
    *)
        exec "$@"
        ;;
esac
