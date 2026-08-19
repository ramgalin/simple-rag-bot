# Runnable image: `docker run --env-file .env -p 8000:8000 ragbot`
#
# Two things are baked in at build time so the first run is not a five-minute
# silence: the ONNX embedding model (~240 MB) and an index built from docs/.
# Both are otherwise fetched or computed on first use.

FROM python:3.12-slim

# fastembed defaults its cache to /tmp/fastembed_cache, which is a poor place for
# a 240 MB download: on many systems /tmp is wiped on reboot, so the model would
# be re-fetched. Pin it somewhere permanent and bake the model into that layer.
ENV FASTEMBED_CACHE_PATH=/opt/models \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: this layer is the slow one and only needs rebuilding when
# pyproject.toml changes, not on every source edit.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[gui]"

# Download the embedding model into the image. Importing the package is enough —
# FastEmbedEmbeddings fetches on construction.
RUN python -c "from ragbot.embeddings import build_embeddings; build_embeddings()"

COPY docs/ ./docs/
COPY app.py chainlit_store.py ./
COPY .chainlit/config.toml ./.chainlit/
COPY public/ ./public/

# Build the demo index so the bot answers out of the box. Runs against docs/
# only — Confluence is not configured at build time, and must not be: its
# credentials belong to whoever runs the container, not to the image.
RUN python -m ragbot.ingest

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Conversations, the vector store and uploaded elements all live here. Mount a
# volume over it to keep them across `docker run`s.
VOLUME /app/data

EXPOSE 8000

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["serve"]
