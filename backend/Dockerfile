# The backend as one container for Google Cloud Run (docs/deployment.md).
# Build and run locally from backend/:
#   docker build -t movie-copilot-api .
#   docker run --rm -p 8080:8080 --env-file .env movie-copilot-api
# Only runtime dependencies are installed: no Hugging Face datasets, no pytest, no tests or data scripts.
# The MCP server (app/mcp_server.py) runs inside this same container as a child process of the API.

FROM python:3.11-slim

# uv installs the exact versions in uv.lock. Same uv version as in CI (.github/workflows/tests.yml).
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /bin/uv

# No .pyc files on a read-mostly container; log lines reach Cloud Logging at once instead of being buffered.
# uv: use the image's Python, never download one; copy files, because the cache is on another filesystem.
ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     UV_PYTHON_DOWNLOADS=never     UV_LINK_MODE=copy     UV_NO_CACHE=1

WORKDIR /app

# Dependencies first, so a code change does not reinstall them (Docker reuses this layer). --locked fails the build
# when uv.lock does not match pyproject.toml; --no-dev leaves out the data scripts' and tests' packages.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev
ENV PATH="/app/.venv/bin:$PATH"

COPY app ./app
COPY migrations ./migrations
COPY alembic.ini .

# Not root: if the app were ever exploited, the attacker would not own the container.
RUN useradd --create-home --uid 1000 appuser
USER appuser

# Cloud Run tells the container which port to listen on in PORT (8080 by default).
ENV PORT=8080
EXPOSE 8080

# `exec` makes uvicorn process 1, so it receives Cloud Run's stop signal and shuts the MCP server down cleanly.
# No --forwarded-allow-ips='*': uvicorn would then take the client IP from the leftmost X-Forwarded-For entry,
# which the client can write itself. The rate limit reads the header on its own terms (app/rate_limit.py).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
