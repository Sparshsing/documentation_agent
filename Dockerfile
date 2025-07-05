FROM python:3.11-slim AS builder

# Enable best practices for Python in Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies only in the builder stage
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy dependency declarations first for better cache utilisation
COPY pyproject.toml ./

# Install uv (fast dependency resolver/installer) and build wheels
RUN pip install --upgrade pip uv \
    # Resolve dependencies for the current (Linux) platform and build CPU-only wheels
    && uv pip compile pyproject.toml -o requirements.lock --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip wheel --no-cache-dir -r requirements.lock -w /wheels --extra-index-url https://download.pytorch.org/whl/cpu

###############################################################################
# Final stage – lightweight runtime image                                         
###############################################################################
FROM python:3.11-slim

# Re-declare runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_HOST="0.0.0.0" \
    API_PORT="8000" \
    API_RELOAD="false"

# Create a non-root user to run the app
RUN addgroup --system app && adduser --system --group app

WORKDIR /app

# Copy built wheel files from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy the backend source code (frontend is excluded via .dockerignore)
COPY backend ./backend
COPY core ./core
COPY pyproject.toml ./

# --- NEW SECTION ---
# Create the directory for ChromaDB and set ownership for the 'app' user.
# This ensures that when we mount a volume here, the user has permissions.
# We use /app/processed_data as the path inside the container.
RUN mkdir -p /app/processed_data && chown -R app:app /app/processed_data
# --- END NEW SECTION ---

# Expose the API port
EXPOSE 8000

# Switch to non-root user
USER app

# Use gunicorn with uvicorn workers for production-grade serving
CMD ["gunicorn", "backend.main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "-b", "0.0.0.0:8000"] 