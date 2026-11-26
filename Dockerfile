FROM python:3.10-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NL2SQL_HOME=/app NL2SQL_OUTPUT_DIR=/app/outputs
COPY requirements.lock ./
RUN python -m pip install --no-cache-dir -r requirements.lock
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN python -m pip install --no-build-isolation --no-deps . && \
    useradd --system --uid 10001 --create-home appuser && \
    mkdir -p /app/outputs && chown appuser:appuser /app/outputs
USER appuser
EXPOSE 8765
CMD ["python", "-m", "nl2sql_handbook.delivery_demo.container_start"]
