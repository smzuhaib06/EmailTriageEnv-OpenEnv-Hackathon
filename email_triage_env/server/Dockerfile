FROM python:3.11-slim

WORKDIR /app

# Copy server requirements first for layer caching
COPY email_triage_env/server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire package
COPY email_triage_env/ ./email_triage_env/
COPY openenv.yaml .

# HuggingFace Spaces uses port 7860
EXPOSE 7860

ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["uvicorn", "email_triage_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
