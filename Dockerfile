FROM python:3.11-slim

WORKDIR /app

COPY email_triage_env/server/requirements.txt .
RUN pip install --no-cache-dir --timeout=120 --retries=5 -r requirements.txt

COPY email_triage_env/ ./email_triage_env/
COPY openenv.yaml .

EXPOSE 7860

ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["uvicorn", "email_triage_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
