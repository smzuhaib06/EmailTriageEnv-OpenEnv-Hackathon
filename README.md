---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# EmailTriageEnv 📧

## Overview

Customer support teams at companies of all sizes face a relentless stream of incoming emails every day. Triaging that queue — deciding what's urgent, what category it belongs to, and what action to take — is one of the most time-consuming and cognitively demanding tasks a support agent performs. A single misclassified critical billing issue can mean a churned customer; a missed security escalation can become a PR crisis.

EmailTriageEnv models this real-world workflow as a reinforcement learning environment. An AI agent is presented with a queue of realistic synthetic customer emails and must assign each one a priority level, a category, and a recommended action. The environment scores the agent's decisions against ground-truth labels, providing a clear signal for learning and evaluation.

This environment is designed for the OpenEnv ecosystem and runs as a self-contained FastAPI server, making it easy to deploy locally, in Docker, or on HuggingFace Spaces. It ships with three task difficulties — from simple category identification up to full triage — so agents can be evaluated at multiple capability levels.

## Environment Description

The environment follows a standard reset-step loop:

1. Call `POST /reset` with a task name to start a new episode. The server loads a queue of synthetic emails and returns the first one as an observation.
2. The agent reads the observation (email subject, body, sender, etc.) and submits a triage decision via `POST /step`.
3. The server grades the decision, returns a reward and the next email observation.
4. Steps repeat until all emails in the queue are processed, at which point `done=true` is returned.
5. Call `POST /reset` again to start a fresh episode.

## Action Space

| Field | Type | Valid Values |
|-------|------|-------------|
| email_id | string | current email ID |
| priority | string | critical, high, medium, low |
| category | string | billing, technical, general, spam |
| action | string | respond, escalate, archive, delete |
| reasoning | string | free text (not graded) |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| email_id | string | Unique email identifier |
| subject | string | Email subject line |
| body | string | Email body text |
| sender | string | Sender name and email |
| timestamp | string | ISO 8601 timestamp |
| queue_position | integer | Current position in queue (1-based) |
| total_in_queue | integer | Total emails this episode |
| emails_processed | integer | Already handled count |
| current_score | float | Running average score |
| task_name | string | Active task name |

## Tasks

| Task | Difficulty | Emails | What's Graded | Success Threshold |
|------|-----------|--------|--------------|-------------------|
| category-identification | Easy | 5 | Category only | 0.70 |
| priority-classification | Medium | 8 | Category + Priority | 0.60 |
| full-triage | Hard | 10 | Category + Priority + Action | 0.50 |

## Reward Function

Scoring is deterministic and based on ground-truth labels baked into the synthetic dataset.

- Easy: `1.0` if category is correct, `0.0` otherwise.
- Medium: `category × 0.4 + priority × 0.6`. Priority gets partial credit (`0.5`) for being one level off (e.g., predicting `high` when truth is `critical`).
- Hard: `category × 0.3 + priority × 0.4 + action × 0.3`. Same partial credit rule applies to priority.

All step scores are averaged across the episode to produce the final episode score.

## API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| GET | /health | Health check |
| GET | / | Root info |
| POST | /reset | Start new episode |
| POST | /step | Submit triage action |
| GET | /state | Current state |
| GET | /tasks | List all tasks |
| GET | /web | Web UI |

## Setup & Installation

### Option 1: Docker (Recommended)

```bash
docker build -t email-triage-env -f email_triage_env/server/Dockerfile .
docker run -p 7860:7860 email-triage-env
```

### Option 2: Local Python

```bash
pip install -e .
uvicorn email_triage_env.server.app:app --port 7860
```

### Running Inference

```bash
export HF_TOKEN=your_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## Baseline Scores

| Task | Difficulty | Baseline Score (Qwen2.5-72B) |
|------|-----------|-------------------------------|
| category-identification | Easy | ~0.75 |
| priority-classification | Medium | ~0.55 |
| full-triage | Hard | ~0.40 |

## Deployment to HuggingFace Spaces

1. Create a new Space with Docker SDK
2. Push this repository
3. Set `HF_TOKEN` in Space secrets
4. Space auto-deploys and responds at `/health`
