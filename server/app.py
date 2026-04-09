"""
app.py - FastAPI application for EmailTriageEnv.
Single global environment instance, CORS enabled, lifespan managed.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from email_triage_env.models import EmailObservation, TriageAction, TriageReward
from email_triage_env.server.email_triage_environment import EmailTriageEnvironment
from email_triage_env.server.tasks import TASKS

# ── Global environment instance ───────────────────────────────────────────────
env = EmailTriageEnvironment()


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pre-warm with default task
    env.reset("category-identification")
    yield
    # Shutdown: nothing to clean up


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EmailTriageEnv",
    description="OpenEnv environment for AI email triage agents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception handlers ────────────────────────────────────────────────────────
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    raise HTTPException(status_code=400, detail=str(exc))


# ── Request models ────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_name: str = "category-identification"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "env": "email-triage-env", "version": "1.0.0"}


@app.get("/")
def root():
    return {"message": "EmailTriageEnv", "docs": "/docs", "health": "/health"}


@app.post("/reset", response_model=EmailObservation)
def reset(request: ResetRequest = ResetRequest()):
    try:
        obs = env.reset(request.task_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return obs


@app.post("/step")
def step(action: TriageAction):
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }


@app.get("/state")
def state():
    return env.state()


@app.get("/tasks")
def list_tasks():
    # Exclude the grader callable — not JSON serialisable
    return [
        {
            "name":              t["name"],
            "description":       t["description"],
            "difficulty":        t["difficulty"],
            "n_emails":          t["n_emails"],
            "success_threshold": t["success_threshold"],
        }
        for t in TASKS
    ]


@app.get("/web", response_class=HTMLResponse)
def web():
    """Simple HTML status page — satisfies base_path: /web in openenv.yaml."""
    s = env.state()
    task_rows = "".join(
        f"<tr><td>{t['name']}</td><td>{t['difficulty']}</td>"
        f"<td>{t['n_emails']}</td><td>{t['success_threshold']}</td></tr>"
        for t in TASKS
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EmailTriageEnv</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 40px auto; padding: 0 20px; background: #f8fafc; color: #1e293b; }}
    h1   {{ color: #2563eb; }}
    .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.8em; background: #dbeafe; color: #1d4ed8; }}
    .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin: 16px 0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #e2e8f0; }}
    th {{ background: #f1f5f9; }}
    .stat {{ font-size: 1.4em; font-weight: bold; color: #2563eb; }}
  </style>
</head>
<body>
  <h1>📧 EmailTriageEnv <span class="badge">v1.0.0</span></h1>
  <p>A realistic email triage environment where AI agents sort, prioritise, and route customer support emails.</p>

  <div class="card">
    <h2>Current Episode State</h2>
    <table>
      <tr><th>Field</th><th>Value</th></tr>
      <tr><td>Task</td><td>{s['task_name']}</td></tr>
      <tr><td>Difficulty</td><td>{s['difficulty']}</td></tr>
      <tr><td>Episode ID</td><td>{s['episode_id']}</td></tr>
      <tr><td>Emails Processed</td><td>{s['emails_processed']} / {s['queue_length']}</td></tr>
      <tr><td>Emails Remaining</td><td>{s['emails_remaining']}</td></tr>
      <tr><td>Current Score</td><td><span class="stat">{s['current_score']:.2f}</span></td></tr>
      <tr><td>Done</td><td>{'✅ Yes' if s['done'] else '⏳ No'}</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Available Tasks</h2>
    <table>
      <tr><th>Name</th><th>Difficulty</th><th>Emails</th><th>Success Threshold</th></tr>
      {task_rows}
    </table>
  </div>

  <div class="card">
    <h2>API Endpoints</h2>
    <table>
      <tr><th>Method</th><th>Path</th><th>Description</th></tr>
      <tr><td>GET</td><td><a href="/health">/health</a></td><td>Health check</td></tr>
      <tr><td>GET</td><td><a href="/">/</a></td><td>Root info</td></tr>
      <tr><td>POST</td><td>/reset</td><td>Start new episode</td></tr>
      <tr><td>POST</td><td>/step</td><td>Submit triage action</td></tr>
      <tr><td>GET</td><td><a href="/state">/state</a></td><td>Current state</td></tr>
      <tr><td>GET</td><td><a href="/tasks">/tasks</a></td><td>List all tasks</td></tr>
      <tr><td>GET</td><td><a href="/docs">/docs</a></td><td>Interactive API docs</td></tr>
    </table>
  </div>
</body>
</html>"""
    return html


def main():
    import uvicorn
    uvicorn.run("email_triage_env.server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
