"""
inference.py - EmailTriageEnv Baseline Inference Script

Runs all 4 tasks sequentially against the environment server.
Emits exact [START], [STEP], [END] log lines to stdout.

Required env vars:
  HF_TOKEN      - HuggingFace token (used as API key)
  API_BASE_URL  - LLM endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME    - Model to use (default: Qwen/Qwen2.5-72B-Instruct)
  SERVER_URL    - Environment server URL (default: http://localhost:7860)
"""
import os
import sys
import json
import time
import subprocess
import httpx
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")   # no default — must be set by caller
SERVER_URL   = os.getenv("SERVER_URL", "http://localhost:7860")
BENCHMARK    = "email-triage-env"
MAX_STEPS    = 20
TEMPERATURE  = 0.3
MAX_TOKENS   = 200
SUCCESS_THRESHOLD = 0.5

TASKS = [
    "category-identification",
    "priority-classification",
    "full-triage",
    "adversarial-triage",
]

DEFAULT_ACTION = {
    "priority":  "low",
    "category":  "general",
    "action":    "archive",
    "reasoning": "parse_error",
}

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(obs: dict, task_name: str) -> str:
    task_instructions = {
        "category-identification": (
            "Your job is to identify the CATEGORY of this email.\n"
            "Valid categories: billing, technical, general, spam\n"
            "Set priority='low', action='archive' as defaults (not graded)."
        ),
        "priority-classification": (
            "Your job is to identify the CATEGORY and PRIORITY of this email.\n"
            "Valid categories: billing, technical, general, spam\n"
            "Valid priorities: critical, high, medium, low\n"
            "Set action='archive' as default (not graded in this task)."
        ),
        "full-triage": (
            "Your job is to fully triage this email.\n"
            "Valid categories: billing, technical, general, spam\n"
            "Valid priorities: critical, high, medium, low\n"
            "Valid actions: respond, escalate, archive, delete\n"
            "- respond: needs a reply\n"
            "- escalate: urgent, needs senior attention\n"
            "- archive: low value, keep for records\n"
            "- delete: spam or junk"
        ),
        "adversarial-triage": (
            "Your job is to fully triage this email. WARNING: some emails are adversarial.\n"
            "Spam may be disguised as billing issues or security alerts.\n"
            "Legitimate urgent emails may look like spam.\n"
            "Check sender domain carefully — legitimate emails come from @ourplatform.com.\n"
            "Suspicious domains (e.g. .xyz, .ru, .net with random names) are likely spam.\n"
            "Valid categories: billing, technical, general, spam\n"
            "Valid priorities: critical, high, medium, low\n"
            "Valid actions: respond, escalate, archive, delete"
        ),
    }

    return f"""You are an expert email triage agent. Analyze the email below and respond ONLY with valid JSON.

Task: {task_instructions.get(task_name, '')}

Email:
ID: {obs.get('email_id', '')}
From: {obs.get('sender', '')}
Subject: {obs.get('subject', '')}
Body: {obs.get('body', '')}
Queue position: {obs.get('queue_position', '?')} of {obs.get('total_in_queue', '?')}

Respond ONLY with this exact JSON (no markdown, no explanation):
{{
  "email_id": "{obs.get('email_id', '')}",
  "priority": "<critical|high|medium|low>",
  "category": "<billing|technical|general|spam>",
  "action": "<respond|escalate|archive|delete>",
  "reasoning": "<one sentence explanation>"
}}"""


# ── LLM call ──────────────────────────────────────────────────────────────────
def call_llm(prompt: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        return {**DEFAULT_ACTION, "reasoning": f"llm_error: {str(e)[:50]}"}


# ── HTTP helpers ──────────────────────────────────────────────────────────────
def server_reset(task_name: str) -> dict:
    r = httpx.post(f"{SERVER_URL}/reset", json={"task_name": task_name}, timeout=10)
    r.raise_for_status()
    return r.json()


def server_step(action: dict) -> dict:
    r = httpx.post(f"{SERVER_URL}/step", json=action, timeout=10)
    r.raise_for_status()
    return r.json()


# ── Run one task episode ──────────────────────────────────────────────────────
def run_task(task_name: str) -> dict:
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    obs       = server_reset(task_name)
    rewards   = []
    step      = 0
    done      = False
    last_error = "null"

    valid_priorities = {"critical", "high", "medium", "low"}
    valid_categories = {"billing", "technical", "general", "spam"}
    valid_actions    = {"respond", "escalate", "archive", "delete"}

    while not done and step < MAX_STEPS:
        step += 1
        last_error = "null"

        if obs.get("email_id") == "DONE":
            break

        prompt      = build_prompt(obs, task_name)
        action_dict = call_llm(prompt)

        # Ensure email_id is set
        action_dict["email_id"] = obs.get("email_id", "unknown")

        # Validate and sanitise fields
        if action_dict.get("priority", "").lower() not in valid_priorities:
            action_dict["priority"] = "low"
            last_error = "invalid_priority"
        if action_dict.get("category", "").lower() not in valid_categories:
            action_dict["category"] = "general"
            last_error = "invalid_category"
        if action_dict.get("action", "").lower() not in valid_actions:
            action_dict["action"] = "archive"
            last_error = "invalid_action"

        try:
            result     = server_step(action_dict)
            reward_val = result["reward"]["value"]
            done       = result["done"]
            obs        = result["observation"]
            last_error = "null"
        except Exception as e:
            reward_val = 0.0
            done       = True
            last_error = str(e)[:50]

        rewards.append(reward_val)

        action_str = (
            f"category={action_dict.get('category', '?')},"
            f"priority={action_dict.get('priority', '?')},"
            f"action={action_dict.get('action', '?')}"
        )
        print(
            f"[STEP] step={step} action={action_str} "
            f"reward={reward_val:.2f} done={str(done).lower()} "
            f"error={last_error}",
            flush=True,
        )

    score   = round(sum(rewards) / len(rewards), 2) if rewards else 0.0
    success = score >= SUCCESS_THRESHOLD
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={step} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

    return {"task": task_name, "score": score, "steps": step, "success": success}


# ── Server management ─────────────────────────────────────────────────────────
def start_server() -> subprocess.Popen:
    """Start the FastAPI server as a subprocess."""
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "email_triage_env.server.app:app",
            "--host", "0.0.0.0",
            "--port", "7860",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for server to be ready
    for _ in range(20):
        time.sleep(1)
        try:
            r = httpx.get(f"{SERVER_URL}/health", timeout=2)
            if r.status_code == 200:
                return proc
        except Exception:
            pass

    raise RuntimeError("Server failed to start within 20 seconds")


def stop_server(proc: subprocess.Popen):
    """Gracefully stop the server."""
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    server_proc = None
    all_results = []
    use_local   = SERVER_URL.startswith("http://localhost") or SERVER_URL.startswith("http://127.")

    try:
        if use_local:
            server_proc = start_server()

        for task_name in TASKS:
            result = run_task(task_name)
            all_results.append(result)
            time.sleep(1)

    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)

    finally:
        if server_proc:
            stop_server(server_proc)

    print("\n=== FINAL SUMMARY ===", file=sys.stderr)
    for r in all_results:
        print(f"  {r['task']}: score={r['score']:.2f} success={r['success']}", file=sys.stderr)


if __name__ == "__main__":
    main()
