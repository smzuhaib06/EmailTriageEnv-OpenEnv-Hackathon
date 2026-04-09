"""
client.py - HTTP client for EmailTriageEnv server.
Uses httpx for synchronous calls. Falls back gracefully if openenv-core unavailable.
"""
import httpx
from .models import EmailObservation, TriageAction, TriageReward


class EmailTriageEnvClient:
    """Synchronous HTTP client for the EmailTriageEnv FastAPI server."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def reset(self, task_name: str = "category-identification") -> EmailObservation:
        """Start a new episode and return the first observation."""
        resp = self._client.post("/reset", json={"task_name": task_name})
        resp.raise_for_status()
        return EmailObservation(**resp.json())

    def step(self, action: TriageAction) -> tuple:
        """Submit a triage action and return (obs, reward, done, info)."""
        resp = self._client.post("/step", json=action.model_dump())
        resp.raise_for_status()
        data = resp.json()
        obs = EmailObservation(**data["observation"])
        reward = TriageReward(**data["reward"])
        return obs, reward, data["done"], data["info"]

    def state(self) -> dict:
        """Return the current environment state."""
        resp = self._client.get("/state")
        resp.raise_for_status()
        return resp.json()

    def tasks(self) -> list:
        """Return available task configurations."""
        resp = self._client.get("/tasks")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        """Check server health."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
