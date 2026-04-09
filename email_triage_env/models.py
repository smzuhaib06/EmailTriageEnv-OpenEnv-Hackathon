"""
models.py - Pydantic v2 models for EmailTriageEnv.
Falls back gracefully if openenv-core is not installed.
"""
try:
    from openenv.core.models import BaseObservation, BaseAction, BaseReward  # type: ignore
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False

from pydantic import BaseModel


class EmailObservation(BaseModel):
    """What the agent sees at each step."""
    email_id: str
    subject: str
    body: str
    sender: str
    timestamp: str
    queue_position: int       # 1-based index of current email
    total_in_queue: int       # total emails in this episode
    emails_processed: int     # how many the agent has already handled
    current_score: float      # running average score so far
    task_name: str            # which task is active


class TriageAction(BaseModel):
    """What the agent submits for each email."""
    email_id: str
    priority: str    # critical / high / medium / low
    category: str    # billing / technical / general / spam
    action: str      # respond / escalate / archive / delete
    reasoning: str = ""  # agent's explanation, not graded


class TriageReward(BaseModel):
    """Reward returned after each step."""
    value: float          # score for this step (0.0 to 1.0)
    breakdown: dict       # {"category": 0.4, "priority": 0.3, "action": 0.3}
    done: bool
    info: dict            # extra metadata
