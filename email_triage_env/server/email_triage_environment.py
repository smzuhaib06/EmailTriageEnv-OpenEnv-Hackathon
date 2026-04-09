"""
email_triage_environment.py - Core environment state machine.
Pure logic, no HTTP. Called by the FastAPI app.
"""
from .data_generator import generate_email_queue
from .tasks import TASK_MAP
from ..models import EmailObservation, TriageAction, TriageReward

VALID_PRIORITIES = {"critical", "high", "medium", "low"}
VALID_CATEGORIES = {"billing", "technical", "general", "spam"}
VALID_ACTIONS    = {"respond", "escalate", "archive", "delete"}


class EmailTriageEnvironment:
    """
    Core environment logic — no HTTP here, just pure state machine.
    Called by the FastAPI app.
    """

    def __init__(self):
        self.task_name   = "category-identification"
        self.task_config = TASK_MAP[self.task_name]
        self.email_queue: list  = []
        self.current_index: int = 0
        self.scores: list       = []
        self.done: bool         = False
        self.episode_id: int    = 0
        self.step_count: int    = 0

    def reset(self, task_name: str = "category-identification") -> EmailObservation:
        """Reset to a clean episode. Returns first email observation."""
        if task_name not in TASK_MAP:
            raise ValueError(f"Unknown task: {task_name}. Valid: {list(TASK_MAP.keys())}")

        self.task_name   = task_name
        self.task_config = TASK_MAP[task_name]
        n = self.task_config["n_emails"]

        adversarial = self.task_config.get("adversarial", False)
        self.email_queue   = generate_email_queue(n=n, seed=42, adversarial=adversarial)
        self.current_index = 0
        self.scores        = []
        self.done          = False
        self.episode_id   += 1
        self.step_count    = 0

        return self._make_observation()

    def step(self, action: TriageAction) -> tuple[EmailObservation, TriageReward, bool, dict]:
        """Process one triage action. Returns (observation, reward, done, info)."""
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if self.current_index >= len(self.email_queue):
            self.done = True
            raise RuntimeError("No more emails. Episode complete.")

        current_email = self.email_queue[self.current_index]

        # Validate action fields — penalise invalid values
        invalid = False
        if action.priority.lower() not in VALID_PRIORITIES:
            invalid = True
        if action.category.lower() not in VALID_CATEGORIES:
            invalid = True
        if action.action.lower() not in VALID_ACTIONS:
            invalid = True

        if invalid:
            step_score = 0.0
            breakdown  = {"error": "invalid_action_fields"}
        else:
            grader = self.task_config["grader"]
            agent_dict = {
                "priority": action.priority.lower(),
                "category": action.category.lower(),
                "action":   action.action.lower(),
            }
            step_score, breakdown = grader(agent_dict, current_email)

        self.scores.append(step_score)
        self.step_count    += 1
        self.current_index += 1

        done      = self.current_index >= len(self.email_queue)
        self.done = done

        current_avg = sum(self.scores) / len(self.scores)

        reward = TriageReward(
            value=step_score,
            breakdown=breakdown,
            done=done,
            info={
                "email_id": current_email["id"],
                "ground_truth": {
                    "priority": current_email["ground_truth_priority"],
                    "category": current_email["ground_truth_category"],
                    "action":   current_email["ground_truth_action"],
                },
                "step":        self.step_count,
                "running_avg": round(current_avg, 4),
            },
        )

        if done:
            obs = EmailObservation(
                email_id="DONE",
                subject="Episode Complete",
                body=f"All {len(self.email_queue)} emails processed.",
                sender="system",
                timestamp="",
                queue_position=len(self.email_queue),
                total_in_queue=len(self.email_queue),
                emails_processed=len(self.scores),
                current_score=round(current_avg, 4),
                task_name=self.task_name,
            )
        else:
            obs = self._make_observation()

        return obs, reward, done, reward.info

    def state(self) -> dict:
        """Return full current state as a plain dict."""
        return {
            "task_name":        self.task_name,
            "episode_id":       self.episode_id,
            "step_count":       self.step_count,
            "queue_length":     len(self.email_queue),
            "emails_processed": self.current_index,
            "emails_remaining": max(0, len(self.email_queue) - self.current_index),
            "current_score":    round(sum(self.scores) / len(self.scores), 4) if self.scores else 0.0,
            "scores_history":   self.scores,
            "done":             self.done,
            "difficulty":       self.task_config.get("difficulty", "unknown"),
        }

    def _make_observation(self) -> EmailObservation:
        """Build observation from current email in queue."""
        email = self.email_queue[self.current_index]
        return EmailObservation(
            email_id=email["id"],
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            timestamp=email["timestamp"],
            queue_position=self.current_index + 1,
            total_in_queue=len(self.email_queue),
            emails_processed=self.current_index,
            current_score=round(sum(self.scores) / len(self.scores), 4) if self.scores else 0.0,
            task_name=self.task_name,
        )
