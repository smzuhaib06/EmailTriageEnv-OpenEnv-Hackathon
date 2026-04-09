"""
tasks.py - Task definitions for EmailTriageEnv.
"""
from .graders import grade_easy, grade_medium, grade_hard, grade_adversarial

TASKS = [
    {
        "name": "category-identification",
        "description": "Identify the correct category for each email (billing/technical/general/spam)",
        "difficulty": "easy",
        "grader": grade_easy,
        "n_emails": 8,
        "success_threshold": 0.7,
        "adversarial": False,
    },
    {
        "name": "priority-classification",
        "description": "Identify both the category and urgency priority for each email",
        "difficulty": "medium",
        "grader": grade_medium,
        "n_emails": 12,
        "success_threshold": 0.6,
        "adversarial": False,
    },
    {
        "name": "full-triage",
        "description": "Complete triage: assign category, priority, and recommended action",
        "difficulty": "hard",
        "grader": grade_hard,
        "n_emails": 15,
        "success_threshold": 0.5,
        "adversarial": False,
    },
    {
        "name": "adversarial-triage",
        "description": (
            "Full triage with adversarial emails: spam disguised as billing/technical issues, "
            "and legitimate urgent emails that look like spam. "
            "Bonus for catching disguised spam, penalty for falling for it."
        ),
        "difficulty": "expert",
        "grader": grade_adversarial,
        "n_emails": 15,
        "success_threshold": 0.45,
        "adversarial": True,
    },
]

TASK_MAP = {t["name"]: t for t in TASKS}
