"""
graders.py - Scoring functions for each task difficulty level.
All graders return (float, dict) tuples.
"""

PRIORITY_ORDER = ["critical", "high", "medium", "low"]


def grade_easy(agent_action: dict, email: dict) -> tuple[float, dict]:
    """
    EASY: Only grade category identification.
    Returns (score, breakdown_dict)
    """
    correct_category = email["ground_truth_category"]
    agent_category = agent_action.get("category", "").lower().strip()

    score = 1.0 if agent_category == correct_category else 0.0
    return score, {"category": score}


def grade_medium(agent_action: dict, email: dict) -> tuple[float, dict]:
    """
    MEDIUM: Grade category (40%) + priority (60%).
    Priority has partial credit for adjacent levels.
    Returns (score, breakdown_dict)
    """
    correct_category = email["ground_truth_category"]
    correct_priority = email["ground_truth_priority"]

    agent_category = agent_action.get("category", "").lower().strip()
    agent_priority = agent_action.get("priority", "").lower().strip()

    cat_score = 1.0 if agent_category == correct_category else 0.0

    if agent_priority == correct_priority:
        pri_score = 1.0
    elif agent_priority in PRIORITY_ORDER and correct_priority in PRIORITY_ORDER:
        diff = abs(PRIORITY_ORDER.index(agent_priority) - PRIORITY_ORDER.index(correct_priority))
        pri_score = 0.5 if diff == 1 else 0.0
    else:
        pri_score = 0.0

    total = (cat_score * 0.4) + (pri_score * 0.6)
    return round(total, 4), {"category": cat_score, "priority": pri_score}


def grade_hard(agent_action: dict, email: dict) -> tuple[float, dict]:
    """
    HARD: Grade category (30%) + priority (40%) + action (30%).
    Priority still has partial credit for adjacent levels.
    Returns (score, breakdown_dict)
    """
    correct_category = email["ground_truth_category"]
    correct_priority = email["ground_truth_priority"]
    correct_action = email["ground_truth_action"]

    agent_category = agent_action.get("category", "").lower().strip()
    agent_priority = agent_action.get("priority", "").lower().strip()
    agent_action_val = agent_action.get("action", "").lower().strip()

    cat_score = 1.0 if agent_category == correct_category else 0.0
    act_score = 1.0 if agent_action_val == correct_action else 0.0

    if agent_priority == correct_priority:
        pri_score = 1.0
    elif agent_priority in PRIORITY_ORDER and correct_priority in PRIORITY_ORDER:
        diff = abs(PRIORITY_ORDER.index(agent_priority) - PRIORITY_ORDER.index(correct_priority))
        pri_score = 0.5 if diff == 1 else 0.0
    else:
        pri_score = 0.0

    total = (cat_score * 0.3) + (pri_score * 0.4) + (act_score * 0.3)
    return round(total, 4), {"category": cat_score, "priority": pri_score, "action": act_score}


def grade_adversarial(agent_action: dict, email: dict) -> tuple[float, dict]:
    """
    ADVERSARIAL: Full triage with bonus/penalty for adversarial emails.
    - Correctly identifying disguised spam: +0.2 bonus (capped at 1.0)
    - Falling for spam disguised as billing/technical: -0.3 penalty
    - Legitimate emails mistakenly marked as spam: -0.2 penalty
    Base scoring same as hard: category(30%) + priority(40%) + action(30%)
    """
    correct_category = email["ground_truth_category"]
    correct_priority = email["ground_truth_priority"]
    correct_action   = email["ground_truth_action"]
    is_adversarial   = email.get("adversarial", False)
    adv_type         = email.get("adversarial_type", "")

    agent_category = agent_action.get("category", "").lower().strip()
    agent_priority = agent_action.get("priority", "").lower().strip()
    agent_action_val = agent_action.get("action", "").lower().strip()

    cat_score = 1.0 if agent_category == correct_category else 0.0
    act_score = 1.0 if agent_action_val == correct_action else 0.0

    if agent_priority == correct_priority:
        pri_score = 1.0
    elif agent_priority in PRIORITY_ORDER and correct_priority in PRIORITY_ORDER:
        diff = abs(PRIORITY_ORDER.index(agent_priority) - PRIORITY_ORDER.index(correct_priority))
        pri_score = 0.5 if diff == 1 else 0.0
    else:
        pri_score = 0.0

    base = (cat_score * 0.3) + (pri_score * 0.4) + (act_score * 0.3)

    # Adversarial bonus/penalty
    bonus = 0.0
    if is_adversarial:
        if adv_type in ("spam_as_billing", "spam_as_technical"):
            if agent_category == "spam":
                bonus = 0.2   # correctly caught disguised spam
            else:
                bonus = -0.3  # fell for the disguise
        elif adv_type == "legitimate_looks_like_spam":
            if agent_category == "spam" and correct_category != "spam":
                bonus = -0.2  # wrongly flagged legitimate email as spam

    total = round(min(1.0, max(0.0, base + bonus)), 4)
    return total, {
        "category": cat_score,
        "priority": pri_score,
        "action": act_score,
        "adversarial_bonus": bonus,
    }
