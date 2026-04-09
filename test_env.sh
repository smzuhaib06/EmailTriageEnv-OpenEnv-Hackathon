#!/usr/bin/env bash
# Quick test — runs a full episode against the live HF Space

BASE="https://smzu890-email-triage-env.hf.space"
TASK="${1:-category-identification}"

echo "=== Starting task: $TASK ==="

# Reset and get first email
OBS=$(curl -s -X POST "$BASE/reset" \
  -H "Content-Type: application/json" \
  -d "{\"task_name\": \"$TASK\"}")

echo "First email: $(echo $OBS | python3 -c "import sys,json; o=json.load(sys.stdin); print(o['subject'])")"

DONE="false"
STEP=0

while [ "$DONE" = "false" ]; do
  STEP=$((STEP + 1))
  EMAIL_ID=$(echo $OBS | python3 -c "import sys,json; print(json.load(sys.stdin)['email_id'])")
  SUBJECT=$(echo $OBS  | python3 -c "import sys,json; print(json.load(sys.stdin)['subject'])")
  BODY=$(echo $OBS     | python3 -c "import sys,json; print(json.load(sys.stdin)['body'])")

  echo ""
  echo "--- Step $STEP | $EMAIL_ID ---"
  echo "Subject: $SUBJECT"

  # Simple rule-based triage (replace with your logic)
  CATEGORY="general"
  PRIORITY="medium"
  ACTION="respond"

  echo "$BODY $SUBJECT" | grep -qi "payment\|charge\|invoice\|billing\|refund\|subscription" && CATEGORY="billing"
  echo "$BODY $SUBJECT" | grep -qi "error\|bug\|crash\|not working\|broken\|technical\|login\|password" && CATEGORY="technical"
  echo "$BODY $SUBJECT" | grep -qi "offer\|discount\|unsubscribe\|click here\|free\|win\|prize" && CATEGORY="spam"

  echo "$BODY $SUBJECT" | grep -qi "urgent\|critical\|immediately\|asap\|emergency" && PRIORITY="critical"
  echo "$BODY $SUBJECT" | grep -qi "today\|soon\|important\|negative\|declined" && PRIORITY="high"

  [ "$CATEGORY" = "spam" ] && ACTION="delete"
  [ "$PRIORITY" = "critical" ] && ACTION="escalate"

  echo "Decision: category=$CATEGORY priority=$PRIORITY action=$ACTION"

  RESULT=$(curl -s -X POST "$BASE/step" \
    -H "Content-Type: application/json" \
    -d "{\"email_id\":\"$EMAIL_ID\",\"priority\":\"$PRIORITY\",\"category\":\"$CATEGORY\",\"action\":\"$ACTION\",\"reasoning\":\"rule-based\"}")

  REWARD=$(echo $RESULT | python3 -c "import sys,json; print(json.load(sys.stdin)['reward']['value'])")
  DONE=$(echo $RESULT   | python3 -c "import sys,json; print(str(json.load(sys.stdin)['done']).lower())")
  OBS=$(echo $RESULT    | python3 -c "import sys,json; print(__import__('json').dumps(json.load(sys.stdin)['observation']))")

  echo "Reward: $REWARD"
done

echo ""
echo "=== Episode complete ==="
curl -s "$BASE/state" | python3 -c "
import sys, json
s = json.load(sys.stdin)
print(f\"Final score : {s['current_score']:.2f}\")
print(f\"Emails done : {s['emails_processed']}/{s['queue_length']}\")
print(f\"Scores      : {s['scores_history']}\")
"
