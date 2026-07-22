#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
PHASE="${1:?usage: init_experiment.sh <phase0_isaac|phase1_calvin> <short_name>}"
NAME="${2:?usage: init_experiment.sh <phase0_isaac|phase1_calvin> <short_name>}"
STAMP="$(date +%F_%H%M%S)"
RUN_DIR="$ROOT/experiments/$PHASE/${STAMP}_${NAME}"

mkdir -p "$RUN_DIR"/{logs,metrics,artifacts}
cat > "$RUN_DIR/command.sh" <<RUNEOF
#!/usr/bin/env bash
set -euo pipefail

# Fill in the exact command used for this run.
RUNEOF
chmod +x "$RUN_DIR/command.sh"
cat > "$RUN_DIR/notes.md" <<RUNEOF
# ${STAMP}_${NAME}

## Question

## Setup

## Result

## Interpretation

## Next
RUNEOF
cat > "$RUN_DIR/env.txt" <<RUNEOF
created_at: $STAMP
host: $(hostname)
phase: $PHASE
RUNEOF
cat > "$RUN_DIR/git_status.txt" <<RUNEOF
Fill this with the relevant git commit and dirty files before launching the run.
RUNEOF

echo "$RUN_DIR"
