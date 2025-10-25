#!/bin/bash

# Script to initialize a WandB sweep and start an agent.

# --- Configuration ---
DEFAULT_ENTITY="a530029885-southeast-university" # CHANGE THIS
DEFAULT_PROJECT_PREFIX="GridSE"
DEFAULT_COUNT=20 # Number of trials for the agent

# --- Argument Parsing ---
SWEEP_CONFIG_YAML=""
ENTITY=""
PROJECT=""
COUNT=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) SWEEP_CONFIG_YAML="$2"; shift ;;
        --entity) ENTITY="$2"; shift ;;
        --project) PROJECT="$2"; shift ;;
        --count) COUNT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Validate Input ---
if [ -z "$SWEEP_CONFIG_YAML" ]; then
  echo "Error: Sweep configuration YAML file must be provided via --config <path>"
  exit 1
fi
if [ ! -f "$SWEEP_CONFIG_YAML" ]; then
  echo "Error: File not found: $SWEEP_CONFIG_YAML"
  exit 1
fi

# Use default entity/project/count if not provided
ENTITY=${ENTITY:-$DEFAULT_ENTITY}
# Infer project from yaml name if not provided (e.g., pgr_sweep.yaml -> GridSE_PGR)
if [ -z "$PROJECT" ]; then
  BASENAME=$(basename "$SWEEP_CONFIG_YAML" .yaml)
  MODEL_NAME=$(echo "$BASENAME" | sed 's/_sweep//') # Extract model name
  PROJECT="${DEFAULT_PROJECT_PREFIX}_${MODEL_NAME^}" # Capitalize first letter
fi
COUNT=${COUNT:-$DEFAULT_COUNT}

# --- Login to WandB (optional, assumes already logged in) ---
# wandb login YOUR_API_KEY

# --- Initialize Sweep (robust parsing) ---
echo "Initializing WandB sweep..."
echo "Config: $SWEEP_CONFIG_YAML"
echo "Entity: $ENTITY"
echo "Project: $PROJECT"

# 1) 也抓取 stderr，因为 wandb 把提示信息写在 stderr
SWEEP_OUTPUT=$(wandb sweep "$SWEEP_CONFIG_YAML" --project "$PROJECT" --entity "$ENTITY" 2>&1)

# 2) 去掉 ANSI 颜色码，避免 grep 失败
SWEEP_OUTPUT_CLEAN=$(echo "$SWEEP_OUTPUT" | sed -r 's/\x1B\[[0-9;]*[mK]//g')

# 3) 优先从 “Run sweep agent with:” 这一行提取 agent 路径（最稳定）
AGENT_LINE=$(echo "$SWEEP_OUTPUT_CLEAN" | grep -E 'Run sweep agent with:')
AGENT_PATH=$(echo "$AGENT_LINE" | awk '{print $NF}')  # 结果形如 entity/project/sweep_id

# 4) 兜底：从 “View sweep at:” 的 URL 解析出 sweep_id
if [ -z "$AGENT_PATH" ]; then
  SWEEP_URL=$(echo "$SWEEP_OUTPUT_CLEAN" | grep -oE 'https://wandb.ai/[^ ]+/sweeps/[^ ]+')
  SWEEP_ID=$(echo "$SWEEP_URL" | awk -F'/' '{print $NF}')
  if [ -n "$SWEEP_ID" ]; then
    AGENT_PATH="$ENTITY/$PROJECT/$SWEEP_ID"
  fi
fi

if [ -z "$AGENT_PATH" ]; then
  echo "Error: Failed to create sweep or extract Sweep ID."
  echo "Output from wandb sweep:"
  echo "$SWEEP_OUTPUT"
  exit 1
fi

echo "Sweep created successfully!"
echo "Sweep Output (cleaned):"
echo "$SWEEP_OUTPUT_CLEAN"
echo "Starting WandB agent for $COUNT trials..."
wandb agent "$AGENT_PATH" --count "$COUNT"
echo "Agent finished."
