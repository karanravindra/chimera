#!/usr/bin/env bash
input=$(cat)

# --- Fields from the Claude Code JSON payload ---
proj_dir=$(echo "$input" | jq -r '.workspace.project_dir // .cwd')
proj=$(basename "$proj_dir")
model=$(echo "$input" | jq -r '.model.display_name')
used=$(echo "$input" | jq -r '.context_window.used_percentage // empty')
cost=$(echo "$input" | jq -r '.cost.total_cost_usd // empty')
added=$(echo "$input" | jq -r '.cost.total_lines_added // 0')
removed=$(echo "$input" | jq -r '.cost.total_lines_removed // 0')

# --- Git branch (skip optional locks to avoid contention) ---
branch=$(git -C "$proj_dir" --no-optional-locks symbolic-ref --short HEAD 2>/dev/null)

# --- Disk usage of the project's filesystem (df is a quick read) ---
disk=$(df -P "$proj_dir" 2>/dev/null | awk 'NR==2{gsub(/%/,"",$5); print $5}')

# --- GPU memory via nvidia-smi (guarded; summed across GPUs) ---
vram=""
if command -v nvidia-smi >/dev/null 2>&1; then
  vram=$(nvidia-smi --query-gpu=memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null | awk -F',' '
      {used+=$1; tot+=$2}
      END{if(tot>0) printf "%d", (100*used+tot/2)/tot}')
fi

# --- Colors (ANSI). Gauges (context, vram, disk) are gray so they recede;
# the diff line count is the only colored signal: green added / red removed. ---
R=$'\033[0m'; DIM=$'\033[2m'; GRAY=$'\033[90m'; BOLD=$'\033[1m'
GREEN=$'\033[32m'; RED=$'\033[31m'; CYAN=$'\033[36m'; MAGENTA=$'\033[35m'
SEP="${DIM} | ${R}"

# Visible length of a string, ignoring ANSI escape codes
vislen() { local s; s=$(printf '%s' "$1" | sed 's/\x1b\[[0-9;]*m//g'); printf '%s' "${#s}"; }

# --- Left group: identity ---
left="${BOLD}${CYAN}${proj}${R}"
[ -n "$branch" ] && left="$left ${MAGENTA}[$branch]${R}"
left="$left${SEP}${model}"

# --- Right group: metrics (joined with SEP) ---
rparts=()
[ -n "$used" ] && rparts+=("${GRAY}$(printf '%.0f' "$used")% ctx${R}")
[ -n "$cost" ] && rparts+=("\$$(printf '%.2f' "$cost")")
rparts+=("${GREEN}+$added${R}/${RED}-$removed${R}")
[ -n "$vram" ] && rparts+=("${GRAY}vram ${vram}%${R}")
[ -n "$disk" ] && rparts+=("${GRAY}disk ${disk}%${R}")
right=""
for p in "${rparts[@]}"; do [ -z "$right" ] && right="$p" || right="$right${SEP}$p"; done

# --- Pad so the right group sits at the terminal's right edge.
# Claude Code sets COLUMNS to the terminal width before running this script. ---
# Reserve one trailing column so the terminal doesn't clip the rightmost
# field (e.g. "disk 45%" rendering as "disk…").
cols=${COLUMNS:-80}
gap=$(( cols - 3 - $(vislen "$left") - $(vislen "$right") ))
[ "$gap" -lt 1 ] && gap=1
printf '%s%*s%s   ' "$left" "$gap" "" "$right"