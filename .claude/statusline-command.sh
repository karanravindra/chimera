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

# --- CPU usage: delta of /proc/stat vs cached sample (no sleep = fast) ---
# lxcfs virtualizes /proc/stat, so this reflects the container's cores.
cpu=""
cache=/tmp/claude-statusline-cpustat
read -r _ u n s idle iowait irq softirq steal _ < /proc/stat
total=$((u + n + s + idle + iowait + irq + softirq + steal))
busy=$((total - idle - iowait))
if [ -r "$cache" ]; then
  read -r ptotal pbusy < "$cache"
  dt=$((total - ptotal))
  db=$((busy - pbusy))
  [ "$dt" -gt 0 ] && cpu=$(( (100 * db + dt / 2) / dt ))
fi
printf '%s %s' "$total" "$busy" > "$cache"

# --- RAM usage: match Proxmox = cgroup memory.current / limit (incl. page cache).
# memory.max is "max" (no limit) on this container, so fall back to MemTotal. ---
ram=""
cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null)
lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)
case "$lim" in ''|max) lim=$(( $(awk '/^MemTotal:/{print $2}' /proc/meminfo) * 1024 ));; esac
if [ -n "$cur" ] && [ "$lim" -gt 0 ] 2>/dev/null; then
  ram=$(( (100 * cur + lim / 2) / lim ))
else
  # Fallback if cgroup is unreadable (e.g. not on Proxmox/cgroup v2)
  ram=$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{if(t>0) printf "%d", (t-a)*100/t}' /proc/meminfo)
fi

# --- Disk usage of the project's filesystem (df is a quick read) ---
disk=$(df -P "$proj_dir" 2>/dev/null | awk 'NR==2{gsub(/%/,"",$5); print $5}')

# --- Colors (ANSI). Gauges (context, cpu, ram, disk) are gray so they recede;
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
[ -n "$cpu" ] && rparts+=("${GRAY}cpu ${cpu}%${R}")
[ -n "$ram" ] && rparts+=("${GRAY}ram ${ram}%${R}")
[ -n "$disk" ] && rparts+=("${GRAY}disk ${disk}%${R}")
right=""
for p in "${rparts[@]}"; do [ -z "$right" ] && right="$p" || right="$right${SEP}$p"; done

# --- Pad so the right group sits at the terminal's right edge.
# Claude Code sets COLUMNS to the terminal width before running this script. ---
cols=${COLUMNS:-80}
gap=$(( cols - $(vislen "$left") - $(vislen "$right") ))
[ "$gap" -lt 1 ] && gap=1
printf '%s%*s%s' "$left" "$gap" "" "$right"
