#!/usr/bin/env bash
# Pause, resume, or inspect one background run started by run_server_background.sh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-status}"
PROTOCOL="${2:-uda}"

case "$ACTION" in
  pause|hold|resume|terminate|status) ;;
  *)
    echo "usage: bash scripts/control_server_run.sh {pause|hold|resume|terminate|status} [protocol]" >&2
    exit 2
    ;;
esac
case "$PROTOCOL" in
  sda|jmda_uda|jmda_oracle_shared_uda|uda|supervised) ;;
  *)
    echo "unknown protocol: $PROTOCOL" >&2
    exit 2
    ;;
esac

LAUNCH_DIR="$HERE/server_logs/launcher"
if [ ! -d "$LAUNCH_DIR" ]; then
  echo "launcher log directory does not exist: $LAUNCH_DIR" >&2
  exit 1
fi

PID_FILE=""
while IFS= read -r candidate; do
  [ -n "$candidate" ] || continue
  candidate_pid="$(tr -d '[:space:]' < "$candidate")"
  if [[ "$candidate_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$candidate_pid" 2>/dev/null; then
    PID_FILE="$candidate"
    break
  fi
done < <(find "$LAUNCH_DIR" -maxdepth 1 -type f -name "${PROTOCOL}_*.pid" \
  -printf '%T@ %p\n' | sort -nr | cut -d' ' -f2-)

if [ -z "$PID_FILE" ]; then
  echo "no live $PROTOCOL background run found under $LAUNCH_DIR" >&2
  exit 1
fi

ROOT_PID="$(tr -d '[:space:]' < "$PID_FILE")"
ROOT_CMD="$(ps -p "$ROOT_PID" -o args= || true)"
EXPECTED="$HERE/scripts/run_server.sh $PROTOCOL"
if [[ "$ROOT_CMD" != *"$EXPECTED"* ]]; then
  echo "refusing to control PID $ROOT_PID: unexpected command" >&2
  echo "expected to contain: $EXPECTED" >&2
  echo "actual: $ROOT_CMD" >&2
  exit 1
fi

collect_tree() {
  local root="$1" changed=1 pid ppid
  local -A selected=(["$root"]=1)
  local snapshot
  snapshot="$(ps -e -o pid=,ppid=)"
  while [ "$changed" -eq 1 ]; do
    changed=0
    while read -r pid ppid; do
      [ -n "${pid:-}" ] || continue
      if [ -n "${selected[$ppid]:-}" ] && [ -z "${selected[$pid]:-}" ]; then
        selected["$pid"]=1
        changed=1
      fi
    done <<< "$snapshot"
  done
  printf '%s\n' "${!selected[@]}" | sort -n
}

show_tree() {
  local pids
  pids="$(collect_tree "$ROOT_PID" | paste -sd, -)"
  echo "PID file: $PID_FILE"
  echo "Root PID: $ROOT_PID"
  ps -o pid,ppid,stat,etime,%cpu,%mem,args -p "$pids"
}

case "$ACTION" in
  terminate)
    # Stop the orchestrator before terminating descendants so it cannot launch
    # another method while the current process tree is being removed.
    kill -STOP "$ROOT_PID" 2>/dev/null || true
    mapfile -t TREE_PIDS < <(collect_tree "$ROOT_PID")
    for pid in "${TREE_PIDS[@]}"; do
      [ "$pid" = "$ROOT_PID" ] || kill -TERM "$pid" 2>/dev/null || true
    done

    # Allow normal CUDA/Python cleanup, excluding zombies which retain no
    # resources and will be reaped when the stopped orchestrator is removed.
    for _ in $(seq 1 20); do
      remaining=0
      for pid in "${TREE_PIDS[@]}"; do
        [ "$pid" = "$ROOT_PID" ] && continue
        stat="$(ps -p "$pid" -o stat= 2>/dev/null | tr -d '[:space:]' || true)"
        if [ -n "$stat" ] && [[ "$stat" != Z* ]]; then
          remaining=1
          break
        fi
      done
      [ "$remaining" -eq 0 ] && break
      sleep 0.5
    done

    for pid in "${TREE_PIDS[@]}"; do
      [ "$pid" = "$ROOT_PID" ] || kill -KILL "$pid" 2>/dev/null || true
    done
    kill -KILL "$ROOT_PID" 2>/dev/null || true
    echo "[TERMINATED] $PROTOCOL run and descendants"
    echo "Logs and checkpoints already written on disk were preserved."
    ;;
  hold)
    # Freeze only the sequential orchestrator. The current trainer may finish
    # and release CUDA memory, but no following method can be launched.
    kill -STOP "$ROOT_PID"
    echo "[HELD] $PROTOCOL orchestrator"
    echo "The current trainer continues, but no next method can start."
    echo "After the current trainer exits, its CUDA memory will be released."
    show_tree
    ;;
  pause)
    # Stop the orchestrator first so it cannot launch the next method while its
    # current child tree is being frozen.
    kill -STOP "$ROOT_PID"
    mapfile -t TREE_PIDS < <(collect_tree "$ROOT_PID")
    for pid in "${TREE_PIDS[@]}"; do
      [ "$pid" = "$ROOT_PID" ] || kill -STOP "$pid" 2>/dev/null || true
    done
    echo "[PAUSED] $PROTOCOL run and descendants"
    echo "Note: SIGSTOP preserves CUDA allocations; GPU memory is not released."
    show_tree
    ;;
  resume)
    mapfile -t TREE_PIDS < <(collect_tree "$ROOT_PID")
    # Resume workers/current trainer before the waiting orchestrator.
    for pid in "${TREE_PIDS[@]}"; do
      [ "$pid" = "$ROOT_PID" ] || kill -CONT "$pid" 2>/dev/null || true
    done
    kill -CONT "$ROOT_PID"
    echo "[RESUMED] $PROTOCOL run and descendants"
    show_tree
    ;;
  status)
    show_tree
    ;;
esac
