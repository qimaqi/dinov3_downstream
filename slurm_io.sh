#!/bin/bash

echo "Scanning blocked processes..."
echo

printf "%-12s %-15s %-10s %-8s %-6s %-30s %s\n" \
    "JOBID" "USER" "PID" "STATE" "CPU" "WCHAN" "COMMAND"

echo "------------------------------------------------------------------------------------------------------"

for PIDDIR in /proc/[0-9]*; do

    PID=${PIDDIR##*/}

    [ -r "/proc/$PID/status" ] || continue
    [ -r "/proc/$PID/cgroup" ] || continue

    STATE=$(awk '/^State:/ {print $2}' /proc/$PID/status 2>/dev/null)

    # We mainly care about D = uninterruptible I/O sleep
    [ "$STATE" = "D" ] || continue

    USER=$(stat -c '%U' /proc/$PID 2>/dev/null)
    WCHAN=$(cat /proc/$PID/wchan 2>/dev/null)

    CMD=$(tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null)
    [ -z "$CMD" ] && CMD=$(cat /proc/$PID/comm 2>/dev/null)

    # Extract Slurm job ID from cgroup
    JOBID=$(grep -oE 'job[_-]?[0-9]+' /proc/$PID/cgroup 2>/dev/null \
        | head -1 \
        | grep -oE '[0-9]+')

    [ -z "$JOBID" ] && JOBID="-"

    CPU=$(ps -o psr= -p "$PID" 2>/dev/null | tr -d ' ')

    printf "%-12s %-15s %-10s %-8s %-6s %-30s %.80s\n" \
        "$JOBID" "$USER" "$PID" "$STATE" "$CPU" "$WCHAN" "$CMD"

done