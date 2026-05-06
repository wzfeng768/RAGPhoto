#!/bin/bash
# =============================================================================
# Parallel RAGAS Metrics Repair Script
# =============================================================================
# Usage:
# # 默认运行 4 个并行任务
# ./run_repair_parallel.sh

# # 指定 8 个并行任务
# ./run_repair_parallel.sh --max-jobs 8

# # 预览将要运行的命令（不实际执行）
# ./run_repair_parallel.sh --dry-run

# # 只处理特定目录
# ./run_repair_parallel.sh --dirs "gpt-4o-mini--" --max-jobs 8

# # 计算所有指标（不只是 RAGAS）
# ./run_repair_parallel.sh --all-metrics
# =============================================================================

# Default settings
MAX_JOBS=4
DRY_RUN=""
RESULT_DIRS="gpt-4o-mini-- DeepSeek-V3.2-Thinking--"
RAGAS_ONLY="--ragas-only"
EVAL_API_KEY=""
EVAL_API_BASE=""
EVAL_MODEL=""
BACKGROUND_MODE=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --max-jobs) MAX_JOBS="$2"; shift ;;
        --dry-run) DRY_RUN="true" ;;
        --dirs) RESULT_DIRS="$2"; shift ;;
        --all-metrics) RAGAS_ONLY="" ;;
        --custom-only) RAGAS_ONLY="--custom-only" ;;
        --eval-api-key) EVAL_API_KEY="$2"; shift ;;
        --eval-api-base) EVAL_API_BASE="$2"; shift ;;
        --eval-model) EVAL_MODEL="$2"; shift ;;
        --background) BACKGROUND_MODE="true" ;;
        --_background_child) BACKGROUND_MODE="child" ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --max-jobs N         Maximum parallel jobs (default: 4)"
            echo "  --dry-run            Preview what will be run without executing"
            echo "  --dirs \"dir1 dir2\"    Specify result subdirs (default: gpt-4o-mini-- DeepSeek-V3.2-Thinking--)"
            echo "  --all-metrics        Compute all metrics, not just RAGAS"
            echo "  --custom-only        Only compute custom metrics (faithfulness, answer_correctness, answer_relevancy)"
            echo "  --eval-api-key KEY   API key for evaluation model"
            echo "  --eval-api-base URL  API base URL for evaluation model"
            echo "  --eval-model MODEL   Model name for evaluation"
            echo "  --background         Run in background (detached from terminal)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Change to evaluation directory
cd "$(dirname "$0")"

# Handle background mode: restart self with nohup and exit
if [ "$BACKGROUND_MODE" = "true" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BG_LOG="logs/repair_parallel_background_${TIMESTAMP}.log"
    mkdir -p logs
    
    # Rebuild args without --background, add --_background_child
    BG_ARGS="--_background_child"
    [ "$MAX_JOBS" != "4" ] && BG_ARGS="$BG_ARGS --max-jobs $MAX_JOBS"
    [ -n "$RESULT_DIRS" ] && BG_ARGS="$BG_ARGS --dirs \"$RESULT_DIRS\""
    [ "$RAGAS_ONLY" = "--custom-only" ] && BG_ARGS="$BG_ARGS --custom-only"
    [ -z "$RAGAS_ONLY" ] && BG_ARGS="$BG_ARGS --all-metrics"
    [ -n "$EVAL_API_KEY" ] && BG_ARGS="$BG_ARGS --eval-api-key \"$EVAL_API_KEY\""
    [ -n "$EVAL_API_BASE" ] && BG_ARGS="$BG_ARGS --eval-api-base \"$EVAL_API_BASE\""
    [ -n "$EVAL_MODEL" ] && BG_ARGS="$BG_ARGS --eval-model \"$EVAL_MODEL\""
    
    echo "=============================================================="
    echo "🚀 Starting in BACKGROUND mode"
    echo "=============================================================="
    echo "Log file: $BG_LOG"
    echo "To monitor: tail -f $BG_LOG"
    echo "To stop: pkill -f 'run_repair_parallel.sh.*_background_child'"
    echo "=============================================================="
    
    eval nohup bash "$0" $BG_ARGS > "$BG_LOG" 2>&1 &
    BG_PID=$!
    echo "Background PID: $BG_PID"
    echo "$BG_PID" > logs/repair_parallel_background.pid
    echo ""
    echo "✅ Script detached. You can close this terminal safely."
    exit 0
fi

# Datasets to process
DATASETS="computational characterization stability materials device structure processing performance"

# Build list of all directories to process
DIRS_TO_PROCESS=""
for result_dir in $RESULT_DIRS; do
    for dataset in $DATASETS; do
        dir_path="results/${result_dir}/${dataset}"
        if [ -d "$dir_path" ]; then
            DIRS_TO_PROCESS="$DIRS_TO_PROCESS $dir_path"
        fi
    done
done

# Count total directories
TOTAL=$(echo $DIRS_TO_PROCESS | wc -w)

echo "=============================================================="
echo "Parallel RAGAS Metrics Repair"
echo "=============================================================="
echo "Max Parallel Jobs: $MAX_JOBS"
echo "Total Directories: $TOTAL"
echo "Mode:              ${RAGAS_ONLY:-all metrics}"
if [ -n "$EVAL_API_KEY" ]; then
    echo "Eval API Key:      ${EVAL_API_KEY:0:10}..."
fi
if [ -n "$EVAL_API_BASE" ]; then
    echo "Eval API Base:     $EVAL_API_BASE"
fi
if [ -n "$EVAL_MODEL" ]; then
    echo "Eval Model:        $EVAL_MODEL"
fi
if [ -n "$DRY_RUN" ]; then
    echo "DRY RUN:           Yes (no actual execution)"
fi
echo "=============================================================="
echo ""

if [ -n "$DRY_RUN" ]; then
    echo "Would run the following commands:"
    for dir in $DIRS_TO_PROCESS; do
        CMD="python repair_null_metrics.py --dir $dir $RAGAS_ONLY"
        [ -n "$EVAL_API_KEY" ] && CMD="$CMD --eval-api-key $EVAL_API_KEY"
        [ -n "$EVAL_API_BASE" ] && CMD="$CMD --eval-api-base $EVAL_API_BASE"
        [ -n "$EVAL_MODEL" ] && CMD="$CMD --eval-model $EVAL_MODEL"
        echo "  $CMD"
    done
    echo ""
    echo "To execute, run without --dry-run"
    exit 0
fi

# Create logs directory
mkdir -p logs

# Record start time
START_TIME=$(date +%s)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Track PIDs
PIDS=""
COMPLETED=0

echo "Launching parallel repairs..."
echo ""

for dir in $DIRS_TO_PROCESS; do
    # Wait if we've hit max jobs
    while [ $(echo $PIDS | wc -w) -ge $MAX_JOBS ]; do
        # Check which jobs are done
        NEW_PIDS=""
        for pid in $PIDS; do
            if kill -0 $pid 2>/dev/null; then
                NEW_PIDS="$NEW_PIDS $pid"
            else
                COMPLETED=$((COMPLETED + 1))
                echo "   [${COMPLETED}/${TOTAL}] Job $pid completed"
            fi
        done
        PIDS=$NEW_PIDS
        sleep 2
    done

    # Extract short name for logging
    short_name=$(echo "$dir" | sed 's|results/||' | tr '/' '_')
    log_file="logs/repair_${short_name}_${TIMESTAMP}.log"
    
    echo "[$(date +%H:%M:%S)] Starting: $dir"

    # Build command with optional API parameters
    CMD="python repair_null_metrics.py --dir \"$dir\" $RAGAS_ONLY"
    [ -n "$EVAL_API_KEY" ] && CMD="$CMD --eval-api-key \"$EVAL_API_KEY\""
    [ -n "$EVAL_API_BASE" ] && CMD="$CMD --eval-api-base \"$EVAL_API_BASE\""
    [ -n "$EVAL_MODEL" ] && CMD="$CMD --eval-model \"$EVAL_MODEL\""

    # Start the job in background
    eval $CMD > "$log_file" 2>&1 &
    
    pid=$!
    PIDS="$PIDS $pid"

    # Small delay to avoid API burst
    sleep 1
done

# Wait for remaining jobs to complete
echo ""
echo "Waiting for remaining jobs to complete..."
echo "   (Check logs/ directory for progress)"

for pid in $PIDS; do
    wait $pid
    COMPLETED=$((COMPLETED + 1))
    if [ $? -eq 0 ]; then
        echo "   [${COMPLETED}/${TOTAL}] Job $pid completed successfully"
    else
        echo "   [${COMPLETED}/${TOTAL}] Job $pid failed"
    fi
done

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=============================================================="
echo "All Parallel Repairs Completed"
echo "=============================================================="
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Logs saved to: logs/repair_*_${TIMESTAMP}.log"
echo "=============================================================="
echo ""

# Show summary
echo "Quick Summary:"
for result_dir in $RESULT_DIRS; do
    echo "  $result_dir:"
    for dataset in $DATASETS; do
        log_file=$(ls logs/repair_${result_dir}_${dataset}_${TIMESTAMP}.log 2>/dev/null)
        if [ -n "$log_file" ]; then
            repaired=$(grep -c "Repaired" "$log_file" 2>/dev/null || echo "0")
            echo "    $dataset: $repaired items repaired"
        fi
    done
done
echo ""
