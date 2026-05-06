#!/bin/bash
# =============================================================================
# RAG Evaluation Script - Parallel Execution (POSIX Compatible)
# =============================================================================
# Usage:
#   ./run_mi.sh --background --resume  后台运行
#   ./run_all_parallel.sh                    # Run all datasets in parallel
#   ./run_all_parallel.sh --resume           # Resume from existing progress
#   ./run_all_parallel.sh --quick            # Quick test (5 questions per dataset)

# 同时启动新的评估 (保持现有任务继续运行)
# ./run_mi.sh \
#   --eval-api-key "" \
#   --eval-api-base "" \
#   --eval-model "mimo-v2-flash-free" \
#   --results-subdir "mimo-v2-flash-free" \
#   --resume

# ./run_mi.sh \
#   --answer-api-key "" \
#   --answer-api-base "" \
#   --answer-model "claude-sonnet-4-5-20250929" \
#   --eval-api-key "" \
#   --eval-api-base "" \
#   --eval-model "mimo-v2-flash-free" \
#   --results-subdir "claude-4.5-mimo" \
#   --resume
#
# API Configuration: Set in .env files (agentic_rag/.env and data_pipeline/.env)
# =============================================================================

NUM_QUESTIONS="all"
MAX_JOBS=50

# Parse command line arguments
RESUME_FLAG=""
QUICK_FLAG=""
EVAL_API_KEY=""
EVAL_API_BASE=""
EVAL_MODEL=""
ANSWER_API_KEY=""
ANSWER_API_BASE=""
ANSWER_MODEL=""
RESULTS_SUBDIR=""
BACKGROUND_MODE=""
GENERATE_ONLY=""
EVAL_ONLY=""
DRY_RUN=""

# Helper function to print usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --resume                 Resume from existing progress"
    echo "  --quick                  Quick test (5 questions per dataset)"
    echo "  --max-jobs N             Limit parallel jobs (default: 4)"
    echo "  --num-questions N        Number of questions per dataset"
    echo "  --eval-api-key KEY       API key for evaluation model"
    echo "  --eval-api-base URL      API base URL for evaluation model"
    echo "  --eval-model MODEL       Model name for evaluation"
    echo "  --answer-api-key KEY     API key for answer generation model"
    echo "  --answer-api-base URL    API base URL for answer generation model"
    echo "  --answer-model MODEL     Model name for answer generation"
    echo "  --results-subdir NAME    Subdirectory for results (e.g., model name)"
    echo "  --background             Run in background (detached from terminal)"
    echo "  --generate-only          Phase 1: Generate answers only (skip evaluation metrics)"
    echo "  --eval-only              Phase 2: Compute metrics on existing answers"
    echo "  --dry-run                Preview which errored questions would be re-generated (use with --resume)"
    echo "  --help                   Show this help message"
    echo ""
    echo "Example: different models for answer and evaluation:"
    echo "  $0 --answer-model gpt-4o --eval-model glm-4.7 --results-subdir gpt4o-answer"
    echo ""
    echo "Example: run in background (keeps running after terminal closes):"
    echo "  $0 --background --resume"
    echo ""
    echo "Example: Two-phase workflow:"
    echo "  Phase 1: $0 --generate-only --answer-model claude-4.5 --results-subdir claude"
    echo "  Phase 2: python repair_null_metrics.py --eval-api-key KEY --dir results/claude"
    echo ""
    echo "Example: Preview which errored questions would be re-generated:"
    echo "  $0 --dry-run --results-subdir glm-4.7--"
    exit 1
}

# Process arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --resume) RESUME_FLAG="--resume" ;;
        --quick) QUICK_FLAG="true"; NUM_QUESTIONS="5" ;;
        --max-jobs) MAX_JOBS="$2"; shift ;;
        --num-questions) NUM_QUESTIONS="$2"; shift ;;
        --eval-api-key) EVAL_API_KEY="$2"; shift ;;
        --eval-api-base) EVAL_API_BASE="$2"; shift ;;
        --eval-model) EVAL_MODEL="$2"; shift ;;
        --answer-api-key) ANSWER_API_KEY="$2"; shift ;;
        --answer-api-base) ANSWER_API_BASE="$2"; shift ;;
        --answer-model) ANSWER_MODEL="$2"; shift ;;
        --results-subdir) RESULTS_SUBDIR="$2"; shift ;;
        --background) BACKGROUND_MODE="true" ;;
        --_background_child) BACKGROUND_MODE="child" ;;  # Internal flag for child process
        --generate-only) GENERATE_ONLY="--generate-only" ;;
        --eval-only) EVAL_ONLY="--eval-only" ;;
        --dry-run) DRY_RUN="--dry-run" ;;
        --help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Change to evaluation directory
cd "$(dirname "$0")"

# Add project root to PYTHONPATH so agentic_rag module can be found
export PYTHONPATH="$(dirname $(pwd)):$PYTHONPATH"

# --dry-run implies --resume (need to load existing results to analyze errors)
if [ -n "$DRY_RUN" ] && [ -z "$RESUME_FLAG" ]; then
    RESUME_FLAG="--resume"
    echo "ℹ️  --dry-run implies --resume, automatically enabled"
fi

# Create logs directory
mkdir -p logs

# Handle background mode: restart self with nohup and exit
if [ "$BACKGROUND_MODE" = "true" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BG_LOG="logs/run_mi_background_${TIMESTAMP}.log"
    
    # Rebuild args without --background, add --_background_child
    BG_ARGS="--_background_child"
    [ -n "$RESUME_FLAG" ] && BG_ARGS="$BG_ARGS --resume"
    [ -n "$QUICK_FLAG" ] && BG_ARGS="$BG_ARGS --quick"
    [ "$MAX_JOBS" != "4" ] && BG_ARGS="$BG_ARGS --max-jobs $MAX_JOBS"
    [ "$NUM_QUESTIONS" != "all" ] && BG_ARGS="$BG_ARGS --num-questions $NUM_QUESTIONS"
    [ -n "$EVAL_API_KEY" ] && BG_ARGS="$BG_ARGS --eval-api-key $EVAL_API_KEY"
    [ -n "$EVAL_API_BASE" ] && BG_ARGS="$BG_ARGS --eval-api-base $EVAL_API_BASE"
    [ -n "$EVAL_MODEL" ] && BG_ARGS="$BG_ARGS --eval-model $EVAL_MODEL"
    [ -n "$ANSWER_API_KEY" ] && BG_ARGS="$BG_ARGS --answer-api-key $ANSWER_API_KEY"
    [ -n "$ANSWER_API_BASE" ] && BG_ARGS="$BG_ARGS --answer-api-base $ANSWER_API_BASE"
    [ -n "$ANSWER_MODEL" ] && BG_ARGS="$BG_ARGS --answer-model $ANSWER_MODEL"
    [ -n "$RESULTS_SUBDIR" ] && BG_ARGS="$BG_ARGS --results-subdir $RESULTS_SUBDIR"
    [ -n "$GENERATE_ONLY" ] && BG_ARGS="$BG_ARGS --generate-only"
    [ -n "$EVAL_ONLY" ] && BG_ARGS="$BG_ARGS --eval-only"
    [ -n "$DRY_RUN" ] && BG_ARGS="$BG_ARGS --dry-run"
    
    echo "=============================================================="
    echo "🚀 Starting in BACKGROUND mode"
    echo "=============================================================="
    echo "Log file: $BG_LOG"
    echo "To monitor: tail -f $BG_LOG"
    echo "To stop: pkill -f 'run_mi.sh.*_background_child'"
    echo "=============================================================="
    
    nohup bash "$0" $BG_ARGS > "$BG_LOG" 2>&1 &
    BG_PID=$!
    echo "Background PID: $BG_PID"
    echo "$BG_PID" > logs/run_mi_background.pid
    echo ""
    echo "✅ Script detached. You can close this terminal safely."
    exit 0
fi

# Define all datasets (space-separated string for POSIX compatibility)
DATASETS="computational characterization stability materials device structure processing performance"

echo "=============================================================="
echo "Starting PARALLEL RAG Evaluation"
echo "=============================================================="
echo "Questions:        $NUM_QUESTIONS"
echo "Max Parallel:     $MAX_JOBS"
echo "Resume Mode:      ${RESUME_FLAG:-disabled}"
if [ -n "$DRY_RUN" ]; then
    echo "Dry-Run Mode:     enabled (preview only, no changes)"
fi
if [ -n "$ANSWER_MODEL" ]; then
    echo "Answer Model:     $ANSWER_MODEL"
fi
if [ -n "$EVAL_MODEL" ]; then
    echo "Eval Model:       $EVAL_MODEL"
fi
if [ -n "$RESULTS_SUBDIR" ]; then
    echo "Results Subdir:   $RESULTS_SUBDIR"
fi
echo "=============================================================="
echo ""

# Record start time
START_TIME=$(date +%s)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Track PIDs
PIDS=""

echo "Launching parallel evaluations..."
echo ""

for dataset in $DATASETS; do
    # Wait if we've hit max jobs
    while [ $(echo $PIDS | wc -w) -ge $MAX_JOBS ]; do
        # Check which jobs are done
        NEW_PIDS=""
        for pid in $PIDS; do
            if kill -0 $pid 2>/dev/null; then
                NEW_PIDS="$NEW_PIDS $pid"
            fi
        done
        PIDS=$NEW_PIDS
        sleep 2
    done

    # Start the job in background
    log_file="logs/${dataset}_${TIMESTAMP}.log"
    echo "[$(date +%H:%M:%S)] Starting: $dataset"

    (
        attempt=1
        while true; do
            if [ $attempt -gt 1 ]; then
                echo "[$(date +%H:%M:%S)] Retry attempt $attempt..."
                # Force resume on retries to continue progress
                CURRENT_FLAGS="$RESUME_FLAG --resume"
            else
                CURRENT_FLAGS="$RESUME_FLAG"
            fi

            # Build eval and answer config arguments
            EVAL_ARGS=""
            if [ -n "$EVAL_API_KEY" ]; then
                EVAL_ARGS="$EVAL_ARGS --eval-api-key $EVAL_API_KEY"
            fi
            if [ -n "$EVAL_API_BASE" ]; then
                EVAL_ARGS="$EVAL_ARGS --eval-api-base $EVAL_API_BASE"
            fi
            if [ -n "$EVAL_MODEL" ]; then
                EVAL_ARGS="$EVAL_ARGS --eval-model $EVAL_MODEL"
            fi
            if [ -n "$ANSWER_API_KEY" ]; then
                EVAL_ARGS="$EVAL_ARGS --answer-api-key $ANSWER_API_KEY"
            fi
            if [ -n "$ANSWER_API_BASE" ]; then
                EVAL_ARGS="$EVAL_ARGS --answer-api-base $ANSWER_API_BASE"
            fi
            if [ -n "$ANSWER_MODEL" ]; then
                EVAL_ARGS="$EVAL_ARGS --answer-model $ANSWER_MODEL"
            fi
            if [ -n "$RESULTS_SUBDIR" ]; then
                EVAL_ARGS="$EVAL_ARGS --results-subdir $RESULTS_SUBDIR"
            fi
            if [ -n "$GENERATE_ONLY" ]; then
                EVAL_ARGS="$EVAL_ARGS $GENERATE_ONLY"
            fi
            if [ -n "$EVAL_ONLY" ]; then
                EVAL_ARGS="$EVAL_ARGS $EVAL_ONLY"
            fi
            if [ -n "$DRY_RUN" ]; then
                EVAL_ARGS="$EVAL_ARGS $DRY_RUN"
            fi

            python evaluate_rag.py \
                --dataset "$dataset" \
                --num_questions "$NUM_QUESTIONS" \
                $CURRENT_FLAGS \
                $EVAL_ARGS

            exit_code=$?
            
            if [ $exit_code -eq 0 ]; then
                break
            fi

            echo "[$(date +%H:%M:%S)] Job failed (exit code $exit_code). Waiting 30 minutes before retry..."
            sleep 1800
            attempt=$((attempt + 1))
        done
    ) > "$log_file" 2>&1 &

    pid=$!
    PIDS="$PIDS $pid"

    # Small delay to avoid API burst
    sleep 2
done

# Wait for all jobs to complete
echo ""
echo "Waiting for all evaluations to complete..."
echo "   (Check logs/ directory for progress)"

for pid in $PIDS; do
    wait $pid
    if [ $? -eq 0 ]; then
        echo "   Job $pid completed"
    else
        echo "   Job $pid failed"
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
echo "All Parallel Evaluations Completed"
echo "=============================================================="
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Logs saved to: logs/"
if [ -n "$RESULTS_SUBDIR" ]; then
    echo "Results saved to: results/${RESULTS_SUBDIR}/<dataset>/"
    RESULTS_PATH="results/${RESULTS_SUBDIR}"
else
    echo "Results saved to: results/<dataset>/"
    RESULTS_PATH="results"
fi
echo "=============================================================="
echo ""

# Show summary of results
echo "Quick Summary:"

# In dry-run mode, extract error info from logs
if [ -n "$DRY_RUN" ]; then
    TOTAL_ERRORS=0
    TOTAL_SUCCESS=0
    for dataset in $DATASETS; do
        # Find the most recent log file for this dataset by filename (includes timestamp)
        # Sort by filename descending to get the newest timestamp
        log_file=$(ls logs/${dataset}_*.log 2>/dev/null | sort -r | head -1)
        if [ -n "$log_file" ]; then
            # Extract the summary info from the log (sum across all modes)
            # Use grep without emoji to be more portable
            success_count=$(grep "Successfully completed:" "$log_file" 2>/dev/null | grep -oP ': \K\d+' | tail -1)
            error_count=$(grep "With errors" "$log_file" 2>/dev/null | grep -oP ': \K\d+' | tail -1)
            
            if [ -z "$success_count" ]; then success_count=0; fi
            if [ -z "$error_count" ]; then error_count=0; fi
            
            if [ "$error_count" -gt 0 ]; then
                echo "   $dataset: ✅ $success_count completed, 🔄 $error_count errors to re-generate"
                TOTAL_ERRORS=$((TOTAL_ERRORS + error_count))
                TOTAL_SUCCESS=$((TOTAL_SUCCESS + success_count))
            elif [ "$success_count" -gt 0 ]; then
                echo "   $dataset: ✅ $success_count completed (no errors)"
                TOTAL_SUCCESS=$((TOTAL_SUCCESS + success_count))
            elif grep -q "No existing results found" "$log_file" 2>/dev/null; then
                echo "   $dataset: 📂 No existing results (start fresh)"
            else
                echo "   $dataset: Check logs for details"
            fi
        else
            echo "   $dataset: No log found"
        fi
    done
    
    echo ""
    echo "📊 Total: $TOTAL_SUCCESS completed, $TOTAL_ERRORS need re-generation"
    if [ "$TOTAL_ERRORS" -gt 0 ]; then
        echo "   Run without --dry-run to actually re-generate"
    fi
else
    # Normal mode: show results availability
    for dataset in $DATASETS; do
        if [ -f "${RESULTS_PATH}/${dataset}/evaluation_summary.json" ]; then
            echo "   $dataset: Results available"
        else
            echo "   $dataset: In progress or failed"
        fi
    done
fi
ech ""
echo ""
echo "============================================="
echo "All tasks completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
