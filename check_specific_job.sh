#!/bin/bash
# check_specific_job.sh - Check detailed progress for a specific job configuration
# Usage: ./check_specific_job.sh [model_type] [algorithm] [lag]
# Example: ./check_specific_job.sh Unified LightGBM 1

MODEL_TYPE=${1:-"Unified"}
ALGORITHM=${2:-"LightGBM"}
LAG=${3:-"1"}

OUTDIR="outputs/${MODEL_TYPE}/${ALGORITHM}/lag${LAG}"

echo "=========================================="
echo "Detailed Progress Check"
echo "=========================================="
echo "Configuration: ${MODEL_TYPE} / ${ALGORITHM} / lag${LAG}"
echo "Output Directory: ${OUTDIR}"
echo ""

if [ ! -d "${OUTDIR}" ]; then
    echo "ERROR: Directory not found: ${OUTDIR}"
    echo ""
    echo "Available configurations:"
    find outputs -type d -name "lag*" 2>/dev/null | sort
    exit 1
fi

# 1. Progress Log
echo "1. PROGRESS LOG"
echo "----------------------------"
if [ -f "${OUTDIR}/progress.log" ]; then
    cat "${OUTDIR}/progress.log"
else
    echo "No progress.log found"
fi
echo ""

# 2. Latest Training Output (last 30 lines)
echo "2. LATEST TRAINING OUTPUT"
echo "----------------------------"
if [ -f "${OUTDIR}/training_output.log" ]; then
    echo "Total lines: $(wc -l < ${OUTDIR}/training_output.log)"
    echo "Last 30 lines:"
    tail -30 "${OUTDIR}/training_output.log"
else
    echo "No training_output.log found"
fi
echo ""

# 3. Checkpoint Status
echo "3. CHECKPOINT STATUS"
echo "----------------------------"
if [ -d "${OUTDIR}/checkpoints" ]; then
    echo "Checkpoint files:"
    ls -lht "${OUTDIR}/checkpoints/" | head -20
    echo ""
    
    # Bayesian optimization progress
    if ls "${OUTDIR}/checkpoints/bayesian_iter"*.pkl 1> /dev/null 2>&1; then
        latest=$(ls -t "${OUTDIR}/checkpoints/bayesian_iter"*.pkl | head -1)
        latest_iter=$(echo $latest | grep -oP 'iter\K[0-9]+')
        echo "Bayesian optimization: ${latest_iter} iterations completed"
    fi
    
    # Test predictions progress
    if ls "${OUTDIR}/checkpoints/test_predictions_"*.pkl 1> /dev/null 2>&1; then
        n_tests=$(ls "${OUTDIR}/checkpoints/test_predictions_"*.pkl | wc -l)
        echo "Test predictions: ${n_tests} / 10 completed"
    fi
else
    echo "No checkpoints directory found"
fi
echo ""

# 4. Summary Files
echo "4. SUMMARY FILES"
echo "----------------------------"
if [ -d "${OUTDIR}/summary" ]; then
    ls -lh "${OUTDIR}/summary/"
    echo ""
    
    # Show best params if exists
    if [ -f "${OUTDIR}/summary/best_params.json" ]; then
        echo "Best Parameters:"
        cat "${OUTDIR}/summary/best_params.json"
        echo ""
    fi
    
    # Show metrics summary if exists
    if [ -f "${OUTDIR}/summary/metrics.csv" ]; then
        echo "Metrics Summary (head):"
        head "${OUTDIR}/summary/metrics.csv"
        echo ""
    fi
else
    echo "No summary directory found"
fi

# 5. Related SLURM Logs
echo "5. RELATED SLURM LOGS"
echo "----------------------------"
echo "Recent log files containing ${MODEL_TYPE}:"
find logs -name "*part*.out" -type f -newermt "1 day ago" 2>/dev/null | while read logfile; do
    if grep -q "${MODEL_TYPE}" "$logfile" 2>/dev/null && grep -q "lag=${LAG}" "$logfile" 2>/dev/null && grep -q "${ALGORITHM}" "$logfile" 2>/dev/null; then
        echo "  - $logfile (modified: $(stat -c %y "$logfile" | cut -d. -f1))"
    fi
done
echo ""

echo "=========================================="
echo "To follow training in real-time, use:"
echo "  tail -f ${OUTDIR}/training_output.log"
echo "=========================================="
