#!/bin/bash
#SBATCH --job-name=train_part2
#SBATCH --partition=xlong
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --array=0-5
#SBATCH -o logs/part2-%A_%a.out
#SBATCH -e logs/part2-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jangjlee7@tamu.edu
#SBATCH --signal=SIGTERM@300
#SBATCH --requeue

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================
JOB_ROOT="/scratch/user/jangjlee7/myproject"
SCRIPT="${JOB_ROOT}/train_optimized_final.py"
PERSIST_OUT="${JOB_ROOT}/outputs"
PERSIST_LOGS="${JOB_ROOT}/logs"

# Local scratch for heavy I/O
LOCAL_RUN_DIR="${SLURM_TMPDIR:-/tmp}/myproj_${SLURM_JOB_ID:-nojid}_${SLURM_ARRAY_TASK_ID:-noidx}"
LOCAL_OUT="${LOCAL_RUN_DIR}/outputs"
LOCAL_LOGS="${LOCAL_RUN_DIR}/logs"

# Create all directories
mkdir -p "${LOCAL_OUT}" "${PERSIST_OUT}" "${PERSIST_LOGS}" "${LOCAL_LOGS}"
cd "${JOB_ROOT}"

# ============================================================================
# LOGGING SETUP
# ============================================================================
LOGFILE="${LOCAL_LOGS}/training_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
exec 1> >(tee -a "${LOGFILE}")
exec 2>&1

echo "=========================================="
echo "Job Started at $(date)"
echo "JobID: ${SLURM_JOB_ID:-N/A}  TaskID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Host: $(hostname)"
echo "WorkDir: ${JOB_ROOT}"
echo "LocalRunDir: ${LOCAL_RUN_DIR}"
echo "PersistOut: ${PERSIST_OUT}"
echo "PersistLogs: ${PERSIST_LOGS}"
echo "=========================================="

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
module purge
module load Anaconda3
module load GCCcore/11.3.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml_env

echo "[env] Python: $(which python)"
python --version

# Thread settings
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_MAX_THREADS=16
export PYTHONUNBUFFERED=1

# ============================================================================
# JOB CONFIGURATION
# ============================================================================
declare -a MODEL_TYPES=("Ensemble" "Ensemble" "Ensemble" "Ensemble" "Ensemble" "Ensemble")
declare -a ALGORITHMS=("LightGBM" "LightGBM" "LightGBM" "XGBoost" "XGBoost" "XGBoost")
declare -a LAGS=(1 12 24 1 12 24)

IDX=${SLURM_ARRAY_TASK_ID}
MODEL_TYPE=${MODEL_TYPES[$IDX]}
ALGORITHM=${ALGORITHMS[$IDX]}
LAG=${LAGS[$IDX]}

DATA_FILE="${JOB_ROOT}/data/merged_NWS_lag${LAG}.parquet"
EXPECTED_OUTDIR="${LOCAL_OUT}/${MODEL_TYPE}/${ALGORITHM}/lag${LAG}"
mkdir -p "${EXPECTED_OUTDIR}"

echo ""
echo "CONFIG: ${MODEL_TYPE} / ${ALGORITHM} / lag${LAG}"
echo "Data: ${DATA_FILE}"
echo "LocalOut: ${EXPECTED_OUTDIR}"
echo "NOTE: Ensemble trains per EVENT_TYPE."
echo ""

# ============================================================================
# DATA VALIDATION
# ============================================================================
if [[ ! -f "${DATA_FILE}" ]]; then
  echo "[ERROR] Missing data file: ${DATA_FILE}"
  echo "[ERROR] Available files in data directory:"
  ls -lh "${JOB_ROOT}/data/" || true
  exit 2
fi

# Copy data to local scratch for faster I/O
echo "[SETUP] Copying data to local scratch..."
LOCAL_DATA="${LOCAL_RUN_DIR}/data"
mkdir -p "${LOCAL_DATA}"
cp -p "${DATA_FILE}" "${LOCAL_DATA}/"
DATA_FILE_LOCAL="${LOCAL_DATA}/$(basename "${DATA_FILE}")"
echo "[SETUP] Data copy complete"

# ============================================================================
# SYNC AND CLEANUP FUNCTIONS
# ============================================================================

# Sync function - called periodically and on exit
sync_to_persistent() {
  local sync_type="${1:-periodic}"
  echo ""
  echo "[SYNC-${sync_type}] Starting rsync at $(date)"
  
  # Sync outputs
  if rsync -a --timeout=300 "${LOCAL_OUT}/" "${PERSIST_OUT}/"; then
    echo "[SYNC-${sync_type}] ✓ Outputs synced successfully"
  else
    echo "[SYNC-${sync_type}] ✗ WARNING: Output sync failed (exit code: $?)"
  fi
  
  # Sync logs
  if rsync -a --timeout=300 "${LOCAL_LOGS}/" "${PERSIST_LOGS}/"; then
    echo "[SYNC-${sync_type}] ✓ Logs synced successfully"
  else
    echo "[SYNC-${sync_type}] ✗ WARNING: Log sync failed (exit code: $?)"
  fi
  
  echo "[SYNC-${sync_type}] Completed at $(date)"
  echo ""
}

# Signal handler for graceful termination
handle_termination() {
  echo ""
  echo "[SIGNAL] Received termination signal at $(date)"
  echo "[SIGNAL] Initiating graceful shutdown..."
  
  # Kill the training process if running
  if [[ -n "${TRAIN_PID:-}" ]]; then
    echo "[SIGNAL] Sending SIGTERM to training process (PID: ${TRAIN_PID})"
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    
    # Wait briefly for graceful shutdown
    for i in {1..30}; do
      if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
        echo "[SIGNAL] Training process terminated gracefully"
        break
      fi
      sleep 1
    done
    
    # Force kill if still running
    if kill -0 "${TRAIN_PID}" 2>/dev/null; then
      echo "[SIGNAL] Force killing training process"
      kill -9 "${TRAIN_PID}" 2>/dev/null || true
    fi
  fi
  
  sync_to_persistent "signal"
  exit 143  # 128 + 15 (SIGTERM)
}

# Exit handler - always runs
cleanup_and_exit() {
  local exit_code=$?
  echo ""
  echo "[CLEANUP] Exit code: ${exit_code}"
  sync_to_persistent "exit"
  
  # Clean up local scratch
  if [[ -d "${LOCAL_RUN_DIR}" ]]; then
    echo "[CLEANUP] Removing local scratch: ${LOCAL_RUN_DIR}"
    rm -rf "${LOCAL_RUN_DIR}" || echo "[CLEANUP] Warning: Failed to remove local scratch"
  fi
  
  echo "[CLEANUP] Job ended at $(date)"
  exit ${exit_code}
}

# Set up traps
trap handle_termination SIGTERM SIGINT
trap cleanup_and_exit EXIT

# ============================================================================
# BACKGROUND SYNC DAEMON
# ============================================================================

# Start background sync process
start_sync_daemon() {
  while true; do
    sleep 600  # Sync every 10 minutes
    sync_to_persistent "daemon"
  done &
  SYNC_DAEMON_PID=$!
  echo "[DAEMON] Background sync started (PID: ${SYNC_DAEMON_PID})"
}

# Stop background sync
stop_sync_daemon() {
  if [[ -n "${SYNC_DAEMON_PID:-}" ]]; then
    echo "[DAEMON] Stopping background sync (PID: ${SYNC_DAEMON_PID})"
    kill "${SYNC_DAEMON_PID}" 2>/dev/null || true
  fi
}

trap 'stop_sync_daemon; cleanup_and_exit' EXIT

# Start the sync daemon
start_sync_daemon

# ============================================================================
# TRAINING EXECUTION
# ============================================================================
echo ""
echo "=========================================="
echo "STARTING TRAINING"
echo "=========================================="
echo ""

# Run training in background so we can handle signals
srun --cpu-bind=cores python -u "${SCRIPT}" \
  "${DATA_FILE_LOCAL}" "${MODEL_TYPE}" "${ALGORITHM}" "${LAG}" "${LOCAL_OUT}" --resume &

TRAIN_PID=$!
echo "[TRAIN] Training process started (PID: ${TRAIN_PID})"

# Wait for training to complete
wait ${TRAIN_PID}
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "TRAINING COMPLETED"
echo "Exit code: ${EXIT_CODE}"
echo "=========================================="
echo ""

# Final sync
stop_sync_daemon
sync_to_persistent "final"

exit ${EXIT_CODE}
