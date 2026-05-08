#!/bin/bash
# Reviewer note: adjust SLURM resource requests below for your cluster if needed.
#SBATCH --job-name=part2_ensemble_all
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --array=0-5
#SBATCH -o logs/part2_ensemble_all-%A_%a.out
#SBATCH -e logs/part2_ensemble_all-%A_%a.err
#SBATCH --signal=SIGTERM@300
#SBATCH --requeue

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_ROOT="${JOB_ROOT:-${REPO_ROOT}}"
SCRIPT="${JOB_ROOT}/src/train_optimized_hpc_singlemode.py"
PERSIST_OUT="${JOB_ROOT}/outputs"
PERSIST_LOGS="${JOB_ROOT}/logs"
FEATURE_SET_MODE="all"
WEATHER_SOURCE_MODE="${WEATHER_SOURCE_MODE:-lagged_observed}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# Local scratch for heavy I/O
LOCAL_RUN_DIR="${SLURM_TMPDIR:-/tmp}/myproj_${SLURM_JOB_ID:-nojid}_${SLURM_ARRAY_TASK_ID:-noidx}_all"
LOCAL_OUT="${LOCAL_RUN_DIR}/outputs"
LOCAL_LOGS="${LOCAL_RUN_DIR}/logs"

mkdir -p "${LOCAL_OUT}" "${PERSIST_OUT}" "${PERSIST_LOGS}" "${LOCAL_LOGS}"
cd "${JOB_ROOT}"

# ============================================================================
# LOGGING SETUP
# ============================================================================
LOGFILE="${LOCAL_LOGS}/training_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_all.log"
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
echo "FeatureSetMode: ${FEATURE_SET_MODE}"
echo "WeatherSourceMode: ${WEATHER_SOURCE_MODE}"
echo "=========================================="


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
VENV_PATH="${VENV_PATH:-${JOB_ROOT}/.venv}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "[ERROR] Python virtual environment not found: ${VENV_PATH}"
  echo "[ERROR] Create it first on the login node:"
  echo "        python3 -m venv .venv"
  echo "        source .venv/bin/activate"
  echo "        python -m pip install --upgrade pip setuptools wheel"
  echo "        python -m pip install --no-cache-dir -r requirements.txt"
  exit 10
fi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

echo "[env] VENV_PATH: ${VENV_PATH}"
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
if (( IDX < 0 || IDX >= ${#LAGS[@]} )); then
  echo "[ERROR] Invalid SLURM_ARRAY_TASK_ID=${IDX} for 6 configured tasks"
  exit 3
fi

MODEL_TYPE=${MODEL_TYPES[$IDX]}
ALGORITHM=${ALGORITHMS[$IDX]}
LAG=${LAGS[$IDX]}

DATA_FILE="${JOB_ROOT}/data/merged_NWS_lag${LAG}.parquet"
EXPECTED_OUTDIR="${LOCAL_OUT}/${MODEL_TYPE}/${ALGORITHM}/lag${LAG}/weather_${WEATHER_SOURCE_MODE}/features_${FEATURE_SET_MODE}"
mkdir -p "${EXPECTED_OUTDIR}"

echo ""
echo "CONFIG: ${MODEL_TYPE} / ${ALGORITHM} / lag${LAG}"
echo "Data: ${DATA_FILE}"
echo "LocalOut: ${EXPECTED_OUTDIR}"
echo "NOTE: Ensemble trains per EVENT_TYPE."
echo "Single feature mode per job: ${FEATURE_SET_MODE}"
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
sync_to_persistent() {
  local sync_type="${1:-periodic}"
  echo ""
  echo "[SYNC-${sync_type}] Starting rsync at $(date)"

  if rsync -a --timeout=300 "${LOCAL_OUT}/" "${PERSIST_OUT}/"; then
    echo "[SYNC-${sync_type}] ✓ Outputs synced successfully"
  else
    echo "[SYNC-${sync_type}] ✗ WARNING: Output sync failed (exit code: $?)"
  fi

  if rsync -a --timeout=300 "${LOCAL_LOGS}/" "${PERSIST_LOGS}/"; then
    echo "[SYNC-${sync_type}] ✓ Logs synced successfully"
  else
    echo "[SYNC-${sync_type}] ✗ WARNING: Log sync failed (exit code: $?)"
  fi

  echo "[SYNC-${sync_type}] Completed at $(date)"
  echo ""
}

handle_termination() {
  echo ""
  echo "[SIGNAL] Received termination signal at $(date)"
  echo "[SIGNAL] Initiating graceful shutdown..."

  if [[ -n "${TRAIN_PID:-}" ]]; then
    echo "[SIGNAL] Sending SIGTERM to training process (PID: ${TRAIN_PID})"
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    for i in {1..30}; do
      if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
        echo "[SIGNAL] Training process terminated gracefully"
        break
      fi
      sleep 1
    done
    if kill -0 "${TRAIN_PID}" 2>/dev/null; then
      echo "[SIGNAL] Force killing training process"
      kill -9 "${TRAIN_PID}" 2>/dev/null || true
    fi
  fi

  sync_to_persistent "signal"
  exit 143
}

cleanup_and_exit() {
  local exit_code=$?
  echo ""
  echo "[CLEANUP] Exit code: ${exit_code}"
  sync_to_persistent "exit"
  if [[ -d "${LOCAL_RUN_DIR}" ]]; then
    echo "[CLEANUP] Removing local scratch: ${LOCAL_RUN_DIR}"
    rm -rf "${LOCAL_RUN_DIR}" || echo "[CLEANUP] Warning: Failed to remove local scratch"
  fi
  echo "[CLEANUP] Job ended at $(date)"
  exit ${exit_code}
}

trap handle_termination SIGTERM SIGINT
trap cleanup_and_exit EXIT

start_sync_daemon() {
  while true; do
    sleep 600
    sync_to_persistent "daemon"
  done &
  SYNC_DAEMON_PID=$!
  echo "[DAEMON] Background sync started (PID: ${SYNC_DAEMON_PID})"
}

stop_sync_daemon() {
  if [[ -n "${SYNC_DAEMON_PID:-}" ]]; then
    echo "[DAEMON] Stopping background sync (PID: ${SYNC_DAEMON_PID})"
    kill "${SYNC_DAEMON_PID}" 2>/dev/null || true
  fi
}

trap 'stop_sync_daemon; cleanup_and_exit' EXIT
start_sync_daemon

# ============================================================================
# TRAINING EXECUTION
# ============================================================================
echo ""
echo "=========================================="
echo "STARTING TRAINING"
echo "=========================================="
echo ""

srun --cpu-bind=cores python -u "${SCRIPT}"   "${DATA_FILE_LOCAL}" "${MODEL_TYPE}" "${ALGORITHM}" "${LAG}" "${LOCAL_OUT}"   --resume   --feature-set="${FEATURE_SET_MODE}"   --weather-source="${WEATHER_SOURCE_MODE}"   ${EXTRA_FLAGS} &

TRAIN_PID=$!
echo "[TRAIN] Training process started (PID: ${TRAIN_PID})"

wait ${TRAIN_PID}
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "TRAINING COMPLETED"
echo "Exit code: ${EXIT_CODE}"
echo "=========================================="
echo ""

stop_sync_daemon
sync_to_persistent "final"

exit ${EXIT_CODE}
