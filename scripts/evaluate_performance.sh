#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SSH_TARGET="${SSH_TARGET:-ascend-box}"
REMOTE_PORT="${REMOTE_PORT:-}"
SSH_KEY="${SSH_KEY:-}"

REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/root/tilelang_eval}"
CONTAINER_NAME="${CONTAINER_NAME:-zyy_cann}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/home/z00893531/tilelang-ascend}"
REMOTE_EVAL_WORKDIR="${REMOTE_EVAL_WORKDIR:-workdir_remote_eval}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-3}"

usage() {
  cat <<'EOF'
Usage: scripts/evaluate_performance.sh [task] [impl] [warmup] [repeat] [seed]

Arguments:
  task      Task directory to benchmark. Defaults to current_task.
  impl      reference | tilelang | ascendc | all. Defaults to all.
  warmup    Warmup iterations. Defaults to 5.
  repeat    Timed iterations. Defaults to 10.
  seed      Random seed forwarded to utils/performance.py. Defaults to 0.

Environment overrides:
  SSH_TARGET                 SSH host or ~/.ssh/config alias
  REMOTE_PORT                Optional SSH port override
  SSH_KEY                    Optional SSH identity file override
  REMOTE_BASE_DIR            Host path used to store uploaded workdir
  CONTAINER_NAME             Target docker container name
  CONTAINER_WORKDIR          Project root inside the container
  REMOTE_EVAL_WORKDIR        Working directory name used inside the container
  ASCEND_RT_VISIBLE_DEVICES  Device id used inside the container

Examples:
  scripts/evaluate_performance.sh
  scripts/evaluate_performance.sh reshape_matmul_rowwise_quant_int8
  scripts/evaluate_performance.sh reshape_matmul_rowwise_quant_int8 tilelang
  REMOTE_EVAL_WORKDIR=workdir_remote_eval_wzz scripts/evaluate_performance.sh reshape_matmul_rowwise_quant_int8 all
  scripts/evaluate_performance.sh reshape_matmul_rowwise_quant_int8 all 2 5 0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TASK="${1:-current_task}"
IMPL="${2:-all}"
WARMUP="${3:-5}"
REPEAT="${4:-10}"
SEED="${5:-0}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_NAME="workdir_${TIMESTAMP}.tar.gz"
LOCAL_ARCHIVE="/tmp/${ARCHIVE_NAME}"
REMOTE_ARCHIVE="/tmp/${ARCHIVE_NAME}"
REMOTE_SESSION_DIR="${REMOTE_BASE_DIR}/${TIMESTAMP}"

if [[ ! -f "${WORKDIR}/utils/performance.py" ]]; then
  echo "Missing performance script: ${WORKDIR}/utils/performance.py" >&2
  exit 1
fi

if [[ ! -d "${WORKDIR}/${TASK}" ]]; then
  echo "Task directory not found: ${WORKDIR}/${TASK}" >&2
  exit 1
fi

if python -c 'import tilelang; import torch; import torch_npu' >/dev/null 2>&1; then
  echo "Detected local TileLang-Ascend environment, running local performance benchmark"
  cd "${WORKDIR}"
  ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" python utils/performance.py "${TASK}" "${IMPL}" "${WARMUP}" "${REPEAT}" "${SEED}"
  exit 0
fi

SSH_OPTS=()
SCP_OPTS=()
if [[ -n "${REMOTE_PORT}" ]]; then
  SSH_OPTS+=(-p "${REMOTE_PORT}")
  SCP_OPTS+=(-P "${REMOTE_PORT}")
fi

if [[ -n "${SSH_KEY}" ]]; then
  if [[ ! -f "${SSH_KEY}" ]]; then
    echo "SSH key not found: ${SSH_KEY}" >&2
    exit 1
  fi
  SSH_OPTS+=(-i "${SSH_KEY}")
  SCP_OPTS+=(-i "${SSH_KEY}")
fi

SSH_CMD=(ssh)
SCP_CMD=(scp)
if [[ ${#SSH_OPTS[@]} -gt 0 ]]; then
  SSH_CMD+=("${SSH_OPTS[@]}")
fi
if [[ ${#SCP_OPTS[@]} -gt 0 ]]; then
  SCP_CMD+=("${SCP_OPTS[@]}")
fi

cleanup() {
  rm -f "${LOCAL_ARCHIVE}"
}
trap cleanup EXIT

echo "[1/4] Packaging ${WORKDIR}"
tar \
  --exclude=".git" \
  --exclude="__pycache__" \
  --exclude=".DS_Store" \
  --exclude=".pytest_cache" \
  --exclude=".mypy_cache" \
  --exclude=".ruff_cache" \
  -C "${WORKDIR}" \
  -czf "${LOCAL_ARCHIVE}" \
  .

echo "[2/4] Uploading archive to ${SSH_TARGET}:${REMOTE_ARCHIVE}"
"${SCP_CMD[@]}" \
  "${LOCAL_ARCHIVE}" \
  "${SSH_TARGET}:${REMOTE_ARCHIVE}"

read -r -d '' REMOTE_SCRIPT <<EOF || true
set -euo pipefail
cleanup_remote() {
  rm -rf "${REMOTE_SESSION_DIR}"
}
trap cleanup_remote EXIT

mkdir -p "${REMOTE_SESSION_DIR}"
tar -xzf "${REMOTE_ARCHIVE}" -C "${REMOTE_SESSION_DIR}"
rm -f "${REMOTE_ARCHIVE}"

docker exec "${CONTAINER_NAME}" /bin/bash -lc 'mkdir -p "${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}"'

docker cp "${REMOTE_SESSION_DIR}/." "${CONTAINER_NAME}:${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}"

docker exec "${CONTAINER_NAME}" /bin/bash -lc '
set -euo pipefail
cd "${CONTAINER_WORKDIR}"
source set_env.sh
cd "${CONTAINER_WORKDIR}/${REMOTE_EVAL_WORKDIR}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" python utils/performance.py "${TASK}" "${IMPL}" "${WARMUP}" "${REPEAT}" "${SEED}"
'
EOF

echo "[3/4] Running performance benchmark inside container ${CONTAINER_NAME}"
"${SSH_CMD[@]}" \
  "${SSH_TARGET}" \
  "${REMOTE_SCRIPT}"

echo "[4/4] Performance benchmark completed"
