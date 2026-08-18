#!/bin/bash
# Pull Bayesian inference outputs from nate (local lab GPU workstation) back
# to the local project. Pulls Results/ and run logs by default.
#
# Usage: ./sync_from_nate.sh [--dry-run] [all|results|job-files|FOLDER_NAME]
#   ./sync_from_nate.sh                              # all Results/ + job_files/
#   ./sync_from_nate.sh "FabD scaling inference"     # one Results/<folder_name>/
#   ./sync_from_nate.sh --dry-run results             # preview Results/ pull
set -euo pipefail
source "$(dirname "$0")/nate_sync_config.sh"

DRY_RUN=""
MODE="results"
RESULTS_FILTER=""

usage() {
    cat >&2 <<'EOF'
Usage: ./sync_from_nate.sh [--dry-run] [all|results|job-files|FOLDER_NAME]

Examples:
  ./sync_from_nate.sh
  ./sync_from_nate.sh results
  ./sync_from_nate.sh job-files
  ./sync_from_nate.sh "FabD scaling inference"
  ./sync_from_nate.sh --dry-run "FabD scaling inference"
EOF
}

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        -h|--help)
            usage
            exit 0
            ;;
        all|results|job-files)
            MODE="$arg"
            ;;
        *)
            MODE="one-result"
            RESULTS_FILTER="${arg%/}"
            ;;
    esac
done

REMOTE_BASE="${NATE_HOST}:${NATE_PROJECT}"

pull_dir() {
    local remote_src="$1"
    local local_dst="$2"

    mkdir -p "$local_dst"
    rsync "${RSYNC_OPTS[@]}" $DRY_RUN \
        "$remote_src" \
        "$local_dst"
}

echo "==> Pulling from ${REMOTE_BASE}"

case "$MODE" in
    all)
        echo "--- Results/"
        pull_dir "$(remote_rsync_path "${NATE_PROJECT}/Results/")" "${LOCAL_BASE}/Results/"
        echo "--- job_files/"
        pull_dir "$(remote_rsync_path "${NATE_PROJECT}/job_files/")" "${LOCAL_BASE}/job_files/"
        ;;
    results)
        echo "--- Results/"
        pull_dir "$(remote_rsync_path "${NATE_PROJECT}/Results/")" "${LOCAL_BASE}/Results/"
        ;;
    job-files)
        echo "--- job_files/"
        pull_dir "$(remote_rsync_path "${NATE_PROJECT}/job_files/")" "${LOCAL_BASE}/job_files/"
        ;;
    one-result)
        [[ -n "$RESULTS_FILTER" ]] || { usage; exit 1; }
        echo "--- Results/${RESULTS_FILTER}/"
        pull_dir "$(remote_rsync_path "${NATE_PROJECT}/Results/${RESULTS_FILTER}/")" "${LOCAL_BASE}/Results/${RESULTS_FILTER}/"
        ;;
    *)
        usage
        exit 1
        ;;
esac

echo "==> Pull complete."
