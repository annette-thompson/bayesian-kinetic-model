#!/bin/bash
# Push local Bayesian Kinetic Model project files to Alpine.
# Syncs selected project items, or all default items if none are given.
#
# Usage: ./sync_to_cluster.sh [--dry-run] [--delete-not-on-source] [FOLDER_OR_FILE ...]
set -euo pipefail
source "$(dirname "$0")/sync_config.sh"

DRY_RUN=""
DELETE_NOT_ON_SOURCE=""
DEFAULT_ITEMS=(
    "Bayesian Inference"
    "Calculation Files"
    "Data"
    "ODE Runner"
    "Reactions"
    "Results"
    "Utilities"
)
SELECTED_ITEMS=()

usage() {
    cat >&2 <<'EOF'
Usage: ./sync_to_cluster.sh [--dry-run] [--delete-not-on-source] [FOLDER_OR_FILE ...]

Options:
    --dry-run                 Show what would be synced without changing files.
    --delete-not-on-source    Delete remote files that are not present locally.

With no folder/file arguments, syncs:
    Bayesian Inference
    Calculation Files
    Data
    ODE Runner
    Reactions
    Results
    Utilities

Examples:
    ./sync_to_cluster.sh
    ./sync_to_cluster.sh --dry-run
    ./sync_to_cluster.sh --dry-run --delete-not-on-source
    ./sync_to_cluster.sh --delete-not-on-source Utilities Data Reactions
    ./sync_to_cluster.sh Utilities Data Reactions
    ./sync_to_cluster.sh "Bayesian Inference"
EOF
}

add_item() {
    local item="$1"
    local existing
    for existing in "${SELECTED_ITEMS[@]:-}"; do
        [[ "$existing" == "$item" ]] && return 0
    done
    SELECTED_ITEMS+=("$item")
}

for arg in "$@"; do
    item="${arg%/}"
    case "$arg" in
        --dry-run)
            DRY_RUN="--dry-run"
            ;;
        --delete-not-on-source)
            DELETE_NOT_ON_SOURCE="--delete"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -e "${LOCAL_BASE}/${item}" ]]; then
                add_item "$item"
            else
                echo "Error: unknown sync item: ${arg}" >&2
                echo "  (checked: ${LOCAL_BASE}/${item})" >&2
                usage
                exit 1
            fi
            ;;
    esac
done

if [[ ${#SELECTED_ITEMS[@]} -eq 0 ]]; then
    SELECTED_ITEMS=("${DEFAULT_ITEMS[@]}")
fi

REMOTE="${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_PROJECT}"

echo "==> Syncing local → ${REMOTE}"
echo "==> Local base: ${LOCAL_BASE}"
if [[ -n "$DELETE_NOT_ON_SOURCE" ]]; then
    echo "==> Deleting remote files that are not present locally."
fi

RSYNC_SOURCES=()

for item in "${SELECTED_ITEMS[@]}"; do
    echo "--- ${item}"
    RSYNC_SOURCES+=("${LOCAL_BASE}/${item}")
done

rsync "${RSYNC_OPTS[@]}" ${DRY_RUN:+$DRY_RUN} ${DELETE_NOT_ON_SOURCE:+$DELETE_NOT_ON_SOURCE} \
    --rsync-path="mkdir -p '$CLUSTER_PROJECT' && rsync" \
    --exclude='__pycache__' \
    --exclude='.ipynb_checkpoints' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    "${RSYNC_SOURCES[@]}" "$(remote_rsync_path "${CLUSTER_PROJECT}/")"

echo "==> Push complete."
