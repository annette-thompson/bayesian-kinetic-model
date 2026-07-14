#!/bin/bash
# Shared Alpine sync configuration for the Bayesian Kinetic Model Python project.
# Source this from sync_to_cluster.sh / sync_from_cluster.sh.

CLUSTER_USER="${CLUSTER_USER:-anth4580}"
CLUSTER_HOST="${CLUSTER_HOST:-login.rc.colorado.edu}"
CLUSTER_PROJECT="${CLUSTER_PROJECT:-/projects/${CLUSTER_USER}/Bayesian}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_BASE="$(cd "$SCRIPT_DIR/.." && pwd)"

RSYNC_OPTS=(-avz --protect-args --progress --human-readable --prune-empty-dirs)

remote_rsync_path() {
	local remote_path="$1"
	printf "%s@%s:%s" "$CLUSTER_USER" "$CLUSTER_HOST" "$remote_path"
}
