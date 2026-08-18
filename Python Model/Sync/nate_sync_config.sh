#!/bin/bash
# Shared sync configuration for nate -- a local lab Ubuntu workstation (Xeon
# Silver 4208 + RTX 3080) reached via `ssh nate`. Source this from
# sync_to_nate.sh / sync_from_nate.sh.
#
# Assumes an SSH config entry named "nate" (~/.ssh/config) that already
# resolves HostName/User/IdentityFile, so no user@host is needed here.

NATE_HOST="${NATE_HOST:-nate}"
# Relative to nate's remote-user home directory (ssh/rsync resolve this on
# the remote side), so no guess about nate's absolute filesystem layout is
# needed here.
NATE_PROJECT="${NATE_PROJECT:-Bayesian}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_BASE="$(cd "$SCRIPT_DIR/.." && pwd)"

RSYNC_OPTS=(-avz --protect-args --progress --human-readable --prune-empty-dirs)
# nate is a personal workstation (not an always-on login node), so it may be
# asleep or off-VPN -- fail fast instead of hanging.
RSYNC_OPTS+=(-e "ssh -o ConnectTimeout=10 -o BatchMode=yes")

remote_rsync_path() {
	local remote_path="$1"
	printf "%s:%s" "$NATE_HOST" "$remote_path"
}
