#!/usr/bin/env bash
#
# check_push_creds.sh — prove this machine has GitHub creds that can PUSH to a
# private repo (default: ACMAD-Niamey/acmadDL), without changing anything.
#
# Creds are a machine-level SSH key (~/.ssh/id_ed25519), not a session secret,
# so any session/agent on THIS machine inherits them. A remote/cloud agent or a
# different computer does NOT — it would need its own key or a PAT.
#
# Usage:
#   scripts/check_push_creds.sh                       # checks ACMAD-Niamey/acmadDL
#   scripts/check_push_creds.sh owner/repo            # checks any repo
#
# Exit status: 0 = push access confirmed, non-zero = not confirmed.

set -uo pipefail

REPO="${1:-ACMAD-Niamey/acmadDL}"
REMOTE="git@github.com:${REPO}.git"
TMP_REF="refs/heads/__perm-check-delete-me"
SSH="ssh -o BatchMode=yes"   # never prompt; fail fast if the key isn't usable

echo "Checking push credentials for: ${REPO}"
echo

# 1. Identity — who does the SSH key authenticate as?
echo "1. SSH identity"
ident="$($SSH -T git@github.com 2>&1)"
echo "   ${ident}"
case "$ident" in
  *"successfully authenticated"*) ;;  # GitHub returns exit 1 here; that's normal
  *) echo "   FAIL: SSH did not authenticate to github.com"; exit 1 ;;
esac
# Extract just the username from "Hi <user>! ..."
user="$(printf '%s' "$ident" | sed -n 's/^Hi \([^!]*\)!.*/\1/p')"
user="${user:-unknown}"
echo

# 2. Read access — can we list refs on the private repo?
echo "2. Read access"
if GIT_SSH_COMMAND="$SSH" git ls-remote --heads "$REMOTE" >/dev/null 2>&1; then
  echo "   OK: can read ${REPO}"
else
  echo "   FAIL: cannot read ${REPO} (no access, or repo path wrong)"; exit 1
fi
echo

# 3. Write access — dry-run push of a temp ref. Creates NOTHING on the remote;
#    GitHub authorizes the write at connection time, so this reveals push
#    permission safely. We point at any commit via HEAD (or origin/HEAD).
echo "3. Push access (dry-run; nothing is created)"
sha="$(GIT_SSH_COMMAND="$SSH" git ls-remote "$REMOTE" HEAD 2>/dev/null | awk '{print $1}')"
[ -z "${sha:-}" ] && sha="$(git rev-parse HEAD 2>/dev/null)"
if [ -z "${sha:-}" ]; then
  echo "   FAIL: could not resolve a commit to test-push"; exit 1
fi
out="$(GIT_SSH_COMMAND="$SSH" git push --dry-run "$REMOTE" "${sha}:${TMP_REF}" 2>&1)"
echo "${out}" | sed 's/^/   /'
if echo "${out}" | grep -qiE "new branch|up-to-date"; then
  echo
  echo "RESULT: push access CONFIRMED for ${REPO} (as ${user})"
  exit 0
else
  echo
  echo "RESULT: push access NOT confirmed for ${REPO}"
  exit 1
fi
