#!/bin/bash
# Initialize git repo locally for WATCH v3_1 reproduction
# Usage: bash git_init_local.sh
# After this, you can `git remote add origin ...` to push to remote

set -e
cd "$(dirname "$0")"

echo "=== Step 1: Configure git user (skip if already set) ==="
if ! git config user.name > /dev/null 2>&1; then
    read -p "Enter git user name: " GIT_NAME
    read -p "Enter git email: " GIT_EMAIL
    git config user.name "$GIT_NAME"
    git config user.email "$GIT_EMAIL"
else
    echo "Already set: $(git config user.name) <$(git config user.email)>"
fi

echo ""
echo "=== Step 2: Verify .gitignore is in place ==="
if [ ! -f ".gitignore" ]; then
    echo "ERROR: .gitignore not found. Aborting."
    exit 1
fi
echo "Found .gitignore"

echo ""
echo "=== Step 3: Check what will be committed ==="
echo "(should be ~270 files, ~3MB total — NO inputs/outputs/checkpoints)"
TOTAL_FILES=$(git ls-files --others --exclude-standard | wc -l)
echo "Files to add: $TOTAL_FILES"
TOTAL_SIZE=$(git ls-files --others --exclude-standard | xargs du -ch 2>/dev/null | tail -1 | awk '{print $1}')
echo "Total size: $TOTAL_SIZE"

if [ "$TOTAL_FILES" -gt 1000 ]; then
    echo ""
    echo "WARNING: $TOTAL_FILES files is too many — likely inputs/outputs not ignored."
    echo "Aborting. Check .gitignore."
    exit 1
fi

echo ""
read -p "Continue with git add + commit? [y/N] " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "=== Step 4: git add . ==="
git add .

echo ""
echo "=== Step 5: Verify staged files ==="
git status --short | head -20
echo "... (use 'git status' for full list)"

echo ""
echo "=== Step 6: Initial commit ==="
git commit -m "WATCH v3_1 reproduced baseline (6 bug fixes)

Fixes:
1. embedder registration order (imgseq vs cam_traj)
2. bedlam.py cam_transl_local: use raw get_local_vel_mat (no CamVelAugmentorV2)
3. AMASS base_dataset.py: same fix
4. h36m.py: same fix
5. 3DPW threedpw_motion_train.py: same fix
6. pipeline f_condition key order: f_imgseq before f_cam_transl

Reproduction (e419 + postproc):
- EMDB-2 wa2: 267.4 vs original 268.4 (-1.0 mm)
- RICH wa2: 116.2 vs original 117.2 (-1.0 mm)
- 3DPW PA: 35.9 vs original 35.3 (+0.6 mm)
- EMDB-1 PA: 43.4 vs original 42.9 (+0.5 mm)
- All 5 core metrics within ±1mm of original.

Best ckpt e399 even better:
- EMDB-2 wa2: 264.3 (better than original by 4.1 mm)
- RICH wa2: 116.1 (better than original by 1.1 mm)"

echo ""
echo "=== Step 7: Tag this milestone ==="
git tag -a v3_1-reproduced -m "WATCH v3_1 reproduced baseline (6 bug fixes), 2026-04-28"

echo ""
echo "=== Done! ==="
echo ""
echo "Local git repo initialized. To push to remote:"
echo ""
echo "  git remote add origin <YOUR_GIT_URL>"
echo "  git branch -M main"
echo "  git push -u origin main"
echo "  git push --tags"
echo ""
echo "Examples:"
echo "  GitHub:  git remote add origin git@github.com:USERNAME/watch.git"
echo "  GitLab:  git remote add origin git@gitlab.com:USERNAME/watch.git"
