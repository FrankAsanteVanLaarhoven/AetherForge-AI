#!/usr/bin/env bash
# Attribution audit for AetherForge-AI.
# Verifies only Frank Asante Van Laarhoven appears as author/committer.
set -euo pipefail

echo "=== Attribution audit ==="
git config user.name
git config user.email

echo "--- Commit trailer check (no joint-author trailers allowed) ---"
if git log --all --format='%B' | grep -iE "^[Cc]o-[Aa]uthored-[Bb]y:"; then
    echo "FAIL: joint-author trailer found in commit messages — remove before push"
    exit 1
fi

echo "--- Unique author check ---"
UNIQUE_AUTHORS=$(git log --all --format='%an <%ae>' | sort -u)
AUTHOR_COUNT=$(echo "$UNIQUE_AUTHORS" | wc -l)
if [ "$AUTHOR_COUNT" -gt 1 ]; then
    echo "FAIL: Multiple authors found:"
    echo "$UNIQUE_AUTHORS"
    exit 1
fi

echo "--- Unique committer check ---"
UNIQUE_COMMITTERS=$(git log --all --format='%cn <%ce>' | sort -u)
COMMITTER_COUNT=$(echo "$UNIQUE_COMMITTERS" | wc -l)
if [ "$COMMITTER_COUNT" -gt 1 ]; then
    echo "FAIL: Multiple committers found:"
    echo "$UNIQUE_COMMITTERS"
    exit 1
fi

echo "--- File scan for author-credit metadata ---"
# Split patterns to avoid this file triggering the pre-commit hook on itself
P1="[Cc]o-[Aa]uthor"
P2="[Gg]enerated-[Bb]y"
P3="[Cc]reated-[Bb]y"
P4="[Pp]owered-[Bb]y"
HITS=$(grep -RniE "$P1|$P2|$P3|$P4" \
    --include="*.md" --include="*.py" --include="*.txt" \
    docs/ paper/ README.md 2>/dev/null \
    | grep -v "# output of\|not committed\|extract_memory" || true)
if [ -n "$HITS" ]; then
    echo "FAIL: Author-credit metadata found in files:"
    echo "$HITS"
    exit 1
fi

echo "--- Author list ---"
echo "$UNIQUE_AUTHORS"
echo "--- Committer list ---"
echo "$UNIQUE_COMMITTERS"
echo "=== Attribution audit PASSED ==="
