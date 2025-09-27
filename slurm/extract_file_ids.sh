#!/bin/bash
# Usage: ./extract_file_ids.sh /path/to/parent
# Creates file_ids.json in the same dir as this script.

set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: $0 /path/to/parent"
  exit 1
fi

PARENT_DIR="$1"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUT="$SCRIPT_DIR/file_ids.json"

# Collect file IDs into a variable
FILE_IDS=$(find "$PARENT_DIR" -type f -printf "%f\n" \
| sed 's/\..*$//' \
| awk -F'_' '
  NF==4 &&
  $1 ~ /^V[0-9]{2}$/ &&
  $2 ~ /^S[0-9]{4}$/ &&
  $3 ~ /^I[0-9]+$/ &&
  $4 ~ /^P[0-9]{4}$/ {print}
' \
| sort -u)

# Write JSON
echo "$FILE_IDS" | jq -R . | jq -s . > "$OUT"

# Count entries
COUNT=$(echo "$FILE_IDS" | wc -l)

echo "✅ Created $OUT with $COUNT unique file IDs"

