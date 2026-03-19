#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run.sh [input.jsonl] [--dry-run]
#
# Run from the directory containing config.yml and your .jsonl file.
#
# Output: <name>.csv (same directory)

IMAGE="${VALIDATOR_IMAGE:-}"
if [[ -z "$IMAGE" ]]; then
    IMAGE="$(docker images hanati-pretrain-data-validator --format "{{.Repository}}:{{.Tag}}" | head -1)"
    if [[ -z "$IMAGE" ]]; then
        echo "Error: no local image found for 'hanati-pretrain-data-validator'" >&2
        exit 1
    fi
fi

# --- arg parsing ---
WORKDIR="$(pwd)"
INPUT_FILE=""
DRY_RUN=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        *.jsonl)   INPUT_FILE="$arg" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# --- auto-detect jsonl if not specified ---
if [[ -z "$INPUT_FILE" ]]; then
    mapfile -t jsonl_files < <(ls "$WORKDIR"/*.jsonl 2>/dev/null)
    if [[ ${#jsonl_files[@]} -eq 0 ]]; then
        echo "Error: no .jsonl file found in $WORKDIR" >&2
        exit 1
    elif [[ ${#jsonl_files[@]} -gt 1 ]]; then
        echo "Error: multiple .jsonl files found in $WORKDIR, specify one explicitly:" >&2
        printf "  %s\n" "${jsonl_files[@]##*/}" >&2
        exit 1
    fi
    INPUT_FILE="${jsonl_files[0]##*/}"
fi

# --- validate workdir contents ---
if [[ ! -f "$WORKDIR/config.yml" ]]; then
    echo "Error: config.yml not found in $WORKDIR" >&2
    exit 1
fi
if [[ ! -f "$WORKDIR/$INPUT_FILE" ]]; then
    echo "Error: $INPUT_FILE not found in $WORKDIR" >&2
    exit 1
fi

OUTPUT_FILE="${INPUT_FILE%.jsonl}.csv"

echo "workdir : $WORKDIR"
echo "input   : $INPUT_FILE"
echo "output  : $OUTPUT_FILE"
echo "image   : $IMAGE"
[[ -n "$DRY_RUN" ]] && echo "(dry-run mode)"
echo ""

docker run --rm \
    --network host \
    -v "$WORKDIR:/data" \
    "$IMAGE" \
    --config /data/config.yml \
    --input  "/data/$INPUT_FILE" \
    --output "/data/$OUTPUT_FILE" \
    $DRY_RUN
