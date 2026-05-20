#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run.sh [input.jsonl] [--config config.yml] [--output output.csv] [--format csv|jsonl] [--mode prompt|messages]
#
# Run from the directory containing config.yml and your .jsonl files.

IMAGE="${VALIDATOR_IMAGE:-}"
if [[ -z "$IMAGE" ]]; then
    IMAGE="$(docker images hanati-pretrain-data-validator --format "{{.Repository}}:{{.Tag}}" | head -1)"
    if [[ -z "$IMAGE" ]]; then
        echo "Error: no local image found for 'hanati-pretrain-data-validator'" >&2
        exit 1
    fi
fi

WORKDIR="$(pwd)"

INPUT_FILE=""
CONFIG_FILE=""
OUTPUT_FILE=""
FORMAT=""
MODE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)  shift; CONFIG_FILE="$1" ;;
        --output)  shift; OUTPUT_FILE="$1" ;;
        --format)  shift; FORMAT="$1" ;;
        --mode)    shift; MODE="$1" ;;
        *.jsonl)   INPUT_FILE="$1" ;;
        -h|--help)
            echo "Usage: ./run.sh [input.jsonl] [--config config.yml] [--output output.csv] [--format csv|jsonl] [--mode prompt|messages]" >&2
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done

# auto-detect jsonl if not specified
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

if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_FILE="config.yml"
fi
if [[ ! -f "$WORKDIR/$CONFIG_FILE" ]]; then
    echo "Error: $CONFIG_FILE not found in $WORKDIR" >&2
    exit 1
fi
if [[ ! -f "$WORKDIR/$INPUT_FILE" ]]; then
    echo "Error: $INPUT_FILE not found in $WORKDIR" >&2
    exit 1
fi

if [[ -z "$FORMAT" ]]; then
    FORMAT="csv"
fi
if [[ -z "$OUTPUT_FILE" ]]; then
    if [[ "$FORMAT" == "jsonl" ]]; then
        OUTPUT_FILE="${INPUT_FILE%.jsonl}_out.jsonl"
    else
        OUTPUT_FILE="${INPUT_FILE%.jsonl}.csv"
    fi
fi

echo "workdir : $WORKDIR"
echo "config  : $CONFIG_FILE"
echo "input   : $INPUT_FILE"
echo "output  : $OUTPUT_FILE"
echo "format  : $FORMAT"
echo "mode    : ${MODE:-messages}"
echo "image   : $IMAGE"
echo ""

MODE_ARG=""
if [[ -n "$MODE" ]]; then
    MODE_ARG="--mode $MODE"
fi

docker run --rm \
    --network host \
    -v "$WORKDIR:/data" \
    "$IMAGE" \
    --config "/data/$CONFIG_FILE" \
    --input  "/data/$INPUT_FILE" \
    --output "/data/$OUTPUT_FILE" \
    --format "$FORMAT" \
    $MODE_ARG
