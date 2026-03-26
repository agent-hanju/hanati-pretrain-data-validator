#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run.sh validate [input.jsonl] [--dry-run]
#   ./run.sh sample   <file1.jsonl> [file2.jsonl ...] -n 20 -o output.jsonl [--text-field text] [--seed 42]
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

COMMAND="${1:-}"
if [[ -z "$COMMAND" ]]; then
    echo "Usage:" >&2
    echo "  ./run.sh validate [input.jsonl] [--dry-run]" >&2
    echo "  ./run.sh sample <file1.jsonl> [file2.jsonl ...] -n 20 -o output.jsonl [--seed 42]" >&2
    exit 1
fi
shift

WORKDIR="$(pwd)"

case "$COMMAND" in
validate)
    INPUT_FILE=""
    DRY_RUN=""

    for arg in "$@"; do
        case "$arg" in
            --dry-run) DRY_RUN="--dry-run" ;;
            *.jsonl)   INPUT_FILE="$arg" ;;
            *) echo "Unknown argument: $arg" >&2; exit 1 ;;
        esac
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

    if [[ ! -f "$WORKDIR/config.yml" ]]; then
        echo "Error: config.yml not found in $WORKDIR" >&2
        exit 1
    fi
    if [[ ! -f "$WORKDIR/$INPUT_FILE" ]]; then
        echo "Error: $INPUT_FILE not found in $WORKDIR" >&2
        exit 1
    fi

    OUTPUT_FILE="${INPUT_FILE%.jsonl}.csv"

    echo "command : validate"
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
        validate \
        --config /data/config.yml \
        --input  "/data/$INPUT_FILE" \
        --output "/data/$OUTPUT_FILE" \
        $DRY_RUN
    ;;

sample)
    # Collect all .jsonl input files and pass remaining args through
    DOCKER_ARGS=("sample")
    INPUT_FILES=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            *.jsonl)
                if [[ ! -f "$WORKDIR/$1" ]]; then
                    echo "Error: $1 not found in $WORKDIR" >&2
                    exit 1
                fi
                INPUT_FILES+=("$1")
                DOCKER_ARGS+=("/data/$1")
                ;;
            *)
                DOCKER_ARGS+=("$1")
                ;;
        esac
        shift
    done

    if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
        echo "Error: at least one .jsonl file is required" >&2
        exit 1
    fi

    # Rewrite -o/--output path to /data/
    for i in "${!DOCKER_ARGS[@]}"; do
        if [[ "${DOCKER_ARGS[$i]}" == "-o" || "${DOCKER_ARGS[$i]}" == "--output" ]]; then
            next=$((i + 1))
            if [[ $next -lt ${#DOCKER_ARGS[@]} ]]; then
                DOCKER_ARGS[$next]="/data/${DOCKER_ARGS[$next]}"
            fi
        fi
    done

    echo "command : sample"
    echo "workdir : $WORKDIR"
    echo "inputs  : ${INPUT_FILES[*]}"
    echo "image   : $IMAGE"
    echo ""

    docker run --rm \
        -v "$WORKDIR:/data" \
        "$IMAGE" \
        "${DOCKER_ARGS[@]}"
    ;;

*)
    echo "Unknown command: $COMMAND" >&2
    echo "Available commands: validate, sample" >&2
    exit 1
    ;;
esac
