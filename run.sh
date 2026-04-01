#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run.sh validate [input.jsonl] [--config config.yml] [--output output.csv] [--format csv|xlsx] [--dry-run]
#   ./run.sh convert  <input.csv> [--output output.xlsx]
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
    echo "  ./run.sh validate [input.jsonl] [--config config.yml] [--output output.csv] [--format csv|xlsx] [--dry-run]" >&2
    echo "  ./run.sh convert  <input.csv> [--output output.xlsx]" >&2
    echo "  ./run.sh sample   <file1.jsonl> [file2.jsonl ...] -n 20 -o output.jsonl [--seed 42]" >&2
    exit 1
fi
shift

WORKDIR="$(pwd)"

case "$COMMAND" in
validate)
    INPUT_FILE=""
    CONFIG_FILE=""
    OUTPUT_FILE=""
    FORMAT=""
    DRY_RUN=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run) DRY_RUN="--dry-run" ;;
            --config)  shift; CONFIG_FILE="$1" ;;
            --output)  shift; OUTPUT_FILE="$1" ;;
            --format)  shift; FORMAT="$1" ;;
            *.jsonl)   INPUT_FILE="$1" ;;
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

    # default config
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

    # default format and output
    if [[ -z "$FORMAT" ]]; then
        FORMAT="csv"
    fi
    if [[ -z "$OUTPUT_FILE" ]]; then
        if [[ "$FORMAT" == "xlsx" ]]; then
            OUTPUT_FILE="${INPUT_FILE%.jsonl}.xlsx"
        else
            OUTPUT_FILE="${INPUT_FILE%.jsonl}.csv"
        fi
    fi

    FORMAT_ARG=""
    if [[ -n "$FORMAT" ]]; then
        FORMAT_ARG="--format $FORMAT"
    fi

    echo "command : validate"
    echo "workdir : $WORKDIR"
    echo "config  : $CONFIG_FILE"
    echo "input   : $INPUT_FILE"
    echo "output  : $OUTPUT_FILE"
    echo "format  : $FORMAT"
    echo "image   : $IMAGE"
    [[ -n "$DRY_RUN" ]] && echo "(dry-run mode)"
    echo ""

    docker run --rm \
        --network host \
        -v "$WORKDIR:/data" \
        "$IMAGE" \
        validate \
        --config "/data/$CONFIG_FILE" \
        --input  "/data/$INPUT_FILE" \
        --output "/data/$OUTPUT_FILE" \
        $FORMAT_ARG \
        $DRY_RUN
    ;;

convert)
    CSV_FILE=""
    OUTPUT_FILE=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --output) shift; OUTPUT_FILE="$1" ;;
            *.csv)    CSV_FILE="$1" ;;
            *) echo "Unknown argument: $1" >&2; exit 1 ;;
        esac
        shift
    done

    if [[ -z "$CSV_FILE" ]]; then
        echo "Error: input CSV file is required" >&2
        exit 1
    fi
    if [[ ! -f "$WORKDIR/$CSV_FILE" ]]; then
        echo "Error: $CSV_FILE not found in $WORKDIR" >&2
        exit 1
    fi
    if [[ -z "$OUTPUT_FILE" ]]; then
        OUTPUT_FILE="${CSV_FILE%.csv}.xlsx"
    fi

    echo "command : convert"
    echo "workdir : $WORKDIR"
    echo "input   : $CSV_FILE"
    echo "output  : $OUTPUT_FILE"
    echo "image   : $IMAGE"
    echo ""

    docker run --rm \
        -v "$WORKDIR:/data" \
        "$IMAGE" \
        convert \
        --input  "/data/$CSV_FILE" \
        --output "/data/$OUTPUT_FILE"
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
    echo "Available commands: validate, convert, sample" >&2
    exit 1
    ;;
esac
