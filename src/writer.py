import csv
import sys
from pathlib import Path
from typing import IO, Any
from types import TracebackType

from .config import Config


def build_comment(config: Config, input_path: str) -> str:
    api = config["api"]
    gen = config["generation"]
    return (
        f"# model={api['model']} "
        f"max_tokens={gen['max_tokens']} "
        f"temperature={gen['temperature']} "
        f"top_p={gen['top_p']} "
        f"repetition_penalty={gen['repetition_penalty']} "
        f"seed={gen['seed']} "
        f"input={input_path}"
    )


class CsvWriter:
    """Context manager that streams rows to CSV as they are produced.

    Avoids buffering all results in memory before writing.
    """

    def __init__(self, config: Config, fieldnames: list[str], output_path: str, input_path: str) -> None:
        self._path = Path(output_path)
        self._fieldnames = fieldnames
        self._comment = build_comment(config, input_path)
        self._file: IO[str] | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._count = 0

    def __enter__(self) -> "CsvWriter":
        try:
            self._file = open(self._path, "w", encoding="utf-8", newline="")
        except OSError as e:
            print(f"[ERROR] Failed to open output file: {e}")
            sys.exit(1)
        self._file.write(self._comment + "\n")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        self._writer.writeheader()
        return self

    def write_row(self, row: dict[str, Any]) -> None:
        assert self._writer is not None, "write_row called outside context manager"
        assert self._file is not None, "write_row called outside context manager"
        self._writer.writerow(row)
        self._file.flush()  # survive crashes mid-run
        self._count += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._file is not None:
            self._file.close()
        if exc_type is None:
            print(f"\nSaved {self._count} rows → {self._path.resolve()}")
