#!/usr/bin/env python3

import os
import sys
from pathlib import Path


def main() -> int:
    local_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    if local_python.is_file() and sys.prefix == sys.base_prefix:
        os.execv(str(local_python), [str(local_python), *sys.argv])

    from pricing_mapper.cli import run_cli

    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
