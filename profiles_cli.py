#!/usr/bin/env python3
"""
Small interactive wrapper for the two profile scripts:
- pipe_profile_generator_LOGOS.py (generate inlet profile: laminar/turbulent)
- read_velocity_profile.py     (read/normalize/plot a profile)

This wrapper is intentionally simple: it delegates parameter collection to the
underlying scripts (their interactive modes), so you can just pick an action.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _here() -> Path:
    return Path(__file__).resolve().parent


def _python_exe() -> str:
    return sys.executable or "python"


def _run(cmd: list[str]) -> int:
    try:
        return subprocess.call(cmd, cwd=str(_here()))
    except FileNotFoundError:
        print("Error: Python executable not found.")
        return 1


def main() -> int:
    generator = _here() / "pipe_profile_generator_LOGOS.py"
    reader = _here() / "read_velocity_profile.py"

    if not generator.exists():
        print(f"Error: missing script: {generator}")
        return 1
    if not reader.exists():
        print(f"Error: missing script: {reader}")
        return 1

    print("=== Profiles CLI ===")
    print("1) Generate pipe inlet profile (CSV)")
    print("2) Read/normalize/plot velocity profile (CSV/TXT + PNG)")
    print("0) Exit")

    choice = input("Select action [1/2/0]: ").strip()

    if choice == "0" or choice.lower() in {"q", "quit", "exit"}:
        return 0
    if choice == "1":
        # Delegate prompting to the generator script.
        cmd = [_python_exe(), str(generator), "--interactive"]
        return _run(cmd)
    if choice == "2":
        # The reader prompts for the input path when omitted.
        # It also auto-saves PNG next to the input file.
        cmd = [_python_exe(), str(reader)]
        return _run(cmd)

    print("Unknown choice. Please enter 1, 2 or 0.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

