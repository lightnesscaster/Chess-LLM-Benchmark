"""Preserve the standalone Codex release layout in a deployment artifact."""

import argparse
from pathlib import Path
import shutil


def package_runtime(binary: Path, destination: Path) -> Path:
    """Copy Codex and its companion resources, resolving the install symlink."""
    binary = binary.resolve(strict=True)
    if binary.parent.name != "bin" or not (binary.parent / "codex-code-mode-host").is_file():
        raise ValueError("Incomplete Codex release: missing bin/codex-code-mode-host")
    release = binary.parent.parent
    destination = destination.resolve()
    if destination == release or release in destination.parents or destination in release.parents:
        raise ValueError("Codex artifact must be separate from the installed release")
    shutil.copytree(release, destination, dirs_exist_ok=True)
    return destination / "bin" / "codex"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    package_runtime(args.binary, args.destination)
