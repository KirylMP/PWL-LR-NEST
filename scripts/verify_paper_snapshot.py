#!/usr/bin/env python3
"""Verify the committed paper model files and their legacy copies."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    snapshot = root / "models" / "paper_frozen_v1"
    legacy = root / "astrocyte_lab" / "paper_models"
    try:
        manifest = json.loads((snapshot / "manifest.json").read_text())
        models = manifest["models"]
        if not models:
            raise ValueError("manifest contains no models")
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: cannot read snapshot manifest: {exc}", file=sys.stderr)
        return 1
    errors = []
    for name, metadata in sorted(models.items()):
        for filename, expected in sorted(metadata["sha256"].items()):
            for base in (snapshot, legacy):
                path = base / name / filename
                try:
                    actual = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError as exc:
                    errors.append(f"{path.relative_to(root)}: {exc}")
                    continue
                if actual != expected:
                    errors.append(f"{path.relative_to(root)}: SHA-256 mismatch")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"OK: {len(models)} models match the manifest in both locations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
