#!/usr/bin/env python3
"""Zip a v4 agent folder into a complete-project ZIP the platform can prepare and run as is.

    python3 pack_agent.py [--agent agent] [--out my-agent.zip] [--include-env]

Upload the ZIP on the website's Participate page (Submit a complete project -> private ZIP). The kit's
baseline agent runs there without any model key (USE_LLM is "0" in its manifest).

Package rules (mirroring the platform's project checks):
  * observer.project.json sits at the zip root with "protocol": "jsonl-v4". The kit ships one (image
    python:3.12-slim, run `python3 -u baseline_agent.py`, no build steps); if the folder has none, a default one
    is generated for the entry script (baseline_agent.py, agent.py or main.py, first found);
  * every other file in the folder is packaged next to it;
  * requirements.txt is NOT installed automatically: add a `build` step to observer.project.json when your
    agent needs packages (README.md, "Upload a complete project"). It may only list installable
    distribution names with optional version specifiers (no local paths, URLs, -e, or pip options);
  * .env is left out: the platform rejects ZIPs that contain .env files, and permanent keys never belong in
    a ZIP. On the platform your agent gets OPENAI_BASE_URL / OPENAI_API_KEY (the platform's model proxy and a
    temporary run credential); set your own model key on the Participate page. --include-env packs .env for
    a local-only copy and prints a warning;
  * __pycache__, .venv, .deps, *.pyc, scratch/, run_output/ and editor/OS clutter are never packaged.

Standard library only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parent
ENTRY_CANDIDATES = ("baseline_agent.py", "agent.py", "main.py")
PROTOCOL = "jsonl-v4"
EXCLUDED_DIRS = {"__pycache__", ".venv", "venv", ".deps", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache", "scratch", "run_output", ".idea", ".vscode"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}
EXCLUDED_NAMES = {".DS_Store", "Thumbs.db"}
ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}
REQUIREMENT_LINE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"  # distribution name (PEP 508)
    r"(?:\[[A-Za-z0-9._,\s-]+\])?"                    # optional extras
    r"\s*(?:(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+(?:\s*,\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+)*)?"
    r"\s*(?:;.*)?$"                                    # optional environment marker
)
MAX_PACKAGE_BYTES = 20 << 20

# The platform's manifest rules (project_platform/manifest.py), repeated here so the kit stays dependency-free.
MANIFEST_NAME = "observer.project.json"
MANIFEST_SCHEMA = "observer-project-v1"
DEFAULT_IMAGE = "python:3.12-slim"
MANIFEST_FIELDS = {"schema_version", "image", "run", "build", "working_directory", "environment", "protocol"}
IMAGE_REFERENCE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/:-]*(?:@sha256:[0-9a-f]{64})?$")
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
PRIVATE_ENV = re.compile(r"(SECRET|PASSWORD|TOKEN|API_KEY|PRIVATE_KEY)")


def find_entry(agent_dir: Path) -> Path:
    for name in ENTRY_CANDIDATES:
        if (agent_dir / name).is_file():
            return agent_dir / name
    raise SystemExit(f"{agent_dir} must contain one of {', '.join(ENTRY_CANDIDATES)} at its top level")


def check_entry(entry: Path) -> None:
    try:
        ast.parse(entry.read_text(encoding="utf-8"), filename=str(entry))
    except SyntaxError as exc:
        raise SystemExit(f"{entry.name} has a syntax error: {exc}")
    text = entry.read_text(encoding="utf-8")
    if "__main__" not in text:
        print(f"warning: {entry.name} has no `if __name__ == \"__main__\":` block; the platform runs it as a script", file=sys.stderr)


def check_requirements(path: Path) -> list[str]:
    problems = []
    for number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        lowered = line.lower()
        if line.startswith("-") or "://" in line or lowered.startswith(("file:", "git+", "./", "../", "/")) or " @ " in line or line.endswith((".whl", ".tar.gz", ".zip")):
            problems.append(f"line {number}: {line!r} is not an installable distribution name")
        elif not REQUIREMENT_LINE.match(line):
            problems.append(f"line {number}: {line!r} is not a valid requirement specifier")
    return problems


def default_manifest(entry: Path) -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "protocol": PROTOCOL,
        "image": DEFAULT_IMAGE,
        "build": [],
        "run": ["python3", "-u", entry.name],
        "working_directory": ".",
        "environment": {"PYTHONDONTWRITEBYTECODE": "1", "USE_LLM": "0"},
    }


def _is_argv(value) -> bool:
    return (isinstance(value, list) and 1 <= len(value) <= 128 and
            all(isinstance(v, str) and v and "\x00" not in v and len(v) <= 8192 for v in value) and
            not value[0].startswith("-"))


def check_manifest(raw: bytes, agent_dir: Path) -> list[str]:
    """Return what the platform would reject in this observer.project.json (empty when it is valid)."""
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError):
        return [f"{MANIFEST_NAME} is not valid JSON"]
    if not isinstance(data, dict):
        return [f"{MANIFEST_NAME} must be a JSON object"]
    problems = []
    if set(data) - MANIFEST_FIELDS:
        problems.append("unknown fields: " + ", ".join(sorted(set(data) - MANIFEST_FIELDS)))
    if data.get("schema_version") != MANIFEST_SCHEMA:
        problems.append(f"schema_version must be {MANIFEST_SCHEMA}")
    image = data.get("image")
    if not isinstance(image, str) or len(image) > 256 or not IMAGE_REFERENCE.fullmatch(image):
        problems.append(f"image must be a container image reference such as {DEFAULT_IMAGE}")
    if data.get("protocol") != PROTOCOL:
        problems.append(f'protocol must be "{PROTOCOL}" (v4 cards speak participant-agent-protocol-v4)')
    if not _is_argv(data.get("run")):
        problems.append('run must be a nonempty array of command arguments, e.g. ["python3", "-u", "baseline_agent.py"]')
    build = data.get("build", [])
    if not isinstance(build, list) or len(build) > 16 or not all(_is_argv(command) for command in build):
        problems.append("build must be a list of at most 16 command-argument arrays")
    workdir = data.get("working_directory", ".")
    if (not isinstance(workdir, str) or workdir.startswith("/") or "\\" in workdir or ":" in workdir or
            (workdir != "." and any(part in ("", ".", "..") for part in workdir.split("/"))) or
            not (agent_dir / workdir).is_dir()):
        problems.append("working_directory must be an existing relative folder inside the project")
    env = data.get("environment", {})
    if not isinstance(env, dict) or len(env) > 32:
        problems.append("environment must be an object with at most 32 settings")
    else:
        for key, value in env.items():
            if (not ENV_NAME.fullmatch(key) or key.startswith(("GITHUB_", "SUPABASE_", "OBSERVER_", "ACTIONS_")) or
                    PRIVATE_ENV.search(key)):
                problems.append(f"environment {key}: credentials and platform settings must not be placed in the manifest")
            elif not isinstance(value, str) or "\x00" in value or len(value) > 4096:
                problems.append(f"environment {key} must be a short string")
    return problems


def collect(agent_dir: Path, include_env: bool) -> list[Path]:
    files = []
    for path in sorted(agent_dir.rglob("*")):
        rel = path.relative_to(agent_dir)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if not path.is_file() or path.name in EXCLUDED_NAMES or path.suffix in EXCLUDED_SUFFIXES:
            continue
        if path.name == ".env" and not include_env:
            continue
        if path.name.startswith(".env.") and path.name not in ENV_TEMPLATES:
            continue
        files.append(path)
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", type=Path, default=KIT_ROOT / "agent", help="agent folder (default: agent/)")
    parser.add_argument("--out", type=Path, default=KIT_ROOT / "my-agent.zip", help="zip path (default: my-agent.zip)")
    parser.add_argument("--include-env", action="store_true",
                        help="also pack agent/.env, for a local-only copy (the platform rejects ZIPs with .env; never upload it)")
    parser.add_argument("--no-env", action="store_true", help=argparse.SUPPRESS)  # the default now; kept for old scripts
    parser.add_argument("--allow-large", action="store_true", help=f"skip the {MAX_PACKAGE_BYTES >> 20} MB size guard")
    args = parser.parse_args(argv)

    agent_dir = args.agent.resolve()
    if not agent_dir.is_dir():
        raise SystemExit(f"agent folder not found: {agent_dir}")
    entry = find_entry(agent_dir)
    check_entry(entry)
    requirements = agent_dir / "requirements.txt"
    if requirements.is_file():
        problems = check_requirements(requirements)
        if problems:
            raise SystemExit("requirements.txt cannot be installed on the platform:\n  " + "\n  ".join(problems))
    manifest_path = agent_dir / MANIFEST_NAME
    generated = not manifest_path.is_file()
    manifest_bytes = ((json.dumps(default_manifest(entry), indent=2) + "\n").encode() if generated
                      else manifest_path.read_bytes())
    problems = check_manifest(manifest_bytes, agent_dir)
    if problems:
        raise SystemExit(f"{MANIFEST_NAME} would be rejected by the platform:\n  " + "\n  ".join(problems))
    manifest = json.loads(manifest_bytes)
    include_env = args.include_env and not args.no_env
    files = [path for path in collect(agent_dir, include_env=include_env) if path != manifest_path]
    out = args.out.resolve()
    if out.parent == agent_dir or agent_dir in out.parents:
        raise SystemExit("write the zip outside the agent folder, otherwise it would package itself")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, manifest_bytes)
        for path in files:
            archive.write(path, arcname=path.relative_to(agent_dir).as_posix())
    size = out.stat().st_size
    if size > MAX_PACKAGE_BYTES and not args.allow_large:
        out.unlink()
        raise SystemExit(f"package is {size / (1 << 20):.1f} MB; move data out of the agent folder or pass --allow-large")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"packed {len(files) + 1} files from {agent_dir} -> {out} ({size} bytes, sha256 {digest[:16]}...)")
    print(f"  {MANIFEST_NAME}{' (generated)' if generated else ''}: image {manifest['image']}, run {' '.join(manifest['run'])}")
    for path in files:
        print(f"  {path.relative_to(agent_dir).as_posix()}")
    if (agent_dir / ".env").is_file():
        if include_env:
            print("WARNING: .env is included. Keep this ZIP on your own computer: the platform rejects ZIPs that contain "
                  ".env, and permanent keys must never be uploaded. Set your model key on the Participate page instead.",
                  file=sys.stderr)
        else:
            print("  note: .env left out (keys never go into the ZIP; set your model key on the Participate page)")
    if requirements.is_file() and not manifest.get("build"):
        print("  note: requirements.txt is installed on the platform only by a build step in observer.project.json;"
              " the baseline agent needs none")
    if not include_env:
        print("  next: upload this ZIP on the website's Participate page (Submit a complete project -> private ZIP)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
