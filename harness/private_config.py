"""Where operator-private material lives, and why it is not here.

Running this harness against a real agent means touching things that must never
enter this repository: an employer's behavioural rules, an internal rubric, a
room naming convention, credentials. The harness is built so none of that has to
be in the tree to be used.

**The default location is outside the repository.** ``~/.duplex-harness`` by
default, overridable with ``DUPLEX_HARNESS_PRIVATE_DIR``. This is deliberate and
it is the whole point of the module.

A ``.gitignore`` entry is a good second line of defence and a bad first one. It
stops ``git add .`` and stops nothing else: ``git add -f`` overrides it, a
rewritten ignore file silently un-protects every file it was covering, and an
editor, an indexer, a backup agent or an AI assistant with repository access
reads the working tree without consulting it at all. A file that is never in the
working tree cannot be committed by any of those routes. So confidential
material is kept out of the tree, and the ignore rules exist to catch the case
where someone puts it there anyway.

:func:`private_dir` warns loudly if the configured directory turns out to sit
inside the repository, because at that point the protection has been reduced to
the ignore file alone and the operator should know it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ENV_VAR = "DUPLEX_HARNESS_PRIVATE_DIR"
DEFAULT_PRIVATE_DIR = Path.home() / ".duplex-harness"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PrivateConfigError(RuntimeError):
    """Raised when private configuration is requested but unusable."""


def private_dir(create: bool = False) -> Path:
    """Resolve the private directory, warning if it sits inside the repository."""
    raw = os.environ.get(ENV_VAR, "").strip()
    path = Path(raw).expanduser().resolve() if raw else DEFAULT_PRIVATE_DIR

    try:
        inside = path.resolve().is_relative_to(PROJECT_ROOT)
    except (OSError, ValueError):
        inside = False
    if inside:
        print(
            f"warning: {ENV_VAR} points inside the repository ({path}).\n"
            "         Private material there is protected only by .gitignore, "
            "which any\n"
            "         forced add or rewritten ignore file defeats. Prefer a "
            "directory\n"
            f"         outside the project, such as {DEFAULT_PRIVATE_DIR}.",
            file=sys.stderr,
        )

    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def private_path(name: str, create_dir: bool = False) -> Path:
    """Path to a named file in the private directory."""
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise PrivateConfigError(
            f"private file name must be a simple relative name, got {name!r}"
        )
    return private_dir(create=create_dir) / name


def load_private_json(name: str, required: bool = False) -> Optional[Dict[str, Any]]:
    """Read a JSON file from the private directory.

    Returns ``None`` when the file is absent and ``required`` is False, so the
    public path stays fully functional for anyone who has no private
    configuration at all -- which is everybody except the operator.
    """
    path = private_path(name)
    if not path.exists():
        if required:
            raise PrivateConfigError(
                f"required private file {name!r} not found at {path}. "
                f"Create it, or point {ENV_VAR} at the directory holding it."
            )
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PrivateConfigError(f"{path} is not valid JSON: {exc}") from exc


def load_env_file(path: Optional[Path] = None, override: bool = False) -> int:
    """Load ``KEY=VALUE`` lines from a .env into the environment.

    The project documents a ``.env`` and gitignores it, so it has to actually be
    read: telling someone to fill in a file that nothing loads produces a
    credentials error that points at the environment while the values sit on
    disk, correct, a metre away.

    Written against the standard library rather than pulling in python-dotenv.
    The format that matters here is fifteen lines of parsing, and the whole test
    suite still runs on a test runner alone.

    Values already present in the environment win by default, so an explicit
    ``export`` or a CI secret is never silently replaced by a stale file.
    Returns how many variables were set.
    """
    path = path or (PROJECT_ROOT / ".env")
    if not path.exists():
        return 0

    loaded = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key:
            continue
        value = value.strip()
        # Strip one matching pair of surrounding quotes, as every other .env
        # reader does; a quoted value is quoting, not part of the secret.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if override or not os.environ.get(key):
            os.environ[key] = value
            loaded += 1
    return loaded


def describe() -> str:
    """Human-readable summary, for the CLI to print when asked."""
    path = private_dir()
    present = sorted(p.name for p in path.glob("*")) if path.exists() else []
    lines = [
        f"private directory: {path}",
        f"  set with       : {ENV_VAR}",
        f"  exists         : {path.exists()}",
        f"  contains       : {', '.join(present) if present else '(nothing)'}",
        "",
        "Nothing in this directory is read by the public code paths, committed,",
        "or referenced by any test. It is where an operator's own rules and",
        "settings live so that they never enter the repository.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
