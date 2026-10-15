"""Keep pyproject.toml honest about what the code actually imports.

A dependency manifest drifts silently: someone imports a package, adds it to the
environment by hand, and the manifest keeps claiming the project needs less than it
does. Nothing fails until someone builds a fresh checkout.

So this walks every ``src/`` and ``tools/`` module, collects its top-level imports,
resolves each to the distribution that provides it via ``importlib.metadata`` (no
hardcoded mapping table), and checks both directions:

* every import resolves to a declared dependency or extra -- nothing undeclared
* every *core* dependency is installed -- extras are optional by definition, so a
  missing extra is reported as a note rather than a failure, otherwise this check
  stays red forever on any machine that skipped ``.[brain]``
* the console-script entry points name callables that exist
* ``[tool.setuptools] packages`` covers every package on disk

Run it after touching pyproject.toml or any import:

    python tools/check_deps.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, packages_distributions, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PYPROJECT = ROOT / "pyproject.toml"
REPORT = ROOT / "logs" / "deps_check.json"

results = []


def check(name, ok, detail=""):
    results.append({"name": name, "ok": bool(ok), "detail": detail})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, (" -- " + detail) if detail else ""),
          file=sys.stderr)
    return bool(ok)


def note(text):
    print("      note: %s" % text, file=sys.stderr)


def dep_name(spec: str) -> str:
    """'flask>=3.0' -> 'flask'. Strips specifiers, extras and environment markers.

    Deliberately not a regex over the whole string: normalising first would turn the
    dot in ">=3.0" into a dash and produce a name no distribution can match.
    """
    head = re.split(r"[\s\[<>=!;@]", spec.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", head).lower()


def top_level_imports(path: Path) -> set:
    """Top-level module names imported by one file, ignoring relative imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # from .x import y -> intra-package
                continue
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def main() -> int:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})
    core = {dep_name(d): d for d in project.get("dependencies", [])}
    extras = {
        dep_name(d): d
        for spec in (project.get("optional-dependencies", {}) or {}).values()
        for d in spec
    }
    known = set(core) | set(extras)

    local = {"src", "tools"}
    stdlib = set(sys.stdlib_module_names)

    imports = {}
    # rglob rather than glob: tools/smb1/ is a real package with its own modules,
    # and a flat scan silently stops checking the moment a sub-package appears.
    for folder in (ROOT / "src", ROOT / "tools"):
        for path in sorted(folder.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            imports[str(path.relative_to(ROOT)).replace("\\", "/")] = top_level_imports(path)

    mapping = packages_distributions()
    resolved = {}      # dist name -> example import that needed it
    unresolved = {}    # import name -> where it was imported from
    for source, names in imports.items():
        for name in sorted(names):
            if name in stdlib or name in local:
                continue
            dists = mapping.get(name)
            if dists:
                for dist in dists:
                    resolved.setdefault(dep_name(dist), "%s (import %s)" % (source, name))
            else:
                unresolved.setdefault(name, source)

    check(
        "every third-party import is either installed or a declared dependency",
        all(dep_name(n) in known for n in unresolved),
        "not declared: %s" % sorted(n for n in unresolved if dep_name(n) not in known)
        if any(dep_name(n) not in known for n in unresolved)
        else "%d installed, %d declared-but-absent (%s)"
        % (len(resolved), len(unresolved), ", ".join(sorted(unresolved)) or "none"),
    )

    undeclared = sorted(d for d in resolved if d not in known)
    check(
        "every imported distribution is declared in pyproject",
        not undeclared,
        "undeclared: %s" % undeclared if undeclared else "all declared (%d)" % len(resolved),
    )

    missing_core = []
    for dist, spec in sorted(core.items()):
        try:
            version(dist)
        except PackageNotFoundError:
            missing_core.append(spec)
    check(
        "every core dependency is installed",
        not missing_core,
        "missing: %s" % missing_core if missing_core else "%d/%d present" % (len(core), len(core)),
    )

    missing_extra = []
    for dist, spec in sorted(extras.items()):
        try:
            version(dist)
        except PackageNotFoundError:
            missing_extra.append(spec)
    if missing_extra:
        note(
            "optional extra packages not installed: %s -- the reflex policy and the "
            "dashboard work without them; run pip install -e \".[brain]\" for laya" % missing_extra
        )
    else:
        note("brain extra fully installed: %s" % sorted(extras))

    bad_scripts = []
    for name, target in (project.get("scripts") or {}).items():
        module_name, _, func = target.partition(":")
        try:
            module = __import__(module_name, fromlist=["_"])
        except Exception as exc:  # noqa: BLE001
            bad_scripts.append("%s -> %s (%r)" % (name, target, exc))
            continue
        if not callable(getattr(module, func, None)):
            bad_scripts.append("%s -> %s (%s is not callable)" % (name, target, func))
    check("console-script entry points resolve", not bad_scripts,
          "; ".join(bad_scripts) if bad_scripts else ", ".join(project.get("scripts") or {}))

    # setuptools spells a sub-package with dots ("tools.smb1"); normalise both sides
    # to paths so a dotted declaration is not reported as missing.
    packages = {
        name.replace(".", "/")
        for name in (data.get("tool", {}).get("setuptools", {}) or {}).get("packages", [])
    }
    # Every directory holding an __init__.py is a package, at any depth -- a flat
    # top-level scan misses tools/smb1/ and would let it ship undeclared.
    on_disk = {
        str(path.parent.relative_to(ROOT)).replace("\\", "/")
        for path in ROOT.rglob("__init__.py")
        if not set(path.parts) & {"logs", ".venv", "roms", "__pycache__", ".git"}
    }
    missing = sorted(on_disk - packages)
    check("[tool.setuptools] packages covers every package on disk",
          not missing, "missing from packages: %s" % missing if missing else ", ".join(sorted(packages)))

    stale = []
    for path in list(ROOT.glob("*.md")) + list(ROOT.glob("roms/*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "pip install -r" in text:
            stale.append(path.name)
    check("no file still tells the reader to pip install -r requirements.txt",
          not stale, "stale: %s" % stale if stale else "clean")

    passed = sum(1 for r in results if r["ok"])
    payload = {
        "passed": passed,
        "total": len(results),
        "core": sorted(core),
        "extras": sorted(extras),
        "resolved_from_imports": sorted(resolved),
        "imported_but_not_installed": sorted(unresolved),
        "results": results,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n%d/%d passed -> %s" % (passed, len(results), REPORT), file=sys.stderr)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
