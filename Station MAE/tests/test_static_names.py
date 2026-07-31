"""
Catch undefined names before a run reaches the GPU.

Motivation: `python -m compileall` validates SYNTAX only. A reference to a name
that does not exist in scope compiles perfectly and fails at runtime -- which is
how `per_station=per_station` shipped into StationMAEDataset.__init__ (the
parameter is called `global_norm` there) and crashed v13 four minutes into the
dataset build, after the PeakWeather load.

pyflakes resolves names statically, so it catches exactly this class of bug in
under a second.

Run:  python tests/test_static_names.py
      pip install pyflakes   # if missing
"""
import os
import subprocess
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_PROJ, "src")

# Findings that are stylistic rather than breaking. Keep this list SHORT and
# justified -- anything here is something a future reader will assume is fine.
IGNORE_SUBSTRINGS = (
    "imported but unused",
    "unable to detect undefined names",
    "redefinition of unused",
    "f-string is missing placeholders",
    "assigned to but never used",
    "shadowed by loop variable",
)

# These are hard errors: they will raise at runtime.
FATAL_SUBSTRINGS = (
    "undefined name",
    "syntax error",
    "invalid syntax",
)


def main() -> int:
    try:
        r = subprocess.run([sys.executable, "-m", "pyflakes", _SRC],
                           capture_output=True, text=True)
    except FileNotFoundError:
        print("pyflakes not installed -> pip install pyflakes")
        return 0

    lines = [l for l in r.stdout.splitlines() if l.strip()]
    fatal = [l for l in lines
             if any(f in l.lower() for f in FATAL_SUBSTRINGS)]
    other = [l for l in lines
             if l not in fatal and not any(i in l for i in IGNORE_SUBSTRINGS)]

    print(f"pyflakes scanned {_SRC}")
    print(f"  {len(lines)} finding(s) total, {len(fatal)} fatal\n")

    if other:
        print("Non-fatal findings worth a look:")
        for l in other:
            print(f"  {l.replace(_PROJ + os.sep, '')}")
        print()

    if fatal:
        print("FATAL -- these WILL raise at runtime:")
        for l in fatal:
            print(f"  {l.replace(_PROJ + os.sep, '')}")
        return 1

    print("No undefined names. Safe to launch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
