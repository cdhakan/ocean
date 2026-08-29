"""Ensure SWIG-generated BMCSimulator.py files are valid UTF-8.

On the Windows runner, SWIG emits a cp1252 byte (e.g. an em-dash 0x97) into
BMCSimulator.py, which makes PyInstaller abort during analysis with
"SyntaxError: invalid or missing encoding declaration".  This re-encodes to
UTF-8 (no-op if already UTF-8) BOTH the installed copy and any file paths passed
as arguments (e.g. the in-place `build_ext --inplace` output, which PyInstaller
also scans).
"""
import importlib.util
import os
import sys


def _fix(path):
    if not (path and path.endswith(".py") and os.path.isfile(path)):
        return
    data = open(path, "rb").read()
    try:
        data.decode("utf-8")
        print(f"  UTF-8 OK: {path}")
    except UnicodeDecodeError:
        # cp1252 maps every byte, so this never fails; SWIG output is cp1252/ASCII.
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(data.decode("cp1252"))
        print(f"  re-encoded -> UTF-8: {path}")


# The installed copy (site-packages / egg).
spec = importlib.util.find_spec("BMCSimulator")
_fix(getattr(spec, "origin", None) if spec else None)

# Any explicit paths (e.g. the in-place build output ./BMCSimulator.py).
for p in sys.argv[1:]:
    _fix(os.path.abspath(p))
