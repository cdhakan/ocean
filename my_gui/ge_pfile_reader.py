"""
ge_pfile_reader.py — GE P-file (raw) reconstruction for OCEAN via Orchestra.
=========================================================================
The GE Orchestra SDK is proprietary and its compiled Python module is tied to a
specific Python / NumPy ABI, so it is NOT imported into OCEAN's process.
Instead the reconstruction runs out-of-process in a dedicated environment
(``install_orchestra.sh`` → conda env ``orchestra``, Python 3.10 + NumPy 1.x)
through ``my_gui/orchestra_worker.py``; this module locates that environment,
runs the worker and converts its output into OCEAN's usual contract:

    read_pfile_cest(path) → (stack (Y, X, n_slices, n_frames) float32,
                             M0 (Y, X, n_slices), info dict)

Discovery (first hit wins):
    OCEAN_ORCHESTRA_PY / OCEAN_ORCHESTRA_SDK environment variables,
    ~/.ocean/orchestra.json  {"python": "...", "sdk": "..."}  (written by the
    install script), conda env named ``orchestra`` + an ``orchestra-sdk*``
    folder in ~ or ~/Downloads.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

_CONFIG = os.path.join(os.path.expanduser("~"), ".ocean", "orchestra.json")
_CONDA_ROOTS = ("miniforge3_default", "miniforge3", "miniconda3", "anaconda3", "mambaforge", "micromamba")
_LIBS = ("GERecon.so.mac", "GERecon.so", "GERecon.pyd", "GERecon.so.python36")


def config_path() -> str:
    """~/.ocean/orchestra.json (OCEAN_ORCHESTRA_CONFIG overrides, used by tests)."""
    return os.environ.get("OCEAN_ORCHESTRA_CONFIG") or _CONFIG


def _config() -> dict:
    try:
        with open(config_path()) as f:
            return json.load(f) or {}
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# One-click setup (what install_orchestra.sh does, in pure Python so the frozen
# app can offer it from the Scan Directory)
# ──────────────────────────────────────────────────────────────────────────────

def sdk_python_version() -> str:
    """GE built the macOS/Windows libraries for Python 3.10, the Linux one
    (GERecon.so.python36) for 3.6."""
    return "3.6" if sys.platform.startswith("linux") else "3.10"


def find_conda() -> "str | None":
    """conda executable: PATH, then the usual user-level installs."""
    import shutil
    p = os.environ.get("CONDA_EXE")          # set by `conda init` shells
    if p and os.path.isfile(p):
        return p
    for name in ("conda", "mamba"):
        p = shutil.which(name)
        if p:
            return p
    home = os.path.expanduser("~")
    exe = "conda.exe" if sys.platform == "win32" else "conda"
    sub = ("Scripts",) if sys.platform == "win32" else ("bin",)
    cands = [os.path.join(home, r, *sub, exe) for r in _CONDA_ROOTS]
    cands += [os.path.join("/opt", r, "bin", exe) for r in ("miniforge3", "miniconda3", "anaconda3", "homebrew/Caskroom/miniforge/base")]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def setup_orchestra(sdk_dir: str, env_name: str = "orchestra", log_cb=None,
                    python_version: "str | None" = None) -> "tuple[str, str]":
    """Create the isolated conda env, clear macOS quarantine on the SDK library,
    write the config and self-test.  Returns (python, sdk).  Raises with a
    user-readable message on any failure.  ``log_cb(str)`` receives progress."""
    def log(m):
        if log_cb:
            log_cb(m)
    sdk_dir = os.path.abspath(sdk_dir)
    if not _has_lib(sdk_dir):
        raise RuntimeError("That folder has no GERecon library (GERecon.so.mac / GERecon.so / "
                           "GERecon.pyd) — pick the orchestra-sdk-*.python folder from GE.")
    conda = find_conda()
    if not conda:
        raise RuntimeError("conda was not found. Install Miniforge first "
                           "(https://github.com/conda-forge/miniforge), then try again.")
    pyver = python_version or sdk_python_version()
    log(f"conda: {conda}")
    # does the env already exist?
    r = subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True)
    envs = []
    try:
        envs = json.loads(r.stdout).get("envs", [])
    except Exception:
        pass
    env_dir = next((e for e in envs if os.path.basename(e.rstrip(os.sep)) == env_name), None)
    if env_dir is None:
        log(f"Creating conda env '{env_name}' (Python {pyver}, NumPy 1.x) — this takes a few minutes…")
        p = subprocess.Popen([conda, "create", "-y", "-n", env_name, "-c", "conda-forge",
                              f"python={pyver}", "numpy<2"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            line = line.rstrip()
            if line:
                log("  " + line[:120])
        if p.wait() != 0:
            raise RuntimeError("conda create failed — see the log above.")
        r = subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True)
        envs = json.loads(r.stdout).get("envs", [])
        env_dir = next((e for e in envs if os.path.basename(e.rstrip(os.sep)) == env_name), None)
        if env_dir is None:
            raise RuntimeError(f"env '{env_name}' was created but cannot be located")
    else:
        log(f"conda env '{env_name}' already exists: {env_dir}")
    py = (os.path.join(env_dir, "python.exe") if sys.platform == "win32"
          else os.path.join(env_dir, "bin", "python"))
    if not os.path.isfile(py):
        raise RuntimeError(f"python not found in {env_dir}")
    if sys.platform == "darwin":
        lib = os.path.join(sdk_dir, "GERecon.so.mac")
        if os.path.isfile(lib):
            subprocess.run(["xattr", "-d", "com.apple.quarantine", lib], capture_output=True)
            log("macOS quarantine flag cleared on GERecon.so.mac")
    cfg = config_path()
    os.makedirs(os.path.dirname(cfg), exist_ok=True)
    with open(cfg, "w") as f:
        json.dump({"python": py, "sdk": sdk_dir}, f)
    log(f"config written: {cfg}")
    log("self-test: importing GERecon in the new env…")
    r = subprocess.run([py, _worker_path(), "--check", "--sdk", sdk_dir],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError("self-test failed:\n" + (r.stderr or r.stdout)[-600:])
    log(r.stdout.strip())
    return py, sdk_dir


def _has_lib(d: str) -> bool:
    return bool(d) and os.path.isdir(d) and any(os.path.isfile(os.path.join(d, n)) for n in _LIBS)


def find_orchestra() -> "tuple[str | None, str | None]":
    """(python executable, SDK folder) or (None, None) parts when not found."""
    cfg = _config()
    home = os.path.expanduser("~")
    exe = "python.exe" if sys.platform == "win32" else "bin/python"

    py = os.environ.get("OCEAN_ORCHESTRA_PY") or cfg.get("python")
    if not (py and os.path.isfile(py)):
        py = None
        for root in _CONDA_ROOTS:
            cand = os.path.join(home, root, "envs", "orchestra", exe)
            if os.path.isfile(cand):
                py = cand
                break

    sdk = os.environ.get("OCEAN_ORCHESTRA_SDK") or cfg.get("sdk")
    if not _has_lib(sdk or ""):
        sdk = None
        for pat in (os.path.join(home, "orchestra-sdk*"), os.path.join(home, "Downloads", "orchestra-sdk*"),
                    "/opt/orchestra*"):
            for d in sorted(glob.glob(pat), reverse=True):
                if _has_lib(d):
                    sdk = d
                    break
            if sdk:
                break
    return py, sdk


def is_available() -> bool:
    py, sdk = find_orchestra()
    return bool(py and sdk)


def availability_message() -> str:
    py, sdk = find_orchestra()
    if py and sdk:
        return f"Orchestra env: {py}  |  SDK: {sdk}"
    missing = []
    if not py:
        missing.append("the 'orchestra' Python env")
    if not sdk:
        missing.append("the Orchestra SDK folder")
    return ("GE P-file reconstruction needs " + " and ".join(missing) +
            " — click 'Set up GE Orchestra…' next to Format (one-time; needs the GE SDK folder "
            "and conda), or run:  bash install_orchestra.sh /path/to/orchestra-sdk-2.1-1.python")


def _worker_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "orchestra_worker.py")
    if os.path.isfile(cand):
        return cand
    # frozen app: the worker is bundled as data next to the package
    base = getattr(sys, "_MEIPASS", here)
    return os.path.join(base, "my_gui", "orchestra_worker.py")


def run_worker(pfile: str, gradwarp: bool = True, timeout_s: int = 1800) -> "tuple[np.ndarray, dict]":
    """Reconstruct in the Orchestra env; returns (stack, meta)."""
    py, sdk = find_orchestra()
    if not (py and sdk):
        raise RuntimeError(availability_message())
    with tempfile.TemporaryDirectory(prefix="ocean_orchestra_") as td:
        out = os.path.join(td, "recon.npz")
        cmd = [py, _worker_path(), pfile, out, "--sdk", sdk]
        if not gradwarp:
            cmd.append("--no-gradwarp")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)            # keep OCEAN's env out of the worker
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
        if r.returncode != 0 or not os.path.isfile(out):
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-6:]
            raise RuntimeError("Orchestra reconstruction failed:\n" + "\n".join(tail))
        z = np.load(out, allow_pickle=False)
        stack = np.asarray(z["stack"], dtype=np.float32)
        meta = json.loads(str(z["meta"]))
    return stack, meta


def read_pfile_cest(path: str, slice_idx: "int | None" = None,
                    gradwarp: bool = True) -> tuple:
    """Reconstruct a GE P-file CEST / WASSR scan.

    Returns (stack (Y, X, n_slices, n_frames) float32, M0 (Y, X, n_slices),
    info).  ``slice_idx`` keeps one slice (None = all).  Frames are the
    acquisition phases (× echoes) — one per saturation offset; the ppm list is
    not stored in the P-file (supply a .txt or the sequence's offsets).
    """
    stack, meta = run_worker(path, gradwarp=gradwarp)
    if slice_idx is not None and stack.shape[2] > 1:
        stack = stack[:, :, [min(int(slice_idx), stack.shape[2] - 1)], :]
    n_off = stack.shape[3]
    fs = stack.reshape(-1, n_off).mean(axis=0)
    m0_idx = int(np.argmax(fs))
    M0 = stack[:, :, :, m0_idx].copy()
    md = meta.get("metadata", {})
    px = meta.get("pixsize_mm") or [None, None]
    info = {
        "n_offsets":    n_off,
        "n_phases":     int(md.get("phases", 1)),
        "n_echoes":     int(md.get("echoes", 1)),
        "n_slices":     int(stack.shape[2]),
        "n_coils":      int(md.get("channels", 1)),
        "matrix":       (int(stack.shape[0]), int(stack.shape[1])),
        "m0_frame_idx": m0_idx,
        "fov_mm":       (meta.get("fov_mm"), meta.get("fov_mm"), meta.get("slice_thickness_mm")),
        "voxel_mm":     (px[0], px[1], meta.get("slice_thickness_mm")),
        "te_ms":        (meta.get("te_us") or 0) / 1000.0,
        "tr_ms":        (meta.get("tr_us") or 0) / 1000.0,
        "psdname":      meta.get("psdname"),
        "b0_t":         (meta.get("magstrength_gauss") or 0) / 1e4,
        "gradwarp":     meta.get("gradwarp"),
        "user_cv":      meta.get("user_cv_rec"),
        "source":       "GE P-file (Orchestra)",
        "recon_seconds": meta.get("recon_seconds"),
    }
    return stack, M0, info


def convert_pfile_to_acquired_mat(pfile: str, out_dir: "str | None" = None) -> str:
    """P-file → acquired_data.mat (CEST-MRF layout, see bruker_reader.save_acquired_data)."""
    from my_gui.bruker_reader import save_acquired_data
    stack, M0, info = read_pfile_cest(pfile)
    if out_dir is None:
        base = os.path.splitext(os.path.basename(pfile))[0]
        out_dir = os.path.join(os.path.dirname(os.path.abspath(pfile)), base)
    os.makedirs(out_dir, exist_ok=True)

    def _clean(v):
        if isinstance(v, (tuple, list)):
            return [np.nan if x is None else x for x in v]
        return np.nan if v is None else v
    meta = {k: _clean(v) for k, v in info.items() if k != "user_cv"}
    meta["pfile"] = os.path.abspath(pfile)
    return save_acquired_data(os.path.join(out_dir, "acquired_data"), stack, meta, None, fmt="mat")
