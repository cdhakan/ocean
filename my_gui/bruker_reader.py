"""
bruker_reader.py
Python equivalent of MATLAB read2dseq.m + readPars.m.

Reads Bruker ParaVision binary 2dseq image files and parameter files
(method, acqp) for MRF ('dictmatch') and CEST ('cest'/'wassr') datasets.

Bruker directory structure assumed:
    <study>/
    └── <scan_num>/          ← scan root (contains method, acqp)
        └── pdata/
            └── 1/           ← scan_dir (pass this path in)
                └── 2dseq    ← binary image data

Usage:
    from my_gui.bruker_reader import read_2dseq_mrf, read_2dseq_cest, save_acquired_data
"""

from __future__ import annotations
import re
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Parameter file parsing  (equivalent to readPars.m)
# ─────────────────────────────────────────────────────────────────────────────

def read_bruker_params(scan_dir: str | Path, parfile: str, parnames: list[str]) -> dict[str, str | None]:
    """
    Read parameter values from a Bruker parameter file.

    Args:
        scan_dir : Path to the directory containing 2dseq (pdata/1/).
                   The method/acqp files are located two levels up.
        parfile  : 'method' or 'acqp'
        parnames : list of parameter name strings, e.g. ['##$PVM_Matrix', ...]

    Returns:
        dict  parname -> string value (None if not found)
    """
    param_path = (Path(scan_dir) / ".." / ".." / parfile).resolve()

    try:
        with open(param_path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Parameter file not found: {param_path}")

    results = {name: None for name in parnames}

    for parname in parnames:
        for i, line in enumerate(lines):
            if "=" not in line:
                continue
            token, _, rest = line.partition("=")
            if token.strip() != parname:
                continue
            rest = rest.strip()
            if rest.startswith("("):
                # Multi-line value: collect until next ## or $$ line
                val_parts: list[str] = []
                j = i + 1
                while j < len(lines):
                    nl = lines[j].strip()
                    if nl.startswith("#") or nl.startswith("$"):
                        break
                    val_parts.append(nl)
                    j += 1
                results[parname] = " ".join(val_parts)
            else:
                results[parname] = rest
            break

    return results


def _parse_numbers(s: str | None) -> np.ndarray:
    """Extract all numbers from a string into a float numpy array."""
    if not s:
        return np.array([], dtype=np.float64)
    tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return np.array([float(t) for t in tokens], dtype=np.float64)


def _expand_pv360(value: str | None) -> str | None:
    """
    Expand compressed PV360 array notation like '(3) 1.0 2.0 3.0'
    or '@3*(1.5)' repeat syntax into a flat list string.
    Not all variants handled — returns value unchanged if not recognised.
    """
    if not value:
        return value
    # Handle '@N*(val)' repeat syntax
    expanded = re.sub(
        r"@(\d+)\*\(([^)]+)\)",
        lambda m: (m.group(2) + " ") * int(m.group(1)),
        value,
    )
    # Strip leading size hint like '(30)'
    expanded = re.sub(r"^\s*\(\d+\)\s*", "", expanded)
    return expanded.strip()


def _detect_cest_b1(scan_dir: "str | Path", pv360: bool) -> tuple[float, str]:
    """
    Robustly extract the CEST saturation B1 amplitude in µT from the Bruker
    method file.

    Priority order (all expected to be in µT):
      1. ##$Fp_SatPows            — OCEAN/QUESP sequences (may use @N*(val) syntax)
      2. ##$PVM_SatTransPulseAmpl_uT — PV360 and some non-360 sequences
      3. Any ##$PVM_*Power* key whose name suggests µT (case-insensitive 'ut' or 'uT')
      4. Any remaining ##$PVM_*Power* key  (last resort; units may not be µT)
      5. ##$PVM_MagTransPower     — legacy, often in Watts — treated as last resort

    Returns (b1_ut, key_used).  Returns (0.0, '') if nothing found.
    """
    method_path = (Path(scan_dir) / ".." / ".." / "method").resolve()
    try:
        method_text = method_path.read_text(errors="replace")
    except (FileNotFoundError, OSError):
        return 0.0, ""

    def _extract_key(text: str, key: str) -> float:
        """
        Pull the value for one ##$KEY from the method file text.
        Handles both single-line (##$KEY=value) and multi-line
        (##$KEY=( N )\n@N*(val) or ##$KEY=( N )\nval1 val2 ...) forms.
        Always applies _expand_pv360 before _parse_numbers so that the
        @N*(val) run-length encoding is expanded before numeric extraction.
        """
        # Build a pattern that captures everything on the key line plus the
        # continuation lines (up to the next ##$ / $$ record).
        pattern = re.compile(
            r"^##\$" + re.escape(key.lstrip("#$")) + r"=([^\n]*(?:\n(?!##|\$\$)[^\n]*)*)",
            re.MULTILINE,
        )
        m = pattern.search(text)
        if not m:
            return 0.0
        raw = m.group(1).strip()
        # Strip leading size hint like "( 50 )" that precedes the actual data
        raw = re.sub(r"^\(\s*\d+\s*\)\s*", "", raw).strip()
        expanded = _expand_pv360(raw) or ""
        nums = _parse_numbers(expanded)
        if len(nums) == 0:
            return 0.0
        return float(nums[0])

    # ── Priority 1 & 2: well-known µT keys ────────────────────────────────
    for key in ("##$Fp_SatPows", "##$PVM_SatTransPulseAmpl_uT"):
        val = _extract_key(method_text, key)
        if val > 0.0:
            return val, key

    # ── Priority 3 & 4: scan all ##$PVM_*Power* keys ──────────────────────
    # Collect all power-related PVM keys that appear in the file.
    all_pvm_power = re.findall(r"^(##\$PVM_\w*[Pp]ow\w*)=", method_text, re.MULTILINE)
    seen: set[str] = set()
    ut_keys, other_keys = [], []
    for k in all_pvm_power:
        if k in seen:
            continue
        seen.add(k)
        # Keys with 'uT' or 'ut' in their name are most likely already in µT
        if re.search(r"[Uu][Tt]", k):
            ut_keys.append(k)
        else:
            other_keys.append(k)

    for key in ut_keys + other_keys:
        val = _extract_key(method_text, key)
        # Sanity-check: typical CEST B1 is 0.1 – 30 µT
        if 0.05 < val < 50.0:
            return val, key

    # ── Priority 5: last resort ────────────────────────────────────────────
    val = _extract_key(method_text, "##$PVM_MagTransPower")
    if val > 0.0:
        return val, "##$PVM_MagTransPower"

    return 0.0, ""


# ─────────────────────────────────────────────────────────────────────────────
# RECO parameter helpers (pdata/1/reco)
# ─────────────────────────────────────────────────────────────────────────────

def _read_reco_params(scan_dir: Path) -> dict[str, str | None]:
    """
    Read RECO_size, RECO_wordtype, RECO_byte_order from pdata/1/reco.

    ``scan_dir`` is the directory that contains 2dseq (i.e. pdata/1/).
    Returns a dict with those three keys; any missing key is None.
    Never raises — silently returns all-None on any I/O error.
    """
    reco_path = scan_dir / "reco"
    result: dict[str, str | None] = {
        "RECO_size":       None,
        "RECO_wordtype":   None,
        "RECO_byte_order": None,
    }
    try:
        with open(reco_path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except (FileNotFoundError, OSError):
        return result

    targets = {
        "##$RECO_size":       "RECO_size",
        "##$RECO_wordtype":   "RECO_wordtype",
        "##$RECO_byte_order": "RECO_byte_order",
    }
    for key, out_key in targets.items():
        for i, line in enumerate(lines):
            if "=" not in line:
                continue
            token, _, rest = line.partition("=")
            if token.strip() != key:
                continue
            rest = rest.strip()
            if rest.startswith("("):
                # multi-line value
                val_parts: list[str] = []
                j = i + 1
                while j < len(lines):
                    nl = lines[j].strip()
                    if nl.startswith("#") or nl.startswith("$"):
                        break
                    val_parts.append(nl)
                    j += 1
                result[out_key] = " ".join(val_parts)
            else:
                result[out_key] = rest
            break

    return result


def _reco_dtype(wordtype: str | None, byte_order: str | None) -> "np.dtype":
    """
    Map Bruker RECO_wordtype + RECO_byte_order to a numpy dtype.

    Falls back to little-endian int16 if the wordtype is unrecognised.
    """
    _wtype_map: dict[str, str] = {
        "_8BIT_UNSGN_INT": "u1",
        "_16BIT_SGN_INT":  "i2",
        "_32BIT_SGN_INT":  "i4",
        "_32BIT_FLOAT":    "f4",
        "_64BIT_FLOAT":    "f8",
    }
    base = _wtype_map.get((wordtype or "").strip(), "i2")
    _big = {"bigEndian", "ieee-be", "big", "big_endian", "BIG_ENDIAN"}
    prefix = ">" if (byte_order or "").strip() in _big else "<"
    return np.dtype(f"{prefix}{base}")


# ─────────────────────────────────────────────────────────────────────────────
# Common geometry reader
# ─────────────────────────────────────────────────────────────────────────────

def _read_geometry(scan_dir: Path) -> tuple[int, int, int, int]:
    """Return (nx, ny, nslices, niter).

    Image dimensions are taken from ``pdata/1/reco`` (``RECO_size``) which
    reflects any zero-filling applied during reconstruction.  If the reco file
    is absent or its values are invalid the method-file ``PVM_Matrix`` is used
    as a fallback.  This makes the reader transparent to 64 / 96 / 128 / 256
    matrix sizes without any user intervention.
    """
    pars = read_bruker_params(scan_dir, "method", [
        "##$PVM_Matrix",
        "##$PVM_SPackArrNSlices",
        "##$PVM_NRepetitions",
        "##$Number_fp_Experiments",
        "##$PVM_SatTransRepetitions",
    ])

    # --- Matrix size: prefer RECO_size (actual reconstructed size) ---
    reco_pars = _read_reco_params(scan_dir)
    reco_sz = _parse_numbers(reco_pars.get("RECO_size")).astype(int)
    if len(reco_sz) >= 2 and reco_sz[0] > 0 and reco_sz[1] > 0:
        nx, ny = int(reco_sz[0]), int(reco_sz[1])
    else:
        # Fall back to k-space encoding size from method file
        mat = _parse_numbers(pars.get("##$PVM_Matrix")).astype(int)
        if len(mat) < 2:
            raise ValueError(
                f"Could not read PVM_Matrix from method file in {scan_dir}. "
                f"Got: {pars.get('##$PVM_Matrix')!r}"
            )
        nx, ny = int(mat[0]), int(mat[1])

    # --- Number of slices ---
    nsl_arr = _parse_numbers(pars.get("##$PVM_SPackArrNSlices"))
    nslices = int(nsl_arr[0]) if len(nsl_arr) > 0 else 1

    # --- Number of repetitions / experiments ---
    candidates = [
        _parse_numbers(pars.get("##$PVM_NRepetitions")),
        _parse_numbers(pars.get("##$Number_fp_Experiments")),
        _parse_numbers(pars.get("##$PVM_SatTransRepetitions")),
    ]
    niter = max(int(v[0]) if len(v) > 0 else 0 for v in candidates)

    # Fallback: derive niter from actual file size when parameters are absent
    if niter == 0:
        fpath = scan_dir / "2dseq"
        try:
            reco_pars2 = _read_reco_params(scan_dir)
            dt = _reco_dtype(reco_pars2.get("RECO_wordtype"), reco_pars2.get("RECO_byte_order"))
            file_bytes = fpath.stat().st_size
            denom = nx * ny * nslices * dt.itemsize
            niter = int(file_bytes // denom) if denom > 0 else 1
        except Exception:
            niter = 1

    return nx, ny, nslices, niter


def _read_raw(scan_dir: Path, nx: int, ny: int, nslices: int, niter: int) -> np.ndarray:
    """Read the 2dseq binary file, returns shape (nx, ny, nslices, niter).

    The data type is auto-detected from ``pdata/1/reco`` (``RECO_wordtype`` /
    ``RECO_byte_order``).  Defaults to little-endian int16 when the reco file
    is absent or the wordtype is unrecognised.

    If the file size does not match (nx × ny × nslices × niter) a warning is
    issued and ``niter`` is silently re-derived from the actual file size so
    the reshape never fails.
    """
    import warnings
    fpath = scan_dir / "2dseq"

    reco_pars = _read_reco_params(scan_dir)
    dt = _reco_dtype(reco_pars.get("RECO_wordtype"), reco_pars.get("RECO_byte_order"))

    raw = np.fromfile(str(fpath), dtype=dt)
    expected = nx * ny * nslices * niter

    if raw.size != expected:
        denom = nx * ny * nslices
        if denom > 0 and raw.size > 0:
            niter_actual = raw.size // denom
            warnings.warn(
                f"2dseq size mismatch in {scan_dir}: "
                f"file has {raw.size} values (dtype={dt}), "
                f"expected {expected} ({nx}×{ny}×{nslices}×{niter}). "
                f"Re-deriving niter={niter_actual} from file size.",
                stacklevel=3,
            )
            niter = niter_actual
            expected = nx * ny * nslices * niter
            raw = raw[:expected]
        else:
            raw = np.zeros(max(expected, 1), dtype=dt)

    return raw.reshape(nx, ny, nslices, niter, order="F")


# ─────────────────────────────────────────────────────────────────────────────
# MRF / dictmatch reader  (equivalent to read2dseq 'dictmatch')
# ─────────────────────────────────────────────────────────────────────────────

def read_seq_defs_from_bruker(scan_dir: str | Path, pv360: bool = False) -> tuple[dict, dict]:
    """
    Read MRF sequence definitions from Bruker method + acqp files.

    Ports the MATLAB pipeline logic exactly, including the overpowered-B1
    correction (single-digit power tokens in PpgPowerList1/ppgPowerList1).

    Args:
        scan_dir : path to directory containing 2dseq (pdata/1/).
                   method/acqp are found two levels up.
        pv360    : True for ParaVision 360 format

    Returns:
        seq_defs : dict with keys:
                     num_meas, n_pulses, tp (s), td (s), Trec (s),
                     B1pa (µT), excFA (°), SLFA (°), SLflag (bool),
                     offsets_ppm, DCsat, Trec_M0, M0_offset, B0
        info     : dict with keys:
                     B0 (T), schedule (filename string)
    """
    scan_dir = Path(scan_dir)

    # ── Parameter keys ──────────────────────────────────────────────────────
    par_keys = [
        "##$Method",                # [0]  method name
        "##$Number_fp_Experiments", # [1]  num_meas
        "##$Fp_SatDur",             # [2]  tp  (ms)
        "##$Fp_TRDels",             # [3]  (unused directly)
        "##$Fp_SatPows",            # [4]  B1 amplitudes  (µT)
        "##$PVM_FrqWork",           # [5]  Larmor freq  (MHz)
        "##$Fp_SatOffset",          # [6]  offsets  (Hz or ppm for PV360)
        "##$Fp_SLflag",             # [7]  spin-lock flag
        "##$PVM_RefPowCh1",         # [8]  reference power  (W)
        "##$Fp_FlipAngle",          # [9]  excitation flip angle
        "##$Fp_SLFlipAngle",        # [10] spin-lock flip angle
        "##$Fp_FileName",           # [11] schedule filename
    ]
    parread = read_bruker_params(scan_dir, "method", par_keys)

    if pv360:
        more_keys = [
            "##$PVM_SatTransInterPulseDelay",  # td  (ms)
            "##$PVM_SatTransNPulses",           # n_pulses
            "##$PpgPowerList1",                 # actual pulse powers  (W)
        ]
    else:
        more_keys = [
            "##$PVM_MagTransInterDelay",        # td  (ms)
            "##$PVM_MagTransPulsNumb",          # n_pulses
            "##$PVM_ppgPowerList1",             # actual pulse powers  (W)
        ]
    moreparread = read_bruker_params(scan_dir, "method", more_keys)
    trread      = read_bruker_params(scan_dir, "acqp", ["##$ACQ_vd_list"])

    # ── PV360 array-format expansion ─────────────────────────────────────────
    for d in (parread, moreparread, trread):
        for k in d:
            d[k] = _expand_pv360(d[k])

    # ── Larmor frequency ─────────────────────────────────────────────────────
    nu_0_arr = _parse_numbers(parread.get("##$PVM_FrqWork"))
    nu_0 = float(nu_0_arr[0]) if len(nu_0_arr) > 0 else 297.0   # MHz; 297 ≈ 7 T

    # ── Scalar / array parameters ────────────────────────────────────────────
    seq_defs: dict = {}

    _nm = _parse_numbers(parread.get("##$Number_fp_Experiments"))
    seq_defs["num_meas"] = int(_nm[0]) if len(_nm) > 0 else 1

    more_vals = list(moreparread.values())          # [td_str, npulses_str, powlist_str]
    _np_arr   = _parse_numbers(more_vals[1]) if len(more_vals) > 1 else np.array([])
    seq_defs["n_pulses"] = int(_np_arr[0]) if len(_np_arr) > 0 else 1

    seq_defs["tp"]   = _parse_numbers(parread.get("##$Fp_SatDur")) / 1000.0   # ms → s
    _td_raw          = _parse_numbers(more_vals[0]) if len(more_vals) > 0 else np.array([0.0])
    seq_defs["td"]   = _td_raw / 1000.0                                        # ms → s
    seq_defs["Trec"] = _parse_numbers(trread.get("##$ACQ_vd_list"))            # already in s

    seq_defs["excFA"] = _parse_numbers(parread.get("##$Fp_FlipAngle"))
    slfa_raw          = _parse_numbers(parread.get("##$Fp_SLFlipAngle"))

    # SLFA convention: isSL=1 → 90°, isSL=0 → 0°
    # Bruker stores SLFA in ##$Fp_SLFlipAngle; if absent or all-zero,
    # apply the physical rule directly.
    if len(slfa_raw) > 0 and np.any(slfa_raw != 0):
        seq_defs["SLFA"] = slfa_raw
    else:
        # Build per-measurement SLFA: 90° where SL flag will be True, else 0°
        # (SLflag not yet computed here; we apply after SLflag is set below)
        seq_defs["SLFA"] = slfa_raw if len(slfa_raw) > 0 else np.zeros_like(seq_defs["excFA"])

    # ── B1 amplitudes — with overpowered-entry correction ────────────────────
    # Bruker stores overloaded power values as a single-digit token (not '0')
    # in the PPG power list.  When detected, recalculate from the raw power:
    #   B1 = (0.25 / 0.001 / 42.577) × sqrt(P_raw / P_ref)   [µT]
    b1_arr = _parse_numbers(parread.get("##$Fp_SatPows")).copy()

    pow_str    = (more_vals[2] if len(more_vals) > 2 else "") or ""
    pow_tokens = pow_str.split()
    seq_pows:    list[float] = []
    overpow_idx: list[int]   = []
    for tok in pow_tokens:
        try:
            seq_pows.append(float(tok))
            if len(tok) == 1 and tok != "0":          # single-digit → overflow
                overpow_idx.append(len(seq_pows) - 1)
        except ValueError:
            pass
    seq_pows_arr = np.array(seq_pows, dtype=float)

    if overpow_idx:
        import warnings
        warnings.warn(
            f"Overpowered RF entries detected at indices {overpow_idx} — "
            "recalculating B1pa from raw PPG power list."
        )
        refpow_arr = _parse_numbers(parread.get("##$PVM_RefPowCh1"))
        refpow = float(refpow_arr[0]) if len(refpow_arr) > 0 else 1.0
        for i in overpow_idx:
            if i < len(b1_arr) and refpow > 0:
                b1_arr[i] = 0.25 / 0.001 / 42.577 * np.sqrt(seq_pows_arr[i] / refpow)

    seq_defs["B1pa"] = b1_arr

    # ── Saturation offsets ────────────────────────────────────────────────────
    offsets_raw = _parse_numbers(parread.get("##$Fp_SatOffset"))
    if pv360:
        offsets_hz = offsets_raw * nu_0       # PV360 stores in ppm → convert to Hz
    else:
        offsets_hz = offsets_raw              # older PV stores directly in Hz

    # ── Spin-lock flag ────────────────────────────────────────────────────────
    is_sl_str  = (parread.get("##$Fp_SLflag") or "").strip()
    method_str = (parread.get("##$Method")    or "").strip()
    if "fp_EPI" in method_str or not is_sl_str:
        sl_flag = np.zeros(len(offsets_hz), dtype=bool)
        sl_flag[offsets_hz == 0] = True
    else:
        sl_flag = _parse_numbers(is_sl_str).astype(bool)
    seq_defs["SLflag"] = sl_flag

    # ── Apply SLFA rule now that SLflag is known ──────────────────────────────
    # Convention: isSL=1 → 90°, isSL=0 → 0°
    # If Bruker stored explicit non-zero SLFA values, keep them; otherwise enforce rule.
    slfa_cur = seq_defs["SLFA"]
    if not np.any(slfa_cur != 0):
        slfa_fixed = np.where(sl_flag, 90.0, 0.0)
        seq_defs["SLFA"] = slfa_fixed

    # ── Derived fields ────────────────────────────────────────────────────────
    seq_defs["offsets_ppm"] = offsets_hz / nu_0 if nu_0 != 0 else offsets_hz.copy()
    tp, td = seq_defs["tp"], seq_defs["td"]
    denom  = tp + td
    seq_defs["DCsat"]    = np.where(denom > 0, tp / denom, np.zeros_like(tp))
    seq_defs["Trec_M0"]  = np.nan
    seq_defs["M0_offset"] = np.nan
    seq_defs["B0"]        = round(nu_0 / 42.577, 1)

    info = {
        "B0":      seq_defs["B0"],
        "schedule": (parread.get("##$Fp_FileName") or "").strip(),
    }

    return seq_defs, info


def load_scan_first_frame(scan_dir, pv360: bool = False):
    """Return the 1st 2-D image frame of a Bruker scan as a float array oriented
    like the app's maps (Y, X), or ``None`` if it cannot be read.

    Accepts either a ``pdata/1`` directory or a scan-number directory
    (``pdata/1`` is appended automatically).  Used as a grayscale background for
    the "ROIs + Bkg" overlay so the user can pick any Scan-Directory image."""
    base = Path(scan_dir)
    for d in (base, base / "pdata" / "1"):
        if (d / "2dseq").exists():
            try:
                nx, ny, nsl, nit = _read_geometry(d)
                raw = np.asarray(_read_raw(d, nx, ny, nsl, nit), dtype=float)
                if raw.ndim == 4:
                    frame = raw[:, :, 0, 0]
                else:
                    frame = raw.reshape(nx, ny, -1)[:, :, 0]
                return frame.T                     # (nx, ny) -> display (Y, X)
            except Exception:
                return None
    return None


def read_2dseq_mrf(scan_dir: str | Path, pv360: bool = False) -> tuple[np.ndarray, dict, dict]:
    """
    Read Bruker 2dseq MRF data and extract sequence parameters.

    Args:
        scan_dir : path to directory containing 2dseq (pdata/1/)
        pv360    : True for ParaVision 360 format

    Returns:
        acquired_data : float32 ndarray (Y, X, slices, niter)
        info          : dict  { 'size', 'B0', 'schedule' }
        seq_defs      : dict  (see read_seq_defs_from_bruker)
    """
    scan_dir = Path(scan_dir)
    nx, ny, nslices, niter = _read_geometry(scan_dir)
    rawdata = _read_raw(scan_dir, nx, ny, nslices, niter)

    # Permute x/y to match Bruker orientation
    acquired_data = np.transpose(rawdata, (1, 0, 2, 3)).astype(np.float32)

    # Reuse the dedicated seq_defs reader (avoids code duplication)
    seq_defs, seq_info = read_seq_defs_from_bruker(scan_dir, pv360=pv360)

    info = {
        "size":     [nx, ny, nslices, niter],
        "B0":       seq_info["B0"],
        "schedule": seq_info["schedule"],
    }

    return acquired_data, info, seq_defs


# ─────────────────────────────────────────────────────────────────────────────
# CEST / z-spectroscopy reader  (equivalent to read2dseq 'cest'/'wassr')
# ─────────────────────────────────────────────────────────────────────────────

def read_2dseq_cest(scan_dir: str | Path, pv360: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Read Bruker 2dseq CEST / z-spectroscopy data.

    Args:
        scan_dir : path to directory containing 2dseq (pdata/1/)
        pv360    : True for ParaVision 360 format

    Returns:
        image    : float64 ndarray (Y, X, slices, n_offsets) sorted low→high ppm
        M0image  : float64 ndarray (Y, X, slices)
        info     : dict  { 'w_offset1' (Hz), 'w_offsetPPM', 'omega_0',
                           'satpwr_uT', 'size' }
    """
    import warnings
    scan_dir = Path(scan_dir)
    nx, ny, nslices, niter = _read_geometry(scan_dir)
    rawdata = _read_raw(scan_dir, nx, ny, nslices, niter)

    # ── Larmor key ────────────────────────────────────────────────────────
    frq_key = "##$PVM_FrqWork"

    # ── Offset key priority list ──────────────────────────────────────────
    #
    #  PV360   : ##$PVM_SatTransFreqValues  — already in ppm
    #  Non-PV360 new style: ##$Fp_SatOffset — already in ppm  (user data)
    #             ##$Fp_SatOffsetHz         — in Hz (fallback, Hz)
    #  Non-PV360 old style: ##$SatFreqList  — in Hz
    #
    # ── Power key priority list ───────────────────────────────────────────
    #
    #  PV360   : ##$PVM_SatTransPulseAmpl_uT — directly in µT
    #  Non-PV360 (tried in order, first non-zero wins):
    #    1. ##$Fp_SatPows              — OCEAN/QUESP sequence, already µT
    #    2. ##$PVM_SatTransPulseAmpl_uT— some non-360 sequences also use this
    #    3. ##$PVM_MagTransPower       — legacy fallback; units may not be µT
    #
    if pv360:
        freq_candidates = [("##$PVM_SatTransFreqValues", True)]   # (key, is_ppm)
        pwr_candidates  = ["##$PVM_SatTransPulseAmpl_uT"]
    else:
        freq_candidates = [
            ("##$Fp_SatOffset",    True),   # new style — ppm
            ("##$Fp_SatOffsetHz",  False),  # new style — Hz
            ("##$SatFreqList",     False),  # old style — Hz
        ]
        pwr_candidates  = [
            "##$Fp_SatPows",               # OCEAN/QUESP seq — already µT  ← first choice
            "##$PVM_SatTransPulseAmpl_uT", # some non-360 scanners
            "##$PVM_MagTransPower",         # legacy last resort (may not be µT)
        ]

    # Read all candidate keys plus larmor
    all_keys = [fc[0] for fc in freq_candidates] + pwr_candidates + [frq_key]
    satpars = read_bruker_params(scan_dir, "method", all_keys)
    for k in satpars:
        satpars[k] = _expand_pv360(satpars[k])

    # Find the first key that returns a non-empty array
    w_offset    = np.array([], dtype=np.float64)
    freq_is_ppm = False
    freq_key_used = ""
    for key, is_ppm in freq_candidates:
        arr = _parse_numbers(satpars.get(key))
        if len(arr) > 0:
            w_offset     = arr
            freq_is_ppm  = is_ppm
            freq_key_used = key
            break

    # ── Larmor frequency (MHz → used to convert Hz ↔ ppm) ────────────────
    omega_0_arr = _parse_numbers(satpars.get(frq_key))
    omega_0 = float(omega_0_arr[0]) if len(omega_0_arr) > 0 else 297.0

    # ── Trim / pad w_offset to niter ─────────────────────────────────────
    if len(w_offset) == 0:
        warnings.warn(
            f"CEST: no saturation offset key found in method file of {scan_dir}. "
            f"Tried: {[fc[0] for fc in freq_candidates]}. Using zeros."
        )
        w_offset = np.zeros(niter)
    if len(w_offset) != niter:
        warnings.warn(
            f"CEST: offset array length ({len(w_offset)}) != niter ({niter}). "
            f"Trimming/padding.  Key used: {freq_key_used!r}"
        )
        padded = np.zeros(niter)
        n = min(len(w_offset), niter)
        padded[:n] = w_offset[:n]
        w_offset = padded

    # ── Convert to ppm ────────────────────────────────────────────────────
    if freq_is_ppm or pv360:
        w_offset_ppm_raw = w_offset.copy()
        w_offset_hz_raw  = w_offset * omega_0
    else:
        w_offset_hz_raw  = w_offset.copy()
        w_offset_ppm_raw = w_offset / omega_0

    # ── Store COMPLETE dataset (original order, includes M0 frame) ────────
    # Transpose rawdata to (Y, X, slices, niter) for all-frames storage
    img_all = np.transpose(rawdata, (1, 0, 2, 3)).astype(np.float64)

    # ── Identify M0 frame (largest |ppm| offset) ──────────────────────────
    #
    # CEST datasets have an explicit off-resonance reference at a very large
    # offset (typically ±100 ppm).  That frame is used as M0 and is REMOVED
    # from the analysis subset so fitting is not contaminated.
    #
    # WASSR datasets cover only a small range (≤ ±2 ppm) for B0 mapping and
    # have NO separate M0 frame.  The frame at the largest |ppm| within the
    # acquisition is used as a proxy M0 for normalisation, but it is KEPT in
    # the analysis subset so all acquired offsets are available for B0 fitting.
    abs_ppm = np.abs(w_offset_ppm_raw)
    m0_idx = int(np.argmax(abs_ppm))   # frame at largest |ppm| — always the best proxy

    if abs_ppm.max() > 50.0:
        # True CEST: explicit far-off-resonance M0 (e.g. 100 ppm)
        has_m0 = True
    else:
        # WASSR / no explicit M0: keep ALL frames in analysis subset
        has_m0 = False

    # Use .copy() so M0image is independent of img_all.
    # Without this, M0image is a view; storing img_all as _z_img_all in the
    # tab would mean any in-place write to that array silently corrupts M0.
    M0image = img_all[:, :, :, m0_idx].copy()

    # ── Build analysis array ───────────────────────────────────────────────
    # CEST: remove the M0 frame, then sort remaining offsets high → low ppm
    #       (descending: first positive, then negative — matches MATLAB convention).
    # WASSR: keep ALL frames (no M0 to remove), sorted high → low ppm.
    keep = np.ones(niter, dtype=bool)
    if has_m0:
        keep[m0_idx] = False   # remove M0 only for true CEST datasets
    w_rest_ppm = w_offset_ppm_raw[keep]
    w_rest_hz  = w_offset_hz_raw[keep]
    sort_idx   = np.argsort(-w_rest_ppm)   # descending: [+9, ..., 0, ..., -9]
    w_sorted_ppm = w_rest_ppm[sort_idx]
    w_sorted_hz  = w_rest_hz[sort_idx]
    image = img_all[:, :, :, keep][:, :, :, sort_idx]

    # Detect saturation B1 using the robust scanner — handles @N*(val) syntax,
    # PV360 and non-PV360 keys, and falls back to a global PVM_*Power scan.
    satpwr_ut, pwr_key_used = _detect_cest_b1(scan_dir, pv360)

    info = {
        # Analysis arrays (M0 removed, sorted high → low ppm — MATLAB convention)
        "w_offset1":    w_sorted_hz,
        "w_offsetPPM":  w_sorted_ppm,
        "omega_0":      omega_0,
        "satpwr_uT":    satpwr_ut,
        "size":         [nx, ny, nslices, int(np.sum(keep))],
        # Complete original-order arrays (for scroll display / M0 selection)
        "img_all":      img_all,            # (Y, X, slices, niter) full dataset
        "ppm_all":      w_offset_ppm_raw,   # shape (niter,) in original order
        "hz_all":       w_offset_hz_raw,
        "m0_frame_idx": m0_idx,
        "has_explicit_m0": has_m0,
        "freq_key_used":   freq_key_used,
        "pwr_key_used":    pwr_key_used,    # for diagnostics
    }

    return image, M0image, info


# ─────────────────────────────────────────────────────────────────────────────
# QUESP reader  (equivalent to read2dseq 'quesp')
# ─────────────────────────────────────────────────────────────────────────────

def read_2dseq_quesp(scan_dir: str | Path, pv360: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Read Bruker 2dseq QUESP data (quantitative exchange saturation power).

    Ports read2dseq.m 'quesp' case from MATLAB.

    Returns:
        image    : float64 ndarray (Y, X, slices, n_sat) — saturation images only
        M0image  : float64 ndarray (Y, X [, slices]) — unsaturated reference
        info     : dict with sat_amplitudes, sat_offsets, sat_duration,
                   sat_powers, ref_power, LarmorFreq, Trec, size
    """
    import warnings
    scan_dir = Path(scan_dir)
    nx, ny, nslices, niter = _read_geometry(scan_dir)
    rawdata = _read_raw(scan_dir, nx, ny, nslices, niter)

    # Read saturation parameters
    sat_keys = [
        "##$Fp_SatPows",        # sat amplitudes (µT)
        "##$Fp_SatOffset",      # sat offsets (Hz or ppm depending on PV version)
        "##$Fp_SatDur",         # saturation duration (ms)
        "##$PVM_FrqWork",       # Larmor freq (MHz)
        "##$PVM_RefPowCh1",     # reference power (W)
    ]
    satpars = read_bruker_params(scan_dir, "method", sat_keys)
    for k in satpars:
        satpars[k] = _expand_pv360(satpars[k])

    trread = read_bruker_params(scan_dir, "acqp", ["##$ACQ_vd_list"])
    for k in trread:
        trread[k] = _expand_pv360(trread[k])

    if pv360:
        pwr_keys = ["##$PpgPowerList1"]
    else:
        pwr_keys = ["##$PVM_ppgPowerList1"]
    pwrread = read_bruker_params(scan_dir, "method", pwr_keys)
    for k in pwrread:
        pwrread[k] = _expand_pv360(pwrread[k])

    nu_0_arr = _parse_numbers(satpars.get("##$PVM_FrqWork"))
    nu_0 = float(nu_0_arr[0]) if len(nu_0_arr) > 0 else 297.0  # MHz

    sat_amplitudes = _parse_numbers(satpars.get("##$Fp_SatPows"))
    _pwr_vals = list(pwrread.values())
    sat_powers = _parse_numbers(_pwr_vals[0]) if _pwr_vals else np.array([])
    ref_power_arr  = _parse_numbers(satpars.get("##$PVM_RefPowCh1"))
    ref_power      = float(ref_power_arr[0]) if len(ref_power_arr) > 0 else 1.0
    sat_offsets    = _parse_numbers(satpars.get("##$Fp_SatOffset"))

    if pv360:
        sat_offsets = sat_offsets * nu_0   # ppm → Hz

    sat_duration = _parse_numbers(satpars.get("##$Fp_SatDur"))
    trec         = _parse_numbers(trread.get("##$ACQ_vd_list"))

    # Check for RF power overload (stored value length == 1 char → recalculate)
    ref_pow_str = (satpars.get("##$PVM_RefPowCh1") or "").strip()
    _pwr_raw = (_pwr_vals[0] if _pwr_vals else None) or ""
    pow_tokens = _pwr_raw.split()
    overpower_idx = []
    for i, tok in enumerate(pow_tokens):
        if len(tok) == 1 and tok != "0":
            overpower_idx.append(i)
    if overpower_idx:
        warnings.warn("Saturation amplitudes exceeding max RF power detected — recalculating.")
        for idx in overpower_idx:
            if idx < len(sat_powers) and ref_power > 0:
                sat_amplitudes[idx] = (0.25 / 0.001 / 42.577) * np.sqrt(sat_powers[idx] / ref_power)

    # Identify M0 images (sat_amplitudes < 1e-3 µT)
    # Always build a boolean mask of exactly *niter* elements so we can safely
    # index rawdata whose last axis == niter.
    if len(sat_amplitudes) == 0 or niter == 0:
        # No amplitude info or no frames — treat first frame as M0 (if any)
        m0_mask = np.zeros(niter, dtype=bool)
        if niter > 0:
            m0_mask[0] = True
    elif len(sat_amplitudes) == niter:
        # Happy path: one amplitude per frame
        m0_mask = sat_amplitudes < 1e-3
        if not np.any(m0_mask):
            m0_mask[0] = True   # fallback: first image
    else:
        # Length mismatch — build a niter-length mask from the shorter array
        warnings.warn(
            f"QUESP: sat_amplitudes length ({len(sat_amplitudes)}) != niter ({niter}). "
            "Using available amplitudes for M0 detection."
        )
        n = min(len(sat_amplitudes), niter)
        m0_mask = np.zeros(niter, dtype=bool)
        m0_mask[:n] = sat_amplitudes[:n] < 1e-3
        if not np.any(m0_mask):
            m0_mask[0] = True   # fallback: first image

    # Guard: nothing to extract
    if niter == 0 or rawdata.shape[3] == 0:
        empty = np.zeros((rawdata.shape[1], rawdata.shape[0], max(rawdata.shape[2], 1)), dtype=np.float64)
        _trec_arr = np.array(trec).ravel()
        info = {
            "sat_amplitudes": sat_amplitudes,
            "sat_offsets":    sat_offsets,
            "sat_duration":   float(sat_duration[0]) if len(sat_duration) > 0 else 0.0,
            "sat_powers":     sat_powers,
            "ref_power":      ref_power,
            "LarmorFreq":     nu_0 * 1e6,
            "omega_0":        nu_0 * 1e6,
            "Trec":           float(_trec_arr[0]) if len(_trec_arr) > 0 else 0.0,
            "size":           [rawdata.shape[0], rawdata.shape[1], rawdata.shape[2], 0],
        }
        return empty, empty[..., 0], info

    M0image = np.squeeze(np.transpose(rawdata[:, :, :, m0_mask], (1, 0, 2, 3))).astype(np.float64)

    # Remove M0 from arrays
    # m0_mask has exactly *niter* elements; trim sat_* arrays to niter first
    keep = ~m0_mask   # shape (niter,)
    if len(sat_offsets) != niter:
        # Pad / trim to match niter
        padded = np.zeros(niter, dtype=sat_offsets.dtype)
        n = min(len(sat_offsets), niter)
        padded[:n] = sat_offsets[:n]
        sat_offsets = padded
    if len(sat_amplitudes) != niter:
        padded = np.zeros(niter, dtype=sat_amplitudes.dtype)
        n = min(len(sat_amplitudes), niter)
        padded[:n] = sat_amplitudes[:n]
        sat_amplitudes = padded
    sat_offsets    = sat_offsets[keep]
    sat_amplitudes = sat_amplitudes[keep]
    rawdata_rest   = rawdata[:, :, :, keep]

    image = np.squeeze(np.transpose(rawdata_rest, (1, 0, 2, 3))).astype(np.float64)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    info = {
        "sat_amplitudes": sat_amplitudes,
        "sat_offsets":    sat_offsets,
        "sat_duration":   sat_duration,
        "sat_powers":     sat_powers,
        "ref_power":      ref_power,
        "LarmorFreq":     nu_0,
        "Trec":           trec,
        "size":           [nx, ny, nslices, int(np.sum(keep))],
    }

    return image, M0image, info


def save_quesp_data(
    output_path: str | Path,
    image: np.ndarray,
    M0image: np.ndarray,
    info: dict,
    fmt: str = "mat",
) -> str:
    """Save QUESP image + M0image + info to .mat or .npz."""
    output_path = Path(output_path)

    def _to_saveable(v):
        if v is None:
            return np.array([])
        if isinstance(v, str):
            return np.bytes_(v)
        return np.array(v)

    if fmt == "npz":
        fpath = str(output_path) + ".npz"
        flat: dict = {"image": image, "M0image": M0image}
        for k, v in info.items():
            flat[f"info_{k}"] = _to_saveable(v)
        np.savez_compressed(fpath, **flat)
        return fpath
    else:
        from scipy.io import savemat
        fpath = str(output_path) + ".mat"
        savemat(fpath, {
            "image":   image,
            "M0image": M0image,
            "info":    {k: _to_saveable(v) for k, v in info.items()},
        })
        return fpath


# ─────────────────────────────────────────────────────────────────────────────
# Save functions — HDF5, NPZ, or MAT
# ─────────────────────────────────────────────────────────────────────────────

def save_acquired_data(
    output_path: str | Path,
    acquired_data: np.ndarray,
    info: dict,
    seq_defs: dict | None = None,
    fmt: str = "mat",
) -> str:
    """
    Save MRF acquired_data + metadata to disk.

    Args:
        output_path : path without extension (extension appended automatically)
        acquired_data : float32 ndarray (Y, X, slices, niter)
        info          : dict from read_2dseq_mrf
        seq_defs      : dict from read_2dseq_mrf
        fmt           : 'h5' (HDF5), 'npz' (NumPy), or 'mat' (MATLAB)

    Returns:
        Absolute path to the saved file.
    """
    output_path = Path(output_path)

    def _to_saveable(v):
        if v is None:
            return np.array([])
        if isinstance(v, str):
            return np.bytes_(v)
        return np.array(v)

    if fmt == "npz":
        fpath = str(output_path) + ".npz"
        flat: dict = {"acquired_data": acquired_data}
        for k, v in info.items():
            flat[f"info_{k}"] = _to_saveable(v)
        if seq_defs:
            for k, v in seq_defs.items():
                flat[f"seq_{k}"] = _to_saveable(v)
        np.savez_compressed(fpath, **flat)
        return fpath

    elif fmt == "mat":
        from scipy.io import savemat
        fpath = str(output_path) + ".mat"
        mat_data = {
            "acquired_data": acquired_data,
            "info": {k: _to_saveable(v) for k, v in info.items()},
        }
        if seq_defs:
            sd_clean = {}
            for k, v in seq_defs.items():
                sd_clean[k] = np.array(v).astype(float) if isinstance(v, np.ndarray) and v.dtype == bool else v
            mat_data["seq_defs"] = sd_clean
        savemat(fpath, mat_data)
        return fpath

    else:
        raise ValueError(f"Unknown format '{fmt}'. Choose 'npz' or 'mat'.")


def load_acquired_data(fpath: str | Path) -> tuple[np.ndarray, dict, dict]:
    """
    Load previously saved acquired_data file (HDF5, NPZ, or MAT).

    Returns:
        acquired_data, info, seq_defs
    """
    fpath = Path(fpath)
    suffix = fpath.suffix.lower()

    if suffix == ".npz":
        d = np.load(str(fpath), allow_pickle=True)
        acquired_data = d["acquired_data"]
        info = {k[5:]: d[k] for k in d if k.startswith("info_")}
        seq_defs = {k[4:]: d[k] for k in d if k.startswith("seq_")}
        return acquired_data, info, seq_defs

    elif suffix == ".mat":
        from scipy.io import loadmat
        d = loadmat(str(fpath))
        acquired_data = d["acquired_data"]
        info = {}
        seq_defs = {}
        return acquired_data, info, seq_defs

    else:
        raise ValueError(f"Unsupported file extension '{suffix}'. Use '.npz' or '.mat'.")


# ─────────────────────────────────────────────────────────────────────────────
# T1 / T2 / B1 parametric mapping loaders
# ─────────────────────────────────────────────────────────────────────────────

def read_2dseq_t1_rarevtr(scan_dir: str | Path, pv360: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Bruker RAREVTR scan for T1 mapping.

    Returns:
        image   : float32 ndarray (Y, X, slices, nTR)
        trs_ms  : float64 ndarray (nTR,) — TR values in ms

    Tries many parameter-name variants across PV5/PV6/PV360 and falls back
    gracefully with a warning instead of raising an exception.
    """
    import warnings
    scan_dir = Path(scan_dir)

    # ── TR parameter names across PV versions ────────────────────────────
    tr_keys_method = [
        "##$PVM_VTRArr",           # most common: variable-TR array (ms)
        "##$PVM_VTRArrSeconds",    # PV360 variant (s → convert to ms)
        "##$PVM_RepetitionTime",   # single-value fallback
        "##$RAREVTR_TRArr",        # alternate sequence naming
        "##$VTRArr",               # short form some sequences use
    ]
    tr_keys_acqp = [
        "##$ACQ_vd_list",          # variable-delay list in acqp (s)
        "##$ACQ_repetition_time",  # single value in acqp
    ]

    pars = read_bruker_params(scan_dir, "method", [
        "##$PVM_Matrix", "##$PVM_SPackArrNSlices", "##$PVM_NRepetitions",
        *tr_keys_method,
    ])
    for k in list(pars.keys()):
        pars[k] = _expand_pv360(pars[k])

    # Also read acqp for alternative TR sources
    try:
        acqp_pars = read_bruker_params(scan_dir, "acqp", tr_keys_acqp)
        for k in list(acqp_pars.keys()):
            acqp_pars[k] = _expand_pv360(acqp_pars[k])
    except Exception:
        acqp_pars = {k: None for k in tr_keys_acqp}

    # Prefer RECO_size (reflects zero-filling) over PVM_Matrix
    _reco_p = _read_reco_params(scan_dir)
    _reco_sz = _parse_numbers(_reco_p.get("RECO_size")).astype(int)
    if len(_reco_sz) >= 2 and _reco_sz[0] > 0 and _reco_sz[1] > 0:
        nx, ny = int(_reco_sz[0]), int(_reco_sz[1])
    else:
        mat    = _parse_numbers(pars["##$PVM_Matrix"]).astype(int)
        nx, ny = int(mat[0]), int(mat[1])
    nslices = int(_parse_numbers(pars["##$PVM_SPackArrNSlices"])[0])

    # ── Try to find TR values ─────────────────────────────────────────────
    trs_ms: np.ndarray | None = None

    # 1. Method-file keys
    for key in tr_keys_method:
        arr = _parse_numbers(pars.get(key))
        if len(arr) > 0:
            trs_ms = arr * (1000.0 if "Seconds" in key else 1.0)
            break

    # 2. acqp ACQ_vd_list (variable delays in seconds for RAREVTR)
    if trs_ms is None or len(trs_ms) == 0:
        vd = _parse_numbers(acqp_pars.get("##$ACQ_vd_list"))
        if len(vd) > 0:
            trs_ms = vd * 1000.0   # s → ms

    # 3. acqp single repetition time
    if trs_ms is None or len(trs_ms) == 0:
        rt = _parse_numbers(acqp_pars.get("##$ACQ_repetition_time"))
        if len(rt) > 0:
            n_reps_arr = _parse_numbers(pars.get("##$PVM_NRepetitions"))
            n_reps = int(n_reps_arr[0]) if len(n_reps_arr) > 0 else 1
            trs_ms = np.full(max(n_reps, 1), float(rt[0]))

    # 4. Single PVM_RepetitionTime with NRepetitions
    if trs_ms is None or len(trs_ms) == 0:
        rt = _parse_numbers(pars.get("##$PVM_RepetitionTime"))
        if len(rt) > 0:
            n_reps_arr = _parse_numbers(pars.get("##$PVM_NRepetitions"))
            n_reps = int(n_reps_arr[0]) if len(n_reps_arr) > 0 else 1
            trs_ms = np.full(max(n_reps, 1), float(rt[0]))

    # 5. Last-resort fallback with warning (never crash)
    if trs_ms is None or len(trs_ms) == 0:
        warnings.warn(
            "Could not read TR values from method/acqp file.\n"
            "Parameters tried: " + ", ".join(tr_keys_method + tr_keys_acqp) + "\n"
            "Using default TRs [500, 1000, 1500, 2000, 2500, 3000] ms — "
            "please verify or enter TR values manually.",
            stacklevel=2,
        )
        trs_ms = np.array([500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0])

    nTR     = len(trs_ms)
    rawdata = _read_raw(scan_dir, nx, ny, nslices, nTR)
    image   = np.transpose(rawdata, (1, 0, 2, 3)).astype(np.float32)
    return image, trs_ms


def read_2dseq_t2_msme(scan_dir: str | Path, pv360: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Bruker MSME scan for T2 mapping.

    Returns:
        image   : float32 ndarray (Y, X, slices, nTE)
        tes_ms  : float64 ndarray (nTE,) — TE values in ms
    """
    scan_dir = Path(scan_dir)

    # TE values — multiple names across PV versions
    te_keys = [
        "##$PVM_VcList",          # explicit TE list (older PV)
        "##$EchoTime",            # single TE (MSME uses n_echoes × this)
        "##$PVM_EchoTime",        # PV parameter variant
        "##$PVM_EffEchoTime1",    # effective first echo
    ]
    n_echo_keys = [
        "##$PVM_NEchoImages",
        "##$NEchoes",
        "##$NechImages",
        "##$PVM_Matrix",          # included so _parse_numbers can work below
    ]
    pars = read_bruker_params(scan_dir, "method", [
        "##$PVM_Matrix", "##$PVM_SPackArrNSlices",
        *te_keys, *n_echo_keys,
    ])
    for k in pars:
        pars[k] = _expand_pv360(pars[k])

    # Prefer RECO_size (reflects zero-filling) over PVM_Matrix
    _reco_p = _read_reco_params(scan_dir)
    _reco_sz = _parse_numbers(_reco_p.get("RECO_size")).astype(int)
    if len(_reco_sz) >= 2 and _reco_sz[0] > 0 and _reco_sz[1] > 0:
        nx, ny = int(_reco_sz[0]), int(_reco_sz[1])
    else:
        mat    = _parse_numbers(pars["##$PVM_Matrix"]).astype(int)
        nx, ny = int(mat[0]), int(mat[1])
    nslices = int(_parse_numbers(pars["##$PVM_SPackArrNSlices"])[0])

    # Try explicit TE list first
    tes_ms: np.ndarray | None = None
    vc = _parse_numbers(pars.get("##$PVM_VcList"))
    if len(vc) > 1:
        tes_ms = vc  # multi-value list → use directly

    if tes_ms is None:
        # Build from first TE + echo count
        n_echoes = 0
        for key in n_echo_keys[:-1]:
            arr = _parse_numbers(pars.get(key))
            if len(arr) > 0:
                n_echoes = int(arr[0])
                break

        te1: float = 0.0
        for key in te_keys[1:]:
            arr = _parse_numbers(pars.get(key))
            if len(arr) > 0:
                te1 = float(arr[0])
                break

        if n_echoes > 0 and te1 > 0:
            tes_ms = np.array([te1 * (i + 1) for i in range(n_echoes)])

    if tes_ms is None or len(tes_ms) == 0:
        raise ValueError(
            "Could not read TE values from method file. "
            "Parameters tried: " + ", ".join(te_keys)
        )

    nTE = len(tes_ms)
    rawdata = _read_raw(scan_dir, nx, ny, nslices, nTE)
    image = np.transpose(rawdata, (1, 0, 2, 3)).astype(np.float32)
    return image, tes_ms


def read_2dseq_b1_rare(scan_dir: str | Path, pv360: bool = False) -> np.ndarray:
    """
    Load a single RARE scan (one frame per slice) for B1 mapping.

    Returns:
        image : float32 ndarray (Y, X, slices)
    """
    scan_dir = Path(scan_dir)
    pars = read_bruker_params(scan_dir, "method", [
        "##$PVM_Matrix", "##$PVM_SPackArrNSlices",
    ])
    # Prefer RECO_size (reflects zero-filling) over PVM_Matrix
    _reco_p = _read_reco_params(scan_dir)
    _reco_sz = _parse_numbers(_reco_p.get("RECO_size")).astype(int)
    if len(_reco_sz) >= 2 and _reco_sz[0] > 0 and _reco_sz[1] > 0:
        nx, ny = int(_reco_sz[0]), int(_reco_sz[1])
    else:
        mat    = _parse_numbers(pars["##$PVM_Matrix"]).astype(int)
        nx, ny = int(mat[0]), int(mat[1])
    nslices = int(_parse_numbers(pars["##$PVM_SPackArrNSlices"])[0])

    rawdata = _read_raw(scan_dir, nx, ny, nslices, 1)
    image = np.transpose(rawdata[:, :, :, 0], (1, 0, 2)).astype(np.float32)
    return image
