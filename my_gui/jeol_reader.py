"""
jeol_reader.py  —  JEOL NMR ultrafast Z-spectroscopy (UFZS) reader
==================================================================
Reads JEOL Delta `.jdx` (JCAMP-DX 5.00, NTUPLES, DIF/SQZ compressed) ultrafast
Z-spectroscopy datasets and returns per-B1 Z-spectra ready for the CEST /
inverse-Z / QUESP analysis tabs.

Ports the MATLAB pipeline in ultrafastZspecCEST_sequence_processing-main:
  ReadJEOLjdx → jcampreadJEOL  (JCAMP-DX parse, FID extraction)
  ProcessFIDdata               (apodize + zerofill + FFT)
  extractUFZSDataPars          (sweep/freq/offset, SAT_B1 → µT)
  normalizeAllSpectra          (divide by the no-saturation reference)
  calcZspecMTRasym             (Z-spectrum within ±ppm window)

A UFZS dataset encodes the saturation-offset axis along the *spectral*
dimension, so each acquired spectrum IS a full Z-spectrum; the second
(NTUPLES Y) dimension arrays the saturation B1 amplitude.

Reference: Bioinformatics Toolbox jcampread (MathWorks); JCAMP-DX 4.24/5.0 spec.
"""
from __future__ import annotations

import re
import numpy as np

_GAMMA_HZ_UT = 42.576375          # ¹H gyromagnetic ratio, Hz/µT  (gamma_.m)


# ─────────────────────────────────────────────────────────────────────────────
# JCAMP-DX ASDF (compressed) line decoder  —  SQZ / DIF / DUP / PAC
# ─────────────────────────────────────────────────────────────────────────────
# SQZ: leading digit carries the sign  (@=+0 A-I=+1..9 ; a-i=-1..-9)
# DIF: difference from previous ordinate (%=0 J-R=+1..9 ; j-r=-1..-9)
# DUP: repeat the previous ordinate/difference  (S-Z=1..8, s=9)
_SQZ = {'@': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
        'a': -1, 'b': -2, 'c': -3, 'd': -4, 'e': -5, 'f': -6, 'g': -7, 'h': -8, 'i': -9}
_DIF = {'%': 0, 'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'O': 6, 'P': 7, 'Q': 8, 'R': 9,
        'j': -1, 'k': -2, 'l': -3, 'm': -4, 'n': -5, 'o': -6, 'p': -7, 'q': -8, 'r': -9}
_DUP = {'S': 1, 'T': 2, 'U': 3, 'V': 4, 'W': 5, 'X': 6, 'Y': 7, 'Z': 8, 's': 9}


def _decode_asdf_line(line: str):
    """Decode one JCAMP-DX ASDF data line.

    Returns (abscissa_index, ordinates_list).  The leading field is the
    abscissa (running point index); ordinates are SQZ-led (first = absolute)
    with DIF differences within the line.  Lines are self-contained.
    """
    toks = []           # list of (mode, value): mode in {'SQZ','DIF','DUP'}
    i, n = 0, len(line)
    # ── Read the leading abscissa field (plain integer index) ──
    a0 = i
    if i < n and line[i] in '+-':
        i += 1
    while i < n and line[i].isdigit():     # abscissa = plain integer index
        i += 1
    try:
        absc = int(line[a0:i])
    except ValueError:
        absc = None
    if absc is not None and (absc < 0 or absc > 10_000_000):
        absc = None
    # ── Tokenize the ordinate fields ──
    while i < n:
        c = line[i]
        if c in _SQZ:
            j = i + 1
            while j < n and line[j].isdigit():
                j += 1
            mag = int(line[i + 1:j]) if j > i + 1 else 0
            sign = -1 if c in 'abcdefghi' else 1
            toks.append(('SQZ', sign * (abs(_SQZ[c]) * (10 ** (j - i - 1)) + mag)))
            i = j
        elif c in _DIF:
            j = i + 1
            while j < n and line[j].isdigit():
                j += 1
            mag = int(line[i + 1:j]) if j > i + 1 else 0
            sign = -1 if c in 'jklmnopqr' else 1
            toks.append(('DIF', sign * (abs(_DIF[c]) * (10 ** (j - i - 1)) + mag)))
            i = j
        elif c in _DUP:
            j = i + 1
            while j < n and line[j].isdigit():
                j += 1
            mag = int(line[i + 1:j]) if j > i + 1 else 0
            toks.append(('DUP', _DUP[c] * (10 ** (j - i - 1)) + mag if j > i + 1
                         else _DUP[c]))
            i = j
        else:
            i += 1   # whitespace / unknown separator

    # ── Build ordinate sequence (first = absolute SQZ, rest = DIF within line)
    out = []
    y = None
    last_diff = 0
    for mode, val in toks:
        if mode == 'SQZ':
            y = val
            out.append(y); last_diff = 0
        elif mode == 'DIF':
            y = (y if y is not None else 0) + val
            out.append(y); last_diff = val
        elif mode == 'DUP':
            for _ in range(val - 1):           # repeat previous (val) times total
                if last_diff != 0:
                    y = y + last_diff
                out.append(y)
    return absc, out


def _decode_page(lines):
    """Decode all ASDF data lines of one NTUPLES page → 1-D float array.

    Each line begins with the running abscissa index, so ordinates are placed
    by index (the first ordinate of each line repeats the previous line's last
    — a Y-value check — and overwrites it identically). This is robust to the
    off-by-one line boundaries of the DIF format.
    """
    placed: dict = {}
    maxidx = -1
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        absc, ords = _decode_asdf_line(ln)
        if absc is None:
            continue
        for k, v in enumerate(ords):
            placed[absc + k] = v
            if absc + k > maxidx:
                maxidx = absc + k
    arr = np.zeros(maxidx + 1, dtype=float)
    for idx, v in placed.items():
        arr[idx] = v
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# JCAMP-DX header parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_jeol_param(value: str):
    """Parse a JEOL parameter value: scalar+unit, bracketed array, or string."""
    value = value.strip()
    if value.startswith('{'):                       # array: {0[Hz], 50[Hz], …}
        inner = value[value.find('{') + 1:value.rfind('}')]
        out = []
        for tok in inner.split(','):
            m = re.match(r'\s*([-+0-9.eE]+)', tok)
            if m:
                out.append(float(m.group(1)))
        return np.asarray(out, dtype=float)
    if '$$' in value:                               # scalar + unit  e.g. "2 $$s"
        num, _, unit = value.partition('$$')
        try:
            v = float(num.strip())
        except ValueError:
            return value
        u = unit.strip()[:1]
        return {'u': v / 1e6, 'm': v / 1e3, 'k': v * 1e3, 'M': v * 1e6}.get(u, v)
    try:
        return float(value)
    except ValueError:
        return value


def read_jeol_jdx(path: str):
    """Parse a JEOL `.jdx` NTUPLES file → (fid, header).

    fid    : complex ndarray (n_b1, n_points)
    header : dict of parsed parameters (keys without the leading ##/##$)
    """
    with open(path, 'r', encoding='latin-1') as fh:
        raw = fh.read().splitlines()

    header: dict = {}
    pages: list = []           # list of (symbol_RI, [data lines])
    factor = None
    cur_lines = None
    cur_kind = None

    i = 0
    while i < len(raw):
        line = raw[i]
        s = line.strip()
        if s.startswith('##FACTOR'):
            nums = re.findall(r'[-+0-9.eE]+', s.split('=', 1)[1])
            factor = [float(x) for x in nums]
        elif s.startswith('##DATA TABLE') or s.startswith('##DATATABLE'):
            cur_lines = []
            cur_kind = 'R' if '(R..R)' in s or 'R..R' in s else 'I'
            pages.append((cur_kind, cur_lines))
        elif s.startswith('##PAGE'):
            cur_lines = None
        elif s.startswith('##END'):
            cur_lines = None
        elif s.startswith('##'):
            # header label — store parsed value (strip ## and ##$)
            key, _, val = s[2:].partition('=')
            key = key.lstrip('$').strip()
            key = re.sub(r'[^0-9A-Za-z]+', '_', key).strip('_')
            val = val.strip()
            # A bracketed array (e.g. SAT_B1) may wrap across several lines —
            # keep appending until the closing '}' is reached.
            if val.startswith('{') and '}' not in val:
                while i + 1 < len(raw) and '}' not in val:
                    i += 1
                    val += raw[i].strip()
            if val:
                header[key] = _parse_jeol_param(val)
            cur_lines = None
        elif cur_lines is not None:
            cur_lines.append(line)
        i += 1

    if not pages:
        raise ValueError("No JCAMP-DX data tables found — is this a JEOL .jdx FID?")

    # FACTOR = [T1, T2, R, I];  R/I share columns 3/4
    facR = factor[2] if factor and len(factor) > 2 else 1.0
    facI = factor[3] if factor and len(factor) > 3 else 1.0

    # Decode pages; R and I alternate (real page, imag page) per B1
    reals, imags = [], []
    for kind, lines in pages:
        arr = _decode_page(lines)
        (reals if kind == 'R' else imags).append(arr)

    n = min(len(reals), len(imags))
    npts = min(min(len(r) for r in reals[:n]), min(len(im) for im in imags[:n]))
    fid = np.zeros((n, npts), dtype=complex)
    for k in range(n):
        fid[k] = reals[k][:npts] * facR + 1j * imags[k][:npts] * facI
    return fid, header


# ─────────────────────────────────────────────────────────────────────────────
# FID → spectra  (ProcessFIDdata.m, ultrafast/echo-centered branch)
# ─────────────────────────────────────────────────────────────────────────────

def _process_fid(fid: np.ndarray, dwell_s: float,
                 filter_type: str = 'gaussian', ap_hz: float = 5.0,
                 edge: float = 1.0, zf: int = 2) -> np.ndarray:
    """Apodize (echo-centered), zero-fill, FFT each FID row → spectra."""
    n_b1, npts = fid.shape
    if filter_type == 'exponential':
        t = np.arange(int(np.ceil(npts / 2))) * dwell_s
        left = np.exp((t - t.max()) * ap_hz * np.pi)
        right = np.exp(-np.arange(npts - len(left)) * dwell_s * ap_hz * np.pi)
        apod = np.concatenate([left, right])[:npts]
    else:  # gaussian (echo at centre)
        gmean = npts / 2 + 0.5
        sigma = np.sqrt(5.0 / np.log(10) * (gmean - 1) ** 2 / max(edge, 1e-6))
        apod = np.exp(-((np.arange(1, npts + 1) - gmean) ** 2) / 2.0 / sigma ** 2)

    pad = int(np.floor(npts * (zf - 1) / 2))
    out = []
    for row in fid:
        f = row * apod
        f = np.concatenate([np.zeros(pad, complex), f, np.zeros(pad, complex)])
        out.append(np.fft.fftshift(np.fft.fft(np.fft.fftshift(f))))
    return np.asarray(out)


def _center_fid(fid: np.ndarray) -> np.ndarray:
    """Circularly shift so the echo max sits at the centre (Load_Preprocess)."""
    a0 = np.abs(fid[0])
    maxind = int(np.argmax(a0))
    npts = fid.shape[1]
    shift = int(np.ceil(npts / 2)) - maxind
    if maxind > 0 and maxind < npts - 1 and a0[maxind - 1] > a0[maxind + 1]:
        shift += 1
    return np.flip(np.roll(fid, shift, axis=1), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — read a JEOL UFZS dataset → per-B1 Z-spectra
# ─────────────────────────────────────────────────────────────────────────────

def read_jeol_ufzs(path: str, ppm_window: float = 6.0,
                   filter_type: str = 'gaussian', ap_hz: float = 5.0,
                   zf: int = 2) -> dict:
    """
    Read a JEOL ultrafast Z-spectroscopy `.jdx` file.

    Returns a dict:
      ppm        : (M,) saturation-offset axis (ppm), descending
      datasets   : list of {b1_ut, b1_hz, z, ppm} — one per saturated B1 power
      b1_uT      : (K,) B1 amplitudes (µT) of the saturated spectra
      omega0_MHz : ¹H Larmor frequency
      tsat_s, rd_s : saturation pulse + recovery delay (s)
    """
    fid, hdr = read_jeol_jdx(path)

    # ── Acquisition parameters ──
    sw_hz   = float(hdr.get('X_SWEEP', hdr.get('x_X_SWEEP', 0.0)))
    freq_hz = float(hdr.get('X_FREQ', 0.0))
    omega0  = freq_hz / 1e6 if freq_hz > 1e3 else float(hdr.get('X_FREQ', 0.0))
    if omega0 <= 0:                                   # X_FREQ already in MHz
        omega0 = float(hdr.get('X_FREQ', 500.0))
    sw_ppm  = (sw_hz / omega0) if (sw_hz and omega0) else 100.0

    npfid = fid.shape[1]
    np_spec = npfid * zf
    dwell = (sw_hz and (1.0 / sw_hz)) or 1e-5

    # ── SAT_B1 (Hz) → µT ──
    sat_hz = np.atleast_1d(np.asarray(hdr.get('SAT_B1', []), dtype=float)).ravel()
    if sat_hz.size == 0:
        sat_hz = np.zeros(fid.shape[0])
    sat_ut = sat_hz / _GAMMA_HZ_UT

    # ── FID → spectra ──
    fid_c = _center_fid(fid)
    spec = _process_fid(fid_c, dwell, filter_type, ap_hz, zf=zf)
    np_spec = spec.shape[1]
    specppm = np.linspace(sw_ppm / 2, -sw_ppm / 2, np_spec)

    # ── Normalize by the highest-amplitude no-saturation spectrum ──
    nosat = np.where(sat_hz < 1e-3)[0]
    aspec = np.abs(spec)
    if nosat.size:
        refind = nosat[int(np.argmax([aspec[k].max() for k in nosat]))]
    else:
        refind = 0
    ref = aspec[refind]

    # ── Constrain to ±ppm_window and build per-B1 Z-spectra ──
    wdw = np.abs(specppm) < ppm_window
    ppm_w = specppm[wdw]
    datasets = []
    b1_list = []
    for k in range(spec.shape[0]):
        if k in nosat:
            continue
        z = aspec[k][wdw] / np.clip(ref[wdw], 1e-12, None)
        datasets.append(dict(b1_ut=float(sat_ut[k]), b1_hz=float(sat_hz[k]),
                             ppm=ppm_w, z=z))
        b1_list.append(float(sat_ut[k]))

    return dict(
        ppm=ppm_w, datasets=datasets, b1_uT=np.asarray(b1_list),
        omega0_MHz=omega0,
        tsat_s=float(hdr.get('SAT_DELAY', 0.0)),
        rd_s=float(hdr.get('CEST_RELAXATION_DELAY', 0.0)),
        title=str(hdr.get('TITLE', 'JEOL UFZS')),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test against a real file
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/cbd/Downloads/260526_Glucoamylase/260527_Glucoamylase_UFZ_CEST-1-1.jdx"
    fid, hdr = read_jeol_jdx(p)
    print(f"FID shape: {fid.shape}  (expect ~12 × 1024)")
    print(f"first R (×factor): {fid[0,0].real:.6e}  (##FIRST expects -8.39e-04)")
    print(f"SAT_B1 (Hz): {hdr.get('SAT_B1')}")
    r = read_jeol_ufzs(p)
    print(f"omega0 = {r['omega0_MHz']:.2f} MHz, Tsat={r['tsat_s']}s, RD={r['rd_s']}s")
    print(f"B1 powers (µT): {np.round(r['b1_uT'],2)}")
    print(f"# saturated Z-spectra: {len(r['datasets'])}, ppm span "
          f"{r['ppm'].min():.1f}..{r['ppm'].max():.1f} ({len(r['ppm'])} pts)")
