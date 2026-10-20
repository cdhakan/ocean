"""
seq_rebuild.py
==============

Parse a Pulseq-CEST ``.seq`` (pulseq-cest-library or any writer) into
*measurements* → *saturation units* and regenerate it from edited sequence
definitions — the ``defs`` dictionary of the library's generator scripts
(b0, b1pa, n_pulses, tp, td, trec, trec_m0, m0_offset, offsets_ppm, spoiling,
seq_id_string …).

Only the [DEFINITIONS] block of a .seq is *descriptive*: the simulator plays
the RF / delay / gradient blocks.  So editing a definition must re-write the
physics:

* ``b1pa``        → RF amplitude of every saturation pulse (the file's own
                    waveform shape is kept — block, Gaussian, SincGauss,
                    Hanning, optimal-control …; spin-lock tip pulses are
                    re-tipped to atan(B1 / (Δppm·B0)) as the library does)
* ``offsets_ppm`` → RF frequency offset (ppm · B0 · γ) and the accumulated
                    inter-pulse phase
* ``b0``          → every frequency offset (and FREQ)
* ``tp``          → pulse duration (waveform resampled)
* ``td``          → delay between pulses (spin-lock tips keep fitting inside)
* ``n_pulses``    → number of pulse units per measurement
* ``trec`` / ``trec_m0`` / ``m0_offset`` → recovery delays
* ``spoiling``    → spoiler gradient block after the preparation
* length of ``offsets_ppm`` → number of measurements (templates cloned)

Everything the editor does not understand (readout pulses, gradients, ADC,
extra definitions) is copied verbatim, so the regenerated file keeps the
library's structure.  Pure Python/NumPy — no Qt — so it is unit-testable.
"""
from __future__ import annotations

import contextlib
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import numpy as np

GAMMA_HZ = 42.5764            # Hz/µT  (pulseq-cest: Opts(gamma=42576400) · 1e-6)
_TWO_PI = 2.0 * math.pi

# canonical (python, lower-case) name → definition keys as different writers
# spell them; the first key present in the file wins and is written back under
# that same spelling.  MATLAB_NAME gives the pulseq-cest spelling used when a
# definition has to be added to a file that did not have it.
CANON_KEYS: "OrderedDict[str, tuple]" = OrderedDict([
    ("b0",            ("B0", "b0")),
    ("b1pa",          ("B1pa", "b1pa", "B1peak", "MRF_B1_uT", "B1_uT", "b1_uT", "B1", "b1")),
    ("n_pulses",      ("n_pulses", "MRF_NumSatPulses", "numSatPulses", "npulses")),
    ("tp",            ("tp", "PulseDuration")),
    ("td",            ("td", "MRF_InterPulseGap_s")),
    ("trec",          ("Trec", "trec")),
    ("trec_m0",       ("Trec_M0", "trec_m0")),
    ("m0_offset",     ("M0_offset", "m0_offset")),
    ("offsets_ppm",   ("offsets_ppm", "offsets", "PulseOffsetPPM", "ppm")),
    ("dcsat",         ("DCsat", "dcsat")),
    ("num_meas",      ("num_meas",)),
    ("tsat",          ("Tsat", "tsat", "MRF_Tsat_s")),
    ("seq_id_string", ("seq_id_string",)),
    ("spoiling",      ("spoiling",)),
    ("freq",          ("FREQ", "freq")),
    ("b1rms",         ("B1rms", "b1rms", "B1cwpe")),
])
MATLAB_NAME = {"b0": "B0", "b1pa": "B1pa", "trec": "Trec", "trec_m0": "Trec_M0",
               "m0_offset": "M0_offset", "tsat": "Tsat", "dcsat": "DCsat",
               "freq": "FREQ", "b1rms": "B1rms"}
EDITABLE = ("b0", "b1pa", "n_pulses", "tp", "td", "trec", "trec_m0", "m0_offset",
            "offsets_ppm", "spoiling", "seq_id_string")
DERIVED = ("dcsat", "num_meas", "tsat", "freq")
PER_MEAS = ("b1pa", "n_pulses", "tp", "td", "trec", "offsets_ppm")   # scalar or vector(num_meas)

DESCRIPTIONS = {
    "b0": ("T",   "B0 field strength"),
    "b1pa": ("µT", "B1 peak amplitude (per measurement or one value for all)"),
    "n_pulses": ("", "number of saturation pulses per measurement"),
    "tp": ("s",   "pulse duration"),
    "td": ("s",   "interpulse delay (one value, one per measurement, or a gap pattern e.g. 0.001, 0.01)"),
    "trec": ("s", "recovery time before the saturation"),
    "trec_m0": ("s", "recovery time before the M0 measurement"),
    "m0_offset": ("ppm", "offset that marks the M0 measurement"),
    "offsets_ppm": ("ppm", "offset vector — its length sets the number of measurements"),
    "spoiling": ("", "spoiler gradient after the preparation (1 / 0)"),
    "seq_id_string": ("", "unique sequence id (file name of the exported .seq)"),
    "dcsat": ("", "duty cycle tp / (tp + td)  — derived"),
    "num_meas": ("", "number of measurements  — derived"),
    "tsat": ("s", "saturation time n_pulses·(tp + td) − td  — derived"),
    "freq": ("Hz", "Larmor frequency B0·γ  — derived"),
    "b1rms": ("µT", "B1 rms over tp + td  — derived"),
}


@contextlib.contextmanager
def vendored_pypulseq():
    """``import pypulseq`` → the bundled Pulseq-1.3.1 build (the format the
    C++ BMCSimulator and the pulseq-cest-library use)."""
    from my_gui.worker import _vendored_pypulseq
    with _vendored_pypulseq():
        import pypulseq as pp
        yield pp


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Shape:
    """Peak-normalised complex RF waveform at the RF raster.  ``samples`` is
    None for a constant (block) pulse so multi-second CW pulses cost nothing.
    ``tail`` = trailing zero samples pypulseq appends for the coil ring-down;
    they are not part of the pulse duration ``tp`` but are written back."""
    n: int
    samples: Optional[np.ndarray]
    raster: float = 1e-6
    tail: int = 0
    const: complex = 1 + 0j                   # unit phase factor of a constant (block) waveform

    @property
    def duration(self) -> float:              # active pulse duration (= tp)
        return self.n * self.raster

    @property
    def total_duration(self) -> float:        # incl. ring-down zeros (what calc_duration sees)
        return (self.n + self.tail) * self.raster

    @property
    def t_nonzero(self) -> float:
        if self.samples is None:
            return self.duration
        return int(np.count_nonzero(np.abs(self.samples) > 0)) * self.raster

    @property
    def mean_abs(self) -> float:
        return 1.0 if self.samples is None else float(np.mean(np.abs(self.samples)))

    @property
    def abs_integral(self) -> float:          # ∫|shape| dt  [s]
        return self.duration if self.samples is None else float(np.sum(np.abs(self.samples))) * self.raster

    @property
    def sq_integral(self) -> float:           # ∫|shape|² dt [s]
        return self.duration if self.samples is None else float(np.sum(np.abs(self.samples) ** 2)) * self.raster

    def resampled(self, duration: float, tp_ref: Optional[float] = None) -> "Shape":
        """Waveform stretched by duration / tp_ref.  ``tp_ref`` is the file's own
        tp: trailing zeros beyond it are ring-down padding (kept as is); zeros
        inside it are part of the pulse shape (e.g. a sinc that ends at a zero
        crossing).  Writers differ in how many samples a "tp" pulse has (block:
        tp + 30 µs of zeros; pypulseq-1.3.1 sinc/gauss: tp + 30 µs of signal),
        so the whole active part is scaled proportionally.  An unchanged tp
        returns the shape untouched."""
        n_total = self.n + self.tail
        if tp_ref is None or tp_ref <= 0:
            tp_ref = self.duration
        if abs(duration - tp_ref) <= 0.5 * self.raster:
            return self
        n_tp = int(round(tp_ref / self.raster))
        ring = min(self.tail, max(0, n_total - n_tp))
        n_act = n_total - ring                         # active samples incl. natural zeros
        n_new = max(1, int(round(n_act * duration / tp_ref)))
        if n_new == n_act:
            return self
        inner_zeros = n_act - self.n
        if self.samples is None and inner_zeros == 0:
            return Shape(n_new, None, self.raster, ring, self.const)
        act = self.samples if self.samples is not None else np.full(self.n, self.const, dtype=complex)
        if inner_zeros > 0:
            act = np.concatenate([act, np.zeros(inner_zeros, dtype=complex)])
        x_old = np.linspace(0.0, 1.0, n_act)
        x_new = np.linspace(0.0, 1.0, n_new)
        s = (np.interp(x_new, x_old, act.real) + 1j * np.interp(x_new, x_old, act.imag)).astype(complex)
        nz = np.flatnonzero(np.abs(s) > 0)
        keep = int(nz[-1] + 1) if nz.size else n_new
        return Shape(keep, s[:keep], self.raster, ring + (n_new - keep))

    def signal(self, amp: float) -> np.ndarray:
        core = np.full(self.n, complex(amp) * self.const) if self.samples is None else amp * self.samples
        if self.tail:
            core = np.concatenate([core, np.zeros(self.tail, dtype=complex)])
        return core


@dataclass
class RFItem:
    role: str                     # 'sat' | 'pre' | 'post'  (pre/post = prep pulses around a sat pulse)
    shape: Shape
    amp: float                    # peak amplitude [Hz]
    freq: float                   # [Hz]
    phase: float                  # [rad]
    delay: float
    dead_time: float
    ringdown_time: float
    use: Optional[str] = None

    @property
    def flip(self) -> float:      # |∫ B1| in rad (block: 2π·amp·tp)
        return _TWO_PI * self.amp * self.shape.abs_integral

    @property
    def duration(self) -> float:              # block duration as calc_duration sees it
        return self.delay + self.shape.total_duration

    def event(self, pp, amp: float, freq: float, phase: float, shape: Optional[Shape] = None):
        sh = shape or self.shape
        rf = SimpleNamespace()
        rf.type = "rf"
        rf.signal = sh.signal(amp)
        rf.t = np.arange(1, sh.n + sh.tail + 1) * sh.raster
        rf.freq_offset = float(freq)
        rf.phase_offset = float(phase)
        rf.dead_time = self.dead_time
        rf.ringdown_time = self.ringdown_time
        rf.delay = self.delay
        if self.use:
            rf.use = self.use
        return rf


@dataclass
class Unit:
    """One saturation pulse with the prep pulses / pauses that belong to it."""
    pre: list = field(default_factory=list)    # [('rf', RFItem) | ('delay', s)]
    sat: Optional[RFItem] = None
    post: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        d = sum(it.duration if k == "rf" else it for k, it in self.pre + self.post)
        return d + (self.sat.duration if self.sat else 0.0)

    @property
    def aux_time(self) -> float:               # time of prep pulses + pauses (what td must contain)
        return self.duration - (self.sat.duration if self.sat else 0.0)


@dataclass
class Meas:
    rec_delay: float                           # leading delay(s) before the train [s]
    units: list                                # [] → no saturation train (B1 = 0 measurement)
    gaps: list                                 # delays between units [s]  (len = n_units-1)
    post: list                                 # verbatim blocks after the train (spoiler, readout, ADC)
    post_duration: float = 0.0
    spoiler_idx: list = field(default_factory=list)   # indices into post that are gradient-only blocks
    pre_blocks: list = field(default_factory=list)    # verbatim blocks between the recovery delay and the train
    lead_delays: list = field(default_factory=list)   # the individual leading delay blocks [s]
    head_blocks: list = field(default_factory=list)   # verbatim blocks played BEFORE the recovery delay
    train_delays: list = field(default_factory=list)  # B1 = 0 measurement: delays that stand in for the train

    @property
    def n_pulses(self) -> int:
        return len(self.units)

    @property
    def sat_duration(self) -> float:
        return self.units[0].sat.shape.duration if self.units else 0.0

    @property
    def train_duration(self) -> float:
        return sum(u.duration for u in self.units) + sum(self.gaps)

    def b1_estimate(self) -> float:
        """Mean |B1| of the saturation pulse in µT (block convention, which is
        also pulseq-cest's flip-angle normalisation for shaped pulses)."""
        if not self.units:
            return 0.0
        s = self.units[0].sat
        return s.amp * s.shape.mean_abs / GAMMA_HZ


@dataclass
class SeqModel:
    path: str
    version: str
    definitions: "OrderedDict"                 # raw [DEFINITIONS] (np arrays / strings)
    preamble: list                             # verbatim blocks before the first measurement
    meas: list                                 # [Meas]
    raster: float
    n_blocks: int
    segmentation: str                          # 'adc' | 'train'
    notes: list = field(default_factory=list)
    rep: int = 1                               # consecutive measurements per offset (per-slice saturation, pHw)
    trailer: list = field(default_factory=list)   # verbatim blocks after the last measurement

    # ── definitions as the library's python ``defs`` dict ────────────────────
    def file_defs(self) -> dict:
        """Canonical-name → value for every definition the file carries."""
        out = {}
        for canon, keys in CANON_KEYS.items():
            for k in keys:
                if k in self.definitions:
                    out[canon] = _def_value(self.definitions[k])
                    break
        return out

    def file_key(self, canon: str) -> Optional[str]:
        for k in CANON_KEYS[canon]:
            if k in self.definitions:
                return k
        return None

    def structure_defs(self, b0: Optional[float] = None) -> dict:
        """The same parameters measured from the blocks (fallback when a file
        does not define them)."""
        N = len(self.meas)
        fd = self.file_defs()
        b0 = b0 if b0 else _scalar(fd.get("b0"), 0.0)
        if not b0 and fd.get("freq"):
            b0 = _scalar(fd["freq"], 0.0) / GAMMA_HZ
        out = {"num_meas": N, "b0": b0 or 3.0}
        out["b1pa"] = _compact([m.b1_estimate() for m in self.meas])
        out["n_pulses"] = _compact([m.n_pulses for m in self.meas])
        out["tp"] = _compact([m.sat_duration for m in self.meas if m.units] or [0.0])
        gaps = []
        for m in self.meas:
            if len(m.units) > 1:
                gaps.append(m.gaps[0] + m.units[0].aux_time)
            elif m.units:
                gaps.append(m.units[0].aux_time)
        out["td"] = _compact(gaps or [0.0])
        out["trec"] = _compact([m.rec_delay for m in self.meas])
        g = (out["b0"] or 3.0) * GAMMA_HZ
        out["offsets_ppm"] = [round(m.units[0].sat.freq / g, 6) if m.units else 0.0 for m in self.meas]
        out["spoiling"] = int(any(m.spoiler_idx for m in self.meas))
        out["seq_id_string"] = os.path.splitext(os.path.basename(self.path))[0]
        return out

    def current_defs(self, b0: Optional[float] = None) -> "tuple[dict, dict]":
        """(defs, source) — file definitions first, structure fallbacks for the
        rest; ``source[name]`` is the file key or 'from sequence'."""
        fd = self.file_defs()
        sd = self.structure_defs(b0)
        defs, src = {}, {}
        for name in EDITABLE + DERIVED:
            if name in fd:
                defs[name] = fd[name]
                src[name] = self.file_key(name)
            elif name in sd:
                defs[name] = sd[name]
                src[name] = "from sequence"
        # per-measurement vectors that the file gives as scalars stay scalars
        defs.setdefault("spoiling", sd["spoiling"])
        return defs, src

    def summary(self) -> str:
        N = len(self.meas)
        with_train = [m for m in self.meas if m.units]
        if with_train:
            m0 = with_train[0]
            sat = m0.units[0].sat
            kind = "block (CW)" if sat.shape.samples is None else "shaped"
            tips = sum(1 for k, _ in m0.units[0].pre + m0.units[0].post if k == "rf")
            info = (f"{N} measurements · {m0.n_pulses} × {sat.shape.duration*1e3:.3g} ms {kind} pulse"
                    + (f" + {tips} prep pulse(s)/unit" if tips else ""))
        else:
            info = f"{N} measurements · no saturation pulses found"
        spoil = any(m.spoiler_idx for m in self.meas)
        ro = any(any(hasattr(b, "rf") for b in m.post) for m in self.meas)
        info += f" · spoiler {'yes' if spoil else 'no'} · readout RF {'yes' if ro else 'pseudo-ADC only'}"
        info += f" · Pulseq {self.version} · {self.n_blocks} blocks"
        return info


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _def_value(v):
    """np.array / list-of-str from Sequence.read → float, list[float] or str."""
    if isinstance(v, np.ndarray):
        vals = [float(x) for x in v.ravel()]
        return vals[0] if len(vals) == 1 else vals
    if isinstance(v, (list, tuple)):
        if all(isinstance(x, str) for x in v):
            try:
                vals = [float(x) for x in v]
                return vals[0] if len(vals) == 1 else vals
            except ValueError:
                return " ".join(v)
        vals = [float(x) for x in v]
        return vals[0] if len(vals) == 1 else vals
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return float(v)


def _scalar(v, default=0.0) -> float:
    if v is None:
        return default
    if isinstance(v, (list, tuple, np.ndarray)):
        return float(v[0]) if len(v) else default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _compact(vals: list):
    """list → scalar when all values agree (within 1e-9), else the list."""
    if not vals:
        return 0.0
    a = np.asarray(vals, dtype=float)
    if np.all(np.abs(a - a[0]) <= 1e-9 * max(1.0, abs(a[0]))):
        v = float(a[0])
        return int(v) if v.is_integer() else v
    return [float(x) for x in a]


def broadcast(v, n: int, name: str) -> list:
    """scalar or length-n vector → list of n floats (raises otherwise)."""
    if isinstance(v, (list, tuple, np.ndarray)):
        vals = [float(x) for x in np.asarray(v, dtype=float).ravel()]
        if len(vals) == 1:
            return vals * n
        if len(vals) != n:
            raise ValueError(f"{name}: {len(vals)} values given but offsets_ppm has {n} "
                             f"entries — give one value or exactly {n}")
        return vals
    return [float(v)] * n


class TdSpec:
    """Interpulse delay ``td``: one value, one value per measurement, or a
    gap pattern repeated between the pulses of every measurement (the
    library's APTw_003 uses td = [1 ms, 10 ms])."""

    def __init__(self, v, n_meas: int):
        vals = ([float(x) for x in np.asarray(v, dtype=float).ravel()]
                if isinstance(v, (list, tuple, np.ndarray)) else [float(v)])
        if not vals:
            vals = [0.0]
        if len(vals) == 1:
            self.mode, self.vals = "meas", vals * n_meas
        elif len(vals) == n_meas:
            self.mode, self.vals = "meas", vals
        else:
            self.mode, self.vals = "pattern", vals
        self.n_meas = n_meas

    def gap(self, k: int, i: int) -> float:
        """td before pulse i+1 of measurement k."""
        if self.mode == "pattern":
            return self.vals[i % len(self.vals)]
        return self.vals[min(k, len(self.vals) - 1)]

    def mean(self, k: int) -> float:
        return float(np.mean(self.vals)) if self.mode == "pattern" else self.gap(k, 0)

    def train_gaps(self, k: int, n_pulses: int) -> float:
        return sum(self.gap(k, i) for i in range(max(n_pulses - 1, 0)))

    def value(self):
        """What goes into [DEFINITIONS]."""
        if self.mode == "pattern":
            return [float(x) for x in self.vals]
        return _vec_or_scalar(self.vals)

    def same_as(self, other: "TdSpec", k: int, n_pulses: int) -> bool:
        return all(abs(self.gap(k, i) - other.gap(k, i)) <= 1e-12 for i in range(max(n_pulses - 1, 1)))


def _events(block) -> list:
    return [getattr(block, a) for a in ("rf", "gx", "gy", "gz", "adc", "delay") if hasattr(block, a)]


def _kind(block) -> str:
    has_rf = hasattr(block, "rf")
    has_g = any(hasattr(block, c) for c in ("gx", "gy", "gz"))
    if hasattr(block, "adc"):
        return "adc"
    if has_rf and not has_g:
        return "rf"
    if has_g and not has_rf:
        return "grad"
    if has_rf and has_g:
        return "other"
    if hasattr(block, "delay"):
        return "delay"
    return "empty"


def _rf_item(rf, role: str, raster: float) -> RFItem:
    sig = np.asarray(rf.signal)
    mag = np.abs(sig)
    amp = float(np.max(mag)) if mag.size else 0.0
    nz = np.flatnonzero(mag > 0)
    tail = int(sig.size - (nz[-1] + 1)) if nz.size else 0      # ring-down zero padding
    core = sig[: sig.size - tail]
    if amp > 0 and core.size and np.allclose(core, core[0], rtol=0, atol=1e-9 * amp):
        c0 = complex(core[0])
        amp = abs(c0)
        # constant waveform: keep its phase factor in the shape (MATLAB writers
        # store e.g. π there) so the phase column is written back unchanged
        shape = Shape(int(core.size), None, raster, tail, c0 / amp)
        extra_phase = 0.0
    else:
        shape = Shape(int(core.size), (core / amp).astype(complex) if amp > 0 else core.astype(complex),
                      raster, tail)
        extra_phase = 0.0
    return RFItem(role=role, shape=shape, amp=amp, freq=float(rf.freq_offset),
                  phase=float(rf.phase_offset) + extra_phase, delay=float(getattr(rf, "delay", 0.0)),
                  dead_time=float(getattr(rf, "dead_time", 0.0)),
                  ringdown_time=float(getattr(rf, "ringdown_time", 0.0)),
                  use=getattr(rf, "use", None))


# ─────────────────────────────────────────────────────────────────────────────
# Parse
# ─────────────────────────────────────────────────────────────────────────────

def seq_version(path: str) -> "tuple[int, int, str]":
    """(major, minor, revision) from the [VERSION] block, read as text."""
    major = minor = 0
    rev = ""
    try:
        with open(path, errors="replace") as f:
            in_v = False
            for ln in f:
                s = ln.strip()
                if s.startswith("["):
                    if in_v:
                        break
                    in_v = s == "[VERSION]"
                    continue
                if in_v and s:
                    k, _, v = s.partition(" ")
                    if k == "major":
                        major = int(float(v))
                    elif k == "minor":
                        minor = int(float(v))
                    elif k == "revision":
                        rev = v.strip()
    except (OSError, ValueError):
        pass
    return major, minor, rev


def _quick_definitions(path: str) -> dict:
    """[DEFINITIONS] as {key: [floats]} (text only; non-numeric values skipped)."""
    out: dict = {}
    try:
        with open(path, errors="replace") as f:
            in_d = False
            for ln in f:
                s = ln.strip()
                if s.startswith("["):
                    if in_d:
                        break
                    in_d = s == "[DEFINITIONS]"
                    continue
                if in_d and s and not s.startswith("#"):
                    parts = s.split()
                    try:
                        out[parts[0]] = [float(x) for x in parts[1:]]
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def parse_seq(path: str) -> SeqModel:
    major, minor, rev = seq_version(path)
    if (major, minor) > (1, 3):
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        sib = [os.path.basename(p) for p in sorted(
            __import__("glob").glob(os.path.join(os.path.dirname(path), "*.seq")))
            if p != path and seq_version(p)[:2] == (1, 3)
            and (os.path.basename(p).lower().startswith(stem)
                 or stem.startswith(os.path.splitext(os.path.basename(p))[0].lower()))]
        hint = (f"  The same folder has a Pulseq 1.3 version: {', '.join(sib)} — open that one."
                if sib else "  Re-export it as Pulseq 1.3.1 (pypulseq 1.3.1 / MATLAB Pulseq 1.3).")
        raise ValueError(f"This file is Pulseq {major}.{minor}.{rev}. OCEAN's simulator and editor read "
                         f"Pulseq 1.3 files only (the BMCSimulator cannot run 1.4/1.5 files).{hint}")
    _dk = _quick_definitions(path)
    if (("TI" in _dk or "TE" in _dk) and all(abs(x) < 1e-12 for x in _dk.get("offsets_ppm", [0.0]))):
        raise ValueError("This is a T1 / T2 preparation sequence (TI / TE list, all offsets 0 ppm). "
                         "The definitions editor rebuilds saturation sequences (CEST, MRF, WASABI); "
                         "TI / TE lists are not editable here.")
    with vendored_pypulseq() as pp:
        seq = pp.Sequence()
        seq.read(path)
        raster = float(seq.rf_raster_time)
        nb = len(seq.dict_block_events)
        blocks = [seq.get_block(i) for i in range(1, nb + 1)]
        durs = [float(pp.calc_duration(*_events(b))) if _events(b) else 0.0 for b in blocks]
        version = f"{seq.version_major}.{seq.version_minor}.{seq.version_revision}"
        definitions = OrderedDict(seq.dict_definitions)
    kinds = [_kind(b) for b in blocks]
    notes: list = []

    # ── saturation pulses: RF-only blocks that are not short prep/readout pulses
    rf_idx = [i for i, k in enumerate(kinds) if k == "rf"]
    if not rf_idx:
        raise ValueError("no RF pulses in this .seq — nothing to edit")
    max_rf = max(durs[i] for i in rf_idx)
    thr = max(2.5e-3, 0.04 * max_rf)
    is_sat = {i: durs[i] >= thr for i in rf_idx}
    # the file's own tp identifies the saturation pulse best (adiabatic spin-lock
    # pulses of 12 ms around a 120 ms lock pulse are long but are prep pulses)
    tp_vals = []
    for key in CANON_KEYS["tp"]:
        if key in definitions:
            v = _def_value(definitions[key])
            tp_vals = [float(x) for x in (v if isinstance(v, list) else [v])
                       if isinstance(x, (int, float)) and float(x) > 0] if not isinstance(v, str) else []
            break
    if tp_vals:
        tps = sorted(set(round(x, 9) for x in tp_vals))
        by_tp = {i: any(abs(durs[i] - tv) <= max(0.03 * tv, 0.5e-3) for tv in tps) for i in rf_idx}
        if any(by_tp.values()):
            is_sat = by_tp
    sat_idx = [i for i in rf_idx if is_sat[i]]
    if not sat_idx:
        sat_idx = rf_idx
        is_sat = {i: True for i in rf_idx}
        notes.append("all RF pulses are short — treating every pulse as a saturation pulse")

    # ── trains: consecutive sat pulses with only RF/delay blocks in between
    trains: list[list[int]] = []
    for s in sat_idx:
        if trains and all(kinds[j] in ("rf", "delay") for j in range(trains[-1][-1] + 1, s)):
            trains[-1].append(s)
        else:
            trains.append([s])

    # ── measurement boundaries
    adc_idx = [i for i, k in enumerate(kinds) if k == "adc"]
    fd_num = None
    for k in CANON_KEYS["num_meas"]:
        if k in definitions:
            fd_num = int(_scalar(_def_value(definitions[k])))
    n_off_def = None
    for k in CANON_KEYS["offsets_ppm"]:
        if k in definitions:
            v = _def_value(definitions[k])
            n_off_def = len(v) if isinstance(v, list) else 1
            break
    n_adc, n_tr = len(adc_idx), len(trains)
    n_rep = 1
    for k in ("nSlices", "n_slices"):
        if k in definitions:
            try:
                n_rep = max(1, int(_scalar(_def_value(definitions[k]))))
            except (TypeError, ValueError):
                n_rep = 1
    if not (n_rep > 1 and n_off_def and n_tr == n_off_def * n_rep):
        n_rep = 1
    if n_rep > 1:
        use_adc = False                      # one saturation per slice readout: group by train
    elif not adc_idx:
        use_adc = False
    elif n_adc == n_tr or n_tr == 0:
        use_adc = True                       # one readout per saturation
    elif n_tr in (fd_num, n_off_def):
        use_adc = False                      # multi-slice readout: several ADCs per saturation (pHw)
    else:
        use_adc = bool(fd_num) and n_adc % fd_num == 0
    bounds: list[tuple[int, int]] = []     # (start, end) block index ranges, end exclusive
    if use_adc:
        start = 0
        for a in adc_idx:
            bounds.append((start, a + 1))
            start = a + 1
        seg = "adc"
    else:
        # a measurement starts at the delay run that precedes a train's first RF
        starts = []
        for tr in trains:
            first_rf = tr[0]
            j = first_rf
            while j - 1 >= 0 and kinds[j - 1] in ("rf", "delay"):
                j -= 1
            # j = first block of the (delay/rf) run containing the train; walk
            # forward past RF-less delays?  No — leading delays belong to it.
            starts.append(j)
        for n, s in enumerate(starts):
            e = starts[n + 1] if n + 1 < len(starts) else len(blocks)
            bounds.append((s, e))
        seg = "train"
    preamble = blocks[:bounds[0][0]] if bounds else blocks
    if preamble:
        notes.append(f"{len(preamble)} block(s) before the first measurement are kept verbatim")

    meas: list[Meas] = []
    for (s0, e0) in bounds:
        rng = list(range(s0, e0))
        tr = next((t for t in trains if s0 <= t[0] < e0), None)
        # head: anything that is neither RF nor delay before the train (e.g. the
        # saturation-recovery pre-pulses + crushers of WASABIT1) is kept verbatim
        # and played BEFORE the recovery delay that follows it
        head_blocks: list = []
        i = s0
        if tr is not None:
            non_rd = [j for j in range(s0, tr[0]) if kinds[j] not in ("rf", "delay")]
            if non_rd and non_rd[-1] + 1 < tr[0] and kinds[non_rd[-1] + 1] == "delay":
                head_blocks = blocks[s0:non_rd[-1] + 1]
                i = non_rd[-1] + 1
        # leading delays (= recovery)
        rec = 0.0
        lead = []
        while i < e0 and kinds[i] == "delay":
            rec += durs[i]
            lead.append(durs[i])
            i += 1
        if tr is None:
            post = blocks[i:e0]
            m = Meas(rec_delay=rec, units=[], gaps=[], post=post,
                     post_duration=sum(durs[i:e0]), lead_delays=lead)
            m.spoiler_idx = [k for k, b in enumerate(post) if _kind(b) == "grad"]
            meas.append(m)
            continue
        # the train region: from the first RF-only block after the leading
        # delays (prep pulses included) through the last unit's post pulses.
        # Anything else between the leading delays and that RF (e.g. a crusher
        # before the saturation) is kept verbatim in ``pre_blocks``.
        first_rf = i
        while first_rf < tr[0] and kinds[first_rf] != "rf":
            first_rf += 1
        pre_blocks = blocks[i:first_rf]
        units: list[Unit] = []
        gaps: list[float] = []
        # region before the first sat
        u = Unit()
        for j in range(first_rf, tr[0]):
            u.pre.append(("rf", _rf_item(blocks[j].rf, "pre", raster)) if kinds[j] == "rf" else ("delay", durs[j]))
        u.sat = _rf_item(blocks[tr[0]].rf, "sat", raster)
        units.append(u)
        pre_count = sum(1 for k, _ in u.pre if k == "rf")
        for a, b in zip(tr[:-1], tr[1:]):
            region = list(range(a + 1, b))
            dl = [j for j in region if kinds[j] == "delay"]
            if dl:
                gap_j = max(dl, key=lambda j: durs[j])
                post_r = [j for j in region if j < gap_j]
                pre_r = [j for j in region if j > gap_j]
                gap = durs[gap_j]
            else:
                # no delay between the two pulses: the RF blocks in between are
                # this unit's post pulses followed by the next unit's pre pulses
                # (as many pre pulses as unit 0 had)
                rf_in = [j for j in region if kinds[j] == "rf"]
                n_post = max(0, len(rf_in) - pre_count)
                cut = (rf_in[n_post - 1] + 1) if n_post > 0 else (a + 1)
                post_r = [j for j in region if j < cut]
                pre_r = [j for j in region if j >= cut]
                gap = 0.0
            for j in post_r:
                units[-1].post.append(("rf", _rf_item(blocks[j].rf, "post", raster)) if kinds[j] == "rf" else ("delay", durs[j]))
            gaps.append(gap)
            nu = Unit()
            for j in pre_r:
                nu.pre.append(("rf", _rf_item(blocks[j].rf, "pre", raster)) if kinds[j] == "rf" else ("delay", durs[j]))
            nu.sat = _rf_item(blocks[b].rf, "sat", raster)
            units.append(nu)
        # post pulses of the LAST unit: as many RF as the earlier units had
        # (or as many as the pre pulses when there is a single unit)
        k_post = (sum(1 for k, _ in units[0].post if k == "rf") if len(units) > 1 else pre_count)
        j = tr[-1] + 1
        taken_rf = 0
        pending: list = []
        while j < e0 and taken_rf < k_post and kinds[j] in ("rf", "delay"):
            if kinds[j] == "rf":
                for p in pending:
                    units[-1].post.append(p)
                pending = []
                units[-1].post.append(("rf", _rf_item(blocks[j].rf, "post", raster)))
                taken_rf += 1
            else:
                pending.append(("delay", durs[j]))
            j += 1
        if taken_rf < k_post:
            pending = []          # pattern not found — leave those blocks to the readout
        post_start = j if taken_rf == k_post else tr[-1] + 1
        # (delays that were pending after the last taken RF stay in the post region)
        post = blocks[post_start:e0]
        m = Meas(rec_delay=rec, units=units, gaps=gaps, post=post,
                 post_duration=sum(durs[post_start:e0]), pre_blocks=pre_blocks, lead_delays=lead,
                 head_blocks=head_blocks)
        m.spoiler_idx = [k for k, b in enumerate(post) if _kind(b) == "grad"]
        meas.append(m)

    if not meas:
        raise ValueError("could not find any measurement (saturation + readout) in this .seq")
    trailer = blocks[bounds[-1][1]:] if bounds else []
    model = SeqModel(path=path, version=version, definitions=definitions, preamble=preamble,
                     meas=meas, raster=raster, n_blocks=nb, segmentation=seg, notes=notes,
                     rep=n_rep, trailer=trailer)
    _split_zero_b1(model)
    return model


def _split_zero_b1(model: "SeqModel"):
    """A B1 = 0 measurement has no pulses: writers put delays of the train's
    length where the pulses would be, right after the recovery delay.  Split
    the leading delays into recovery (``rec_delay``) and that stand-in
    (``train_delays``) using the file's n_pulses / tp / td."""
    fd = model.file_defs()
    N = len(model.meas)
    if not any(not m.units for m in model.meas) or "tp" not in fd:
        return
    try:
        tp = broadcast(fd["tp"], N, "tp")
        td = TdSpec(fd.get("td", 0.0), N)
        npl = broadcast(fd.get("n_pulses", 1), N, "n_pulses")
    except ValueError:
        return
    for k, m in enumerate(model.meas):
        if m.units or not m.lead_delays:
            continue
        n = max(0, int(round(npl[k])))
        expect = n * tp[k] + td.train_gaps(k, n)
        if expect <= 0:
            continue
        lead = m.lead_delays
        for j in range(len(lead)):
            tail = sum(lead[j:])
            if abs(tail - expect) <= 1e-6 * (len(lead) - j) + 1e-9 * expect:
                m.train_delays = list(lead[j:])
                m.rec_delay = sum(lead[:j])
                break


# ─────────────────────────────────────────────────────────────────────────────
# Rebuild
# ─────────────────────────────────────────────────────────────────────────────

def _phase_model(phases: list, inc_expected: float) -> "tuple[float, float, str]":
    """(phase0, increment, mode) for a list of per-unit phases.
    mode 'accum' → increment follows Δω·2π·t_nonzero (pulseq-cest APTw scripts);
    'const' → fixed increment (phase cycling, e.g. 50° in the SL MRF scripts)."""
    if not phases:
        return 0.0, 0.0, "accum"
    p0 = float(phases[0])
    if len(phases) < 2:
        return p0, inc_expected, "accum"
    d = np.diff(np.asarray(phases, dtype=float))
    inc = float(np.angle(np.mean(np.exp(1j * d))))
    dist = abs(np.angle(np.exp(1j * (inc - inc_expected))))
    if dist < 1e-3:
        return p0, inc, "accum"
    return p0, inc, "const"


def rebuild(model: SeqModel, defs: dict, out_path: str, opts=None, author: str = "OCEAN") -> dict:
    """Regenerate ``model`` with the edited ``defs`` and write ``out_path``.
    Returns an info dict (num_meas, n_blocks, duration_s, warnings)."""
    warnings: list = []
    fd = model.file_defs()
    sd = model.structure_defs()
    n_orig = len(model.meas)

    def _lst(v):
        return [float(x) for x in (v if isinstance(v, (list, tuple, np.ndarray)) else [v])]

    # multi-B1 layout (library MultiPool_9p4T_001): b1pa = list of B1 levels and
    # the whole offsets_ppm list is played once per level (num_meas = per level)
    fb1, foff = fd.get("b1pa"), fd.get("offsets_ppm")
    file_levels = 1
    if (isinstance(fb1, list) and isinstance(foff, list) and len(fb1) > 1
            and len(fb1) != len(foff) and len(fb1) * len(foff) == n_orig):
        file_levels = len(fb1)

    # layout: measurement k ↔ (B1 level, offset j, repetition r) with
    #   k = level·(n_off·R) + j·R + r      (R = per-slice saturations, pHw)
    R = max(1, int(getattr(model, "rep", 1)))

    def _expand(vals_off, levels, reps):
        return [v for _lv in range(levels) for v in vals_off for _r in range(reps)]

    # ── new values (scalars broadcast to the measurement count)
    offs_in = _lst(defs.get("offsets_ppm", fd.get("offsets_ppm", sd["offsets_ppm"])))
    if len(offs_in) < 1:
        raise ValueError("offsets_ppm is empty")
    b1_in = _lst(defs.get("b1pa", fd.get("b1pa", sd["b1pa"])))
    n_levels = len(b1_in) if file_levels > 1 else 1        # in a multi-B1 file b1pa lists the levels
    n_off = len(offs_in)
    offs = _expand(offs_in, n_levels, R) if (n_levels > 1 or R > 1) else offs_in
    N = len(offs)

    def per_meas(v, name, n_each=n_off, levels=n_levels, reps=R):
        """scalar | one per offset (repeated per level / slice) | one per measurement"""
        vals = _lst(v)
        if (levels > 1 or reps > 1) and len(vals) == n_each:
            return _expand(vals, levels, reps)
        return broadcast(vals if len(vals) > 1 else vals[0], n_each * levels * reps, name)

    b0 = float(defs.get("b0", fd.get("b0", sd["b0"])))
    if b0 <= 0:
        raise ValueError("b0 must be > 0")
    freq = b0 * GAMMA_HZ
    if n_levels > 1:
        b1 = [b for b in b1_in for _ in range(n_off * R)]      # level-major, like the generator loops
    else:
        b1 = per_meas(b1_in if len(b1_in) > 1 else b1_in[0], "b1pa")
    npl = [max(0, int(round(x))) for x in per_meas(defs.get("n_pulses", fd.get("n_pulses", sd["n_pulses"])), "n_pulses")]
    tp = per_meas(defs.get("tp", fd.get("tp", sd["tp"])), "tp")
    _td_in = defs.get("td", fd.get("td", sd["td"]))
    _td_l = _lst(_td_in)
    td = TdSpec(_expand(_td_l, n_levels, R) if ((n_levels > 1 or R > 1) and len(_td_l) == n_off and n_off > 1)
                else _td_in, N)
    trec = per_meas(defs.get("trec", fd.get("trec", sd["trec"])), "trec")
    m0_off = defs.get("m0_offset", fd.get("m0_offset"))
    m0_off = None if m0_off in (None, "") or (isinstance(m0_off, float) and math.isnan(m0_off)) else float(_scalar(m0_off))
    trec_m0 = defs.get("trec_m0", fd.get("trec_m0"))
    trec_m0 = None if trec_m0 in (None, "") or (isinstance(trec_m0, float) and math.isnan(trec_m0)) else float(_scalar(trec_m0))
    spoiling = bool(int(float(_scalar(defs.get("spoiling", fd.get("spoiling", sd["spoiling"])), 0))))
    seq_id = str(defs.get("seq_id_string", fd.get("seq_id_string", sd["seq_id_string"]))).strip() or "seq"
    if any(x <= 0 for x in tp):
        raise ValueError("tp must be > 0")

    # ── reference (original) values per measurement, for the delta rules
    if file_levels > 1 or R > 1:
        foff_l = _lst(fd.get("offsets_ppm", sd["offsets_ppm"]))
        n_off_f = len(foff_l)
        if file_levels > 1:
            b1_ref = [b for b in _lst(fb1) for _ in range(n_off_f * R)]
        else:
            b1_ref = per_meas(fd.get("b1pa", sd["b1pa"]), "b1pa (file)", n_each=n_off_f, levels=1)
        off_ref = _expand(foff_l, file_levels, R)

        def ref_meas(v, name):
            return per_meas(v, name, n_each=n_off_f, levels=file_levels)
        _tdr = _lst(fd.get("td", sd["td"]))
        td_ref = TdSpec(_expand(_tdr, file_levels, R) if (len(_tdr) == n_off_f and n_off_f > 1)
                        else fd.get("td", sd["td"]), n_orig)
    else:
        b1_ref = broadcast(fd.get("b1pa", sd["b1pa"]), n_orig, "b1pa (file)") if n_orig else []
        off_ref = broadcast(fd.get("offsets_ppm", sd["offsets_ppm"]), n_orig, "offsets_ppm (file)")

        def ref_meas(v, name):
            return broadcast(v, n_orig, name)
        td_ref = TdSpec(fd.get("td", sd["td"]), n_orig)
    tp_ref = ref_meas(fd.get("tp", sd["tp"]), "tp (file)")
    npl_ref = ref_meas(fd.get("n_pulses", sd["n_pulses"]), "n_pulses (file)")
    trec_ref = ref_meas(fd.get("trec", sd["trec"]), "trec (file)")
    b0_ref = float(_scalar(fd.get("b0"), 0.0)) or float(sd["b0"])
    m0_ref = fd.get("m0_offset")
    m0_ref = None if m0_ref is None or (isinstance(m0_ref, float) and math.isnan(m0_ref)) else float(_scalar(m0_ref))
    trec_m0_ref = fd.get("trec_m0")
    trec_m0_ref = None if trec_m0_ref is None or (isinstance(trec_m0_ref, float) and math.isnan(trec_m0_ref)) else float(_scalar(trec_m0_ref))
    has_trec_def = "trec" in fd

    def applicable_ref(k):
        if m0_ref is not None and trec_m0_ref is not None and abs(off_ref[k] - m0_ref) < 1e-9:
            return trec_m0_ref
        return trec_ref[k]

    def applicable_new(k):
        if m0_off is not None and abs(offs[k] - m0_off) < 1e-9:
            return trec_m0 if trec_m0 is not None else trec[k]
        return trec[k]

    with_train = [i for i, m in enumerate(model.meas) if m.units]
    if not with_train:
        raise ValueError("the sequence has no saturation pulses to edit")

    def _follows_sat(it, sat) -> bool:
        """prep pulse played at the saturation frequency (adiabatic spin-lock)"""
        return abs(sat.freq) > 1.0 and abs(it.freq - sat.freq) <= 1e-6 * abs(sat.freq) + 1e-3

    # chain mode: prep pulses carry the offset frequency and every pulse's phase
    # accumulates along pre → sat → post (bmctool adiabatic SL, DGErho)
    chain_mode = any(_follows_sat(it, u.sat) for m in model.meas for u in m.units
                     for r, it in (u.pre + u.post) if r == "rf")

    def template_index(k):
        """Original measurement whose structure measurement k is built from."""
        if k < n_orig and model.meas[k].units:
            own_ok = True
            if chain_mode and np.sign(offs[k]) != np.sign(off_ref[k]) and offs[k] != 0:
                own_ok = False              # sign-specific adiabatic shapes (pre_pos / pre_neg)
            if own_ok:
                return k
        cand = with_train
        if chain_mode and offs[k] != 0:
            same = [i for i in with_train if np.sign(off_ref[i]) == np.sign(offs[k])]
            cand = same or with_train
        # nearest measurement with a train (earlier ones preferred)
        kk = min(k, n_orig - 1)
        return min(cand, key=lambda i: (abs(i - kk), i > kk))

    with vendored_pypulseq() as pp:
        seq = pp.Sequence(system=opts) if opts is not None else pp.Sequence()
        seq.rf_raster_time = model.raster
        total = 0.0
        for b in model.preamble:
            ev = _events(b)
            if ev:
                seq.add_block(*ev)
                total += float(pp.calc_duration(*ev))

        spoiler_events = None
        for k in range(N):
            ti = template_index(k)
            tm = model.meas[ti]
            own = model.meas[k] if k < n_orig else model.meas[ti]
            kref = k if k < n_orig else ti
            # recovery delay — delta rule keeps writer conventions (e.g. Trec − TE,
            # or OCEAN's first measurement that has no recovery delay at all);
            # the file's leading delay blocks are kept as separate blocks
            for b in (own.head_blocks if k < n_orig else tm.head_blocks):
                ev = _events(b)
                if ev:
                    seq.add_block(*ev); total += float(pp.calc_duration(*ev))
            n_rec = len(own.lead_delays) - len(own.train_delays)
            lead_rec = list(own.lead_delays[:max(n_rec, 0)]) or ([own.rec_delay] if own.rec_delay > 0 else [])
            if lead_rec:
                first = lead_rec[0] - applicable_ref(kref) + applicable_new(k)
                if first < -1e-9:
                    warnings.append(f"measurement {k+1}: recovery delay would be negative ({first:.4f} s) — set to 0")
                lead_rec[0] = max(0.0, first)
            for d in lead_rec:
                if d > 0:
                    seq.add_block(pp.make_delay(d))
                    total += d
            for b in (own.pre_blocks if k < n_orig else tm.pre_blocks):
                ev = _events(b)
                if ev:
                    seq.add_block(*ev); total += float(pp.calc_duration(*ev))

            n_units = npl[k]
            b1k = b1[k]
            b1_tmpl = b1_ref[ti] if ti < len(b1_ref) and b1_ref[ti] > 0 else tm.b1_estimate()
            scale = (b1k / b1_tmpl) if b1_tmpl > 0 else 0.0
            sat0 = tm.units[0].sat
            # two frequencies per measurement:
            #   dw_rf   — written to the [RF] freq column (the file's own value when
            #             the offset is unchanged → exact no-op; files round it)
            #   dw / dw_ref — exact ppm·B0·γ, used for every accumulated-phase step
            dw = offs[k] * freq                                        # Hz (exact)
            dw_ref_exact = off_ref[ti] * b0_ref * GAMMA_HZ
            dw_ref = (dw_ref_exact if abs(sat0.freq - dw_ref_exact) <= 2e-5 * abs(dw_ref_exact) + 1e-3
                      else sat0.freq)            # file definitions disagree with its blocks → trust the blocks
            same_off = abs(offs[k] - off_ref[ti]) <= 1e-12 and abs(b0 - b0_ref) <= 1e-12
            dw_rf = sat0.freq if same_off else dw
            if same_off:
                dw = dw_ref
            # chain mode: cumulative non-zero RF time before each template pulse
            t_ref_at = {}
            if chain_mode:
                acc = 0.0
                for ui, u in enumerate(tm.units):
                    for j, (r, it) in enumerate([x for x in u.pre if x[0] == "rf"]):
                        t_ref_at[(ui, "pre", j)] = acc; acc += it.shape.t_nonzero
                    t_ref_at[(ui, "sat", 0)] = acc; acc += u.sat.shape.t_nonzero
                    for j, (r, it) in enumerate([x for x in u.post if x[0] == "rf"]):
                        t_ref_at[(ui, "post", j)] = acc; acc += it.shape.t_nonzero
            t_new_acc = [0.0]

            def _chain_phase(ph_file, key):
                """file phase + change of the accumulated phase 2π·Δf·T at this pulse"""
                d = _TWO_PI * (dw * t_new_acc[0] - dw_ref * t_ref_at.get(key, 0.0))
                return ph_file if abs(d) < 1e-12 else (ph_file + d) % _TWO_PI
            sat_shape = sat0.shape.resampled(tp[k], tp_ref[ti])
            t_nz = sat_shape.t_nonzero
            t_nz_ref = sat0.shape.t_nonzero
            # phase pattern of the saturation pulses across the template units
            # pulseq-cest: accum_phase += 2π·Δf·(number of non-zero samples)·Δt
            inc_ref = (_TWO_PI * dw_ref * t_nz_ref) % _TWO_PI
            p0, inc, mode = _phase_model([u.sat.phase for u in tm.units], inc_ref)
            inc_new = (_TWO_PI * dw * t_nz) % _TWO_PI if mode == "accum" else inc
            # same increment as the file → its own (rounded) phase values are reused verbatim
            same_inc = abs(np.angle(np.exp(1j * (inc_new - (inc_ref if mode == "accum" else inc))))) < 1e-9
            # prep (spin-lock tip) pulses: re-tip when the template follows atan(B1/(Δppm·B0))
            aux_items = [(r, it) for u in tm.units[:1] for r, it in (u.pre + u.post) if r == "rf"]
            sl_mode = False
            if aux_items:
                theta_ref = math.atan(b1_tmpl / (abs(off_ref[ti]) * b0_ref + 1e-8))
                flips = [it.flip for _, it in aux_items]
                sl_mode = all(abs(f - theta_ref) <= 0.03 * max(theta_ref, 1e-6) + 1e-4 for f in flips)
            theta_new = math.atan(b1k / (abs(offs[k]) * b0 + 1e-8))
            theta_tmpl = math.atan(b1_tmpl / (abs(off_ref[ti]) * b0_ref + 1e-8))
            retip = sl_mode and abs(theta_new - theta_tmpl) > 1e-12      # unchanged → keep the file's tips exactly
            accum_ref = (_TWO_PI * dw_ref * sat0.shape.duration) % _TWO_PI
            accum_new = (_TWO_PI * dw * sat_shape.duration) % _TWO_PI
            # per-role phase models for the prep pulses (phase cycling continues)
            pre_phases = {}
            post_phases = {}
            for u in tm.units:
                for j, (r, it) in enumerate([x for x in u.pre if x[0] == "rf"]):
                    pre_phases.setdefault(j, []).append(it.phase)
                for j, (r, it) in enumerate([x for x in u.post if x[0] == "rf"]):
                    post_phases.setdefault(j, []).append(it.phase)

            def _aux_phase(role_phases, j, i):
                ph = role_phases.get(j, [0.0])
                q0, qinc, _ = _phase_model(ph, 0.0) if len(ph) > 1 else (ph[0], 0.0, "const")
                if len(ph) > 1 and i < len(ph):
                    return ph[i]
                return q0 + i * qinc

            unchanged_timing = (k < n_orig and n_units == int(round(npl_ref[kref]))
                                and abs(tp[k] - tp_ref[kref]) <= 1e-12
                                and td.same_as(td_ref, kref, n_units))
            if (n_units == 0 or b1k <= 0) and not own.units and own.train_delays and unchanged_timing:
                for d in own.train_delays:             # the file's own stand-in delays, verbatim
                    seq.add_block(pp.make_delay(d)); total += d
            elif n_units == 0 or b1k <= 0:
                # library convention: the train is replaced by delays of the same length
                for i in range(n_units):
                    u = tm.units[min(i, len(tm.units) - 1)]
                    d_unit = (u.aux_time + sat_shape.total_duration + sat0.delay)
                    if d_unit > 0:
                        seq.add_block(pp.make_delay(d_unit)); total += d_unit
                    if i < n_units - 1:
                        gap = _gap(tm, i, td.gap(k, i), td_ref.gap(ti, i), warnings, k)
                        if gap > 0:
                            seq.add_block(pp.make_delay(gap)); total += gap
            else:
                nt = len(tm.units)

                def _prep(it, ph_default, key, i, role):
                    """amplitude / frequency / phase of one prep pulse"""
                    amp, fq, ph = it.amp, it.freq, ph_default
                    if retip:
                        amp = it.amp * (theta_new / it.flip) if it.flip > 0 else it.amp
                    if chain_mode and _follows_sat(it, u.sat):
                        fq = it.freq if same_off else dw_rf         # follows the offset
                        if abs(it.amp - u.sat.amp) <= 1e-6 * max(u.sat.amp, 1e-12):
                            amp = it.amp * scale                   # amp = b1pa (adiabatic SL)
                        ph = _chain_phase(ph_default, (min(i, nt - 1), role, key))
                    elif sl_mode and role == "post":
                        ph = ph_default - accum_ref + accum_new
                    return amp, fq, ph

                for i in range(n_units):
                    u = tm.units[min(i, nt - 1)]
                    jpre = 0
                    for r, it in u.pre:
                        if r == "rf":
                            amp, fq, ph = _prep(it, _aux_phase(pre_phases, jpre, i), jpre, i, "pre")
                            seq.add_block(it.event(pp, amp, fq, ph)); total += it.duration
                            t_new_acc[0] += it.shape.t_nonzero
                            jpre += 1
                        else:
                            seq.add_block(pp.make_delay(it)); total += it
                    sat = u.sat
                    amp = sat.amp * scale
                    if chain_mode:
                        ph = _chain_phase(sat.phase, (min(i, nt - 1), "sat", 0))
                    elif same_inc and i < nt:      # unchanged → the file's own values
                        ph = tm.units[i].sat.phase
                    elif same_inc or mode == "const":
                        # beyond the file's pulse count: continue from its last pulse
                        ph = (tm.units[-1].sat.phase + (i - nt + 1) * inc_new) % _TWO_PI
                    else:                          # accumulated phase follows the (new) offset
                        ph = (p0 + i * inc_new) % _TWO_PI          # the library stores accum mod 2π
                    seq.add_block(sat.event(pp, amp, dw_rf if same_off else dw, ph, sat_shape)); total += sat.delay + sat_shape.total_duration
                    t_new_acc[0] += sat_shape.t_nonzero
                    jpost = 0
                    for r, it in u.post:
                        if r == "rf":
                            amp, fq, ph = _prep(it, _aux_phase(post_phases, jpost, i), jpost, i, "post")
                            seq.add_block(it.event(pp, amp, fq, ph)); total += it.duration
                            t_new_acc[0] += it.shape.t_nonzero
                            jpost += 1
                        else:
                            seq.add_block(pp.make_delay(it)); total += it
                    if i < n_units - 1:
                        gap = _gap(tm, i, td.gap(k, i), td_ref.gap(ti, i), warnings, k)
                        if gap > 0:
                            seq.add_block(pp.make_delay(gap)); total += gap
            # post region (spoiler, readout, ADC) verbatim — spoiler toggled
            post_src = own if k < n_orig else tm
            post_ev = []                                   # [(events, is_spoiler)]
            for idx, b in enumerate(post_src.post):
                ev = _events(b)
                if not ev:
                    continue
                is_sp = idx in post_src.spoiler_idx
                if is_sp and spoiler_events is None:
                    spoiler_events = ev
                post_ev.append((ev, is_sp))
            had_spoiler = any(sp for _, sp in post_ev)
            if not spoiling:
                post_ev = [(ev, sp) for ev, sp in post_ev if not sp]
            elif not had_spoiler:
                if spoiler_events is None:
                    spoiler_events = _make_spoiler(pp, opts)
                adc_pos = next((n for n, (ev, _) in enumerate(post_ev)
                                if any(getattr(e, "type", "") == "adc" for e in ev)), len(post_ev))
                post_ev.insert(adc_pos, (spoiler_events, True))
            for ev, _ in post_ev:
                seq.add_block(*ev); total += float(pp.calc_duration(*ev))

        for b in getattr(model, "trailer", []):          # blocks after the last measurement
            ev = _events(b)
            if ev:
                seq.add_block(*ev); total += float(pp.calc_duration(*ev))

        # ── definitions: keep the file's keys / order, update what changed
        new_defs = OrderedDict(model.definitions)

        def put(canon, value):
            key = model.file_key(canon) or MATLAB_NAME.get(canon, canon)
            new_defs[key] = value

        def per_off(vals):
            """one value per offset (multi-B1 levels / per-slice repeats collapsed)"""
            if n_levels > 1 or R > 1:
                vals = [vals[j * R] for j in range(n_off)]
            return _vec_or_scalar(vals)

        put("b0", b0)
        put("b1pa", [float(x) for x in b1_in] if n_levels > 1 else _vec_or_scalar(b1))
        put("n_pulses", per_off(npl))
        put("tp", per_off(tp))
        if (n_levels > 1 or R > 1) and td.mode == "meas":
            put("td", per_off(td.vals))
        else:
            put("td", td.value())
        put("trec", per_off(trec))
        if trec_m0 is not None:
            put("trec_m0", trec_m0)
        if m0_off is not None:
            put("m0_offset", m0_off)
        put("offsets_ppm", [float(x) for x in offs_in])
        # keep the file's convention (pHw counts the M0 played before the offsets)
        _nm_f, _no_f = fd.get("num_meas"), fd.get("offsets_ppm")
        _extra = 0
        if isinstance(_nm_f, (int, float)) and _no_f is not None:
            _extra = int(round(_nm_f)) - len(_lst(_no_f))
            _extra = _extra if 0 < _extra <= 2 else 0
        put("num_meas", n_off + _extra)
        dcs = [tp[k] / (tp[k] + td.mean(k)) if (tp[k] + td.mean(k)) > 0 else 1.0 for k in range(N)]
        put("dcsat", per_off(dcs))
        put("tsat", per_off([npl[k] * tp[k] + td.train_gaps(k, npl[k]) if npl[k] > 0 else 0.0
                             for k in range(N)]))
        put("seq_id_string", seq_id)
        put("spoiling", "1" if spoiling else "0")
        if model.file_key("freq"):
            put("freq", freq)
        if model.file_key("b1rms"):
            # rescale the file's own B1rms (keeps the writer's convention exactly)
            if file_levels > 1 and isinstance(fd["b1rms"], list) and len(fd["b1rms"]) == file_levels:
                rms_file = [r for r in fd["b1rms"] for _ in range(len(foff) * R)]
            elif R > 1:
                rms_file = per_meas(fd["b1rms"], "b1rms (file)", n_each=len(_lst(fd.get("offsets_ppm", [0]))), levels=1)
            else:
                rms_file = broadcast(fd["b1rms"], n_orig, "b1rms (file)")

            def _rms(b1v, shape, tpv, tdv):
                per = tpv + tdv
                return b1v * math.sqrt(shape.sq_integral / shape.duration * tpv / per) if per > 0 and shape.duration > 0 else b1v

            rms = []
            for k in range(N):
                ti = template_index(k)
                sh_ref = model.meas[ti].units[0].sat.shape
                sh_new = sh_ref.resampled(tp[k], tp_ref[ti])
                new = _rms(b1[k], sh_new, tp[k], td.mean(k))
                ref = _rms(b1_ref[ti], sh_ref, tp_ref[ti], td_ref.mean(ti))
                rms.append(rms_file[ti] * new / ref if ref > 0 else new)
            if n_levels > 1:
                put("b1rms", [float(rms[lv * n_off * R + (n_off - 1) * R]) for lv in range(n_levels)])
            else:
                put("b1rms", _vec_or_scalar(rms))
        seq.dict_definitions = new_defs
        seq.version_revision = 1                 # MATLAB CEST-MRF / BMCSim compatible header
        seq.write(out_path)
        n_blocks = len(seq.dict_block_events)
    _insert_header(out_path, author)
    return {"num_meas": N, "n_blocks": n_blocks, "duration_s": total, "warnings": warnings,
            "path": out_path, "seq_id_string": seq_id}


def _gap(tm: Meas, i: int, td_new: float, td_ref: float, warnings: list, k: int) -> float:
    g_ref = tm.gaps[min(i, len(tm.gaps) - 1)] if tm.gaps else max(0.0, td_ref - tm.units[0].aux_time)
    g = g_ref + (td_new - td_ref)
    if g < -1e-9:
        warnings.append(f"measurement {k+1}: td = {td_new:g} s is shorter than the prep pulses "
                        f"need — delay between pulses set to 0")
    return max(0.0, g)


def _vec_or_scalar(vals):
    a = np.asarray(vals, dtype=float)
    if a.size == 1 or np.all(np.abs(a - a[0]) <= 1e-12):
        v = float(a[0])
        return int(v) if v.is_integer() else v
    return [int(x) if float(x).is_integer() else float(x) for x in a]


def _make_spoiler(pp, opts):
    """pulseq-cest library spoiler: 0.8·max_grad, 6.5 ms, 1 ms ramps on x, y, z."""
    sysx = opts if opts is not None else pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=130,
                                                  slew_unit="T/m/s")
    amp = 0.8 * sysx.max_grad
    return [pp.make_trapezoid(channel=c, system=sysx, amplitude=amp, duration=6.5e-3, rise_time=1e-3)
            for c in ("x", "y", "z")]


def _insert_header(path: str, author: str):
    """pulseq-cest style header lines after '# Created by …'."""
    try:
        with open(path) as f:
            lines = f.readlines()
        out, done = [], False
        for ln in lines:
            out.append(ln)
            if not done and ln.startswith("# Created by"):
                out.append("\n# Created for Pulseq-CEST\n# https://pulseq-cest.github.io/\n"
                           f"# Created by: {author} (OCEAN Seq Parameter Definitions)\n")
                done = True
        with open(path, "w") as f:
            f.writelines(out)
    except OSError:
        pass


def derived(defs: dict) -> dict:
    """dcsat / num_meas / tsat / freq from an edited ``defs`` dict (no file)."""
    offs = defs.get("offsets_ppm", [])
    N = len(offs) if isinstance(offs, (list, tuple, np.ndarray)) else 1
    try:
        tp = broadcast(defs.get("tp", 0.0), N, "tp")
        td = TdSpec(defs.get("td", 0.0), N)
        npl = [int(round(x)) for x in broadcast(defs.get("n_pulses", 1), N, "n_pulses")]
    except ValueError:
        return {"num_meas": N}
    dcs = [tp[k] / (tp[k] + td.mean(k)) if (tp[k] + td.mean(k)) > 0 else 1.0 for k in range(N)]
    tsat = [npl[k] * tp[k] + td.train_gaps(k, npl[k]) if npl[k] > 0 else 0.0 for k in range(N)]
    b0 = float(_scalar(defs.get("b0", 0.0)))
    return {"dcsat": _vec_or_scalar(dcs), "num_meas": N, "tsat": _vec_or_scalar(tsat),
            "freq": b0 * GAMMA_HZ}
