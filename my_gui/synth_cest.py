"""
synth_cest.py
=============
Self-contained **synthetic CEST Z-spectrum generator** (Bloch-McConnell) for the
"Synthetic CEST MRI" sub-tab.

A user defines a pool system — a water pool, any number of CEST/NOE pools, and an
optional **MT pool** (SuperLorentzian / Lorentzian lineshape) — plus scanner /
saturation settings, and this module returns the CEST Z-spectrum.  It can also
build a synthetic phantom (controllable tiles or random shapes) whose regions
carry varying pool parameters, and assemble the per-pixel Z-spectra into a
Z-stack that looks like real acquired CEST data.

Physics: this drives BMCTool's validated ``BlochMcConnellSolver`` directly (block
saturation, no *.seq / pypulseq needed), so it runs in any environment that has
``bmctool`` installed.  The MT pool that was commented out in the original MATLAB
CEST-Generator is included here as a first-class, optional pool.

Conventions (verified against BMCTool):
  * ``rf_amp`` handed to the solver is in **Hz** = B1[µT] · γ/2π  (42.5764 Hz/µT).
  * ``rf_freq`` (saturation offset) is in **Hz** = Δ[ppm] · B0[T] · γ/2π.
  * Water Mz sits at state index ``2·(n_cest+1)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

GAMMA = 267.5153                 # ¹H gyromagnetic ratio [rad/s/µT]
GAMMA_HZ = GAMMA / (2 * np.pi)   # 42.5764 Hz/µT


# ─────────────────────────────────────────────────────────────────────────────
# Pool / scanner definitions
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CESTPool:
    name: str = "pool"
    f: float = 0.0009            # relative proton fraction (concentration / 111 M)
    k: float = 30.0              # exchange rate [Hz]
    dw: float = 3.5              # chemical shift relative to water [ppm]
    t1: float = 1.0             # T1 [s]
    t2: float = 0.04            # T2 [s]


@dataclass
class MTPool:
    f: float = 0.05             # relative pool size
    k: float = 40.0             # exchange rate [Hz]
    dw: float = 0.0             # centre offset [ppm] (symmetric MT, as in CEST-Generator)
    t2_us: float = 9.1          # T2 [µs]  (→ R2 = 1/9.1µs)
    t1: float = 1.0             # T1 [s]
    lineshape: str = "SuperLorentzian"   # SuperLorentzian | Lorentzian


@dataclass
class WaterPool:
    t1: float = 1.3             # T1 [s]
    t2: float = 0.05            # T2 [s]


@dataclass
class Scanner:
    b0: float = 3.0             # field strength [T]
    b1: float = 1.0             # saturation B1 [µT]
    mode: str = "cw"            # "cw" or "pulsed"
    tsat: float = 2.0           # CW saturation time [s]
    tp: float = 0.1             # pulsed: saturation time per pulse [s]
    dc: float = 0.5             # pulsed: duty cycle (0–1)
    n_pulses: int = 20          # pulsed: number of saturation pulses
    trec: float = 3.0           # recovery time [s]
    ppm_range: float = 6.0      # ± offset range [ppm]
    n_offsets: int = 61         # number of offsets across the range


@dataclass
class PoolSystem:
    water: WaterPool = field(default_factory=WaterPool)
    cest: List[CESTPool] = field(default_factory=list)
    mt: Optional[MTPool] = None


def default_pool_system() -> PoolSystem:
    """A sensible starting system: amide (+3.5), amine/creatine (+2.0), NOE (−3.5)."""
    return PoolSystem(
        water=WaterPool(t1=1.3, t2=0.05),
        cest=[
            CESTPool("Amide",   f=0.0018, k=30.0,   dw=3.5,  t1=1.0, t2=0.04),
            CESTPool("Amine",   f=0.0008, k=1000.0, dw=2.0,  t1=1.0, t2=0.04),
            CESTPool("NOE",     f=0.0006, k=20.0,   dw=-3.5, t1=1.0, t2=0.04),
        ],
        mt=MTPool(f=0.05, k=40.0, dw=0.0, t2_us=9.1, t1=1.0,
                  lineshape="SuperLorentzian"),
    )


def offset_list(scanner: Scanner) -> np.ndarray:
    """Symmetric list of offsets [ppm] from −ppm_range to +ppm_range."""
    return np.linspace(-scanner.ppm_range, scanner.ppm_range, int(scanner.n_offsets))


# ─────────────────────────────────────────────────────────────────────────────
# Bloch-McConnell Z-spectrum
# ─────────────────────────────────────────────────────────────────────────────
def _build_params(scanner: Scanner, water: WaterPool,
                  cest: List[CESTPool], mt: Optional[MTPool]):
    """Assemble a BMCTool ``Params`` object for the given pool system."""
    from bmctool.params import Params
    p = Params()
    p.set_water_pool(r1=1.0 / water.t1, r2=1.0 / water.t2, f=1.0)
    for c in cest:
        p.set_cest_pool(r1=1.0 / c.t1, r2=1.0 / c.t2, k=float(c.k),
                        f=float(c.f), dw=float(c.dw))
    if mt is not None:
        p.set_mt_pool(r1=1.0 / mt.t1, r2=1.0 / (mt.t2_us * 1e-6), k=float(mt.k),
                      f=float(mt.f), dw=float(mt.dw), lineshape=mt.lineshape)
    p.set_scanner(b0=scanner.b0, gamma=GAMMA, b0_inhom=0.0, rel_b1=1.0)
    p.set_options(verbose=False, reset_init_mag=True, max_pulse_samples=1, scale=1.0)
    p.set_m_vec()                     # populate p.m_vec (required by the solver)
    return p


def simulate_zspectrum(scanner: Scanner, water: WaterPool,
                       cest: List[CESTPool], mt: Optional[MTPool] = None,
                       offsets_ppm: Optional[np.ndarray] = None) -> np.ndarray:
    """Return the water Z-spectrum (Mz/M0) for the given pool system.

    Drives BMCTool's solver one offset at a time with block saturation
    (CW = one long pulse; pulsed = ``n_pulses`` pulses of ``tp`` with duty cycle
    ``dc``).  ``offsets_ppm`` defaults to :func:`offset_list`.
    """
    from bmctool.bmc_solver import BlochMcConnellSolver

    if offsets_ppm is None:
        offsets_ppm = offset_list(scanner)
    offsets_ppm = np.asarray(offsets_ppm, dtype=float)

    p = _build_params(scanner, water, cest, mt)
    solver = BlochMcConnellSolver(params=p, n_offsets=1)

    n_p = len(cest)
    mz_w = 2 * (n_p + 1)                     # index of water Mz in the state vector
    m0 = np.asarray(p.m_vec, dtype=float)
    rf_amp = scanner.b1 * GAMMA_HZ           # Hz

    # incomplete-recovery start magnetisation of the water pool
    zi = 1.0 - np.exp(-(1.0 / water.t1) * max(scanner.trec, 0.0))

    if scanner.mode == "pulsed":
        tp = float(scanner.tp)
        dc = float(np.clip(scanner.dc, 1e-3, 1.0))
        n_pulses = max(int(scanner.n_pulses), 1)
        td = tp * (1.0 - dc) / dc if dc < 1.0 else 0.0
    else:
        tp = float(scanner.tsat)
        n_pulses = 1
        td = 0.0

    zero = np.array([0.0])

    # ── Unsaturated M0 reference ──────────────────────────────────────────────
    # Run the SAME recovery + saturation-block timing but with the RF switched
    # OFF, and read the water Mz.  This is the physically correct Z = M_sat / M0
    # reference (exactly how an M0 image is acquired experimentally).  The old
    # code divided by ``zi`` (the *start-of-saturation* magnetisation), but during
    # t_sat the water T1-recovers well above ``zi`` — so far off-resonance
    # (negligible saturation) M_z > zi and Z = M_z/zi rose above 1 (wings ≈ 1.5).
    # Dividing by the RF-off M0 makes the wings sit at 1.0 as they should.
    mag0 = m0.copy()[np.newaxis, :, np.newaxis]
    mag0[0, mz_w, 0] = zi
    for _ in range(n_pulses):
        solver.update_matrix(0.0, zero, zero)
        mag0 = solver.solve_equation(mag0, tp)
        if td > 0.0:
            solver.update_matrix(0.0, zero, zero)
            mag0 = solver.solve_equation(mag0, td)
    m0_ref = float(np.real(mag0[0, mz_w, 0]))

    Z = np.empty(offsets_ppm.size, dtype=float)
    for idx, off in enumerate(offsets_ppm):
        off_hz = np.array([off * scanner.b0 * GAMMA_HZ])
        mag = m0.copy()[np.newaxis, :, np.newaxis]
        mag[0, mz_w, 0] = zi
        for _ in range(n_pulses):
            solver.update_matrix(rf_amp, zero, off_hz)
            mag = solver.solve_equation(mag, tp)
            if td > 0.0:
                solver.update_matrix(0.0, zero, off_hz)
                mag = solver.solve_equation(mag, td)
        Z[idx] = float(np.real(mag[0, mz_w, 0]))

    if m0_ref > 1e-6:
        Z = Z / m0_ref                        # normalise to the unsaturated M0
    return Z


def mtr_asym(offsets_ppm: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """MTR_asym(Δ) = Z(−Δ) − Z(+Δ) on the positive-offset axis.

    Returns (pos_offsets, asym).  Assumes a symmetric offset list.
    """
    offsets_ppm = np.asarray(offsets_ppm, float)
    Z = np.asarray(Z, float)
    pos = offsets_ppm[offsets_ppm >= 0]
    asym = np.empty(pos.size)
    for i, dw in enumerate(pos):
        zp = np.interp(dw, offsets_ppm, Z)
        zn = np.interp(-dw, offsets_ppm, Z)
        asym[i] = zn - zp
    return pos, asym


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic phantoms
# ─────────────────────────────────────────────────────────────────────────────
def generate_phantom(size: int = 64, mode: str = "tiles",
                     n_regions: int = 9, seed: Optional[int] = None) -> np.ndarray:
    """Return an ``(size, size)`` integer label image (0 = background,
    1..K = regions).  ``mode`` is ``"tiles"`` (a grid of circular vials) or
    ``"random"`` (a foreground ellipse filled with random shapes)."""
    rng = np.random.default_rng(seed)
    lbl = np.zeros((size, size), dtype=int)
    yy, xx = np.mgrid[0:size, 0:size]

    if mode == "random":
        # foreground ellipse
        a = size * (0.28 + 0.12 * rng.random())
        b = size * (0.28 + 0.12 * rng.random())
        cy, cx = size / 2, size / 2
        fg = ((yy - cy) / b) ** 2 + ((xx - cx) / a) ** 2 <= 1.0
        # scatter random shapes, each assigned a random region label
        n_shapes = max(n_regions * 3, 12)
        for _ in range(n_shapes):
            lab = int(rng.integers(1, n_regions + 1))
            sy, sx = rng.integers(0, size), rng.integers(0, size)
            if rng.random() < 0.5:                     # circle
                r = rng.integers(3, max(4, size // 6))
                m = (yy - sy) ** 2 + (xx - sx) ** 2 <= r * r
            else:                                      # rectangle
                hy = rng.integers(3, max(4, size // 5))
                hx = rng.integers(3, max(4, size // 5))
                m = (np.abs(yy - sy) <= hy) & (np.abs(xx - sx) <= hx)
            lbl[m & fg] = lab
        lbl[~fg] = 0
        # ensure every region label is present; fall back to tiles if too sparse
        if np.unique(lbl[lbl > 0]).size < max(2, n_regions // 2):
            return generate_phantom(size, "tiles", n_regions, seed)
        return lbl

    # ── tiles: grid of circular vials ────────────────────────────────────────
    g = int(np.ceil(np.sqrt(n_regions)))
    r = 0.42 * size / g
    k = 0
    for iy in range(g):
        for ix in range(g):
            if k >= n_regions:
                break
            cy = (iy + 0.5) * size / g
            cx = (ix + 0.5) * size / g
            lbl[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = k + 1
            k += 1
    return lbl


def vary_pool_systems(base: PoolSystem, n_regions: int,
                     target: str = "none", lo: float = 1.0,
                     hi: float = 1.0) -> List[PoolSystem]:
    """Return ``n_regions`` copies of ``base`` in which one parameter is swept
    linearly from ``lo`` to ``hi`` across regions.

    ``target`` selects what to vary:
      * ``"none"`` — identical regions.
      * ``"cest{i}_f"`` / ``"cest{i}_k"`` / ``"cest{i}_dw"`` — the *(i+1)*-th CEST
        pool's conc / rate / shift (0-based index, e.g. ``"cest0_f"`` for pool 1,
        ``"cest2_dw"`` for pool 3).
      * ``"mt_f"`` — MT pool size.
      * ``"water_t1"`` / ``"water_t2"`` — water relaxation.
    """
    import copy

    def _cest_target(t):
        # "cest0_f" / "cest2_dw" -> (index, attr) ; returns None if not a CEST target
        if not t.startswith("cest") or "_" not in t:
            return None
        head, attr = t.split("_", 1)
        try:
            return int(head[4:]), attr
        except ValueError:
            return None

    ct = _cest_target(target)
    vals = np.linspace(lo, hi, max(n_regions, 1))
    out: List[PoolSystem] = []
    for v in vals:
        ps = copy.deepcopy(base)
        if ct is not None and ps.cest and 0 <= ct[0] < len(ps.cest) and ct[1] in ("f", "k", "dw"):
            setattr(ps.cest[ct[0]], ct[1], float(v))
        elif target == "mt_f" and ps.mt is not None:
            ps.mt.f = float(v)
        elif target == "water_t1":
            ps.water.t1 = float(v)
        elif target == "water_t2":
            ps.water.t2 = float(v)
        out.append(ps)
    return out


def randomize_pool_systems(base: PoolSystem, n_regions: int,
                           ppm_range: float = 5.0,
                           seed: Optional[int] = None) -> List[PoolSystem]:
    """Return ``n_regions`` pool systems in which **every** pool parameter varies
    randomly across regions — the CEST-Generator behaviour (``generate_pool_params.m``
    + ``generateCESTPoolParams.m``), where each region gets its own water/CEST/MT
    relaxation, concentration, exchange-rate and chemical-shift values.

    Ranges follow the MATLAB source: T1 0.5–2.5 s, T2 1–20 ms (pools) / 20–110 ms
    (water), CEST f ≈ (rand+0.5)²·800/111 ‰, k 50–2050 Hz, Δω within ±ppm_range
    (sign anchored to each base pool so amide/NOE identities are kept).
    """
    import copy
    rng = np.random.default_rng(seed)
    out: List[PoolSystem] = []
    for _ in range(max(int(n_regions), 1)):
        ps = copy.deepcopy(base)
        ps.water.t1 = float(rng.uniform(0.5, 2.5))
        ps.water.t2 = float(rng.uniform(0.02, 0.11))
        for c in ps.cest:
            c.f = float((rng.random() + 0.5) ** 2 * 800.0 / 1000.0 / 111.0)
            c.k = float(rng.random() * 2000.0 + 50.0)
            sign = 1.0 if c.dw >= 0 else -1.0        # keep pool identity (down/up-field)
            c.dw = float(sign * rng.uniform(0.3, 1.0) * ppm_range)
            c.t1 = float(rng.uniform(0.5, 2.5))
            c.t2 = float(rng.uniform(0.005, 0.05))
        if ps.mt is not None:
            ps.mt.f = float(rng.uniform(0.02, 0.12))
            ps.mt.k = float(rng.uniform(20.0, 80.0))
        out.append(ps)
    return out


def pool_param_maps(label_img: np.ndarray,
                    systems: List[PoolSystem]) -> "dict":
    """Build per-pool 2-D parameter maps (piecewise-constant per region), for a
    CEST-Generator-style figure.  Background is NaN.  Returns::

        {"Pool A (water)": {"R1": map, "R2": map},
         "Pool B":         {"f": map, "k": map, "dw": map, "R1": map, "R2": map},
         ... , "MT":       {"f": map, "k": map, "dw": map}}
    """
    H, W = label_img.shape
    labels = np.unique(label_img); labels = labels[labels > 0]

    def _blank():
        return np.full((H, W), np.nan, dtype=float)

    s0 = systems[0]
    maps: dict = {"Pool A (water)": {"R1": _blank(), "R2": _blank()}}
    pool_letters = ["B", "C", "D", "E", "F", "G"]
    for pi in range(len(s0.cest)):
        name = f"Pool {pool_letters[pi]}" if pi < len(pool_letters) else f"Pool {pi+2}"
        maps[name] = {k: _blank() for k in ("f", "k", "dw", "R1", "R2")}
    if s0.mt is not None:
        maps["MT"] = {k: _blank() for k in ("f", "k", "dw")}

    for lab in labels:
        sysm = systems[min(int(lab) - 1, len(systems) - 1)]
        msk = label_img == lab
        maps["Pool A (water)"]["R1"][msk] = 1.0 / sysm.water.t1
        maps["Pool A (water)"]["R2"][msk] = 1.0 / sysm.water.t2
        for pi, c in enumerate(sysm.cest):
            name = f"Pool {pool_letters[pi]}" if pi < len(pool_letters) else f"Pool {pi+2}"
            maps[name]["f"][msk] = c.f
            maps[name]["k"][msk] = c.k
            maps[name]["dw"][msk] = c.dw
            maps[name]["R1"][msk] = 1.0 / c.t1
            maps[name]["R2"][msk] = 1.0 / c.t2
        if sysm.mt is not None and "MT" in maps:
            maps["MT"]["f"][msk] = sysm.mt.f
            maps["MT"]["k"][msk] = sysm.mt.k
            maps["MT"]["dw"][msk] = sysm.mt.dw
    return maps


def simulate_phantom(label_img: np.ndarray, systems: List[PoolSystem],
                     scanner: Scanner, offsets_ppm: Optional[np.ndarray] = None,
                     progress=None) -> np.ndarray:
    """Simulate one Z-spectrum per region and fill an ``(H, W, n_off)`` Z-stack.

    ``systems[i]`` is used for region label ``i+1`` (clamped if fewer systems
    than regions).  ``progress(done, total)`` is called after each region.
    """
    if offsets_ppm is None:
        offsets_ppm = offset_list(scanner)
    offsets_ppm = np.asarray(offsets_ppm, float)

    H, W = label_img.shape
    zstack = np.zeros((H, W, offsets_ppm.size), dtype=float)
    labels = np.unique(label_img)
    labels = labels[labels > 0]
    for li, lab in enumerate(labels):
        ps = systems[min(int(lab) - 1, len(systems) - 1)]
        Z = simulate_zspectrum(scanner, ps.water, ps.cest, ps.mt, offsets_ppm)
        zstack[label_img == lab, :] = Z
        if progress is not None:
            progress(li + 1, len(labels))
    return zstack
