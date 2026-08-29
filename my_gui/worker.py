"""
worker.py
Background QThread worker for MRF dictionary simulation and dot-product matching.

Mirrors the logic of the original MRFmatch_mt.py pipeline:
  1.  Load dictpars + seq_defs from acquired_data.mat  (ConfigMT equivalent)
  2.  Build YAML scenario                               (write_yaml_dict)
  3.  Write pulse sequence file                         (write_sequence_sl → write_seq fallback)
  4.  Generate MRF dictionary                           (generate_mrf_cest_dictionary)
       - auto-adds equals=[('fs_0','fs_1',0.6666667)]  when 2 CEST pools present
  5.  Run dot-product matching                          (dot_prod_matching)
  6.  Save quant_maps.mat + mask.npy
  7.  Emit quant_maps dict back to the GUI
"""

from __future__ import annotations
import os
import time

import numpy as np
import scipy.io as sio
from PyQt6.QtCore import QThread, pyqtSignal

from my_gui.paths import output_path, get_output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Vendored Pulseq-1.3.1 context — force .seq writing onto the repo's bundled
# pypulseq (open-py-cest-mrf/cest_mrf/pypulseq) which the C++ BMCSimulator can
# parse, without disturbing the pip pypulseq 1.5 that BMCTool needs.
# ─────────────────────────────────────────────────────────────────────────────
import contextlib as _contextlib


@_contextlib.contextmanager
def _vendored_pypulseq():
    """Temporarily make ``import pypulseq`` resolve to the repo's vendored
    Pulseq-1.3.1 build, then restore whatever was loaded before (pip 1.5)."""
    import sys as _sys
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # …/ocean
    _vend = os.path.join(_repo, 'open-py-cest-mrf', 'cest_mrf', 'pypulseq')
    _saved = {m: _sys.modules[m] for m in list(_sys.modules)
              if m == 'pypulseq' or m.startswith('pypulseq.')}
    _had_path = _vend in _sys.path
    for _m in _saved:
        del _sys.modules[_m]
    if not _had_path:
        _sys.path.insert(0, _vend)
    try:
        yield
    finally:
        if not _had_path:
            try:
                _sys.path.remove(_vend)
            except ValueError:
                pass
        for _m in [m for m in list(_sys.modules)
                   if m == 'pypulseq' or m.startswith('pypulseq.')]:
            del _sys.modules[_m]
        _sys.modules.update(_saved)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — mirror ConfigMT and setup_sequence_definitions from original script
# ─────────────────────────────────────────────────────────────────────────────

def _load_dictpars(acqdata_fn: str) -> dict:
    """
    Load dictpars struct from acquired_data.mat and flatten into a plain dict.
    Matches the ConfigMT.dp_import logic exactly.
    """
    dp_import = sio.loadmat(acqdata_fn)['dictpars']
    dp: dict = {}
    for name in dp_import.dtype.names:
        raw = dp_import[name].flatten()[0].flatten()
        if len(raw) > 1:
            dp[name] = raw.tolist()
        elif isinstance(raw[0], np.integer):
            dp[name] = int(raw[0])
        else:
            try:
                dp[name] = float(raw[0])
            except (ValueError, TypeError):
                val = raw[0]
                dp[name] = val.decode('utf-8') if hasattr(val, 'decode') else str(val)
    return dp


def _build_config_from_dictpars(dp: dict, acqdata_fn: str,
                                  yaml_fn: str, seq_fn: str,
                                  dict_fn: str, quantmaps_fn: str,
                                  num_workers: int, max_pulse_samples: int) -> dict:
    """
    Build the full config dict from dictpars (same as ConfigMT.__init__).
    """
    cfg: dict = {}
    cfg['acqdata_fn']    = acqdata_fn
    cfg['yaml_fn']       = yaml_fn
    cfg['seq_fn']        = seq_fn
    cfg['dict_fn']       = dict_fn
    cfg['quantmaps_fn']  = quantmaps_fn
    cfg['num_workers']   = num_workers
    cfg['max_pulse_samples'] = max_pulse_samples

    # Water pool
    cfg['water_pool'] = {
        't1': dp['water_t1'],
        't2': dp['water_t2'],
        'f':  dp['water_f'],
    }

    # CEST / solute pool(s)
    cfg['cest_pool'] = {}
    if 'cest_amine_f' in dp:
        cfg['cest_pool']['Amine'] = {
            't1': dp['cest_amine_t1'],
            't2': dp['cest_amine_t2'],
            'k':  dp['cest_amine_k'],
            'dw': dp['cest_amine_dw'],
            'f':  dp['cest_amine_f'],
        }
    if 'cest_mt_f' in dp:
        cfg['cest_pool']['MT'] = {
            't1': dp['cest_mt_t1'],
            't2': dp['cest_mt_t2'],
            'k':  dp['cest_mt_k'],
            'dw': dp['cest_mt_dw'],
            'f':  dp['cest_mt_f'],
        }
    if not cfg['cest_pool']:
        cfg.pop('cest_pool')

    # MT pool
    if 'mt_f' in dp:
        cfg['mt_pool'] = {
            't1':        dp['mt_t1'],
            't2':        dp['mt_t2'],
            'k':         dp['mt_k'],
            'dw':        dp['mt_dw'],
            'f':         dp['mt_f'],
            'lineshape': str(dp.get('mt_lineshape', 'SuperLorentzian')),
        }

    # Scanner / magnetisation
    cfg['scale']          = dp.get('magnetization_scale', 1)
    cfg['reset_init_mag'] = dp.get('magnetization_reset', 0)
    cfg['b0']             = dp.get('b0', 3.0)
    cfg['gamma']          = dp.get('gamma', 267.5153)
    cfg['b0_inhom']       = dp.get('b0_inhom', 0.0)
    cfg['rel_b1']         = dp.get('rel_b1', 1.0)
    cfg['verbose']        = 0

    return cfg


def _load_seq_defs(cfg: dict) -> dict:
    """
    Load seq_defs from acquired_data.mat (setup_sequence_definitions equivalent).
    """
    sd_import = sio.loadmat(cfg['acqdata_fn'])['seq_defs']
    seq_defs: dict = {}
    for name in sd_import.dtype.names:
        raw = sd_import[name].flatten()[0].flatten()
        if len(raw) > 1:
            seq_defs[name] = raw.tolist()
        elif isinstance(raw[0], np.integer):
            seq_defs[name] = int(raw[0])
        else:
            seq_defs[name] = float(raw[0])

    # Back-compat additions (DK edits from original script)
    if 'SLflag' not in seq_defs:
        offsets = np.array(seq_defs.get('offsets_ppm', []))
        seq_defs['SLflag'] = (offsets < 1e-3).tolist()
    if 'SLFA' not in seq_defs:
        seq_defs['SLFA'] = seq_defs.get('excFA', 60)

    seq_defs['B0']            = cfg['b0']
    seq_defs['seq_id_string'] = os.path.splitext(cfg['seq_fn'])[1][1:] or 'seq'

    # write_sequence_DK indexes tp/Trec/excFA/SLFA per-measurement.
    # If the .mat stored them as scalars, broadcast to num_meas-length lists.
    num_meas = int(seq_defs.get('num_meas', 1))
    for key in ('tp', 'Trec', 'excFA', 'SLFA'):
        if key in seq_defs and not isinstance(seq_defs[key], (list, np.ndarray)):
            seq_defs[key] = [seq_defs[key]] * num_meas

    return seq_defs


# ─────────────────────────────────────────────────────────────────────────────
# DictWorker
# ─────────────────────────────────────────────────────────────────────────────

class DictWorker(QThread):
    """
    Background worker for the full MRF pipeline:
      dictionary generation  →  dot-product matching  →  save results.

    Two operating modes:
      A) acquired_data_fn provided → reads dictpars/seq_defs from .mat (original pipeline)
      B) explicit cfg + seq_defs   → uses values from GUI Config/Sequence tabs

    Signals:
      log(str)           – progress messages
      finished(dict)     – quant_maps dict on success
      error(str)         – error message on failure
    """

    finished  = pyqtSignal(dict)
    error     = pyqtSignal(str)
    log       = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        # Mode A — acquired_data.mat drives everything
        acquired_data_fn: str | None = None,
        # Mode B — explicit params from GUI tabs
        seq_fn:       str | None = None,
        param_fn:     str | None = None,
        dict_fn:      str | None = None,
        num_workers:  int = 8,
        max_pulse_samples: int = 100,
        # Full cfg dict from app.py (Mode B) — avoids rebuilding in worker
        full_cfg:     dict | None = None,
        # seq_defs passed directly from SequenceTab (Mode B)
        seq_defs:     dict | None = None,
        # Output paths (used in both modes)
        quantmaps_fn: str = "",   # resolved below via output_path()
        yaml_fn:      str = "",   # resolved below via output_path()
    ):
        super().__init__()
        self._acquired_data_fn   = acquired_data_fn
        self._seq_fn             = seq_fn
        self._param_fn           = param_fn
        self._dict_fn            = dict_fn or output_path("dict.mat")
        self._num_workers        = num_workers
        self._max_pulse_samples  = max_pulse_samples
        self._full_cfg           = full_cfg   # complete cfg from ConfigTab (Mode B)
        self._gui_seq_defs       = seq_defs   # from SequenceTab.get_seq_defs()
        self._quantmaps_fn       = quantmaps_fn or output_path("quant_maps.mat")
        self._yaml_fn            = yaml_fn    or output_path("scenario.yaml")

    def stop(self):
        """Request cancellation between pipeline steps."""
        self.requestInterruption()

    # ── main thread entry ─────────────────────────────────────────────────

    def run(self):
        try:
            # Ensure the output directory exists (works in both dev and bundled mode)
            get_output_dir()   # mkdir is done inside; CWD was already set by main.py

            # ── Step 1: build config from GUI tabs (always) ───────────────
            # Pool parameters always come from the Config tab (full_cfg).
            # Sequence timing always comes from the Sequence tab (gui_seq_defs).
            # acquired_data_fn is used ONLY for dot-product matching (Step 5).
            if self._full_cfg is not None:
                cfg = dict(self._full_cfg)
                cfg['yaml_fn']      = self._yaml_fn
                cfg['seq_fn']       = self._seq_fn or cfg.get('seq_fn', output_path('acq_protocol.seq'))
                cfg['dict_fn']      = self._dict_fn
                cfg['quantmaps_fn'] = self._quantmaps_fn
                cfg['num_workers']  = self._num_workers
            elif self._param_fn:
                # Minimal fallback when launched without a full_cfg
                cfg = {
                    'yaml_fn':      self._yaml_fn,
                    'seq_fn':       self._seq_fn or output_path('acq_protocol.seq'),
                    'dict_fn':      self._dict_fn,
                    'quantmaps_fn': self._quantmaps_fn,
                    'num_workers':  self._num_workers,
                }
            else:
                raise ValueError(
                    "No pool configuration available. "
                    "Please set parameters in the Configuration tab."
                )

            self.log.emit(
                f"Config ready — B0={cfg.get('b0', '?')} T, "
                f"pools: {list(cfg.get('cest_pool', {}).keys())}, "
                f"MT: {'mt_pool' in cfg}"
            )

            if self.isInterruptionRequested():
                self.log.emit("Cancelled by user.")
                self.cancelled.emit()
                return

            # ── Step 2: write YAML scenario ───────────────────────────────
            # app.py pre-writes the YAML; only write here if it doesn't exist.
            from cest_mrf.write_scenario import write_yaml_dict
            if not (self._yaml_fn and os.path.isfile(self._yaml_fn)):
                write_yaml_dict(cfg)
                self.log.emit("Configuration saved")
            else:
                self.log.emit("Configuration saved")

            if self.isInterruptionRequested():
                self.log.emit("Cancelled by user.")
                self.cancelled.emit()
                return

            # ── Step 3: write sequence file ──────────────────────────────
            # seq_defs come from the GUI Sequence tab (.txt loaded or the
            # default/edited schedule).  Users load their MRF schedule via
            # "Load from .txt" or "Load from Bruker scan" in the Sequence tab.
            if self._gui_seq_defs is not None:
                seq_defs = dict(self._gui_seq_defs)
                seq_defs.setdefault('B0',            cfg.get('b0', 9.4))
                seq_defs.setdefault('seq_id_string', 'seq')
                loaded_fn = seq_defs.pop('_loaded_fname', '')
                src = f"'{loaded_fn}'" if loaded_fn else "default/edited schedule"
                n_meas = seq_defs.get('num_meas', '?')
                trec0  = seq_defs.get('Trec', [None])[0]
                trec0_s = f"{float(trec0):.4f} s" if trec0 is not None else "?"
                self.log.emit(
                    f"Sequence tab seq_defs loaded ({src}) — "
                    f"num_meas={n_meas}, Trec[0]={trec0_s}"
                )
            else:
                seq_defs = {}
                self.log.emit(
                    "WARNING: no seq_defs from Sequence tab — "
                    "sequence file may be incomplete"
                )

            self._write_sequence(cfg, seq_defs)

            if self.isInterruptionRequested():
                self.log.emit("Cancelled by user.")
                self.cancelled.emit()
                return

            # ── Step 4: generate dictionary ───────────────────────────────
            self.log.emit("Starting dictionary generation …")
            t0 = time.perf_counter()

            from cest_mrf.dictionary.generation import generate_mrf_cest_dictionary

            # equals constraint: when 2 CEST pools present, link fs_0 and fs_1
            eqvals = None
            if 'cest_pool' in cfg and len(cfg['cest_pool']) > 1:
                eqvals = [('fs_0', 'fs_1', 0.6666667)]
                self.log.emit(
                    "2 CEST pools detected → equals constraint: fs_0 = 0.6667 × fs_1"
                )

            generate_mrf_cest_dictionary(
                seq_fn      = cfg['seq_fn'],
                param_fn    = cfg['yaml_fn'],
                dict_fn     = cfg['dict_fn'],
                num_workers = cfg['num_workers'],
                axes        = 'xy',
                equals      = eqvals,
            )
            dt_gen = time.perf_counter() - t0
            self.log.emit(
                f"Dictionary generation complete in {dt_gen:.1f} s → {cfg['dict_fn']}"
            )

            if self.isInterruptionRequested():
                self.log.emit("Cancelled by user (after dictionary generation).")
                self.cancelled.emit()
                return

            # ── Step 5: dot-product matching (only if acquired_data.mat available) ──
            acqdata_fn = self._acquired_data_fn
            if not acqdata_fn or not os.path.isfile(acqdata_fn):
                self.log.emit(
                    "No acquired_data.mat — dictionary generation complete. "
                    "Load acquired data and re-run to perform matching."
                )
                self.finished.emit({'_dict_only': True})
                return

            self.log.emit("Running dot-product matching…")
            t1 = time.perf_counter()

            # Try dot_product_mt first (handles MT-only and mixed pool dicts).
            # Fall back to dot_product for legacy CEST-only dicts.
            try:
                from cest_mrf.metrics.dot_product_mt import dot_prod_matching
            except ImportError:
                from cest_mrf.metrics.dot_product import dot_prod_matching

            quant_maps = dot_prod_matching(
                dict_fn          = cfg['dict_fn'],
                acquired_data_fn = acqdata_fn,
            )
            dt_match = time.perf_counter() - t1
            self.log.emit(
                f"Matching complete in {dt_match:.2f} s — "
                f"maps: {', '.join(quant_maps.keys())}"
            )

            # ── Step 6: save results ──────────────────────────────────────
            sio.savemat(cfg['quantmaps_fn'], quant_maps)
            self.log.emit(f"quant_maps saved → {cfg['quantmaps_fn']}")

            if 'dp' in quant_maps:
                mask = quant_maps['dp'] > 0.99974
                mask_fn = os.path.join(os.path.dirname(cfg['quantmaps_fn']), 'mask.npy')
                np.save(mask_fn, mask)
                self.log.emit(f"Mask saved → {mask_fn}")

            self.finished.emit(quant_maps)

        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")

    # ── sequence writer ───────────────────────────────────────────────────

    # Fields that creation.py treats as plain Python scalars.
    # When loaded from .mat they can arrive as 1-D lists (e.g. tp repeated
    # once per time-point); we collapse them to a single value before passing
    # to write_sequence so that  B1 * gyroRatio_rad * seq_defs['tp']  works.
    _CREATION_SCALAR_FIELDS: dict[str, type] = {
        'tp':       float,
        'td':       float,
        'Trec':     float,
        'Trec_M0':  float,
        'n_pulses': int,
        'B0':       float,
        'SLFA':     float,
        'excFA':    float,
    }

    @staticmethod
    def _sanitize_for_creation(sd: dict) -> dict:
        """
        Return a copy of *sd* where every key listed in _CREATION_SCALAR_FIELDS
        is guaranteed to be a plain Python scalar (not a list or ndarray).
        Arrays/lists are collapsed to their first element; the value is then
        cast to the required type (float or int).
        """
        sd = dict(sd)
        for k, cast in DictWorker._CREATION_SCALAR_FIELDS.items():
            if k not in sd:
                continue
            v = sd[k]
            if isinstance(v, (list, np.ndarray)):
                arr = np.asarray(v, dtype=float).ravel()
                sd[k] = cast(arr[0]) if len(arr) > 0 else cast(0)
            else:
                try:
                    sd[k] = cast(v)
                except (TypeError, ValueError):
                    pass
        return sd

    @staticmethod
    def _sanitize_for_sl(sd: dict) -> dict:
        """
        Prepare seq_defs for sequences_sl.write_sequence_sl.

        write_sequence_sl requirements:
          • All per-measurement fields (tp, Trec, B1pa, excFA, SLFA, SLflag,
            offsets_ppm) must be indexable Python lists (not numpy arrays).
          • td   must be a plain Python float scalar.
          • B0   must be a plain Python float scalar.
          • NaN placeholder keys (Trec_M0, M0_offset) are excluded — pypulseq
            may reject NaN values in seq.set_definition().
          • Internal helpers (_loaded_fname) are stripped.
        """
        import math

        # Keys that pypulseq should NOT receive as definitions
        _EXCLUDE = {'Trec_M0', 'M0_offset', '_loaded_fname'}

        def _to_pylist(v):
            """Convert numpy arrays / other iterables to Python list of floats."""
            if hasattr(v, 'tolist'):          # numpy array
                return v.tolist()
            if isinstance(v, (list, tuple)):
                return [float(x) if not isinstance(x, (int, bool)) else x for x in v]
            return v

        def _to_scalar(v):
            """Collapse array to first element, return plain Python float."""
            if hasattr(v, '__len__') and not isinstance(v, str):
                return float(v[0]) if len(v) > 0 else 0.0
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        sd_out = {}
        for k, v in sd.items():
            if k in _EXCLUDE:
                continue
            if v is None:
                continue
            # Skip NaN scalars (e.g. floats produced by float('nan'))
            try:
                if isinstance(v, float) and math.isnan(v):
                    continue
            except TypeError:
                pass
            # td must be a scalar for pp.make_delay()
            if k == 'td':
                sd_out[k] = _to_scalar(v)
            elif k == 'B0':
                sd_out[k] = _to_scalar(v)
            else:
                sd_out[k] = _to_pylist(v)

        return sd_out

    def _sanitize_for_clinical(self, sd: dict, cfg: dict) -> dict:
        """Map the preclinical seq_defs schema to the keys write_sequence_clinical
        expects: gamma_hz, b0, freq, offsets_ppm, b1, trec, tp, td, n_pulses,
        spoiling.  Per-measurement arrays (b1, trec, offsets) are kept; tp/td/
        n_pulses are collapsed to scalars (the clinical writer treats them so)."""
        gamma_rad = float(cfg.get('gamma', 267.5153))
        gamma_hz  = gamma_rad / (2.0 * np.pi)            # ≈ 42.5764 Hz/µT
        b0        = float(cfg.get('b0', 3.0))

        def _arr(key, default):
            v = sd.get(key, default)
            return np.atleast_1d(np.asarray(v, dtype=float))

        def _sc(key, default):
            a = np.atleast_1d(np.asarray(sd.get(key, default), dtype=float))
            return float(a[0]) if a.size else float(default)

        return {
            'gamma_hz':    gamma_hz,
            'b0':          b0,
            'freq':        b0 * gamma_hz,                # Hz per ppm
            'offsets_ppm': _arr('offsets_ppm', [0.0]),
            'b1':          _arr('B1pa', sd.get('B1', 1.0)),
            'trec':        _arr('Trec', 3.0),
            'tp':          _sc('tp', 0.1),
            'td':          _sc('td', 0.0),
            'n_pulses':    int(_sc('n_pulses', 1)),
            'spoiling':    True,
        }

    def _write_sequence(self, cfg: dict, seq_defs: dict | None):
        """
        Write the .seq pulse-sequence file.

        Priority order:
          1. sequences_sl.write_sequence_sl  (primary — preclinical
             CW spin-lock writer; handles per-measurement arrays for tp, Trec,
             excFA, SLFA, SLflag, B1pa, offsets_ppm)
          2. cest_mrf.sequence.creation.write_sequence   (last-resort fallback —
             generic pypulseq writer; expects scalar fields only)
        """
        seq_fn = cfg['seq_fn']
        sd = seq_defs if seq_defs is not None else {}

        # The .seq MUST be written by the vendored Pulseq-1.3.1 build: the bundled
        # C++ BMCSimulator only parses .seq ≤ v1.3.1, but this env's pip pypulseq
        # (1.5, dragged in by BMCTool) writes a v1.5 file the simulator misreads →
        # empty magnetisation → "index 0 is out of bounds for axis 0 with size 0".
        # We temporarily force `import pypulseq` to the vendored 1.3.1, then restore
        # the pip pypulseq 1.5 (needed elsewhere by BMCTool) — scoped, no reinstall.
        with _vendored_pypulseq():
            # 0 ─ Optional: scanner-limits writer (opt-in via the Config tab's
            #     "Scanner Limits" group). Produces a hardware-valid .seq; the
            #     limit-free simulation path below is used when it's disabled.
            lims_cfg = cfg.get('scanner_lims') if isinstance(cfg, dict) else None
            if lims_cfg:
                try:
                    import pypulseq as pp
                    lims = pp.Opts(
                        max_grad=lims_cfg['max_grad_mT_m'], grad_unit='mT/m',
                        max_slew=lims_cfg['max_slew_T_m_s'], slew_unit='T/m/s',
                        rf_dead_time=lims_cfg['rf_dead_time_us'] * 1e-6,
                        rf_ringdown_time=lims_cfg['rf_ringdown_us'] * 1e-6,
                        adc_dead_time=lims_cfg['adc_dead_time_us'] * 1e-6,
                        rf_raster_time=lims_cfg['rf_raster_us'] * 1e-6,
                        grad_raster_time=lims_cfg['grad_raster_us'] * 1e-6,
                    )
                    from sequences_sl import write_sequence_clinical
                    write_sequence_clinical(
                        seq_defs=self._sanitize_for_clinical(sd, cfg),
                        seq_fn=seq_fn, lims=lims, type='scanner')
                    self.log.emit(f"Sequence saved → {seq_fn}  (scanner-limits writer)")
                    return
                except Exception as exc:
                    self.log.emit(
                        f"WARNING: scanner-limits writer failed ({exc}) — "
                        "using the default simulation writer")

            # 1 ─ Primary: preclinical spin-lock writer
            try:
                from sequences_sl import write_sequence_sl
                sd_clean = self._sanitize_for_sl(sd)
                write_sequence_sl(seq_defs=sd_clean, seq_fn=seq_fn)
                self.log.emit(f"Sequence saved → {seq_fn}")
                return
            except ImportError:
                self.log.emit(
                    "sequences_sl not found — falling back to "
                    "cest_mrf.sequence.creation.write_sequence"
                )
            except Exception as exc:
                self.log.emit(f"WARNING: sequences_sl failed ({exc}) — using fallback")

            # 2 ─ Last-resort fallback: cest_mrf.sequence.creation.write_sequence.
            #     creation.py expects tp / td / Trec / n_pulses as plain scalars;
            #     collapse per-measurement arrays to their first element.
            from cest_mrf.sequence.creation import write_sequence
            sd_scalar = self._sanitize_for_creation(sd)
            write_sequence(seq_defs=sd_scalar, seq_fn=seq_fn)
            self.log.emit(f"Sequence saved → {seq_fn}  (fallback writer)")
