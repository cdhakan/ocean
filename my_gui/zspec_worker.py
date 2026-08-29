"""
zspec_worker.py
QThread worker for z-spectroscopy processing.

Mirrors MATLAB zSpec_load_proc.m:
    zppars.peaktype  = 'Pseudo-Voigt'   ← PV primary (Lorentzian optional)
    zppars.water1st  = false            ← single simultaneous pass
    zppars.pools     = {'water','NOE','MT','amide'}  ← 4 pools
"""

from __future__ import annotations

import threading

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class ZSpecWorker(QThread):
    """
    Background worker for the z-spectroscopy processing pipeline:
        1. (Optional) B0 correction — per-voxel Akima, matching MATLAB B0correction.m
        2. Voxelwise Pseudo-Voigt fitting (+ optional Lorentzian)
        3. MTR asymmetry map — matching MATLAB calcMTRmap.m

    Result dict keys (emitted via finished signal):
        "ppm"              — (N_off,) ppm offset array
        "pools"            — list of pool names fitted (PV / Gaussian)
        "pv_ampl_{pool}"   — (rows, cols, slices) Pseudo-Voigt amplitude map
        "pv_peak_{pool}"   — (rows, cols, slices, N_off) per-pool PV curve (1-Z)
        "pv_peak_sum"      — (rows, cols, slices, N_off) sum of all PV peaks
        "gauss_ampl_{pool}"— (rows, cols, slices) Gaussian amplitude map   [if fit_gaussian]
        "gauss_peak_{pool}"— (rows, cols, slices, N_off) per-pool Gaussian curve
        "gauss_peak_sum"   — (rows, cols, slices, N_off) sum of Gaussian peaks
        "mplf_pools"       — list of pool names from MPLF catalog fit       [if fit_mplf]
        "mplf_ampl_{pool}" — (rows, cols, slices) MPLF peak amplitude map  [if fit_mplf]
        "mplf_peak_{pool}" — (rows, cols, slices, N_off) per-pool MPLF ΔZ  [if fit_mplf]
        "lor_ampl_{pool}"  — (rows, cols, slices) Lorentzian amplitude map  [if fit_lorentzian]
        "lor_peak_{pool}"  — (rows, cols, slices, N_off) per-pool Lor curve [if fit_lorentzian]
        "lor_peak_sum"     — (rows, cols, slices, N_off) sum of Lor peaks   [if fit_lorentzian]
        "avg_zspec"        — (N_off,) mean z-spectrum over fitted voxels
        "mtr_map"          — (rows, cols, slices) MTR asymmetry map
        "mtr_ppm_used"     — float, ppm used for MTR map
    """

    log       = pyqtSignal(str)
    progress  = pyqtSignal(int)
    finished  = pyqtSignal(dict)
    error     = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        z_img: np.ndarray,
        ppm: np.ndarray,
        img_shape: tuple[int, ...],
        snr_mask: np.ndarray,
        pools: list[str],
        n_workers: int,
        b0_map_ppm: np.ndarray | None = None,
        sel_mtr_ppm: float = 3.5,
        ftol: float = 1e-6,
        max_nfev: int = 600,
        fit_lorentzian: bool = False,
        fit_gaussian: bool = True,
        fit_mplf: bool = True,
        parent=None,
    ):
        """
        Args:
            z_img          : (n_vox, N_off) z-spectra for SNR-thresholded voxels
            ppm            : (N_off,) ppm offset array
            img_shape      : (rows, cols, slices) spatial shape
            snr_mask       : bool ndarray (rows*cols*slices,) — which voxels are in z_img
            pools          : pool names to fit (e.g. ['water','NOE','MT','amine'])
            n_workers      : parallel threads for fitting
            b0_map_ppm     : optional (rows*cols*slices,) or (rows,cols,slices) B0 map in ppm
            sel_mtr_ppm    : ppm for MTR asymmetry map (default 3.5)
            ftol           : convergence tolerance (1e-6 fast, 1e-10 precise, 1e-12 MATLAB)
            max_nfev       : max function evaluations per voxel (600 fast, 2000 precise)
            fit_lorentzian : also run Lorentzian fit in parallel (doubles compute time)
            fit_gaussian   : run voxelwise Gaussian fit after PV (default True)
            fit_mplf       : run voxelwise MPLF fit after Gaussian (default True)
        """
        super().__init__(parent)
        self._z_img          = z_img
        self._ppm            = ppm
        self._img_shape      = img_shape
        self._snr_mask       = snr_mask
        self._pools          = pools
        self._n_workers      = n_workers
        self._b0_map_ppm     = b0_map_ppm
        self._sel_mtr_ppm    = sel_mtr_ppm
        self._ftol           = ftol
        self._max_nfev       = max_nfev
        self._fit_lorentzian = fit_lorentzian
        self._fit_gaussian   = fit_gaussian
        self._fit_mplf       = fit_mplf
        self._stop_event     = threading.Event()

    def stop(self):
        """Request cancellation — checked between chunks."""
        self._stop_event.set()

    def run(self):
        try:
            from my_gui.zspec_processing import (
                b0_correction, fit_all_zspec, fit_all_zspec_dual,
                fit_all_zspec_mplf, MPLF_POOL_CATALOG, calc_mtr_map
            )

            ppm           = self._ppm
            z_sel         = self._z_img.copy()
            n_vox_sel     = z_sel.shape[0]
            n_off         = len(ppm)
            rows, cols, slices = self._img_shape
            n_spatial     = rows * cols * slices
            mask_idx      = np.where(self._snr_mask.ravel())[0]

            # ── Count how many fit phases will run (for progress offset) ───
            n_phases = 1  # PV always runs
            if self._fit_gaussian:
                n_phases += 1
            if self._fit_mplf:
                n_phases += 1

            # ── 1. B0 correction (if B0 map was loaded) ────────────────────
            if self._b0_map_ppm is not None:
                self.log.emit("Applying B0 correction (Akima interpolation)...")
                b0_flat = self._b0_map_ppm.ravel()[mask_idx]
                z_sel   = b0_correction(b0_flat, ppm, z_sel)
                self.log.emit("B0 correction done.")

            # ── 2. Fitting ─────────────────────────────────────────────────
            _methods_note = "Pseudo-Voigt"
            if self._fit_gaussian: _methods_note += " + Gaussian"
            if self._fit_mplf:     _methods_note += " + MPLF (Multi-Pool Lorentzian Fitting)"
            if self._fit_lorentzian: _methods_note += " + Lorentzian"
            self.log.emit(
                f"Fitting {n_vox_sel} voxels — {_methods_note}\n"
                f"  pools: {', '.join(self._pools)}   workers: {self._n_workers}"
            )

            # Phase-aware progress: each phase occupies 1/n_phases of total bar.
            # Total emitted range = n_phases * n_vox_sel.
            def _progress_phase(n_done: int, phase_offset: int, phase_name: str):
                self.progress.emit(phase_offset + n_done)
                step = max(1, n_vox_sel // 10)
                if n_done % step == 0 or n_done == n_vox_sel:
                    pct = int(n_done * 100 / n_vox_sel) if n_vox_sel else 0
                    self.log.emit(
                        f"  [{phase_name}] {n_done}/{n_vox_sel} voxels ({pct}%)"
                    )

            _base_kw = dict(
                n_workers=self._n_workers,
                cancelled_fn=self._stop_event.is_set,
                ftol=self._ftol,
                max_nfev=self._max_nfev,
            )

            results: dict = {"ppm": ppm, "pools": self._pools}

            def _store(prefix: str, ampl: dict, indiv: dict,
                       sumv: np.ndarray | None, pool_list: list):
                for pool in pool_list:
                    if pool not in ampl:
                        continue
                    full_a = np.zeros(n_spatial)
                    full_a[mask_idx] = ampl[pool]
                    results[f"{prefix}ampl_{pool}"] = full_a.reshape(rows, cols, slices)

                    full_p = np.zeros((n_spatial, n_off))
                    full_p[mask_idx] = indiv[pool]
                    results[f"{prefix}peak_{pool}"] = full_p.reshape(rows, cols, slices, n_off)

                if sumv is not None:
                    full_s = np.zeros((n_spatial, n_off))
                    full_s[mask_idx] = sumv
                    results[f"{prefix}peak_sum"] = full_s.reshape(rows, cols, slices, n_off)

            # ── Phase 1: Pseudo-Voigt (+ optional Lorentzian) ─────────────
            phase_offset = 0
            self.log.emit("Processing 1/{}  — Pseudo-Voigt…".format(n_phases))
            _pv_kw = dict(
                pools=self._pools,
                progress_cb=lambda n: _progress_phase(n, phase_offset, "PV"),
                **_base_kw,
                xtol=self._ftol, gtol=self._ftol,
            )
            if self._fit_lorentzian:
                lor_ampl, lor_indiv, lor_sum, pv_ampl, pv_indiv, pv_sum = \
                    fit_all_zspec_dual(ppm, z_sel, **_pv_kw)
                _store("lor_", lor_ampl, lor_indiv, lor_sum, self._pools)
            else:
                pv_ampl, pv_indiv, pv_sum = fit_all_zspec(
                    ppm, z_sel, peak_type="Pseudo-Voigt", **_pv_kw)
            _store("pv_", pv_ampl, pv_indiv, pv_sum, self._pools)
            self.log.emit("  Pseudo-Voigt complete.")

            # ── Phase 2: Gaussian ─────────────────────────────────────────
            if self._fit_gaussian:
                phase_offset += n_vox_sel
                self.log.emit("Processing 2/{}  — Gaussian…".format(n_phases))
                _gauss_kw = dict(
                    pools=self._pools,
                    progress_cb=lambda n: _progress_phase(n, phase_offset, "Gaussian"),
                    **_base_kw,
                    xtol=self._ftol, gtol=self._ftol,
                )
                gauss_ampl, gauss_indiv, gauss_sum = fit_all_zspec(
                    ppm, z_sel, peak_type="Gaussian", **_gauss_kw)
                _store("gauss_", gauss_ampl, gauss_indiv, gauss_sum, self._pools)
                self.log.emit("  Gaussian complete.")

            # ── Phase 3: MPLF ─────────────────────────────────────────────
            if self._fit_mplf:
                phase_offset += n_vox_sel
                self.log.emit("Processing {}/{}  — MPLF (Multi-Pool Lorentzian Fitting)…".format(n_phases, n_phases))
                _mplf_pools = [p for p in self._pools if p in MPLF_POOL_CATALOG]
                if not _mplf_pools:
                    _mplf_pools = None   # fit_all_zspec_mplf uses default
                # n_restarts tied to quality preset:
                #   Fast (max_nfev=600)  → 1 restart (same cost as PV)
                #   Balanced (1200)      → 2 restarts
                #   Precise (3000)       → 3 restarts
                _mplf_restarts = 1 if self._max_nfev <= 600 else (
                                  2 if self._max_nfev <= 1200 else 3)
                _mplf_kw = dict(
                    pools=_mplf_pools,
                    progress_cb=lambda n: _progress_phase(n, phase_offset, "MPLF"),
                    n_workers=self._n_workers,
                    cancelled_fn=self._stop_event.is_set,
                    ftol=self._ftol,
                    max_nfev=self._max_nfev,   # same budget as PV/Gaussian
                    n_restarts=_mplf_restarts,
                )
                mplf_ampl, mplf_indiv = fit_all_zspec_mplf(ppm, z_sel, **_mplf_kw)
                _mplf_pools_used = list(mplf_ampl.keys())
                results["mplf_pools"] = _mplf_pools_used
                _store("mplf_", mplf_ampl, mplf_indiv, None, _mplf_pools_used)
                self.log.emit("  MPLF (Multi-Pool Lorentzian Fitting) complete.")

            self.log.emit("All fits complete. Building output maps…")

            # Average z-spectrum
            results["avg_zspec"] = np.mean(z_sel, axis=0)

            # ── MTR asymmetry map ──────────────────────────────────────────
            self.log.emit(f"Computing MTR asymmetry at ±{self._sel_mtr_ppm} ppm...")
            z_full = np.zeros((n_spatial, n_off))
            z_full[mask_idx] = z_sel
            z_vol  = z_full.reshape(rows, cols, slices, n_off)
            mtr_vol, sel_true = calc_mtr_map(z_vol, ppm, sel_ppm=self._sel_mtr_ppm)
            results["mtr_map"]      = mtr_vol
            results["mtr_ppm_used"] = sel_true
            self.log.emit(f"MTR asymmetry map done (ppm used: {sel_true:.2f}).")

            self.finished.emit(results)

        except InterruptedError:
            self.log.emit("Analysis cancelled by user.")
            self.cancelled.emit()
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")
