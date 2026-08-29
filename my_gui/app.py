import os
from PyQt6.QtWidgets import QMainWindow, QTabWidget, QStatusBar, QApplication
from my_gui.paths import output_path, get_output_dir
from my_gui.tabs.scan_dir_tab  import ScanDirTab
from my_gui.tabs.cest_mrf_tab  import CestMrfTab          # grouped CEST-MRF tab
from my_gui.tabs.cest_mri_tab  import CestMriTab          # grouped CEST-MRI tab
from my_gui.tabs.zspec_tab     import ZSpecTab
# HPC Cluster tab temporarily disabled — code kept in my_gui/tabs/hpc_tab.py
# for a future release. Re-enable the import + the two lines flagged "HPC" below.
# from my_gui.tabs.hpc_tab       import HpcTab
from my_gui.tabs.t1t2_tab      import T1T2Tab
from my_gui.tabs.inv_zspec_tab import InvZSpecTab
from my_gui.tabs.quesp_tab     import QUESPTab
from my_gui.worker             import DictWorker
from my_gui.tabs.roi_tab       import ROITab
from my_gui.tabs.equations_tab import EquationsTab
from my_gui.roi_manager        import get_roi_manager


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCEAN")

        # ── Fit window to screen ───────────────────────────────────────────────
        # availableGeometry() excludes the macOS menu bar, Dock, and any notch
        # inset, so the window always opens fully visible regardless of display.
        screen = QApplication.primaryScreen()
        ag = screen.availableGeometry()
        # Width: 95 % of screen, capped at 1 300 px — leaves side breathing room.
        # Height: full available height (top of screen → bottom of Dock/taskbar).
        win_w = min(1300, int(ag.width() * 0.95))
        win_h = ag.height()          # full top-to-bottom of the usable desktop
        self.resize(win_w, win_h)
        # Horizontally centred; pinned to the very top of the available area.
        self.move(
            ag.x() + (ag.width() - win_w) // 2,
            ag.y(),
        )
        self.setMinimumSize(900, 560)   # hard lower bound so nothing clips

        self._worker = None

        # ── CEST MRF grouped tab (Config + Sequence + Simulation + Results) ──
        self.cest_mrf_tab  = CestMrfTab(on_generate_clicked=self._on_generate)
        # Expose inner tabs as direct attributes so all existing wiring works
        self.config_tab    = self.cest_mrf_tab.config_tab
        self.seq_tab       = self.cest_mrf_tab.seq_tab
        self.dict_tab      = self.cest_mrf_tab.dict_tab
        self.results_tab   = self.cest_mrf_tab.results_tab

        # ── CEST MRI grouped tab (Load CEST Datas + Inverse Z + QUESP) ─────────
        self.cest_mri_tab  = CestMriTab()
        # Expose inner tabs as direct attributes so all existing wiring works
        self.zspec_tab     = self.cest_mri_tab.zspec_tab
        self.inv_zspec_tab = self.cest_mri_tab.inv_zspec_tab
        self.quesp_tab     = self.cest_mri_tab.quesp_tab

        # ── Standalone tabs ───────────────────────────────────────────────────
        self.scan_dir_tab  = ScanDirTab()
        self.t1t2_tab      = T1T2Tab()
        # self.hpc_tab       = HpcTab()   # HPC Cluster tab disabled for now (see import note)
        self.roi_tab       = ROITab()
        self.equations_tab = EquationsTab()

        self.tabs = tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        tabs.setStyleSheet("""
            QTabBar::tab {
                padding: 6px 16px;
                font-size: 12px;
                min-width: 110px;
            }
            QTabBar::tab:selected { font-weight: bold; }
        """)
        tabs.addTab(self.scan_dir_tab,  "Scan Directory")
        tabs.addTab(self.roi_tab,       "ROI Manager")
        tabs.addTab(self.t1t2_tab,      "T1 / T2 / B1 / WASABI")
        tabs.addTab(self.cest_mrf_tab,  "CEST MRF")
        tabs.addTab(self.cest_mri_tab,  "CEST MRI")
        # tabs.addTab(self.hpc_tab,       "HPC Cluster")   # HPC Cluster tab disabled for now
        tabs.addTab(self.equations_tab, "Educational")
        self.setCentralWidget(tabs)
        self.setStatusBar(QStatusBar())

        # ── Wire cross-tab callbacks ───────────────────────────────────────
        # "Load quant_maps.mat" in DictTab → push straight to ResultsTab
        self.dict_tab.on_quant_maps_loaded = self.results_tab.set_quant_maps_external

        # Acquired data set in DictTab (any path: scan dir, convert, manual) → MRF Viewer
        self.dict_tab.on_acquired_data_loaded = self.results_tab._try_load_acquired

        # ScanDirTab → push assigned paths to relevant tabs automatically
        self.scan_dir_tab.scan_assigned.connect(self._on_scan_assigned)
        # Let every map tab pull the assigned-scan paths for the "ROIs + Bkg"
        # background-image picker.
        for _mt in (self.results_tab, self.zspec_tab, self.inv_zspec_tab,
                    self.quesp_tab, self.t1t2_tab):
            if hasattr(_mt, "set_scan_paths_getter"):
                _mt.set_scan_paths_getter(self.scan_dir_tab.get_scan_paths)

        # T1T2 tab T1 fit result → push T1 map to QUESP tab + 1/Z tab
        _orig_t1_done = self.t1t2_tab._on_t1_done
        def _t1_done_hook(t1_map):
            _orig_t1_done(t1_map)
            self.quesp_tab.set_t1_map(t1_map)
            # Auto-push mean R1 (s⁻¹) to the 1/Z tab (T1 map is in ms)
            import numpy as np
            valid = t1_map[t1_map > 0]
            if valid.size > 0:
                mean_r1 = 1000.0 / float(np.median(valid))   # ms→s⁻¹
                self.inv_zspec_tab.set_r1_from_map(mean_r1)
        self.t1t2_tab._on_t1_done = _t1_done_hook

        # Wire "From T1 map" button in 1/Z tab → read current T1 map on demand
        def _r1_from_t1_getter():
            t1_map = self.t1t2_tab._t1_map
            if t1_map is None:
                return None
            import numpy as np
            valid = t1_map[t1_map > 0]
            return (1000.0 / float(np.median(valid))) if valid.size > 0 else None
        self.inv_zspec_tab.set_r1_getter(_r1_from_t1_getter)
        # Also expose the getter on zspec_tab so the 1/Z sub-tab inside ROI Spectra
        # can use it without holding a direct reference to inv_zspec_tab
        self.zspec_tab._inv_r1_getter = _r1_from_t1_getter

        # Feed the 1/Z tab the main CEST MRI tab's pool selection so it
        # automatically fits whatever pools the user picked there.
        self.inv_zspec_tab.set_main_pools_getter(
            lambda: self.zspec_tab._global_pools
        )

        # ── ROI Manager — broadcast ROIs to all display tabs ──────────────
        _roi_mgr = get_roi_manager()

        # Wire ROI tab canvas changes → manager
        self.roi_tab.canvas.roi_added.connect(
            lambda _: self.roi_tab._push_to_manager()
        )

        # Deep skull-strip: hand the ROI panel the T1 tab's 3-D volume + slice so
        # "Detect Brain Outline" can use the U-Net (falls back to Otsu otherwise).
        def _brain_volume():
            import numpy as _np
            # 1) Prefer the ROI Manager's loaded reference volume (a 3-D T1 the
            #    user loaded there).  It is cleared on any 2-D load, so it is
            #    always the live volume — regardless of the current view plane.
            rv = getattr(self.roi_tab, "_ref_vol", None)
            if rv is not None:
                v = _np.asarray(rv)
                if v.ndim == 3 and min(v.shape) >= 2:
                    return (v, int(getattr(self.roi_tab, "_ref_slice", v.shape[2] // 2)))
            # 2) Fall back to the T1/T2 tab's 3-D volume.
            t1 = getattr(self.t1t2_tab, "_t1_img", None)
            if t1 is not None:
                v = _np.asarray(t1)
                if v.ndim == 4:             # (Y, X, slices, nTR) → most T1-weighted frame
                    v = v[:, :, :, -1]
                if v.ndim == 3 and min(v.shape) >= 2:
                    return (v, int(getattr(self.t1t2_tab, "_slice_idx", v.shape[2] // 2)))
            return None
        if hasattr(self.roi_tab, "roi_panel") and \
                hasattr(self.roi_tab.roi_panel, "set_brain_volume_getter"):
            self.roi_tab.roi_panel.set_brain_volume_getter(_brain_volume)

        # Wire manager → all display tab canvases
        if hasattr(self.results_tab, 'connect_roi_manager'):
            self.results_tab.connect_roi_manager(_roi_mgr)
        elif hasattr(self.results_tab, 'canvas'):
            _roi_mgr.connect_canvas(self.results_tab.canvas)

        # Auto-navigate to MRF Results when ROIs change and MRF map is loaded —
        # but NEVER hijack the user away from the ROI Manager tab while they are
        # actively drawing / editing ROIs there.
        _roi_tab_idx = self.tabs.indexOf(self.roi_tab)
        def _auto_switch_mrf_results(rois):
            # Skip if the user is currently on the ROI Manager tab
            if self.tabs.currentIndex() == _roi_tab_idx:
                return
            if (rois
                    and hasattr(self.results_tab, '_dc_img_data')
                    and self.results_tab._dc_img_data is not None):
                self.tabs.setCurrentIndex(3)                    # outer: CEST MRF
                self.cest_mrf_tab.sub_tabs.setCurrentIndex(2)  # inner: MRF Viewer
        _roi_mgr.rois_changed.connect(_auto_switch_mrf_results)
        if hasattr(self.zspec_tab, 'connect_roi_manager'):
            self.zspec_tab.connect_roi_manager(_roi_mgr)
        if hasattr(self.inv_zspec_tab, 'connect_roi_manager'):
            self.inv_zspec_tab.connect_roi_manager(_roi_mgr)
        if hasattr(self.quesp_tab, 'connect_roi_manager'):
            self.quesp_tab.connect_roi_manager(_roi_mgr)
        if hasattr(self.t1t2_tab, 'connect_roi_manager'):
            self.t1t2_tab.connect_roi_manager(_roi_mgr)

        # ── ROI tab image sources — register one getter per display tab ────
        # Each getter returns (np.ndarray, title_str) or None if no image loaded.

        def _src_zspec():
            c = self.zspec_tab.canvas
            if hasattr(c, '_img_data') and c._img_data is not None:
                return c._img_data, "M0 (Z-Spec)"
            if hasattr(self.zspec_tab, '_M0_img') and self.zspec_tab._M0_img is not None:
                return self.zspec_tab._M0_img[:, :, 0], "M0 (Z-Spec)"
            return None

        def _src_mrf():
            c = self.results_tab.canvas
            if hasattr(c, '_img_data') and c._img_data is not None:
                return c._img_data, "MRF Results"
            return None

        def _src_t1():
            # Always return the raw acquired T1 images (last TR frame),
            # even after the T1 map has been fitted.  The fitted map is a
            # derived product and should not appear as the ROI reference.
            t = self.t1t2_tab
            if hasattr(t, '_t1_img') and t._t1_img is not None:
                sl = getattr(t, '_slice_idx', 0)
                # _t1_img shape: (Y, X, n_slices, nTR) → last TR, current slice
                img = t._t1_img[:, :, sl, -1]
                return img, "T1 image (last TR)"
            return None

        # Mutable container so the TE-setter closure can update the index
        _t2_te_idx = [0]

        def _src_t2():
            # Return the raw acquired T2 image at the user-selected TE (default: first TE).
            t = self.t1t2_tab
            if hasattr(t, '_t2_img') and t._t2_img is not None:
                sl  = getattr(t, '_slice_idx', 0)
                nTE = t._t2_img.shape[-1]
                te  = min(_t2_te_idx[0], nTE - 1)
                # _t2_img shape: (Y, X, n_slices, nTE)
                img = t._t2_img[:, :, sl, te]
                return img, f"T2 image (TE {te + 1}/{nTE})"
            return None

        def _n_t2_te():
            t = self.t1t2_tab
            if hasattr(t, '_t2_img') and t._t2_img is not None:
                return t._t2_img.shape[-1]
            return 1

        def _set_t2_te(idx: int):
            _t2_te_idx[0] = max(0, idx)

        def _src_b1():
            t = self.t1t2_tab
            if hasattr(t, '_b1_map') and t._b1_map is not None:
                return t._b1_map, "B1 Map (%)"
            return None

        def _src_b1_fa1():
            t = self.t1t2_tab
            if hasattr(t, '_b1_fa1_img') and t._b1_fa1_img is not None:
                img = t._b1_fa1_img
                while img.ndim > 2:
                    img = img[..., 0]
                return img, "B1 FA1 Image"
            return None

        def _src_b1_fa2():
            t = self.t1t2_tab
            if hasattr(t, '_b1_fa2_img') and t._b1_fa2_img is not None:
                img = t._b1_fa2_img
                while img.ndim > 2:
                    img = img[..., 0]
                return img, "B1 FA2 Image"
            return None

        def _src_quesp():
            q = self.quesp_tab
            if hasattr(q, '_m0_img') and q._m0_img is not None:
                return q._m0_img, "M0 (QUESP)"
            if hasattr(q, 'canvas') and hasattr(q.canvas, '_img_data') and q.canvas._img_data is not None:
                return q.canvas._img_data, "QUESP display"
            return None

        def _src_inv_zspec():
            c = self.inv_zspec_tab.canvas
            if hasattr(c, '_img_data') and c._img_data is not None:
                return c._img_data, "Inv Z-Spec"
            return None

        def _src_t1_map():
            t = self.t1t2_tab
            if hasattr(t, '_t1_map') and t._t1_map is not None:
                return t._t1_map, "T1 map (ms)"
            return None

        def _src_t2_map():
            t = self.t1t2_tab
            if hasattr(t, '_t2_map') and t._t2_map is not None:
                return t._t2_map, "T2 map (ms)"
            return None

        self.roi_tab.register_image_source("M0 / Unsaturated CEST Image", _src_zspec)
        self.roi_tab.register_image_source("T1 Images",                  _src_t1)
        self.roi_tab.register_image_source("T2 Images",                  _src_t2)
        self.roi_tab.register_image_source("α₁ Images (B1 FA1)",         _src_b1_fa1)
        self.roi_tab.register_image_source("α₂ Images (B1 FA2)",         _src_b1_fa2)

        # Let the T1/T2 tab pull the CEST MRI tab's MTR-asymmetry map for overlays
        if hasattr(self.t1t2_tab, "set_mtr_source"):
            self.t1t2_tab.set_mtr_source(
                self.zspec_tab.get_mtr_asym_map,
                self.zspec_tab.get_mtr_asym_ppm,
            )
        if hasattr(self.t1t2_tab, "set_cest_image_source"):
            self.t1t2_tab.set_cest_image_source(
                self.zspec_tab.get_current_cest_image
            )

        # Register TE selection support for T2 Images
        self.roi_tab.register_te_source("T2 Images", _n_t2_te, _set_t2_te)

    def _on_generate(self):
        cfg = self.config_tab.get_config()

        # ── Always Mode B: Config tab + Sequence tab drive pool params & schedule ──
        # acquired_data.mat (if present) is used ONLY for dot-product matching,
        # never to load dictpars/seq_defs.  If absent, only the dictionary is built.
        seq_defs = self.seq_tab.get_seq_defs(b0=cfg['b0'])
        seq_defs['B0']                = cfg['b0']
        seq_defs['max_pulse_samples'] = cfg.get('max_pulse_samples', 100)
        seq_defs['seq_id_string']     = 'seq'

        # Pre-write the scenario YAML (pool parameters — no arrays involved)
        try:
            from cest_mrf.write_scenario import write_yaml_dict
            get_output_dir()   # ensure output dir exists before writing
            write_yaml_dict(cfg)
            self.dict_tab.append_log("Configuration saved")
        except Exception as e:
            self.dict_tab.on_error(f"Failed to write scenario YAML: {e}")
            return

        # acquired_data.mat path for matching step only
        acqdata_fn = self.dict_tab.get_acquired_data_path()
        if acqdata_fn and os.path.isfile(acqdata_fn):
            self.dict_tab.append_log("Acquired data ready — matching will run after dictionary generation")
            # Propagate to Results tab so it has the data path ready and images pre-loaded
            from pathlib import Path
            self.results_tab._data_fn = acqdata_fn
            self.results_tab.lbl_data.setText(Path(acqdata_fn).name)
            self.results_tab.lbl_data.setStyleSheet("font-size: 11px; color: green;")
            # Pre-load acquired images into the Results tab immediately
            self.results_tab._try_load_acquired(acqdata_fn)
        else:
            self.dict_tab.append_log(
                "No acquired data — dictionary only (matching skipped)"
            )
            acqdata_fn = None

        self._worker = DictWorker(
            acquired_data_fn  = acqdata_fn,   # None → skip matching
            seq_fn            = cfg['seq_fn'],
            param_fn          = cfg['yaml_fn'],
            dict_fn           = cfg.get('dict_fn', output_path('dict.mat')),
            num_workers       = cfg.get('num_workers', 8),
            max_pulse_samples = cfg.get('max_pulse_samples', 100),
            full_cfg          = cfg,          # pool config from Config tab
            seq_defs          = seq_defs,     # per-measurement arrays from Sequence tab
            quantmaps_fn      = cfg.get('quantmaps_fn', output_path('quant_maps.mat')),
            yaml_fn           = cfg['yaml_fn'],
        )

        self._worker.log.connect(self.dict_tab.append_log)
        self._worker.finished.connect(self._on_dict_done)
        self._worker.cancelled.connect(self._on_dict_cancelled)
        self._worker.error.connect(self.dict_tab.on_error)
        self.dict_tab.set_cancel_callback(self._worker.stop)
        self.dict_tab.set_running(True)
        self._worker.start()

    def _on_dict_done(self, quant_maps: dict):
        """
        Called when DictWorker.finished fires.

        The worker now emits the quant_maps dict directly.
        If matching was skipped (no acquired_data.mat available at matching step)
        the worker emits {'_dict_only': True}.
        """
        self.dict_tab.set_running(False)
        cfg      = self.config_tab.get_config()
        dict_fn  = cfg.get('dict_fn', output_path('dict.mat'))

        if quant_maps.get('_dict_only'):
            # Dictionary generated but matching not run (no acquired_data)
            self.dict_tab.append_log("Dictionary generation complete")
            if os.path.isfile(dict_fn):
                self.results_tab.set_dict_fn(dict_fn)

            # Also propagate acquired data path + dict path so the user can
            # click "Run Dot-Product Matching" from the Results tab directly.
            acq_path = self.dict_tab.get_acquired_data_path()
            if acq_path and os.path.isfile(acq_path):
                self.results_tab._data_fn = acq_path
                from pathlib import Path
                self.results_tab.lbl_data.setText(Path(acq_path).name)
                self.results_tab.lbl_data.setStyleSheet("font-size: 11px; color: green;")
        else:
            # Full pipeline completed — push maps directly to Results tab
            self.dict_tab.append_log(
                f"Pipeline complete — maps: {', '.join(quant_maps.keys())}"
            )
            self.results_tab.set_quant_maps_external(quant_maps)
            if os.path.isfile(dict_fn):
                self.results_tab.set_dict_fn(dict_fn)

            # Auto-switch to MRF Viewer sub-tab to show the parametric maps
            try:
                self.tabs.setCurrentIndex(self.tabs.indexOf(self.cest_mrf_tab))
                self.cest_mrf_tab.show_results()
            except Exception:
                pass

        self.statusBar().showMessage("Dictionary generation complete.", 5000)

    def _on_scan_assigned(self, key: str, path: str):
        """
        Called when ScanDirTab assigns a scan path to a modality.
        Push paths into the relevant tabs and auto-trigger loading where possible.
        """
        if not path:
            return
        pv360 = self.scan_dir_tab.is_pv360()

        vendor = self.scan_dir_tab.get_vendor()

        if key == "cest":
            if vendor == "mr_solutions" and hasattr(self.zspec_tab, 'edit_cest_mrd_path'):
                self.zspec_tab.combo_vendor_cest.setCurrentText("MR Solutions (.MRD)")
                self.zspec_tab._cest_mrd_path = path
                self.zspec_tab.edit_cest_mrd_path.setText(path)
                self.zspec_tab.spin_cest_mrd_mhz.setValue(
                    self.scan_dir_tab.get_mrd_larmor_mhz())
                self.zspec_tab._load_cest()   # auto-load immediately
            elif vendor == "ge_siemens" and hasattr(self.zspec_tab, 'edit_cest_data'):
                self.zspec_tab.combo_vendor_cest.setCurrentText("GE/Siemens")
                self.zspec_tab._cest_data_path = path
                self.zspec_tab.edit_cest_data.setText(path)
                self.zspec_tab._load_cest()   # auto-load immediately
            elif hasattr(self.zspec_tab, 'edit_cest_dir'):
                self.zspec_tab.combo_vendor_cest.setCurrentText("Bruker")
                self.zspec_tab._cest_dir = path
                self.zspec_tab.edit_cest_dir.setText(path)
                self.zspec_tab.combo_pv_cest.setCurrentText("PV360" if pv360 else "PV6 / PV7")
                self.zspec_tab._load_cest()   # auto-load immediately

        elif key == "wassr":
            if vendor == "mr_solutions" and hasattr(self.zspec_tab, 'edit_wassr_mrd_path'):
                self.zspec_tab.combo_vendor_wassr.setCurrentText("MR Solutions (.MRD)")
                self.zspec_tab._wassr_mrd_path = path
                self.zspec_tab.edit_wassr_mrd_path.setText(path)
                self.zspec_tab.spin_wassr_mrd_mhz.setValue(
                    self.scan_dir_tab.get_mrd_larmor_mhz())
                self.zspec_tab._load_wassr()   # auto-load immediately
            elif vendor == "ge_siemens" and hasattr(self.zspec_tab, 'edit_wassr_data'):
                self.zspec_tab.combo_vendor_wassr.setCurrentText("GE/Siemens")
                self.zspec_tab._wassr_data_path = path
                self.zspec_tab.edit_wassr_data.setText(path)
                self.zspec_tab._load_wassr()   # auto-load immediately
            elif hasattr(self.zspec_tab, 'edit_wassr_dir'):
                self.zspec_tab.combo_vendor_wassr.setCurrentText("Bruker")
                self.zspec_tab._wassr_dir = path
                self.zspec_tab.edit_wassr_dir.setText(path)
                self.zspec_tab.combo_pv_wassr.setCurrentText("PV360" if pv360 else "PV6 / PV7")
                self.zspec_tab._load_wassr()   # auto-load immediately

        elif key == "mrf":
            # Update DictTab with the assigned MRF scan path
            if hasattr(self.dict_tab, 'set_mrf_scan_path'):
                self.dict_tab.set_mrf_scan_path(path)
            # Also push any pre-existing acquired_data.mat to the MRF Viewer immediately
            import os as _os
            _acqmat = _os.path.join(path, "acquired_data.mat")
            if _os.path.isfile(_acqmat):
                self.results_tab._data_fn = _acqmat
                self.results_tab._try_load_acquired(_acqmat)

        elif key == "quesp":
            # Auto-load QUESP data
            if hasattr(self.quesp_tab, 'load_from_path'):
                self.quesp_tab.load_from_path(path, pv360=pv360)

        elif key == "t1":
            if hasattr(self.t1t2_tab, 'le_t1_dir'):
                self.t1t2_tab._t1_dir = path
                self.t1t2_tab.le_t1_dir.setText(path)
                # Set scanner format before loading so the correct reader is used:
                # index 0 = Bruker (2dseq), 1 = GE/Siemens DICOM, 2 = GE/Siemens NIfTI
                vendor = self.scan_dir_tab.get_vendor()
                if vendor == 'ge_siemens' and hasattr(self.t1t2_tab, 'combo_t1_fmt'):
                    gs_fmt = self.scan_dir_tab.get_gs_format()
                    fmt_idx = 1 if gs_fmt == 'dicom' else 2
                    self.t1t2_tab.combo_t1_fmt.setCurrentIndex(fmt_idx)
                elif hasattr(self.t1t2_tab, 'combo_t1_fmt'):
                    self.t1t2_tab.combo_t1_fmt.setCurrentIndex(0)
                    self.t1t2_tab.combo_t1_pv.setCurrentText("PV360" if pv360 else "PV6 / PV7")
                else:
                    self.t1t2_tab.combo_t1_pv.setCurrentText("PV360" if pv360 else "PV6 / PV7")
                self.t1t2_tab._load_t1()

        elif key == "t2":
            if hasattr(self.t1t2_tab, 'le_t2_dir'):
                self.t1t2_tab._t2_dir = path
                self.t1t2_tab.le_t2_dir.setText(path)
                vendor = self.scan_dir_tab.get_vendor()
                if vendor == 'ge_siemens' and hasattr(self.t1t2_tab, 'combo_t2_fmt'):
                    gs_fmt = self.scan_dir_tab.get_gs_format()
                    fmt_idx = 1 if gs_fmt == 'dicom' else 2
                    self.t1t2_tab.combo_t2_fmt.setCurrentIndex(fmt_idx)
                elif hasattr(self.t1t2_tab, 'combo_t2_fmt'):
                    self.t1t2_tab.combo_t2_fmt.setCurrentIndex(0)
                    self.t1t2_tab.combo_t2_pv.setCurrentText("PV360" if pv360 else "PV6 / PV7")
                else:
                    self.t1t2_tab.combo_t2_pv.setCurrentText("PV360" if pv360 else "PV6 / PV7")
                self.t1t2_tab._load_t2()

        elif key == "b1_fa1":
            if hasattr(self.t1t2_tab, 'le_b1_fa1'):
                self.t1t2_tab._b1_fa1_dir = path
                self.t1t2_tab.le_b1_fa1.setText(path)

        elif key == "b1_fa2":
            if hasattr(self.t1t2_tab, 'le_b1_fa2'):
                self.t1t2_tab._b1_fa2_dir = path
                self.t1t2_tab.le_b1_fa2.setText(path)

        self.statusBar().showMessage(
            f"Scan Directory: {key.upper()} → {path}", 4000
        )

    def _on_dict_cancelled(self):
        self.dict_tab.set_running(False)
        self.dict_tab.lbl_status.setText("Cancelled.")
        self.dict_tab.append_log("Dictionary generation cancelled.")
        self.statusBar().showMessage("Cancelled.", 4000)