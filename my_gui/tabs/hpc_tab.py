"""
hpc_tab.py
HPC Cluster tab — configure remote SLURM cluster, upload files, submit
jobs via srun.bash → run.bash, and fetch results.

Default SLURM parameters are pre-filled from the provided job scripts:
  srun.bash  — SBATCH header + calls run.bash
  run.bash   — activates conda env, runs MRFmatch_mt.py, reports results

⚠  Neural-network matching integration is a placeholder for future work.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QCheckBox, QFileDialog,
    QScrollArea, QFrame, QSplitter, QTabWidget,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


_GRP_STYLE = (
    "QGroupBox {"
    "  border: 1px solid #3a3a3a;"
    "  border-radius: 5px;"
    "  margin-top: 6px;"
    "  padding: 22px 4px 4px 4px;"
    "  font-weight: bold;"
    "}"
    "QGroupBox::title {"
    "  subcontrol-origin: padding;"
    "  subcontrol-position: top left;"
    "  left: 8px;"
    "  top: 4px;"
    "  padding: 0 4px;"
    "  font-size: 14px;"
    "  color: #c9d1d9;"
    "}"
)


def _mono_font(size: int) -> QFont:
    """
    Return a monospace QFont using the first available family so Qt never
    has to resolve an alias (which triggers the 'Populating font family
    aliases took N ms' warning when 'Courier' is requested but absent).

    Priority: Menlo (macOS) → Consolas (Windows) → DejaVu Sans Mono
    (Linux) → Courier New (last resort, may still alias on some hosts).
    """
    from PyQt6.QtGui import QFontDatabase
    families = set(QFontDatabase.families())
    for candidate in ("Menlo", "Consolas", "DejaVu Sans Mono",
                      "Liberation Mono", "Courier New"):
        if candidate in families:
            return QFont(candidate, size)
    f = QFont()
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPointSize(size)
    return f


# ── Default values from the provided job scripts ──────────────────────────────
_DEFAULTS = {
    "account":   "cestmrf",
    "partition": "dgx-a100,rtx6000",
    "nodes":     1,
    "cpus":      18,
    "gpus":      3,
    "mem_gb":    100,
    "walltime":  "0-02:00:00",
    "job_name":  "MRF_dict",
    "conda_env": "cbdmrf",
    "venv":      "~/cbdmrfpy/bin/activate",
    "remote_code_dir": "~/molecular-mrf-main/molecular-mrf-main",
    "python_script":   "MRFmatch_mt.py",
}


# ─────────────────────────────────────────────────────────────────────────────

class HpcTab(QWidget):
    """
    HPC Cluster tab — SSH connection, SLURM job config, bash-script runner,
    and (future) neural-network matching mode.
    """

    def __init__(self):
        super().__init__()
        self._job_id: str     = ""
        self._ssh_client      = None   # open paramiko.SSHClient (if connected)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer = QHBoxLayout(self)
        outer.addWidget(splitter)

        # ── Left: settings panel (scrollable) ────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(390)
        scroll.setMaximumWidth(500)
        # Always show the vertical scrollbar so users can scroll on macOS
        # (macOS overlay scrollbars are invisible until hover — users miss them)
        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical {"
            "  background: #1e1e1e; width: 8px;"
            "  border-radius: 4px; margin: 0px;"
            "}"
            "QScrollBar::handle:vertical {"
            "  background: #555; border-radius: 4px; min-height: 24px;"
            "}"
            "QScrollBar::handle:vertical:hover { background: #888; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical"
            " { height: 0px; }"
        )

        inner = QWidget()
        left_layout = QVBoxLayout(inner)
        left_layout.setSpacing(5)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.addWidget(self._build_connection_group())
        left_layout.addWidget(self._build_job_group())
        left_layout.addWidget(self._build_scripts_group())
        left_layout.addWidget(self._build_matching_group())
        left_layout.addWidget(self._build_actions_group())
        left_layout.addStretch()
        scroll.setWidget(inner)
        splitter.addWidget(scroll)

        # ── Right: script preview + log tabs ─────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(4)

        tabs = QTabWidget()

        # Log tab
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(4, 4, 4, 4)
        status_row = QHBoxLayout()
        self.lbl_job_status = QLabel("No job submitted.")
        self.lbl_job_status.setStyleSheet("font-weight: bold;")
        status_row.addWidget(self.lbl_job_status)
        status_row.addStretch()
        self.btn_refresh = QPushButton("Refresh Status")
        self.btn_refresh.clicked.connect(self._refresh_status)
        status_row.addWidget(self.btn_refresh)
        log_layout.addLayout(status_row)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(_mono_font(10))
        log_layout.addWidget(self.log)
        tabs.addTab(log_widget, "Job Log")

        # Hidden widgets — kept so _generate_srun_bash() / _generate_run_bash()
        # callers still work; they are never added to the tab bar.
        self.txt_srun_preview = QTextEdit()
        self.txt_run_preview  = QTextEdit()

        # ── Terminal tab (local shell via qtpyTerminal) ───────────────────
        _local_w  = QWidget()
        _local_vl = QVBoxLayout(_local_w)
        _local_vl.setContentsMargins(0, 0, 0, 0)
        _local_vl.setSpacing(0)

        self._local_term = None   # set below if import succeeds

        try:
            import os as _os
            _os.environ.setdefault("QT_API", "pyqt6")
            from qtpyTerminal import qtpyTerminal as _QtpyTerm  # noqa

            # ── toolbar ───────────────────────────────────────────────────
            _lt_bar = QHBoxLayout()
            _lt_bar.setContentsMargins(6, 4, 6, 3)
            _lt_bar.setSpacing(6)

            _shell_path = _os.environ.get("SHELL", "/bin/bash")

            self._btn_local_start = QPushButton("▶  Start Shell")
            self._btn_local_start.setFixedHeight(26)
            self._btn_local_start.setToolTip(
                f"Launch an interactive local shell: {_shell_path}"
            )
            self._btn_local_start.setStyleSheet(
                "QPushButton{background:#238636;color:white;border:none;"
                "border-radius:4px;padding:0 10px;font-weight:bold;}"
                "QPushButton:hover{background:#2ea043;}"
                "QPushButton:disabled{background:#333;color:#666;}"
            )
            _lt_bar.addWidget(self._btn_local_start)

            self._btn_local_stop = QPushButton("■  Stop")
            self._btn_local_stop.setFixedHeight(26)
            self._btn_local_stop.setEnabled(False)
            self._btn_local_stop.setToolTip("Terminate the local shell session")
            self._btn_local_stop.setStyleSheet(
                "QPushButton{background:#6e2828;color:white;border:none;"
                "border-radius:4px;padding:0 10px;}"
                "QPushButton:hover{background:#8b3030;}"
                "QPushButton:disabled{background:#333;color:#666;}"
            )
            _lt_bar.addWidget(self._btn_local_stop)

            _lbl_shell = QLabel(f"Shell: {_shell_path}")
            _lbl_shell.setStyleSheet(
                "color:#8b949e; font-size:11px; padding-left:4px;"
            )
            _lt_bar.addWidget(_lbl_shell, stretch=1)

            _lt_bar_w = QWidget()
            _lt_bar_w.setStyleSheet(
                "background:#161b22; border-bottom:1px solid #30363d;"
            )
            _lt_bar_w.setLayout(_lt_bar)
            _local_vl.addWidget(_lt_bar_w)

            # ── qtpyTerminal widget ───────────────────────────────────────
            self._local_term = _QtpyTerm(_local_w)
            try:
                from PyQt6.QtGui import QColor as _QC
                self._local_term.set_bgcolor(_QC("#0d1117"))
                self._local_term.set_fgcolor(_QC("#c9d1d9"))
            except Exception:
                pass   # colour API optional — falls back to system theme
            _local_vl.addWidget(self._local_term, stretch=1)

            # ── wire buttons ──────────────────────────────────────────────
            def _local_start():
                self._local_term.start()
                self._btn_local_start.setEnabled(False)
                self._btn_local_stop.setEnabled(True)

            def _local_stop():
                self._local_term.stop()
                self._btn_local_start.setEnabled(True)
                self._btn_local_stop.setEnabled(False)

            self._btn_local_start.clicked.connect(_local_start)
            self._btn_local_stop.clicked.connect(_local_stop)

        except ImportError:
            # ── not installed — show instructions ─────────────────────────
            _pip_cmd = (
                'pip install '
                '"qtpyTerminal@git+https://github.com/mguijarr/qtpyTerminal.git" '
                'pyte qtpy'
            )
            _lbl_info = QLabel(
                "<b style='font-size:14px;'>qtpyTerminal not installed</b><br><br>"
                "This tab provides a real interactive local shell (bash/zsh) "
                "embedded directly in the GUI, powered by "
                "<a href='https://github.com/mguijarr/qtpyTerminal' "
                "style='color:#58a6ff;'>qtpyTerminal</a>.<br><br>"
                "Install with:<br><br>"
                f"<code style='background:#161b22;padding:6px 10px;"
                f"border-radius:4px;display:inline-block;'>{_pip_cmd}</code>"
                "<br><br>Then restart OCEAN."
            )
            _lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            _lbl_info.setTextFormat(Qt.TextFormat.RichText)
            _lbl_info.setOpenExternalLinks(True)
            _lbl_info.setWordWrap(True)
            _lbl_info.setStyleSheet(
                "color:#c9d1d9; font-family:Menlo, Consolas, 'DejaVu Sans Mono', 'Courier New'; "
                "font-size:12px; padding:30px;"
            )

            from PyQt6.QtWidgets import QApplication as _QApp
            _btn_copy = QPushButton("📋  Copy install command")
            _btn_copy.setFixedWidth(240)
            _btn_copy.setFixedHeight(30)
            _btn_copy.setStyleSheet(
                "QPushButton{background:#21262d;color:#c9d1d9;border:1px solid #30363d;"
                "border-radius:5px;font-size:12px;}"
                "QPushButton:hover{background:#30363d;color:white;}"
            )
            _btn_copy.setToolTip(_pip_cmd)
            _btn_copy.clicked.connect(
                lambda: _QApp.clipboard().setText(_pip_cmd)
            )

            _cent = QWidget()
            _cent_vl = QVBoxLayout(_cent)
            _cent_vl.addStretch()
            _cent_vl.addWidget(
                _lbl_info, alignment=Qt.AlignmentFlag.AlignHCenter
            )
            _cent_vl.addWidget(
                _btn_copy, alignment=Qt.AlignmentFlag.AlignHCenter
            )
            _cent_vl.addStretch()
            _local_vl.addWidget(_cent)

        tabs.addTab(_local_w, "Terminal")

        right_layout.addWidget(tabs, stretch=1)
        splitter.addWidget(right)
        splitter.setSizes([440, 660])

        # Initial preview render
        self._refresh_previews()
        self._log("HPC tab ready. Configure connection settings and submit a job.")

    # ─────────────────────────────────────────────────────────────────────
    # Group builders
    # ─────────────────────────────────────────────────────────────────────

    def _build_connection_group(self) -> QGroupBox:
        grp = QGroupBox("Cluster Connection (SSH)")
        grp.setStyleSheet(_GRP_STYLE)
        form = QFormLayout(grp)
        form.setContentsMargins(6, 4, 6, 6)
        form.setVerticalSpacing(3)
        form.setHorizontalSpacing(8)

        self.edit_host = QLineEdit()
        self.edit_host.setPlaceholderText("e.g. hpc.university.edu")

        self.spin_port = QSpinBox()
        self.spin_port.setRange(1, 65535)
        self.spin_port.setValue(22)

        self.edit_user = QLineEdit()
        self.edit_user.setPlaceholderText("username")

        key_row = QHBoxLayout()
        self.edit_ssh_key = QLineEdit()
        self.edit_ssh_key.setPlaceholderText("~/.ssh/id_rsa  (leave blank for password auth)")
        key_row.addWidget(self.edit_ssh_key, stretch=1)
        btn_key = QPushButton("Browse…")
        btn_key.clicked.connect(self._browse_ssh_key)
        key_row.addWidget(btn_key)

        self.edit_remote_dir = QLineEdit()
        self.edit_remote_dir.setPlaceholderText("/scratch/username/cest_mrf_run/")
        self.edit_remote_dir.setToolTip(
            "Working directory on the cluster where scripts and data are uploaded\n"
            "and results are downloaded from."
        )

        self.btn_test = QPushButton("Test Connection")
        self.btn_test.clicked.connect(self._test_connection)

        form.addRow("Hostname:", self.edit_host)
        form.addRow("SSH port:", self.spin_port)
        form.addRow("Username:", self.edit_user)
        form.addRow("SSH key:", key_row)
        form.addRow("Remote work dir:", self.edit_remote_dir)
        form.addRow("", self.btn_test)
        return grp

    def _build_job_group(self) -> QGroupBox:
        grp = QGroupBox("SLURM Job Resources")
        grp.setStyleSheet(_GRP_STYLE)
        form = QFormLayout(grp)
        form.setContentsMargins(6, 4, 6, 6)
        form.setVerticalSpacing(3)
        form.setHorizontalSpacing(8)

        self.edit_job_name = QLineEdit(_DEFAULTS["job_name"])
        self.edit_account  = QLineEdit(_DEFAULTS["account"])
        self.edit_partition = QLineEdit(_DEFAULTS["partition"])
        self.edit_partition.setToolTip("Comma-separated list of partitions, e.g. dgx-a100,rtx6000")

        self.spin_nodes = QSpinBox()
        self.spin_nodes.setRange(1, 64); self.spin_nodes.setValue(_DEFAULTS["nodes"])

        self.spin_cpus = QSpinBox()
        self.spin_cpus.setRange(1, 256); self.spin_cpus.setValue(_DEFAULTS["cpus"])
        self.spin_cpus.setToolTip("--cpus-per-task (used for parallel dictionary simulation)")

        self.spin_gpus = QSpinBox()
        self.spin_gpus.setRange(0, 16); self.spin_gpus.setValue(_DEFAULTS["gpus"])
        self.spin_gpus.setToolTip("--gpus (required for neural-network matching)")

        self.spin_mem = QSpinBox()
        self.spin_mem.setRange(1, 1500); self.spin_mem.setValue(_DEFAULTS["mem_gb"])
        self.spin_mem.setSuffix(" GB")

        self.edit_walltime = QLineEdit(_DEFAULTS["walltime"])
        self.edit_walltime.setToolTip("Wall-clock limit  D-HH:MM:SS")

        # Wire all fields → refresh preview on change
        for w in (self.edit_job_name, self.edit_account, self.edit_partition,
                  self.edit_walltime):
            w.textChanged.connect(self._refresh_previews)
        for sb in (self.spin_nodes, self.spin_cpus, self.spin_gpus, self.spin_mem):
            sb.valueChanged.connect(self._refresh_previews)

        form.addRow("Job name:", self.edit_job_name)
        form.addRow("Account:", self.edit_account)
        form.addRow("Partition(s):", self.edit_partition)
        form.addRow("Nodes:", self.spin_nodes)
        form.addRow("CPUs / task:", self.spin_cpus)
        form.addRow("GPUs:", self.spin_gpus)
        form.addRow("Memory:", self.spin_mem)
        form.addRow("Walltime:", self.edit_walltime)
        return grp

    def _build_scripts_group(self) -> QGroupBox:
        grp = QGroupBox("Environment && Script Settings")
        grp.setStyleSheet(_GRP_STYLE)
        form = QFormLayout(grp)
        form.setContentsMargins(6, 4, 6, 6)
        form.setVerticalSpacing(3)
        form.setHorizontalSpacing(8)

        self.edit_conda_env = QLineEdit(_DEFAULTS["conda_env"])
        self.edit_conda_env.setToolTip("Conda environment name on the cluster")

        self.edit_venv = QLineEdit(_DEFAULTS["venv"])
        self.edit_venv.setToolTip("Path to Python venv activate script (leave blank if not used)")

        self.edit_code_dir = QLineEdit(_DEFAULTS["remote_code_dir"])
        self.edit_code_dir.setToolTip("Directory containing MRFmatch_mt.py on the cluster")

        self.edit_py_script = QLineEdit(_DEFAULTS["python_script"])
        self.edit_py_script.setToolTip("Python script to run  (MRFmatch_mt.py or MRFmatch_SL.py)")

        # LARGE_STORAGE_DIR — optional
        self.edit_large_storage = QLineEdit()
        self.edit_large_storage.setPlaceholderText(
            "Leave blank to use OUTPUT_FILES/  (set if cluster has a separate large-storage mount)"
        )
        self.edit_large_storage.setToolTip(
            "If set, dict.mat and quant_maps.mat are written to\n"
            "$LARGE_STORAGE_DIR/MRF_OUTPUT/  (mirrors run.bash logic)"
        )

        # ── Singularity/Apptainer container (optional) ────────────────────
        self.chk_use_container = QCheckBox("Use Singularity / Apptainer container")
        self.chk_use_container.setToolTip(
            "When enabled, python is called via:\n"
            "  singularity exec --nv <container.sif> python ...\n\n"
            "No conda/venv needed on the cluster — everything is inside the container."
        )

        container_row = QHBoxLayout()
        self.edit_container_path = QLineEdit()
        self.edit_container_path.setPlaceholderText(
            "/scratch/shared/ocean-hpc_latest.sif"
        )
        self.edit_container_path.setToolTip(
            "Full path to the .sif file on the cluster.\n"
            "Build it with:  singularity pull docker://yourdockerhubname/ocean-hpc:latest"
        )
        self.edit_container_path.setEnabled(False)
        container_row.addWidget(self.edit_container_path, stretch=1)

        self.chk_container_gpu = QCheckBox("--nv (GPU)")
        self.chk_container_gpu.setChecked(True)
        self.chk_container_gpu.setToolTip(
            "Pass --nv flag to singularity exec so CUDA GPUs are visible inside."
        )
        self.chk_container_gpu.setEnabled(False)
        container_row.addWidget(self.chk_container_gpu)

        # toggle enabled state of container fields
        def _toggle_container(checked: bool):
            self.edit_container_path.setEnabled(checked)
            self.chk_container_gpu.setEnabled(checked)
            # grey out conda/venv when container is active
            for w in (self.edit_conda_env, self.edit_venv):
                w.setEnabled(not checked)

        self.chk_use_container.toggled.connect(_toggle_container)

        # Wire changes → refresh preview
        for w in (self.edit_conda_env, self.edit_venv, self.edit_code_dir,
                  self.edit_py_script, self.edit_large_storage,
                  self.edit_container_path):
            w.textChanged.connect(self._refresh_previews)
        self.chk_use_container.toggled.connect(self._refresh_previews)
        self.chk_container_gpu.toggled.connect(self._refresh_previews)

        form.addRow("Conda env:", self.edit_conda_env)
        form.addRow("Venv activate:", self.edit_venv)
        form.addRow("Code directory:", self.edit_code_dir)
        form.addRow("Python script:", self.edit_py_script)
        form.addRow("Large storage dir:", self.edit_large_storage)
        form.addRow("", self.chk_use_container)
        form.addRow("Container (.sif):", container_row)
        return grp

    def _build_matching_group(self) -> QGroupBox:
        grp = QGroupBox("Matching Mode")
        grp.setStyleSheet(_GRP_STYLE)
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(4)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Run matching:"))
        self.combo_run_mode = QComboBox()
        self.combo_run_mode.addItems([
            "Locally (dot-product, CPU)",
            "HPC cluster (dot-product, CPU)",
            "HPC cluster (neural network, GPU)  [future]",
        ])
        self.combo_run_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.combo_run_mode, stretch=1)
        layout.addLayout(mode_row)

        # NN placeholder panel
        self._nn_widget = QGroupBox("Neural Network Settings  [placeholder]")
        nn_form = QFormLayout(self._nn_widget)

        nn_note = QLabel(
            "⚠  Neural-network matching is not yet implemented.\n"
            "   Settings below are reserved for future integration."
        )
        nn_note.setStyleSheet(
            "color: #e6a817; font-size: 11px; padding: 4px;"
            "background: #2a2000; border-radius: 4px;"
        )
        nn_note.setWordWrap(True)
        nn_form.addRow(nn_note)

        model_row = QHBoxLayout()
        self.edit_nn_model = QLineEdit()
        self.edit_nn_model.setPlaceholderText("path/to/trained_model.pt")
        model_row.addWidget(self.edit_nn_model, stretch=1)
        btn_model = QPushButton("Browse…")
        btn_model.clicked.connect(self._browse_nn_model)
        model_row.addWidget(btn_model)

        self.combo_nn_arch = QComboBox()
        self.combo_nn_arch.addItems([
            "Sequential NN (1 output)",
            "Deep reconstruction CNN",
            "Transformer-based MRF",
        ])
        self.spin_nn_batch = QSpinBox()
        self.spin_nn_batch.setRange(64, 65536); self.spin_nn_batch.setValue(4096)
        self.chk_nn_normalize = QCheckBox("Normalize fingerprints before inference")
        self.chk_nn_normalize.setChecked(True)

        nn_form.addRow("Model file:", model_row)
        nn_form.addRow("Architecture:", self.combo_nn_arch)
        nn_form.addRow("Inference batch:", self.spin_nn_batch)
        nn_form.addRow("", self.chk_nn_normalize)

        layout.addWidget(self._nn_widget)
        self._nn_widget.setVisible(False)
        return grp

    def _build_actions_group(self) -> QGroupBox:
        grp = QGroupBox("Submit Job")
        grp.setStyleSheet(_GRP_STYLE)
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(6, 4, 6, 8)
        layout.setSpacing(5)

        note = QLabel(
            "Clicking Submit will:\n"
            "  1. Generate srun.bash + run.bash from the settings above\n"
            "  2. Upload them (+ acquired_data + config) via SCP\n"
            "  3. Run:  sbatch srun.bash  in the remote working directory"
        )
        note.setStyleSheet("font-size: 11px; color: #888;")
        note.setWordWrap(True)
        layout.addWidget(note)

        btn_row = QHBoxLayout()
        self.btn_submit = QPushButton("Submit Job")
        self.btn_submit.setFixedHeight(38)
        self.btn_submit.setStyleSheet(
            "QPushButton { background: #198754; color: white; font-weight: bold;"
            " border-radius: 5px; }"
            "QPushButton:hover { background: #157347; }"
            "QPushButton:disabled { background: #444; color: #888; }"
        )
        self.btn_submit.clicked.connect(self._submit_job)

        self.btn_cancel = QPushButton("Cancel Job")
        self.btn_cancel.setFixedHeight(38)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_job)

        self.btn_fetch = QPushButton("Fetch Results")
        self.btn_fetch.setFixedHeight(38)
        self.btn_fetch.setEnabled(False)
        self.btn_fetch.clicked.connect(self._fetch_results)

        btn_row.addWidget(self.btn_submit)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_fetch)
        layout.addLayout(btn_row)
        return grp

    # ─────────────────────────────────────────────────────────────────────
    # Script generation (srun.bash + run.bash)
    # ─────────────────────────────────────────────────────────────────────

    def _generate_srun_bash(self) -> str:
        """Build srun.bash content from current UI values."""
        mem = self.spin_mem.value()
        large = self.edit_large_storage.text().strip()
        large_export = f"\nexport LARGE_STORAGE_DIR={large}" if large else ""
        return f"""#!/bin/bash
#SBATCH --job-name={self.edit_job_name.text().strip() or 'MRF_dict'}
#SBATCH --account={self.edit_account.text().strip()}
#SBATCH --qos=normal
#SBATCH --partition={self.edit_partition.text().strip()}
#SBATCH --nodes={self.spin_nodes.value()}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={self.spin_cpus.value()}
#SBATCH --gpus={self.spin_gpus.value()}
#SBATCH --mem={mem}G
#SBATCH --time={self.edit_walltime.text().strip()}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
{large_export}
echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory requested: {mem}G"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Show available memory
free -h

# Run Python processing
bash run.bash

# Show final memory usage
echo "=========================================="
echo "Final memory usage:"
free -h
echo "=========================================="
"""

    def _generate_run_bash(self) -> str:
        """Build run.bash content from current UI values."""
        conda     = self.edit_conda_env.text().strip()
        venv      = self.edit_venv.text().strip()
        code      = self.edit_code_dir.text().strip()
        script    = self.edit_py_script.text().strip()
        large     = self.edit_large_storage.text().strip()
        use_ctr   = self.chk_use_container.isChecked()
        ctr_path  = self.edit_container_path.text().strip()
        ctr_gpu   = self.chk_container_gpu.isChecked()

        # ── environment activation block ──────────────────────────────────
        if use_ctr and ctr_path:
            nv_flag      = "--nv " if ctr_gpu else ""
            # Inside the container /opt/ocean already has everything.
            # We still cd to the user's code dir for I/O paths.
            env_block    = f"# Using Singularity container — no conda/venv needed"
            python_cmd   = f"singularity exec {nv_flag}{ctr_path} python"
            env_verify   = (
                f'singularity exec {nv_flag}{ctr_path} python -c '
                f'"import torch,numpy; print(\'Container: torch\',torch.__version__,'
                f'\'CUDA\',torch.version.cuda)"'
            )
        else:
            venv_line    = f"source {venv}" if venv else "# (no venv configured)"
            env_block    = (
                f"# Activate conda + venv\n"
                f"source ~/.bashrc\n"
                f"conda activate {conda}\n"
                f"{venv_line}"
            )
            python_cmd   = "python"
            env_verify   = (
                'echo "Python: $(which python)"\n'
                'echo "Python version: $(python --version)"\n'
                'echo "Numpy: $(python -c \'import numpy; print(numpy.__version__)\')"'
            )

        # ── storage / cleanup blocks ──────────────────────────────────────
        if large:
            storage_check = (
                "\n# Check if LARGE_STORAGE_DIR is set\n"
                "if [ -n \"$LARGE_STORAGE_DIR\" ]; then\n"
                "    echo \"Large storage enabled: $LARGE_STORAGE_DIR\"\n"
                "    export LARGE_STORAGE_DIR\n"
                "else\n"
                "    echo \"ℹ Using default OUTPUT_FILES directory\"\n"
                "fi"
            )
            cleanup = (
                "\necho \"Cleaning up previous output files...\"\n"
                "if [ -n \"$LARGE_STORAGE_DIR\" ]; then\n"
                "    rm -f $LARGE_STORAGE_DIR/MRF_OUTPUT/dict.mat\n"
                "    rm -f $LARGE_STORAGE_DIR/MRF_OUTPUT/quant_maps.mat\n"
                "else\n"
                "    rm -f OUTPUT_FILES/dict.mat\n"
                "    rm -f OUTPUT_FILES/quant_maps.mat\n"
                "fi"
            )
            output_check = (
                "\n    if [ -n \"$LARGE_STORAGE_DIR\" ]; then\n"
                "        echo \"Output files:\"\n"
                "        ls -lh $LARGE_STORAGE_DIR/MRF_OUTPUT/ 2>/dev/null "
                "|| echo \"  No files in large storage\"\n"
                "    else\n"
                "        echo \"Output files:\"\n"
                "        ls -lh OUTPUT_FILES/*.mat 2>/dev/null "
                "|| echo \"  No .mat files in OUTPUT_FILES\"\n"
                "    fi"
            )
        else:
            storage_check = ""
            cleanup = (
                "\necho \"Cleaning up previous output files...\"\n"
                "rm -f OUTPUT_FILES/dict.mat\n"
                "rm -f OUTPUT_FILES/quant_maps.mat"
            )
            output_check = (
                "\n        echo \"Output files:\"\n"
                "        ls -lh OUTPUT_FILES/*.mat 2>/dev/null "
                "|| echo \"  No .mat files in OUTPUT_FILES\""
            )

        return f"""#!/bin/bash

echo "=========================================="
echo "MRF Dictionary Simulation + Matching"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "=========================================="
{storage_check}
# Show GPU info
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi

{env_block}

# Verify environment
{env_verify}
echo "=========================================="

# Navigate to code directory
cd {code}
echo "Working directory: $(pwd)"
{cleanup}
echo "Cleanup completed."

# Run the Python script
echo "Starting MRF processing..."
{python_cmd} {script}

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "✓ MRF processing completed successfully"
{output_check}
else
    echo "=========================================="
    echo "✗ MRF processing failed with exit code $?"
fi
echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="
"""

    def _refresh_previews(self):
        """No-op — preview tabs have been removed. Scripts are generated on demand."""
        pass

    # ─────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────

    def _on_mode_changed(self, idx: int):
        self._nn_widget.setVisible(idx == 2)

    def _browse_ssh_key(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SSH private key", "", "All files (*)"
        )
        if path:
            self.edit_ssh_key.setText(path)

    def _browse_nn_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model checkpoint",
            "", "Model files (*.pt *.pth *.ckpt);;All files (*)"
        )
        if path:
            self.edit_nn_model.setText(path)

    def _test_connection(self):
        host = self.edit_host.text().strip()
        user = self.edit_user.text().strip()
        if not host or not user:
            self._log("ERROR: Please enter hostname and username first.")
            return
        self._log(f"Testing SSH → {user}@{host}:{self.spin_port.value()} …")
        try:
            import paramiko  # type: ignore
            # Close any previously open client
            if self._ssh_client:
                try:
                    self._ssh_client.close()
                except Exception:
                    pass
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            key_path = self.edit_ssh_key.text().strip() or None
            client.connect(host, port=self.spin_port.value(), username=user,
                           key_filename=key_path, timeout=10)
            _, stdout, _ = client.exec_command("hostname && uname -r && echo READY")
            out = stdout.read().decode().strip()
            self._ssh_client = client
            self._log(f"Connection successful!  Remote:\n  {out}")
            self.lbl_job_status.setText("Connected.")
        except ImportError:
            self._log("paramiko not installed — run:  pip install paramiko")
        except Exception as exc:
            self._log(f"Connection failed: {exc}")


    def _submit_job(self):
        host = self.edit_host.text().strip()
        if not host:
            self._log("ERROR: No cluster hostname configured.")
            return

        mode = self.combo_run_mode.currentIndex()
        if mode == 2:
            self._log("⚠  Neural-network matching is not yet implemented. Use dot-product mode.")
            return
        if mode == 0:
            self._log("Run mode is 'Local'. Switch to an HPC mode to submit a cluster job.")
            return

        user     = self.edit_user.text().strip()
        rdir     = self.edit_remote_dir.text().strip()
        srun_txt = self._generate_srun_bash()
        run_txt  = self._generate_run_bash()

        self._log(f"Preparing to submit job to {user}@{host} …")
        self._log(f"  Remote dir : {rdir}")
        self._log(f"  Script     : {self.edit_py_script.text().strip()}")

        try:
            import paramiko, io  # type: ignore

            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            key_path = self.edit_ssh_key.text().strip() or None
            client.connect(host, port=self.spin_port.value(), username=user,
                           key_filename=key_path, timeout=15)

            sftp = client.open_sftp()

            # Ensure remote working dir exists
            try:
                sftp.stat(rdir)
            except FileNotFoundError:
                client.exec_command(f"mkdir -p {rdir}")

            # Upload generated bash scripts
            sftp.putfo(io.BytesIO(srun_txt.encode()), f"{rdir}/srun.bash")
            sftp.putfo(io.BytesIO(run_txt.encode()),  f"{rdir}/run.bash")
            self._log("Uploaded srun.bash and run.bash to cluster.")

            # Make scripts executable and submit
            _, stdout, stderr = client.exec_command(
                f"cd {rdir} && chmod +x srun.bash run.bash && sbatch srun.bash"
            )
            out = stdout.read().decode().strip()
            err = stderr.read().decode().strip()
            sftp.close()
            client.close()

            if out:
                self._log(f"sbatch output: {out}")
                # Parse job ID from "Submitted batch job 12345"
                for token in out.split():
                    if token.isdigit():
                        self._job_id = token
                        self.lbl_job_status.setText(f"Job submitted — ID: {self._job_id}")
                        break
            if err:
                self._log(f"sbatch stderr: {err}")

            self.btn_cancel.setEnabled(True)
            self.btn_fetch.setEnabled(True)

        except ImportError:
            self._log("paramiko not installed — run:  pip install paramiko")
        except Exception as exc:
            self._log(f"Submission error: {exc}")

    def _cancel_job(self):
        if not self._job_id:
            self._log("No active job ID to cancel.")
            return
        self._log(f"Requesting scancel for job {self._job_id} …")
        host = self.edit_host.text().strip()
        user = self.edit_user.text().strip()
        try:
            import paramiko  # type: ignore
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            key_path = self.edit_ssh_key.text().strip() or None
            client.connect(host, port=self.spin_port.value(), username=user,
                           key_filename=key_path, timeout=10)
            _, stdout, _ = client.exec_command(f"scancel {self._job_id}")
            self._log(stdout.read().decode().strip() or f"scancel {self._job_id} sent.")
            client.close()
        except ImportError:
            self._log("paramiko not installed.")
        except Exception as exc:
            self._log(f"Cancel error: {exc}")
        self.btn_cancel.setEnabled(False)

    def _fetch_results(self):
        """Download quant_maps.mat + dict.mat from cluster via SFTP."""
        host  = self.edit_host.text().strip()
        user  = self.edit_user.text().strip()
        rdir  = self.edit_remote_dir.text().strip()
        code  = self.edit_code_dir.text().strip()
        large = self.edit_large_storage.text().strip()

        remote_mat_dir = f"{large}/MRF_OUTPUT" if large else f"{code}/OUTPUT_FILES"

        local_dir, _ = QFileDialog.getExistingDirectory(
            self, "Choose local directory for downloaded results"
        ) if hasattr(QFileDialog, 'getExistingDirectory') else ("", "")

        # QFileDialog.getExistingDirectory returns a str, not tuple
        if isinstance(local_dir, tuple):
            local_dir = local_dir[0]
        if not local_dir:
            self._log("Fetch cancelled — no local directory chosen.")
            return

        self._log(f"Fetching from {user}@{host}:{remote_mat_dir} …")
        try:
            import paramiko  # type: ignore
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            key_path = self.edit_ssh_key.text().strip() or None
            client.connect(host, port=self.spin_port.value(), username=user,
                           key_filename=key_path, timeout=15)
            sftp = client.open_sftp()
            for fname in ("quant_maps.mat", "dict.mat"):
                remote = f"{remote_mat_dir}/{fname}"
                local  = f"{local_dir}/{fname}"
                try:
                    sftp.get(remote, local)
                    self._log(f"  ✓ {fname} → {local}")
                except FileNotFoundError:
                    self._log(f"  ✗ {fname} not found on cluster yet")
            sftp.close()
            client.close()
        except ImportError:
            self._log("paramiko not installed — run:  pip install paramiko")
        except Exception as exc:
            self._log(f"Fetch error: {exc}")

    def _refresh_status(self):
        if not self._job_id:
            self._log("No active job ID to check.")
            return
        host = self.edit_host.text().strip()
        user = self.edit_user.text().strip()
        if not host or not user:
            self._log("Configure SSH connection first.")
            return
        self._log(f"Checking squeue for job {self._job_id} …")
        try:
            import paramiko  # type: ignore
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            key_path = self.edit_ssh_key.text().strip() or None
            client.connect(host, port=self.spin_port.value(), username=user,
                           key_filename=key_path, timeout=10)
            _, stdout, _ = client.exec_command(
                f"squeue -j {self._job_id} --format='%.18i %.9P %.8j %.8u %.8T %.10M' 2>&1"
            )
            out = stdout.read().decode().strip()
            client.close()
            self._log(out or f"Job {self._job_id} not found in queue (may have finished).")
            self.lbl_job_status.setText(
                f"Job {self._job_id}: " + (out.splitlines()[-1] if out else "done/unknown")
            )
        except ImportError:
            self._log("paramiko not installed.")
        except Exception as exc:
            self._log(f"Status check error: {exc}")

    def _log(self, msg: str):
        self.log.append(msg)
