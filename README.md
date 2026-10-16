# OCEAN

**Open Cross‑platform Environment for Analysis of molecular MRI** — a desktop application for quantitative CEST, CEST‑MRF, and relaxometry analysis, for **macOS, Windows, and Linux**.

## Download

Get the latest build for your operating system from the **[Releases](../../releases)** page:

| Platform | File | How to run |
|----------|------|-----------|
| Windows 10/11 | `OCEAN.zip` | Unzip → run `OCEAN.exe` |
| macOS 11+ | `OCEAN.dmg` | Open → drag to Applications |
| Linux x86_64 | `OCEAN.tar.gz` | Extract → run `./OCEAN/OCEAN` |

> First launch on Windows may show a SmartScreen warning ("Unknown publisher") — click **More info → Run anyway**. On macOS, right‑click → **Open** the first time (unsigned app).

## Features

- **Relaxometry & field mapping** — T1, T2, B1 maps; WASABI / WASSR B0/B1 mapping
- **CEST‑MRI quantification** — Lorentzian, Gaussian, pseudo‑Voigt, multi‑pool Lorentzian (MPLF), PLOF, DROF, inverse‑Z (1/Z), QUESP, and forward synthetic‑CEST simulation
- **Inverse‑Z (AREX)** — spillover/MT/T1‑corrected analysis with per‑pool QUESP *f*ₛ / *k*ₛw maps
- **CEST‑MRF** — Pulseq schedule → Bloch–McConnell dictionary simulation + dot‑product matching
- **Automation** — phantom detection & brain skull‑stripping; ROI statistics with X–Y agreement plots and box plots; 300‑dpi figure export (PNG/JPEG/TIFF/PDF/SVG)
- **Vendor support** — Bruker (ParaVision), Siemens & GE (DICOM / NIfTI), Siemens raw data (TWIX `.dat`, reconstructed in-app), MR Solutions (.MRD), MATLAB `.mat` stacks from any vendor

## Build from source

The Windows `.zip` and Linux `.tar.gz` are built by [GitHub Actions](.github/workflows/build.yml) when a version tag is pushed (`git push origin v1.0`) and attached to the Release. The macOS `.dmg` is built on a Mac with `bash build_dmg.sh` and uploaded to the same Release. To build locally, install the dependencies (`pip install -r requirements.txt`) and run `pyinstaller OCEAN.spec`.

## License

OCEAN is released under the **MIT License** — see [LICENSE](LICENSE).

> The software is provided "as is", without warranty of any kind. Third‑party components (PyQt6, NumPy, SciPy, Matplotlib, BMCTool, PyPulseq, itk‑elastix, the pulseq‑CEST library, …) retain their own licenses, and the bundled reference material is included solely for documentation of the implemented methods.
