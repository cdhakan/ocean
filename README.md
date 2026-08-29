# OCEAN

**Open Cross‑platform Environment for Analysis of molecular MRI** — a desktop application for quantitative CEST, CEST‑MRF, and relaxometry analysis, for **macOS, Windows, and Linux**.

## Download

Get the latest build for your operating system from the **[Releases](../../releases)** page:

| Platform | File | How to run |
|----------|------|-----------|
| Windows 10/11 | `OCEAN_Windows.zip` | Unzip → run `OCEAN.exe` |
| macOS 11+ | `OCEAN_v*.dmg` | Open → drag to Applications |
| Linux x86_64 | `OCEAN_Linux.tar.gz` | Extract → run `./OCEAN/OCEAN` |

> First launch on Windows may show a SmartScreen warning ("Unknown publisher") — click **More info → Run anyway**. On macOS, right‑click → **Open** the first time (unsigned app).

## Features

- **Relaxometry & field mapping** — T1, T2, B1 maps; WASABI / WASSR B0/B1 mapping
- **CEST‑MRI quantification** — Lorentzian, Gaussian, pseudo‑Voigt, multi‑pool Lorentzian (MPLF), PLOF, DROF, inverse‑Z (1/Z), QUESP, and forward synthetic‑CEST simulation
- **Inverse‑Z (AREX)** — spillover/MT/T1‑corrected analysis with per‑pool QUESP *f*ₛ / *k*ₛw maps
- **CEST‑MRF** — Pulseq schedule → Bloch–McConnell dictionary simulation + dot‑product matching
- **Automation** — phantom detection & brain skull‑stripping; ROI statistics with X–Y agreement plots and box plots; 300‑dpi figure export (PNG/JPEG/TIFF/PDF/SVG)
- **Vendor support** — Bruker (ParaVision), Siemens & GE (DICOM), MR Solutions (.MRD)

## Build from source

Builds are produced entirely in the cloud by [GitHub Actions](.github/workflows/build.yml): pushing a version tag (`git tag v1.0.0 && git push origin v1.0.0`) builds all three platforms and publishes a Release. To build locally, install the dependencies and run `pyinstaller OCEAN.spec`.

## License

OCEAN is released under the **MIT License** — see [LICENSE](LICENSE).

> The software is provided "as is", without warranty of any kind. Third‑party components (PyQt6, NumPy, SciPy, Matplotlib, BMCTool, PyPulseq, itk‑elastix, the pulseq‑CEST library, …) retain their own licenses, and the bundled reference material is included solely for documentation of the implemented methods.
