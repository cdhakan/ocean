"""
orchestra_worker.py — GE P-file reconstruction with the GE Orchestra SDK.
=======================================================================
Runs in the Orchestra Python environment (created by install_orchestra.sh:
Python 3.10 + NumPy 1.x), NEVER in OCEAN's own env — the SDK's compiled module
is tied to that interpreter/NumPy ABI, and GE's licence forbids bundling it.
OCEAN calls this script as a subprocess (my_gui/ge_pfile_reader.py) and reads
the .npz it writes.

    python orchestra_worker.py <P12345.7> <out.npz> --sdk <orchestra-sdk dir>
    python orchestra_worker.py --check --sdk <orchestra-sdk dir>

Reconstruction per frame (GE's Cartesian 2-D pipeline, Scripts/Cartesian.py):
    k-space (all channels) → Transformer (iFFT) → ChannelCombiner.SumOfSquares
    → |·| → Gradwarp (corners, FOV scaling) → Orientation.OrientImage
Output ``stack`` is float32 (Y, X, n_slices, n_frames) with
frame = phase * n_echoes + echo — one frame per saturation offset for CEST.
"""
import argparse
import importlib.machinery
import importlib.util
import json
import os
import sys
import time

import numpy as np

_LIB_CANDIDATES = {
    "darwin": ("GERecon.so.mac", "GERecon.so"),
    "linux":  ("GERecon.so", "GERecon.so.python36"),
    "win32":  ("GERecon.pyd",),
}


def load_gerecon(sdk_dir: str):
    """Import the SDK's compiled module straight from its file (no symlink,
    nothing written into the SDK folder)."""
    plat = "darwin" if sys.platform == "darwin" else ("win32" if sys.platform == "win32" else "linux")
    for name in _LIB_CANDIDATES[plat]:
        path = os.path.join(sdk_dir, name)
        if os.path.isfile(path):
            loader = importlib.machinery.ExtensionFileLoader("GERecon", path)
            spec = importlib.util.spec_from_file_location("GERecon", path, loader=loader)
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)
            sys.modules["GERecon"] = mod
            return mod
    raise FileNotFoundError(f"no GERecon library ({', '.join(_LIB_CANDIDATES[plat])}) in {sdk_dir}")


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def reconstruct(pfile_path: str, sdk_dir: str, gradwarp: bool = True) -> "tuple[np.ndarray, dict]":
    G = load_gerecon(sdk_dir)
    t0 = time.time()
    p = G.Pfile(pfile_path)
    md = dict(p.MetaData())
    n_ph, n_sl, n_ec, n_ch = (int(md["phases"]), int(md["slices"]),
                              int(md["echoes"]), int(md["channels"]))
    transformer = G.Transformer(p.TransformerParams())
    combiner = G.ChannelCombiner(p.ChannelCombinerParams())
    gw = G.Gradwarp() if gradwarp else None
    orient = G.Orientation()
    gw_params = p.GradwarpParams() if gradwarp else None

    stack = None
    gradwarp_ok, orient_ok = True, True
    orientation_info = None
    for sl in range(n_sl):
        corners = p.Corners(sl)
        ori = p.Orientation(sl)
        if orientation_info is None:
            orientation_info = {k: ori[k] for k in ori}
        for ph in range(n_ph):
            for ec in range(n_ec):
                if n_ph == 1:
                    ks = np.asarray(p.KSpace(sl, ec))                      # (X, Y, ch)
                else:
                    ks = np.stack([np.asarray(p.KSpace(sl, ec, ch, ph))
                                   for ch in range(n_ch)], axis=-1)
                if ks.ndim == 2:
                    ks = ks[:, :, None]
                img = transformer.Execute(np.ascontiguousarray(ks, dtype=np.complex64))
                mag = np.abs(combiner.SumOfSquares(img)) if img.shape[-1] > 1 else np.abs(img[:, :, 0])
                mag = np.asarray(mag, dtype=np.float32)
                if gw is not None:
                    try:
                        mag = np.asarray(gw.Execute(mag, corners, fovScalingParams=gw_params), dtype=np.float32)
                    except Exception:
                        gradwarp_ok = False
                try:
                    mag = np.asarray(orient.OrientImage(mag, ori), dtype=np.float32)
                except Exception:
                    orient_ok = False
                if stack is None:
                    stack = np.zeros(mag.shape + (n_sl, n_ph * n_ec), dtype=np.float32)
                stack[:, :, sl, ph * n_ec + ec] = mag

    hdr = p.Header()
    img_h = hdr.get("rdb_hdr_image", {}) or {}
    rec_h = hdr.get("rdb_hdr_rec", {}) or {}
    ser_h = hdr.get("rdb_hdr_series", {}) or {}
    ex_h = hdr.get("rdb_hdr_exam", {}) or {}

    def _g(d, k, default=None):
        try:
            return _jsonable(d[k])
        except Exception:
            return default

    meta = {
        "metadata":        _jsonable(md),
        "n_frames":        int(n_ph * n_ec),
        "frame_order":     "phase-major (frame = phase * n_echoes + echo)",
        "gradwarp":        bool(gradwarp and gradwarp_ok),
        "oriented":        bool(orient_ok),
        "orientation":     _jsonable(orientation_info),
        "fov_mm":          _g(img_h, "dfov"),
        "slice_thickness_mm": _g(img_h, "slthick"),
        "pixsize_mm":      [_g(img_h, "pixsize_X"), _g(img_h, "pixsize_Y")],
        "te_us":           _g(img_h, "te"),
        "tr_us":           _g(img_h, "tr"),
        "psdname":         _g(img_h, "psdname"),
        "magstrength_gauss": _g(ex_h, "magstrength"),
        "series_desc":     _g(ser_h, "se_desc"),
        "rdbm_rev":        _g(rec_h, "rdb_hdr_rdbm_rev"),
        "user_cv_image":   [_g(img_h, f"user{i}", 0.0) for i in range(30)],
        "user_cv_rec":     [_g(rec_h, f"rdb_hdr_user{i}", 0.0) for i in range(49)],
        "recon_seconds":   round(time.time() - t0, 2),
        "sdk_dir":         sdk_dir,
        "python":          sys.version.split()[0],
        "numpy":           np.__version__,
    }
    return stack, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="GE P-file → image stack (Orchestra SDK)")
    ap.add_argument("pfile", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--sdk", required=True, help="orchestra-sdk-*.python folder")
    ap.add_argument("--no-gradwarp", action="store_true")
    ap.add_argument("--check", action="store_true", help="only verify the SDK imports")
    a = ap.parse_args(argv)
    if a.check:
        G = load_gerecon(a.sdk)
        n = len(str(G.Options()).splitlines())
        print(f"GERecon OK: {n} API entries, python {sys.version.split()[0]}, numpy {np.__version__}")
        return 0
    if not a.pfile or not a.out:
        ap.error("pfile and out are required (or --check)")
    stack, meta = reconstruct(a.pfile, a.sdk, gradwarp=not a.no_gradwarp)
    np.savez(a.out, stack=stack, meta=json.dumps(meta))
    print(f"OK {stack.shape} → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
