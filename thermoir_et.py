"""ThermoIR-ET: temperature transfer of molecular infrared band spectra.

Inference uses numeric tree arrays only; no pickle or scikit-learn is loaded.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d

VERSION = "1.0.0"
N = 512
U = np.linspace(0, 1, N)
FWHM = 2.0
SIGMA_CM1 = FWHM / np.sqrt(8 * np.log(2))
PNNL_COLUMN = 1e-6 * 101325 / (1.380649e-23 * 296) * 1e-6 * 100
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "thermoir_et_v1.npz"


def unique_sorted(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 2:
        raise ValueError("Each spectrum must contain at least two wavenumber/value pairs.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Spectra must contain finite numbers; missing values are not interpolated.")
    order = np.argsort(x)
    x, y = x[order], y[order]
    ux, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(ux) != len(x):
        y, x = np.bincount(inverse, weights=y) / counts, ux
    if len(x) < 2 or x[0] <= 0:
        raise ValueError("At least two distinct positive wavenumbers are required.")
    return x, y


def native_smooth(x, y, output_x):
    """Apply the benchmark's 2 cm-1 Gaussian on the native sampling grid."""
    x, y = unique_sorted(x, y)
    output_x = np.asarray(output_x, float)
    out = np.full(output_x.shape, np.nan)
    gap_left = np.flatnonzero(np.diff(x) > 5.0)
    if len(output_x) and any(x[i] < output_x.max() and x[i + 1] > output_x.min() for i in gap_left):
        raise ValueError("Requested band contains a sampling gap exceeding 5 cm-1.")
    cuts = np.r_[0, gap_left + 1, len(x)]
    details = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        sx, sy = x[start:end], y[start:end]
        if len(sx) < 2:
            continue
        cover = (output_x >= sx[0]) & (output_x <= sx[-1])
        if not np.any(cover):
            continue
        spacing = (sx[-1] - sx[0]) / (len(sx) - 1)
        median_dx = float(np.median(np.diff(sx)))
        uniform = bool(np.max(np.abs(np.diff(sx) - spacing)) <= max(1e-5, 1e-3 * spacing))
        if uniform:
            grid, values, dx = sx, sy, spacing
            method = "native_uniform_convolution"
        else:
            requested = min(0.25, median_dx)
            count = int(np.ceil((sx[-1] - sx[0]) / requested)) + 1
            if count > 2_000_000:
                raise ValueError("Irregular input sampling requests an excessive convolution grid.")
            grid = np.linspace(sx[0], sx[-1], count)
            dx = (sx[-1] - sx[0]) / (count - 1)
            values = np.interp(grid, sx, sy)
            method = "piecewise_linear_fine_grid_convolution"
        smoothed = gaussian_filter1d(values, SIGMA_CM1 / dx, mode="nearest", truncate=6.0)
        out[cover] = np.interp(output_x[cover], grid, smoothed)
        details.append(dict(native_points=len(sx), native_median_dx_cm1=median_dx,
                            convolution_dx_cm1=float(dx), method=method))
    if not np.all(np.isfinite(out)):
        raise ValueError("Requested band contains uncovered points or a sampling gap exceeding 5 cm-1.")
    return out, details


def to_cross_section(values, units, column_density=None):
    """Return natural-log absorption cross-section in cm2/molecule."""
    values = np.asarray(values, float)
    if units == "cross-section":
        return values
    if units == "pnnl":
        return values * np.log(10) / PNNL_COLUMN
    if column_density is None or not np.isfinite(column_density) or column_density <= 0:
        raise ValueError("Absorbance input requires a positive absorber column density in molecules/cm2.")
    if units == "absorbance-ln":
        return values / column_density
    if units == "absorbance-decadic":
        return values * np.log(10) / column_density
    raise ValueError("Unknown spectrum units.")


def make_features(p, ts, x):
    """Identical 2/3-input features to training; explicit experimental 1-input adapter."""
    p, ts, x = np.asarray(p, float), np.asarray(ts, float), np.asarray(x, float)
    nobs = len(ts)
    if nobs not in (1, 2, 3) or p.shape != (nobs, N) or x.shape != (N,):
        raise ValueError("Expected one to three input spectra on a 512-point band grid.")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(ts)) or np.any(ts <= 0):
        raise ValueError("Input spectra and positive temperatures must be finite.")
    if np.any(np.diff(ts) <= 0):
        raise ValueError("Input temperatures must be distinct and sorted in increasing order.")
    area = np.trapezoid(np.maximum(p[-1], 0), x)
    width = x[-1] - x[0]
    if area <= 0 or width <= 0:
        raise ValueError("The warmest input must have positive integrated absorption in the band.")
    f = p * width / area
    if nobs == 2:
        f = np.stack([f[0], (f[0] + f[-1]) / 2, f[-1]])
        ts = np.array([ts[0], np.mean(ts), ts[-1]])
    elif nobs == 1:
        f = np.repeat(f, 3, axis=0)
    ii = np.arange(N)
    cols, names = [], []
    for j in range(3):
        for off in (-16, -8, -4, -2, 0, 2, 4, 8, 16):
            cols.append(f[j, np.clip(ii + off, 0, N - 1)])
            names.append(f"lowT{j}_patch{off}")
    slope = np.zeros(N) if nobs == 1 else (f[-1] - f[0]) / (ts[-1] - ts[0]) * 100
    cols.extend([slope, U, x / 1000])
    names.extend(["slope_per100K", "relative_wavenumber", "wavenumber_per1000"])
    pos = np.maximum(f[-1], 0)
    mass = np.trapezoid(pos, U)
    center = np.trapezoid(pos * U, U) / mass
    spread = np.sqrt(np.trapezoid(pos * (U - center) ** 2, U) / mass)
    vals = [width / 1000, np.mean(x) / 1000, np.log10(area), np.trapezoid(f[0], U),
            np.trapezoid(f[1], U), center, spread, nobs, ts[0] / 1000, ts[-1] / 1000]
    cols.extend(np.full(N, val) for val in vals)
    names.extend(["window_width", "window_center", "log10_anchor_area", "low_area_ratio",
                  "mid_area_ratio", "centroid", "rms_width", "input_count", "coldest_T", "anchor_T"])
    return np.column_stack(cols).astype("f4"), area, f[-1], slope, names


def features_at(features, anchor_temperature, target_temperature):
    n = len(features)
    return np.column_stack([features, np.full(n, target_temperature / 1000),
                            np.full(n, (target_temperature - anchor_temperature) / 500)]).astype("f4")


class NumericForest:
    """Read fixed numeric arrays and evaluate extremely randomized regression trees."""
    def __init__(self, path=DEFAULT_MODEL):
        path = Path(path)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata_path = path.with_suffix(".json")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf8"))
        if self.sha256 != self.metadata["weights_sha256"]:
            raise ValueError("Model weights do not match the SHA-256 recorded in their metadata.")
        with np.load(path, allow_pickle=False) as z:
            self.offsets = z["offsets"]
            self.feature, self.threshold = z["feature"], z["threshold"]
            self.left, self.right, self.value = z["left"], z["right"], z["value"]
        size = len(self.feature)
        if not (self.offsets[0] == 0 and self.offsets[-1] == size and np.all(np.diff(self.offsets) > 0)):
            raise ValueError("Invalid tree offsets.")
        if any(a.ndim != 1 or len(a) != size for a in (self.feature, self.threshold, self.left, self.right, self.value)):
            raise ValueError("Invalid tree array dimensions.")
        if not np.all(np.isfinite(self.value)) or not np.all(np.isfinite(self.threshold)):
            raise ValueError("Non-finite model arrays.")
        for start, end in zip(self.offsets[:-1], self.offsets[1:]):
            split = self.feature[start:end] >= 0
            if np.any(self.feature[start:end][split] >= 42):
                raise ValueError("Invalid feature index.")
            for children in (self.left[start:end][split], self.right[start:end][split]):
                if np.any(children < 0) or np.any(children >= end - start):
                    raise ValueError("Invalid tree child index.")

    def predict(self, X):
        X = np.asarray(X, dtype="f4")
        if X.ndim != 2 or X.shape[1] != 42 or not np.all(np.isfinite(X)):
            raise ValueError("Prediction requires finite 42-column features.")
        rows = np.arange(len(X))
        total = np.zeros(len(X), float)
        for start, end in zip(self.offsets[:-1], self.offsets[1:]):
            nodes = np.zeros(len(X), dtype=np.int64)
            for _ in range(64):
                features = self.feature[start + nodes]
                active = features >= 0
                if not np.any(active):
                    break
                current = start + nodes[active]
                go_left = X[rows[active], features[active]] <= self.threshold[current]
                nodes[active] = np.where(go_left, self.left[current], self.right[current])
            else:
                raise ValueError("Invalid tree: traversal exceeds maximum supported depth.")
            total += self.value[start + nodes]
        return total / (len(self.offsets) - 1)


def read_spectrum(path):
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    first = next((line for line in lines if line.strip() and not line.lstrip().startswith("#")), "")
    content = first.split("#", 1)[0]
    delimiter = "," if "," in content else None
    fields = content.split(delimiter)
    try:
        [float(v) for v in fields]
        skip = 0
    except ValueError:
        skip = lines.index(first) + 1
    arr = np.loadtxt(path, delimiter=delimiter, skiprows=skip, comments="#", ndmin=2, encoding="utf-8-sig")
    if arr.shape[1] != 2:
        raise ValueError(f"{path.name}: exactly two columns (wavenumber, absorption) are required.")
    return unique_sorted(arr[:, 0], arr[:, 1])


def predict_spectra(spectra, temperatures, targets, units="cross-section", input_columns=None,
                    window=None, forest=None):
    temperatures, targets = np.asarray(temperatures, float), np.asarray(targets, float)
    if len(spectra) not in (1, 2, 3) or len(spectra) != len(temperatures):
        raise ValueError("Supply one to three spectra and one temperature for each spectrum.")
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0) or len(np.unique(temperatures)) != len(temperatures):
        raise ValueError("Input temperatures must be distinct positive Kelvin values.")
    if targets.ndim != 1 or not len(targets) or not np.all(np.isfinite(targets)) or np.any(targets <= temperatures.max()):
        raise ValueError("Every target temperature must exceed the warmest input temperature.")
    if input_columns is None:
        input_columns = [None] * len(spectra)
    elif len(input_columns) == 1:
        input_columns = list(input_columns) * len(spectra)
    if len(input_columns) != len(spectra):
        raise ValueError("Give one shared column density or one for each input spectrum.")
    order = np.argsort(temperatures)
    temperatures = temperatures[order]
    native = [unique_sorted(*spectra[i]) for i in order]
    columns = [input_columns[i] for i in order]
    low, high = max(a[0][0] for a in native), min(a[0][-1] for a in native)
    if window is not None:
        low, high = map(float, window)
        if any(low < a[0][0] or high > a[0][-1] for a in native):
            raise ValueError("The selected band must be covered by every input spectrum.")
    if not np.isfinite(low + high) or high <= low:
        raise ValueError("Input spectra must share a finite nonempty wavenumber interval.")
    x = np.linspace(low, high, N)
    prepared, detail = [], []
    for (nx, ny), col in zip(native, columns):
        y, d = native_smooth(nx, to_cross_section(ny, units, col), x)
        prepared.append(y)
        detail.append(d)
    features, area, anchor, _, _ = make_features(np.asarray(prepared), temperatures, x)
    forest = forest or NumericForest()
    predictions = np.asarray([np.maximum((anchor + forest.predict(features_at(features, temperatures[-1], T))) * area / (high - low), 0) for T in targets])
    flags = []
    if len(temperatures) == 1:
        flags.append("EXPERIMENTAL_SINGLE_INPUT: model fit used two or three inputs; one observed spectrum is repeated in the shape slots with zero slope and input_count=1.")
    if temperatures.min() < 278.15 or temperatures.max() > 323.15:
        flags.append("INPUT_TEMPERATURE_OUTSIDE_PNNL_RANGE: the principal low-temperature training inputs were near 278-323 K.")
    if targets.max() > 1573.15:
        flags.append("TARGET_ABOVE_BENCHMARK_RANGE: no reported held-out evaluation above 1573.15 K.")
    if window is None:
        flags.append("FULL_COMMON_INTERVAL_USED: supply a band window when input files span multiple unrelated absorption bands.")
    metadata = dict(tool="ThermoIR-ET", version=VERSION, model_sha256=forest.sha256,
                    input_temperatures_K=temperatures.tolist(), target_temperatures_K=targets.tolist(),
                    input_units=units, cross_section_units="cm2/molecule (natural-log convention)",
                    output_points=N, window_cm1=[low, high], convolution_fwhm_cm1=FWHM,
                    preprocessing=detail, warnings=flags)
    if units in ("absorbance-ln", "absorbance-decadic"):
        metadata["input_column_density_molecules_per_cm2"] = [float(c) for c in columns]
    elif units == "pnnl":
        metadata["pnnl_reference_column_density_molecules_per_cm2"] = PNNL_COLUMN
    return x, predictions, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum", nargs=2, action="append", required=True, metavar=("FILE", "TEMPERATURE_K"))
    parser.add_argument("--target-temperature", nargs="+", required=True, type=float, metavar="K")
    parser.add_argument("--units", choices=["cross-section", "absorbance-ln", "absorbance-decadic", "pnnl"], required=True)
    parser.add_argument("--input-column-density", nargs="+", type=float, metavar="MOLECULES_PER_CM2")
    parser.add_argument("--output-column-density", type=float, metavar="MOLECULES_PER_CM2")
    parser.add_argument("--window", nargs=2, type=float, metavar=("LOW_CM1", "HIGH_CM1"))
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path; adjacent JSON records provenance")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--plot", action="store_true", help="Also save a PNG (requires matplotlib)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        out, meta_path = args.output, args.output.with_suffix(".json")
        if out.suffix.lower() != ".csv":
            raise ValueError("--output must have the .csv extension.")
        paths = [out, meta_path] + ([out.with_suffix(".png")] if args.plot else [])
        input_paths = {Path(v[0]).resolve() for v in args.spectrum}
        input_paths.update([args.model.resolve(), args.model.with_suffix(".json").resolve()])
        if any(p.resolve() in input_paths for p in paths):
            raise ValueError("Output paths must not overwrite an input spectrum or model file.")
        if not args.overwrite and any(p.exists() for p in paths):
            raise ValueError("An output file already exists; choose another path or use --overwrite.")
        if args.output_column_density is not None and (not np.isfinite(args.output_column_density) or args.output_column_density <= 0):
            raise ValueError("Output column density must be finite and positive.")
        if args.plot:
            import matplotlib.pyplot as plt
            from matplotlib import font_manager
        x, predictions, meta = predict_spectra(
            [read_spectrum(v[0]) for v in args.spectrum], [float(v[1]) for v in args.spectrum],
            args.target_temperature, units=args.units, input_columns=args.input_column_density,
            window=args.window, forest=NumericForest(args.model))
        blocks = []
        header = "wavenumber_cm-1,temperature_K,cross_section_cm2_per_molecule"
        for T, y in zip(args.target_temperature, predictions):
            cols = [x, np.full(N, T), y]
            if args.output_column_density is not None:
                cols.extend([y * args.output_column_density, y * args.output_column_density / np.log(10)])
            blocks.append(np.column_stack(cols))
        if args.output_column_density is not None:
            header += ",absorbance_ln,absorbance_decadic"
            meta["output_column_density_molecules_per_cm2"] = args.output_column_density
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out, np.concatenate(blocks), delimiter=",", header=header, comments="", fmt="%.12g")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf8")
        if args.plot:
            fonts = {f.name for f in font_manager.fontManager.ttflist}
            if "Garamond" in fonts:
                plt.rcParams["font.family"] = "Garamond"
            fig, ax = plt.subplots(figsize=(8, 4.8))
            for T, y in zip(args.target_temperature, predictions):
                ax.plot(x, y, label=f"{T:g} K")
            ax.set(xlabel=r"Wavenumber (cm$^{-1}$)",
                   ylabel=r"Absorption cross-section (cm$^{2}$/molecule)")
            ax.legend(title="Predicted temperature")
            fig.tight_layout()
            fig.savefig(out.with_suffix(".png"), dpi=200)
            plt.close(fig)
        for flag in meta["warnings"]:
            print("Warning: " + flag, file=sys.stderr)
        print(f"Wrote {out} and {meta_path}")
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
