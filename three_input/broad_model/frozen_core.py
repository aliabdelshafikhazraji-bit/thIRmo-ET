"""thIRmo-ET: temperature transfer of molecular infrared band spectra.

Inference uses numeric tree arrays only; no pickle or scikit-learn is loaded.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d

VERSION = "2.5.1"
N = 512
U = np.linspace(0, 1, N)
FWHM = 2.0
SIGMA_CM1 = FWHM / np.sqrt(8 * np.log(2))
PNNL_COLUMN = 1e-6 * 101325 / (1.380649e-23 * 296) * 1e-6 * 100
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "thirmo_et_final.npz"


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
    """Identical 2/3-input features to training; an explicit one-input branch."""
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
