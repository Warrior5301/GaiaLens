import numpy as np
from PIL import Image

SIZE = 256
N_BINS = 64

_WIN = np.outer(np.hanning(SIZE), np.hanning(SIZE)).astype(np.float32)
_yy, _xx = np.indices((SIZE, SIZE))
_RAD = np.hypot(_yy - SIZE // 2, _xx - SIZE // 2)
_EDGES = np.geomspace(1.0, SIZE // 2, N_BINS + 1)
_BIN_IDX = np.digitize(_RAD, _EDGES) - 1  # -1 = below r=1 (DC), N_BINS = beyond r=128
_VALID = (_BIN_IDX >= 0) & (_BIN_IDX < N_BINS)
_COUNTS = np.bincount(_BIN_IDX[_VALID], minlength=N_BINS)


def load_gray(path):
    im = Image.open(path).convert("L")
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32)


def radial_spectrum(img):
    x = img - img.mean()          # remove DC spike
    x = x * _WIN                  # Hann window: no edge artifacts
    F = np.fft.fftshift(np.fft.fft2(x))
    logp = np.log1p(np.abs(F) ** 2)
    sums = np.bincount(_BIN_IDX[_VALID], weights=logp[_VALID], minlength=N_BINS)
    prof = np.zeros(N_BINS, dtype=np.float64)
    ok = _COUNTS > 0
    prof[ok] = sums[ok] / _COUNTS[ok]
    if not ok.all():              # low log-bins can be empty (width < 1 px): interpolate
        idx = np.arange(N_BINS)
        prof[~ok] = np.interp(idx[~ok], idx[ok], prof[ok])
    return prof.astype(np.float32)


def cutoff_feature(prof):
    lo, hi = prof.min(), prof.max()
    above = np.where(prof > lo + 0.1 * (hi - lo))[0]
    return np.float32(above.max() / (N_BINS - 1)) if len(above) else np.float32(0.0)


def extract(img):
    p = radial_spectrum(img)
    return np.concatenate([p, [cutoff_feature(p)]]).astype(np.float32)


def extract_path(path):
    return extract(load_gray(path))
