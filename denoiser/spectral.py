"""
spectral.py – STFT spracovanie a multi-band MMSE-LSA denoising.
"""

import numpy as np
import librosa
import scipy.ndimage

from .profiles import DenoiseProfile
from .noise_estimation import estimate_noise_profile
from .filters import (
    harmonic_mask,
    psychoacoustic_floor,
    detect_transients,
    smooth_mask_time,
)


def _expint(x: np.ndarray) -> np.ndarray:
    """Exponenciálny integrál E1(x) – potrebný pre MMSE-LSA gain."""
    x      = np.maximum(x, 1e-10)
    small  = -np.log(x) - 0.5772 + x - x ** 2 / 4.0 + x ** 3 / 18.0
    safe_x = np.minimum(x, 500.0)
    with np.errstate(over="ignore", invalid="ignore"):
        large = np.exp(-safe_x) / safe_x * (
            1.0 - 1.0 / safe_x + 2.0 / safe_x ** 2 - 6.0 / safe_x ** 3
        )
    large  = np.where(np.isfinite(large), large, 0.0)
    result = np.where(x < 1.0, small, large)
    result = np.where(x > 500.0, 0.0, result)
    return np.maximum(result, -50.0)


def mmse_lsa_gain(
    power: np.ndarray,
    noise_power: np.ndarray,
    alpha_ns: float = 0.98,
) -> np.ndarray:
    """
    MMSE-LSA gain estimátor (Ephraim-Malah 1984) s decision-directed
    a-priori SNR updatom.

    DD update: xi = alpha * G_prev² * |Y_prev|² / N + (1-alpha) * max(gamma-1, 0)
    LSA gain:  G = xi/(1+xi) * exp(0.5 * E1(nu)),  nu = xi/(1+xi) * gamma
    """
    n_bins, n_frames = power.shape

    gain_out   = np.empty_like(power)
    gain_prev  = np.ones(n_bins)
    power_prev = power[:, 0].copy()

    for t in range(n_frames):
        n_t     = noise_power[:, t] + 1e-10
        gamma_t = power[:, t] / n_t  # a-posteriori SNR

        xi_dd = alpha_ns * (gain_prev ** 2) * power_prev / n_t
        xi_ml = (1.0 - alpha_ns) * np.maximum(gamma_t - 1.0, 0.0)
        xi_t  = np.maximum(xi_dd + xi_ml, 1e-10)

        nu_t   = np.maximum(xi_t / (1.0 + xi_t) * gamma_t, 1e-10)
        gain_t = np.clip(xi_t / (1.0 + xi_t) * np.exp(0.5 * _expint(nu_t)), 0.0, 1.0)

        gain_out[:, t] = gain_t
        gain_prev      = gain_t
        power_prev     = power[:, t]

    return gain_out


BANDS: list[tuple[int, int, str]] = [
    (0,     250,   "sub_bass"),
    (250,   2000,  "bass_mid"),
    (2000,  6000,  "upper_mid"),
    (6000,  12000, "highs"),
    (12000, 96000, "air"),
]


def apply_multiband_mmse(
    power: np.ndarray,
    noise_power: np.ndarray,
    freqs: np.ndarray,
    profile: DenoiseProfile,
    scale: float,
) -> np.ndarray:
    """
    Multi-band MMSE-LSA s exponent blendom masky:
        mask = gain ** (strength * scale)

    Strength < 1 → mäkšie filtrovanie, > 1 → agresívnejšie.
    """
    strengths = {
        "sub_bass":  profile.strength_low  * scale,
        "bass_mid":  profile.strength_low  * scale,
        "upper_mid": profile.strength_mid  * scale,
        "highs":     profile.strength_high * scale,
        "air":       profile.strength_high * scale * 0.85,
    }
    strengths = {k: float(np.clip(v, 0.20, 1.50)) for k, v in strengths.items()}

    mask = np.ones_like(power)

    for f_lo, f_hi, band_name in BANDS:
        idx = np.where((freqs >= f_lo) & (freqs < f_hi))[0]
        if len(idx) == 0:
            continue
        lsa_gain = mmse_lsa_gain(power[idx, :], noise_power[idx, :],
                                 alpha_ns=profile.alpha_ns)
        effective = np.power(np.maximum(lsa_gain, 1e-10), strengths[band_name])
        mask[idx, :] = np.clip(effective, profile.mask_floor, 1.0)

    return mask


def spectral_pass(
    y: np.ndarray,
    sr: int,
    profile: DenoiseProfile,
    snr_scale_factor: float,
) -> tuple[np.ndarray, str]:
    """
    Jeden kompletný STFT denoising prechod.

    Tok:
      STFT → noise profile → multi-band MMSE-LSA → harmonic protection (HPSS)
      → psychoacoustic floor → transient protection → temporal smoothing → ISTFT

    Returns:
        (y_clean, noise_source_label)
    """
    # n_fft dynamicky podľa profilu (n_fft_ms v ms), zarovnané na power-of-2
    fft_ms = profile.n_fft_ms / 1000.0
    n_fft  = max(512, min(int(2 ** np.round(np.log2(sr * fft_ms))), 4096))
    hop    = n_fft // 4

    original_len = len(y)
    y_pad        = np.pad(y.astype(np.float32), (0, n_fft))

    D         = librosa.stft(y_pad, n_fft=n_fft, hop_length=hop)
    magnitude = np.abs(D)
    phase     = np.angle(D)
    power     = magnitude ** 2
    freqs     = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_frames  = power.shape[1]

    # Noise profile
    noise_profile_1d, noise_src = estimate_noise_profile(
        y, sr, n_fft, hop,
        window_sec=profile.window_sec,
        bias=profile.bias,
    )
    noise_power = noise_profile_1d[:, np.newaxis] * np.ones((1, n_frames))

    # Multi-band MMSE-LSA maska
    mask = apply_multiband_mmse(power, noise_power, freqs, profile, snr_scale_factor)

    # Harmonic protection cez HPSS
    hmask = harmonic_mask(y, sr, n_fft, hop, protection=0.35)
    if hmask.shape[1] != n_frames:
        hmask = scipy.ndimage.zoom(
            hmask, (1, n_frames / max(hmask.shape[1], 1)), order=1
        )
        hmask = hmask[:power.shape[0], :n_frames]
    mask = np.minimum(mask * np.clip(hmask, 1.0, 1.35), 1.0)

    # Psychoakustický floor
    psych_floor = psychoacoustic_floor(freqs, power, floor_min=0.06, floor_max=0.12)
    mask        = np.clip(mask, psych_floor[:, np.newaxis], 1.0)

    # Transient protection
    is_tr = detect_transients(y, sr, n_fft, hop)
    if len(is_tr) > n_frames:
        is_tr = is_tr[:n_frames]
    elif len(is_tr) < n_frames:
        is_tr = np.pad(is_tr, (0, n_frames - len(is_tr)))

    if np.any(is_tr):
        tr_frames   = power[:, is_tr]
        mean_power  = np.mean(power, axis=1, keepdims=True) + 1e-10
        active_bins = (tr_frames / mean_power) > 3.0
        tr_mask     = mask[:, is_tr].copy()
        tr_mask[active_bins] = np.maximum(tr_mask[active_bins], 0.95)
        mask[:, is_tr] = tr_mask

    # Temporal smoothing mimo transientných rámcov
    smooth = smooth_mask_time(mask, size=5)
    mask   = np.where(is_tr[np.newaxis, :], mask, smooth)
    mask   = np.clip(mask, 0.0, 1.0)

    # ISTFT
    D_clean = mask * magnitude * np.exp(1j * phase)
    y_clean = librosa.istft(D_clean, hop_length=hop, n_fft=n_fft, length=len(y_pad))

    return y_clean[:original_len].astype(np.float32), noise_src
