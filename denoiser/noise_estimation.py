"""
noise_estimation.py – Odhad SNR, detekcia typu šumu a odhad šumového profilu.
"""

import numpy as np
import librosa
import scipy.stats


def estimate_snr(y: np.ndarray, sr: int) -> float:
    """Odhadne SNR z najtiššieho 0.5s okna."""
    window = int(sr * 0.5)
    if len(y) < window:
        window = max(1, len(y))
    step      = max(window // 2, 1)
    min_var   = float("inf")
    noise_var = float(np.var(y)) + 1e-10
    for start in range(0, max(1, len(y) - window), step):
        v = float(np.var(y[start : start + window]))
        if v < min_var:
            min_var   = v
            noise_var = max(v, 1e-10)
    signal_var = float(np.var(y)) + 1e-10
    return 10.0 * np.log10(signal_var / noise_var)


# SNR threshold vypnutý – estimate_snr meria dynamický rozsah, nie SNR voči šumu
SNR_CLEAN_THRESHOLD_DB = 999.0

# DIAGNOSTICKÝ REŽIM
DIAG_MODE = "normal"


def snr_scale(global_snr_db: float) -> float:
    """Sigmoid transformácia SNR → škálovací faktor sily filtrovania."""
    if DIAG_MODE == "aggressive":
        return 1.20
    if global_snr_db >= SNR_CLEAN_THRESHOLD_DB:
        return 0.0
    scale = 0.40 + 0.80 / (1.0 + np.exp(0.20 * (global_snr_db - 12.0)))
    return float(np.clip(scale, 0.40, 1.20))


def detect_noise_type(y: np.ndarray, sr: int) -> tuple[str, dict]:
    """
    Klasifikuje typ šumu: impulsive / stationary / nonstationary.

    Spektrálny sklon sa meria len z tichých úsekov (RMS < 40 % priemeru)
    aby hudobné basy neskreslili výsledok do záporných hodnôt.
    """
    kurt     = float(scipy.stats.kurtosis(y))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    block    = int(sr * 0.5)

    rms_blocks = [
        float(np.sqrt(np.mean(y[i : i + block] ** 2)))
        for i in range(0, len(y) - block, block)
    ]
    rms_cv   = float(np.std(rms_blocks) / (np.mean(rms_blocks) + 1e-10))
    rms_all  = float(np.sqrt(np.mean(y ** 2)))

    # Slope meráme len z tichých úsekov – hudobné basy by skreslili výsledok
    quiet_segs = [
        y[i : i + block]
        for i in range(0, len(y) - block, block)
        if np.sqrt(np.mean(y[i : i + block] ** 2)) < rms_all * 0.4
    ]
    y_ref    = np.concatenate(quiet_segs) if quiet_segs else y
    fft_mag  = np.abs(np.fft.rfft(y_ref))
    fft_freq = np.fft.rfftfreq(len(y_ref), 1.0 / sr)
    valid    = fft_freq > 200
    if np.sum(valid) > 10:
        slope = float(np.polyfit(
            np.log10(fft_freq[valid] + 1e-10),
            np.log10(fft_mag[valid]  + 1e-10), 1
        )[0])
    else:
        slope = 0.0

    # Hudba prirodzene mení hlasitosť → rms_cv 0.2–0.4 je normálne
    # Skutočný nestacionárny šum má rms_cv > 0.5
    noise_cv = max(rms_cv - 0.25, 0.0)

    scores = {
        "impulsive":     min(max((kurt - 5.0) / 20.0, 0.0), 1.0),
        "stationary":    min(max(flatness * 2.0, 0.0), 1.0) * (1.0 - min(noise_cv * 2, 1.0)),
        "nonstationary": min(noise_cv * 2, 1.0),
    }
    dominant = max(scores, key=scores.get)
    scores["spectral_slope"] = slope
    return dominant, scores


def estimate_noise_profile(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    window_sec: float = 0.4,
    bias: float = 2.5,
) -> tuple[np.ndarray, str]:
    """
    Odhadne šumový profil (per-bin výkon).

    1. Ak existujú tiché úseky (< 15 % RMS) → silence metóda
    2. Inak → minimum statistics s window_sec a bias z profilu
    """
    win        = int(sr * 0.5)
    median_rms = float(np.sqrt(np.mean(y ** 2))) + 1e-10
    segments   = []

    for region in [y[:win], y[-win:]]:
        if len(region) >= n_fft:
            if float(np.sqrt(np.mean(region ** 2))) < median_rms * 0.15:
                segments.append(region)

    if segments:
        noise_ref = np.concatenate(segments)
        D_noise   = librosa.stft(noise_ref, n_fft=n_fft, hop_length=hop)
        profile   = np.mean(np.abs(D_noise) ** 2, axis=1)
        if DIAG_MODE == "aggressive":
            return profile * 5.0, "silence-boosted"
        return profile, "silence"

    return _minimum_statistics(y, sr, n_fft, hop, window_sec=window_sec, bias=bias), "minimum-statistics"


def _minimum_statistics(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    window_sec: float = 0.4,
    bias: float = 2.5,
) -> np.ndarray:
    """
    Minimum statistics odhad šumového profilu.
    Kĺzavé minimum každého STFT binu → medián → bias korekcia.
    window_sec a bias sú dynamické – prichádzajú z DenoiseProfile.
    """
    D   = librosa.stft(np.pad(y, (0, n_fft)), n_fft=n_fft, hop_length=hop)
    pwr = np.abs(D) ** 2

    win_frames   = max(int(window_sec * sr / hop), 3)
    n_bins, n_frames = pwr.shape
    min_pwr      = np.empty_like(pwr)

    for t in range(n_frames):
        lo = max(0, t - win_frames // 2)
        hi = min(n_frames, t + win_frames // 2 + 1)
        min_pwr[:, t] = np.min(pwr[:, lo:hi], axis=1)

    profile = np.median(min_pwr, axis=1) * bias

    if DIAG_MODE == "aggressive":
        profile = profile * 4.0

    return profile.astype(np.float32)