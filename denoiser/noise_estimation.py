"""
noise_estimation.py – Odhad SNR, detekcia typu šumu a odhad šumového profilu.
"""

import numpy as np
import librosa
import scipy.ndimage
import scipy.signal
import scipy.stats


# Nad týmto SNR sa považuje signál za čistý a denoising sa preskočí.
SNR_CLEAN_THRESHOLD_DB = 30.0


# ==============================================================================
# Minimum statistics – Martin-style
# ==============================================================================

def _minimum_statistics(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    window_sec: float = 0.8,
    bias: float = 1.5,
) -> np.ndarray:
    """
    Martin 2001 minimum statistics odhad šumového PSD (zjednodušená verzia).

    Postup:
      1. STFT → |Y[k,t]|²
      2. Rekurzívne vyhladenie P_smooth[t] = λ·P_smooth[t-1] + (1-λ)·|Y[t]|²
         s λ = 0.7. Redukuje vplyv transientov na odhad minima.
      3. Running minimum cez okno `window_sec` – sledujeme noise floor.
      4. Medián výsledku cez čas + bias korekcia.
    """
    D   = librosa.stft(np.pad(y, (0, n_fft)), n_fft=n_fft, hop_length=hop)
    pwr = np.abs(D) ** 2
    n_bins, n_frames = pwr.shape

    # Rekurzívne vyhladenie cez čas
    lam      = 0.7
    smoothed = np.empty_like(pwr)
    smoothed[:, 0] = pwr[:, 0]
    for t in range(1, n_frames):
        smoothed[:, t] = lam * smoothed[:, t - 1] + (1.0 - lam) * pwr[:, t]

    win_frames = max(int(window_sec * sr / hop), 10)
    min_pwr    = scipy.ndimage.minimum_filter1d(
        smoothed, size=win_frames, axis=1, mode="nearest"
    )

    return (np.median(min_pwr, axis=1) * bias).astype(np.float32)


def estimate_noise_profile(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    window_sec: float = 0.8,
    bias: float = 1.5,
) -> tuple[np.ndarray, str]:
    """
    Odhadne šumový profil (per-bin výkon).

    1. Ak sú tiché úseky na začiatku/konci (< 15 % RMS) → silence metóda
    2. Inak → minimum statistics
    """
    win        = int(sr * 0.5)
    median_rms = float(np.sqrt(np.mean(y ** 2))) + 1e-10

    segments = [
        region for region in [y[:win], y[-win:]]
        if len(region) >= n_fft
        and float(np.sqrt(np.mean(region ** 2))) < median_rms * 0.15
    ]

    if segments:
        D_noise = librosa.stft(np.concatenate(segments), n_fft=n_fft, hop_length=hop)
        return np.mean(np.abs(D_noise) ** 2, axis=1), "silence"

    return (
        _minimum_statistics(y, sr, n_fft, hop, window_sec=window_sec, bias=bias),
        "minimum-statistics",
    )


# ==============================================================================
# SNR odhad – skutočný signal-to-noise ratio cez minimum statistics
# ==============================================================================

def estimate_snr(y: np.ndarray, sr: int) -> float:
    """
    Odhadne SNR voči šumu cez minimum statistics.

    Šumový výkon per bin cez MS, signálny = max(total - noise, 0),
    SNR = 10·log10(Σ signal / Σ noise). Výsledok klipnutý na [-10, 60] dB.
    """
    if len(y) < 2048:
        return 40.0

    n_fft, hop = 2048, 512
    D          = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    power      = np.abs(D) ** 2
    n_frames   = power.shape[1]

    noise_1d     = _minimum_statistics(y, sr, n_fft, hop, window_sec=0.8, bias=1.5)
    noise_power  = noise_1d[:, np.newaxis]
    signal_power = np.maximum(power - noise_power, 0.0)

    total_signal = float(signal_power.sum()) + 1e-10
    total_noise  = float(noise_power.sum() * n_frames) + 1e-10

    return float(np.clip(10.0 * np.log10(total_signal / total_noise), -10.0, 60.0))


def snr_scale(global_snr_db: float) -> float:
    """Sigmoid transformácia SNR → škálovací faktor sily filtrovania."""
    if global_snr_db >= SNR_CLEAN_THRESHOLD_DB:
        return 0.0
    scale = 0.40 + 0.80 / (1.0 + np.exp(0.20 * (global_snr_db - 12.0)))
    return float(np.clip(scale, 0.40, 1.20))


# ==============================================================================
# Detekcia typu šumu
# ==============================================================================

def detect_noise_type(y: np.ndarray, sr: int) -> tuple[str, dict]:
    """
    Klasifikuje typ šumu: impulsive / stationary / nonstationary.

    Rozhodovacia logika:
      1. impulsive_score > 0.30 → impulsive (clicks/crackle)
      2. nonstat_score   > 0.35 → nonstationary (bursts, vietor)
      3. inak                   → stationary (hiss, pink, white, rumble)

    Flatness a slope sa merajú len z tichých úsekov, aby hudobné basy
    neskreslili výsledok.
    """
    # --- Impulsive features ---
    nyq = sr / 2.0
    if nyq > 4500:
        b, a = scipy.signal.butter(4, 4000 / nyq, btype="high")
        y_hp = scipy.signal.filtfilt(b, a, y.astype(np.float64)).astype(np.float32)
    else:
        y_hp = y
    kurt = float(scipy.stats.kurtosis(y_hp))

    rms_y    = float(np.sqrt(np.mean(y ** 2))) + 1e-10
    peak_y   = float(np.max(np.abs(y)))
    crest_db = 20.0 * np.log10(peak_y / rms_y + 1e-10)

    crest_score    = min(max((crest_db - 18.0) / 10.0, 0.0), 1.0)
    kurtosis_score = min(max((kurt - 3.0) / 12.0, 0.0), 1.0)
    impulsive_score = max(crest_score, kurtosis_score)

    # --- Rozdelenie na tiché a hlasné úseky ---
    block      = int(sr * 0.5)
    rms_all    = float(np.sqrt(np.mean(y ** 2)))
    rms_blocks = [
        float(np.sqrt(np.mean(y[i : i + block] ** 2)))
        for i in range(0, len(y) - block, block)
    ]
    rms_cv = float(np.std(rms_blocks) / (np.mean(rms_blocks) + 1e-10))

    quiet_segs = [
        y[i : i + block]
        for i in range(0, len(y) - block, block)
        if np.sqrt(np.mean(y[i : i + block] ** 2)) < rms_all * 0.4
    ]
    y_ref = np.concatenate(quiet_segs) if quiet_segs else y

    # Spektrálny sklon z tichých úsekov
    fft_mag  = np.abs(np.fft.rfft(y_ref))
    fft_freq = np.fft.rfftfreq(len(y_ref), 1.0 / sr)
    valid    = fft_freq > 200
    if np.sum(valid) > 10:
        slope = float(np.polyfit(
            np.log10(fft_freq[valid] + 1e-10),
            np.log10(fft_mag[valid]  + 1e-10), 1,
        )[0])
    else:
        slope = 0.0

    # Nonstationary score po odčítaní hudobnej variability
    nonstat_score = min(max(rms_cv - 0.10, 0.0), 1.0)

    # Rozhodovanie cez absolútne prahy
    if impulsive_score > 0.30:
        dominant = "impulsive"
    elif nonstat_score > 0.35:
        dominant = "nonstationary"
    else:
        dominant = "stationary"

    scores = {
        "impulsive":      impulsive_score,
        "nonstationary":  nonstat_score,
        "spectral_slope": slope,
    }
    return dominant, scores
