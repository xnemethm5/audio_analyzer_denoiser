"""
noise_estimation.py – Odhad SNR, detekcia typu šumu a odhad šumového profilu.

Opravené:
  - estimate_snr() teraz počíta skutočný SNR voči šumu cez minimum statistics,
    nie dynamický rozsah. Predtým vracal dynamic range a komentár priznával
    že to nie je SNR.
  - _minimum_statistics() používa Martin-style rekurzívne vyhladenie výkonu
    pred hľadaním minima, čo dáva stabilnejší odhad (menej biased nadol).
"""

import numpy as np
import librosa
import scipy.ndimage
import scipy.stats


# ==============================================================================
# Konfigurácia
# ==============================================================================

# Nad týmto SNR sa považuje signál za čistý a denoising sa preskočí.
# 30 dB je reálne čistý hudobný signál – pod tým má denoising zmysel.
SNR_CLEAN_THRESHOLD_DB = 30.0

# DIAGNOSTICKÝ REŽIM: "normal" | "bypass" | "aggressive"
DIAG_MODE = "normal"


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
      2. Rekurzívne vyhladenie cez čas:
             P_smooth[t] = λ · P_smooth[t-1] + (1-λ) · |Y[t]|²
         s λ = 0.7. Toto redukuje vplyv krátkych transientov na následný
         odhad minima a dáva menej biased výsledok (hlavný rozdiel oproti
         pôvodnej verzii, kde sa minimum bralo priamo z raw výkonu).
      3. Running minimum cez okno `window_sec` – sledujeme noise floor.
      4. Medián výsledku cez čas + bias korekcia (Martin má teoretické
         tabuľky pre bias, my používame konštantný 1.5 – v praxi stačí).

    Args:
        y:          mono signál
        sr:         sample rate
        n_fft:      STFT veľkosť okna
        hop:        STFT hop
        window_sec: dĺžka okna pre running minimum (dlhšie = robustnejšie
                    pre stacionárny šum, kratšie = lepšie pre nonstacionárny)
        bias:       bias korekcia (1.3–2.0); nižší ako pôvodný 2.5 lebo
                    rekurzívne vyhladenie už robí časť práce

    Returns:
        noise_profile: np.ndarray (n_bins,) odhad výkonu šumu per bin
    """
    D   = librosa.stft(np.pad(y, (0, n_fft)), n_fft=n_fft, hop_length=hop)
    pwr = np.abs(D) ** 2
    n_bins, n_frames = pwr.shape

    # Rekurzívne vyhladenie – jednorozmerný IIR filter po čase
    lam       = 0.7
    smoothed  = np.empty_like(pwr)
    smoothed[:, 0] = pwr[:, 0]
    for t in range(1, n_frames):
        smoothed[:, t] = lam * smoothed[:, t - 1] + (1.0 - lam) * pwr[:, t]

    # Running minimum
    win_frames = max(int(window_sec * sr / hop), 10)
    min_pwr    = scipy.ndimage.minimum_filter1d(
        smoothed, size=win_frames, axis=1, mode="nearest"
    )

    # 1D profil = medián minima cez čas × bias
    profile = np.median(min_pwr, axis=1) * bias

    if DIAG_MODE == "aggressive":
        profile = profile * 4.0

    return profile.astype(np.float32)


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

    1. Ak existujú tiché úseky na začiatku/konci (< 15 % RMS) → silence metóda
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

    return (
        _minimum_statistics(y, sr, n_fft, hop, window_sec=window_sec, bias=bias),
        "minimum-statistics",
    )


# ==============================================================================
# SNR odhad – SKUTOČNÝ signal-to-noise ratio
# ==============================================================================

def estimate_snr(y: np.ndarray, sr: int) -> float:
    """
    Odhadne SNR voči šumu cez minimum statistics.

    Postup:
      1. STFT → celkový výkon per frame/bin
      2. Šumový výkon per bin cez minimum statistics
      3. Signálny výkon = max(celkový - šum, 0)
      4. SNR = 10·log10(Σ signal / Σ noise)

    Toto je SKUTOČNÝ SNR voči šumu, nie dynamický rozsah (predchádzajúca
    verzia počítala `signal_var / min_window_var`, čo je dynamic range).

    Returns:
        SNR v dB, klipnuté na [-10, 60]
    """
    if len(y) < 2048:
        return 40.0  # krátky klip, predpokladáme čistý

    n_fft = 2048
    hop   = 512

    # Signal STFT
    D     = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    power = np.abs(D) ** 2
    n_frames = power.shape[1]

    # Noise PSD odhad
    noise_1d     = _minimum_statistics(y, sr, n_fft, hop, window_sec=0.8, bias=1.5)
    noise_power  = noise_1d[:, np.newaxis]  # broadcast cez čas

    # Signal power = total - noise (soft subtract)
    signal_power = np.maximum(power - noise_power, 0.0)

    total_signal = float(signal_power.sum()) + 1e-10
    total_noise  = float(noise_power.sum() * n_frames) + 1e-10

    snr_db = 10.0 * np.log10(total_signal / total_noise)
    return float(np.clip(snr_db, -10.0, 60.0))


def snr_scale(global_snr_db: float) -> float:
    """
    Sigmoid transformácia SNR → škálovací faktor sily filtrovania.
    Nízky SNR → silné filtrovanie, vysoký SNR → jemné.
    """
    if DIAG_MODE == "aggressive":
        return 1.20
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

    Rozhodovacia logika (opravená):
      1. Ak impulsive_score > 0.30 → impulsive (clicks/crackle)
      2. Inak porovnanie stationary vs nonstationary cez rms_cv
         (ak je variabilita RMS medzi blokmi nízka → stacionárny šum)

    Impulsive detection: HP kurtosis + crest factor.
    Flatness A slope sa merajú len z TICHÝCH úsekov, aby hudobné basy
    neskreslili výsledok (predtým flatness bola z celého signálu, čo
    dávalo umelo nízke skóre stationary pre hudbu + stacionárny šum).
    """
    from scipy.signal import butter, filtfilt

    # --- Impulsive features ---
    nyq = sr / 2.0
    if nyq > 4500:
        b, a = butter(4, 4000 / nyq, btype="high")
        y_hp = filtfilt(b, a, y.astype(np.float64)).astype(np.float32)
    else:
        y_hp = y
    kurt = float(scipy.stats.kurtosis(y_hp))

    rms_y    = float(np.sqrt(np.mean(y ** 2))) + 1e-10
    peak_y   = float(np.max(np.abs(y)))
    crest_db = 20 * np.log10(peak_y / rms_y + 1e-10)

    crest_score    = min(max((crest_db - 18.0) / 10.0, 0.0), 1.0)
    kurtosis_score = min(max((kurt - 3.0) / 12.0, 0.0), 1.0)
    impulsive_score = max(crest_score, kurtosis_score)

    # --- Rozdelenie na tiché a hlasné úseky ---
    block   = int(sr * 0.5)
    rms_all = float(np.sqrt(np.mean(y ** 2)))

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

    # Flatness z tichých úsekov – tam dominuje šum, nie hudba.
    # Predtým sa počítala z celého signálu a pre hudbu + šum bola
    # vždy nízka (~0.05), takže stationary score bolo umelo potlačené.
    flatness_quiet = float(np.mean(librosa.feature.spectral_flatness(y=y_ref)))

    # Spektrálny sklon takisto z tichých úsekov
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

    # Odčítanie prirodzenej hudobnej variability z nonstationary skóre
    noise_cv = max(rms_cv - 0.10, 0.0)

    # Stationary skóre: flatness z tichých úsekov × (1 - variabilita)
    # Flatness biely šum ≈ 0.6–0.9, ružový ≈ 0.3–0.5, hudba ≈ 0.02–0.08.
    # Škálujeme × 3 aby stacionárny šum dal skóre blízke 1.0.
    stationary_score = min(max(flatness_quiet * 3.0, 0.0), 1.0) * \
                       (1.0 - min(noise_cv, 1.0))

    scores = {
        "impulsive":     impulsive_score,
        "stationary":    stationary_score,
        "nonstationary": min(noise_cv, 1.0),
    }

    # --- Rozhodovacia logika ---
    # Priority (zhora):
    #   1. impulsive_score > 0.30  → clicks/crackle
    #   2. nonstat_score   > 0.35  → bursty / nestacionárny šum (vietor, kroky)
    #   3. inak                    → stationary (hiss, pink, white, rumble)
    #
    # Prahy sú volené tak, aby "stationary" bol default pre stabilný šum
    # (pink, white, hiss), ktorý má nízke rms_cv – typicky < 0.30. Reálne
    # nestacionárne javy (vietor, bursty) majú rms_cv >> 0.45.
    nonstat_score = scores["nonstationary"]

    if impulsive_score > 0.30:
        dominant = "impulsive"
    elif nonstat_score > 0.35:
        dominant = "nonstationary"
    else:
        dominant = "stationary"

    scores["spectral_slope"] = slope
    scores["flatness_quiet"] = flatness_quiet
    return dominant, scores