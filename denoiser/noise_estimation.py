"""
noise_estimation.py – Odhad SNR, detekcia typu šumu a odhad šumového profilu.

Obsahuje tri hlavné funkcie:
  estimate_snr()          – globálny SNR z najtiššieho 0.5s úseku
  snr_scale()             – sigmoid transformácia SNR → sila filtrovania
  detect_noise_type()     – klasifikácia šumu (impulzný / stacionárny / nestacionárny)
  estimate_noise_profile()– per-bin šumový profil pre STFT spracovanie
"""

import numpy as np
import librosa
import scipy.stats


# ==============================================================================
# SNR odhad
# ==============================================================================

def estimate_snr(y: np.ndarray, sr: int) -> float:
    """
    Odhadne SNR porovnaním celkovej variance signálu
    s variancou najtiššieho 0.5s okna.

    Args:
        y:  mono audio pole (float32)
        sr: vzorkovacia frekvencia

    Returns:
        SNR v dB (čím nižšie, tým viac šumu)
    """
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


# ==============================================================================
# SNR-adaptívna modulácia sily filtrovania
# ==============================================================================

# SNR threshold je vypnutý (999 dB) – estimate_snr meria dynamický rozsah hudby,
# nie skutočný SNR voči šumu. Biely šum 0.02 zmení hodnotu len o ~0.8 dB,
# threshold by teda nikdy nespoľahlivo nedetekoval čistú nahrávku.
# Čisté signály sa spracujú korektne – minimum statistics vráti nízky
# noise_profile a MMSE-LSA gain bude prirodzene blízko 1.0.
SNR_CLEAN_THRESHOLD_DB = 999.0

# DIAGNOSTICKÝ REŽIM – nastav na "aggressive", "normal" alebo "bypass"
#   aggressive  → maximálna sila filtrovania, ignoruje žánrový profil
#   normal      → štandardná pipeline
#   bypass      → vráti signál bez zmeny (overenie že problém je v denoiserI)
DIAG_MODE = "normal"


def snr_scale(global_snr_db: float) -> float:
    """
    Prevádza globálny SNR na škálovací faktor sily filtrovania.

    V diagnostickom režime "aggressive" vráti vždy maximálnu hodnotu 1.20.
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
    Klasifikuje dominantný typ šumu v signáli.

    Metódy:
      - Kurtosis  → impulzný šum (praskanie, clicky)
      - Spectral flatness → stacionárny šum (biely/ružový)
      - RMS coefficient of variation → nestacionárny šum (prostredie, vietor)

    Args:
        y:  mono audio pole
        sr: vzorkovacia frekvencia

    Returns:
        (typ: str, skóre: dict)  kde typ ∈ {"impulsive", "stationary", "nonstationary"}
    """
    kurt     = float(scipy.stats.kurtosis(y))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    block    = int(sr * 0.5)

    rms_blocks = [
        float(np.sqrt(np.mean(y[i : i + block] ** 2)))
        for i in range(0, len(y) - block, block)
    ]
    rms_cv = float(np.std(rms_blocks) / (np.mean(rms_blocks) + 1e-10))

    # Spektrálny sklon – rozlíši biely šum (plochý) od ružového (klesajúci)
    # Ružový šum má záporný sklon ~-10 dB/dekádu, biely ~0 dB/dekádu
    fft_mag  = np.abs(np.fft.rfft(y))
    fft_freq = np.fft.rfftfreq(len(y), 1.0 / sr)
    valid    = fft_freq > 100  # ignoruj DC a sub-bas
    if np.sum(valid) > 10:
        log_freq = np.log10(fft_freq[valid] + 1e-10)
        log_mag  = np.log10(fft_mag[valid]  + 1e-10)
        slope    = float(np.polyfit(log_freq, log_mag, 1)[0])
    else:
        slope = 0.0
    # slope < -0.8  → ružový/hnedý šum (veľa basov)
    # slope ~ 0     → biely šum (rovnomerný)
    # slope > 0.5   → high-frequency hum

    scores = {
        "impulsive":     min(max((kurt - 5.0) / 20.0, 0.0), 1.0),
        "stationary":    min(max(flatness * 2.0, 0.0), 1.0) * (1.0 - min(rms_cv, 1.0)),
        "nonstationary": min(rms_cv, 1.0),
    }
    dominant = max(scores, key=scores.get)
    # Uložíme spektrálny sklon do scores pre použitie v profiles.py
    scores["spectral_slope"] = slope
    return dominant, scores


# ==============================================================================
# Odhad šumového profilu (per-bin power)
# ==============================================================================

def estimate_noise_profile(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
) -> tuple[np.ndarray, str]:
    """
    Odhadne šumový profil ako vektor výkonu na každom STFT bine.

    Stratégia:
      1. Ak sú na začiatku/konci nahrávky tiché úseky (< 15 % RMS),
         použijú sa ako referencia – "silence" metóda.
      2. Inak sa použije percentilový odhad z celého spectrogramu:
         - Nízke/stredné frekvencie: 35. percentil
         - Vysoké frekvencie (>6 kHz): 25. percentil × 0.75
         Nižší percentil pre výšky zabraňuje nadhodnoteniu šumu
         kde je hudobná energia prirodzene nižšia.

    Args:
        y:     mono audio pole
        sr:    vzorkovacia frekvencia
        n_fft: veľkosť FFT okna
        hop:   hop length

    Returns:
        (noise_profile_1d: np.ndarray shape (n_fft//2+1,), metóda: str)
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

    return _minimum_statistics(y, sr, n_fft, hop), "minimum-statistics"


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

    Princíp: šum je prítomný neustále, hudba nie. Ak sledujeme minimum
    spektrálneho výkonu každého binu cez krátke časové okno, dostaneme
    odhad šumu – lebo v každom okne je aspoň jeden frame kde hudba
    stíchla ale šum zostal.

    Na rozdiel od percentilu: nepočítame zo všetkých frameov naraz
    ale sledujeme kĺzavé minimum – odolnejší voči hudobným peakom.

    Args:
        y:          mono audio pole
        sr:         vzorkovacia frekvencia
        n_fft:      FFT veľkosť
        hop:        hop length
        window_sec: dĺžka okna pre minimum (0.3–0.5s)
        bias:       korekčný faktor – minimum statistics systematicky
                    podhodnocuje šum, bias ho kompenzuje (1.3–2.0)

    Returns:
        noise_profile: np.ndarray (n_fft//2+1,)
    """
    D   = librosa.stft(np.pad(y, (0, n_fft)), n_fft=n_fft, hop_length=hop)
    pwr = np.abs(D) ** 2                            # (n_bins, n_frames)

    win_frames = max(int(window_sec * sr / hop), 3)

    # Kĺzavé minimum po frameoch pre každý bin
    n_bins, n_frames = pwr.shape
    min_pwr = np.empty_like(pwr)
    for t in range(n_frames):
        lo = max(0, t - win_frames // 2)
        hi = min(n_frames, t + win_frames // 2 + 1)
        min_pwr[:, t] = np.min(pwr[:, lo:hi], axis=1)

    # Mediánom kĺzavých miním cez čas dostaneme robustný odhad šumu
    profile = np.median(min_pwr, axis=1)

    # Bias korekcia – minimum statistics podhodnocuje o ~30–50%
    profile = profile * bias

    if DIAG_MODE == "aggressive":
        profile = profile * 4.0

    return profile.astype(np.float32)