"""
filters.py – DSP filtre a ochranné moduly pre denoising pipeline.

Obsahuje:
  remove_impulses()      – mediánový filter pre impulzný šum (praskanie)
  harmonic_mask()        – ochrana harmonických (pyin F0 detektor)
  psychoacoustic_floor() – ATH + Bark maskovanie ako dolný limit masky
  detect_transients()    – spectral flux detektor transientov
  smooth_mask_time()     – jemné 1D časové vyhladzovanie STFT masky
  kalman_denoise()       – Kalmanov filter pre časové vyhladzenie signálu
  maybe_gate()           – Pedalboard NoiseGate pri veľmi nízkom SNR
"""

import warnings
import numpy as np
import librosa
import scipy.ndimage
import scipy.signal

from .profiles import DenoiseProfile

try:
    from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, NoiseGate
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False


# ==============================================================================
# Mediánový filter – impulzný šum (praskanie, clicky)
# ==============================================================================

def remove_impulses(y: np.ndarray, threshold_sigma: float = 3.5) -> np.ndarray:
    """
    Nahradí impulzné špičky (praskanie, clicky) mediánovou hodnotou okolia.

    Vzorky s amplitúdou > threshold_sigma × std sa považujú za impulzy
    a nahradia sa hodnotou mediánového filtra s oknom 7 vzoriek.

    Args:
        y:               mono audio pole (float32)
        threshold_sigma: prah detekcie v násobkoch std (default 3.5)

    Returns:
        y_out: vyčistený signál (float32)
    """
    std    = float(np.std(y))
    median = scipy.signal.medfilt(y, kernel_size=7)
    spikes = np.abs(y) > threshold_sigma * std
    y_out  = y.copy()
    y_out[spikes] = median[spikes]
    return y_out.astype(np.float32)


# ==============================================================================
# Harmonická ochrana (pyin F0 detektor)
# ==============================================================================

def harmonic_mask(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    protection: float = 0.35,
) -> np.ndarray:
    """
    Generuje ochrannú masku pre harmonické zložky signálu.

    Používa pyin F0 detektor na lokalizáciu základnej frekvencie
    a jej prvých 15 harmoník. Na každej harmonike sa maska zvýši
    o faktor (1.0 + protection) – tieto biny sa budú filtrovať menej.

    Mierne nastavenie (protection=0.35) zabraňuje oslabeniu hudobného tónu
    bez toho, aby blokoval filtrovanie šumu medzi harmonickými.

    Args:
        y:          mono audio pole
        sr:         vzorkovacia frekvencia
        n_fft:      FFT veľkosť okna
        hop:        hop length
        protection: multiplikátor ochrany harmoník (0.2–0.5 odporúčané)

    Returns:
        hmask: np.ndarray (n_bins, n_frames) s hodnotami >= 1.0
    """
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Trying to estimate tuning from empty frequency set",
                category=UserWarning,
            )
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
                hop_length=hop,
                frame_length=n_fft,
            )
    except Exception:
        return np.ones((n_fft // 2 + 1, 1))

    n_frames = len(f0)
    hmask    = np.ones((len(freqs), n_frames), dtype=np.float32)

    for t in range(n_frames):
        if voiced_flag[t] and f0[t] > 0:
            for h in range(1, 16):
                f_harm = f0[t] * h
                if f_harm > sr / 2:
                    break
                idx = int(np.argmin(np.abs(freqs - f_harm)))
                lo  = max(0, idx - 2)
                hi  = min(len(freqs), idx + 3)
                hmask[lo:hi, t] = 1.0 + protection

    return hmask


# ==============================================================================
# Psychoakustický floor (ISO 226 ATH + Bark maskovanie)
# ==============================================================================

def _bark(f: np.ndarray) -> np.ndarray:
    """Prevod Hz → Bark škálu (Zwicker)."""
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def psychoacoustic_floor(
    freqs: np.ndarray,
    power: np.ndarray,
    floor_min: float = 0.06,
    floor_max: float = 0.12,
) -> np.ndarray:
    """
    Vypočíta psychoakustický dolný limit (floor) pre STFT masku.

    Kombinuje dva modely:
      - ATH (ISO 226): absolútny prah sluchu – biny pod prahom sluchu
        nepotrebujú vysokú presnos filtrovania
      - Bark simultánne maskovanie: hlasné frekvencie maskujú susedné biny
        v rozsahu ±1.5 Bark; maskované biny môžu mať vyšší floor

    Výsledok: floor nikdy neklesne pod floor_min (zachová sustain),
    nikdy nepresiahne floor_max (filter zostáva aktívny).

    Args:
        freqs:     frekvencie STFT binov (Hz)
        power:     spektrálny výkon (n_bins, n_frames)
        floor_min: dolná hranica flooru (0.06–0.08)
        floor_max: horná hranica flooru (0.10–0.15)

    Returns:
        psych_floor: np.ndarray (n_bins,) hodnoty v [floor_min, floor_max]
    """
    f   = np.maximum(freqs, 20.0)
    # ISO 226 aproximácia ATH
    ath = (
        3.64 * (f / 1000.0) ** -0.8
        - 6.5 * np.exp(-0.6 * (f / 1000.0 - 3.3) ** 2)
        + 1e-3 * (f / 1000.0) ** 4
    )
    ath      = np.clip(ath, 0.0, 60.0)
    ath_norm = (ath - ath.min()) / (ath.max() - ath.min() + 1e-10)

    # Bark simultánne maskovanie (±1.5 Bark okno)
    bark_f   = _bark(f)
    mean_pdb = 10.0 * np.log10(np.mean(power, axis=1) + 1e-10)
    mask_thr = np.zeros(len(freqs))
    for i, b_i in enumerate(bark_f):
        nb = np.abs(bark_f - b_i) < 1.5
        if np.any(nb):
            mask_thr[i] = np.max(mean_pdb[nb]) - 18.0

    mask_norm = np.clip(
        (mask_thr - mask_thr.min()) / (mask_thr.max() - mask_thr.min() + 1e-10),
        0.0, 1.0,
    )

    combined    = np.maximum(ath_norm, mask_norm)
    psych_floor = floor_max - combined * (floor_max - floor_min)
    return psych_floor.astype(np.float32)


# ==============================================================================
# Transient protection – spectral flux detektor
# ==============================================================================

def detect_transients(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    percentile: int = 88,
) -> np.ndarray:
    """
    Detekuje transienty (bicí, útoky) cez spectral flux.

    Spectral flux = suma kladných zmien výkonu medzi po sebe idúcimi framami.
    Framy nad daným percentilom sa označia ako transienty
    a ich okolie (2 framy) sa rozšíri binárnou dilatáciou.

    Args:
        y:          mono audio pole
        sr:         vzorkovacia frekvencia
        n_fft:      FFT veľkosť
        hop:        hop length
        percentile: prah detekcie (88 = top 12 % framov)

    Returns:
        is_tr: bool np.ndarray (n_frames,) True = transient frame
    """
    D    = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    flux = np.sum(np.maximum(np.diff(np.abs(D) ** 2, axis=1), 0), axis=0)
    flux = np.concatenate([[0], flux])
    thr  = np.percentile(flux, percentile)
    return scipy.ndimage.binary_dilation(flux > thr, iterations=2)


# ==============================================================================
# Jemné 1D časové vyhladzovanie STFT masky
# ==============================================================================

def smooth_mask_time(mask: np.ndarray, size: int = 5) -> np.ndarray:
    """
    Vyhladzuje STFT masku len v časovej osi (nie vo frekvencii).

    Redukuje musical noise (náhodné buľvy v spektrograme) bez straty
    frekvenčnej selektivity. Väčší size = plynulejšia maska, menej artifacts.

    Args:
        mask: STFT maska (n_bins, n_frames) v rozsahu [0, 1]
        size: veľkosť vyhladzovacieho okna (nepárne odporúčané)

    Returns:
        vyhladená maska, orezaná na [0, 1]
    """
    return np.clip(
        scipy.ndimage.uniform_filter1d(mask, size=size, axis=1),
        0.0, 1.0,
    )


# ==============================================================================
# Kalmanov filter – jemné časové vyhladzenie signálu
# ==============================================================================

def kalman_denoise(
    y: np.ndarray,
    process_var: float = 1e-5,
    measurement_var: float = 0.02,
) -> np.ndarray:
    """
    Jednorozmerný Kalmanov filter na časové vyhladzenie audio signálu.

    Konzervatívne nastavenie (vyšší measurement_var = jemnejšie vyhladenie):
      - process_var = 1e-5  → signál sa mení pomaly (konzistentný model)
      - measurement_var = 0.02 → merania sú relatívne dôveryhodné

    Efekt: redukuje náhodné fluktuácie pri zachovaní sustain a ambiencii.
    Nie je agresívny – slúži ako záverečné jemné vyhladenie po MMSE-LSA.

    Args:
        y:               mono audio pole
        process_var:     variancia procesného šumu Q (model šum)
        measurement_var: variancia merania R (dôvera v meranie)

    Returns:
        vyhladzený signál (float32)
    """
    n     = len(y)
    x_est = float(y[0])
    P_est = 1.0
    out   = np.zeros(n, dtype=np.float32)

    for t in range(n):
        # Predikcia
        P_pred = P_est + process_var
        # Kalman gain
        K      = P_pred / (P_pred + measurement_var)
        # Aktualizácia
        x_est  = x_est + K * (float(y[t]) - x_est)
        P_est  = (1.0 - K) * P_pred
        out[t] = x_est

    return out


# ==============================================================================
# Pedalboard NoiseGate – iba pri veľmi nízkym SNR
# ==============================================================================

def maybe_gate(
    y: np.ndarray,
    sr: int,
    profile: DenoiseProfile,
    snr_db: float,
    threshold: float = 7.0,
) -> tuple[np.ndarray, bool]:
    """
    Aplikuje Pedalboard NoiseGate + HP/LP filter len ak SNR < threshold.

    Pedalboard je voliteľná závislosť – ak nie je nainštalovaná alebo
    SNR je dostatočné, funkcia vráti pôvodný signál bez zmeny.

    Pipeline (ak sa aplikuje):
      HighpassFilter(cutoff = profile.highpass_hz)
      LowpassFilter(cutoff  = min(sr/2 - 1000, 21000) Hz)
      NoiseGate(threshold, ratio, attack, release z profilu)

    Args:
        y:         audio pole (mono alebo stereo)
        sr:        vzorkovacia frekvencia
        profile:   žánrový profil s gate parametrami
        snr_db:    aktuálny SNR (po MMSE-LSA spracovaní)
        threshold: SNR prah pod ktorým sa gate aktivuje (default 7 dB)

    Returns:
        (y_clean, gate_used: bool)
    """
    if not PEDALBOARD_AVAILABLE or snr_db >= threshold:
        return y, False

    nyquist    = sr / 2.0
    lowpass_hz = min(nyquist - 1000.0, 21000.0)

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=float(profile.highpass_hz)),
        LowpassFilter(cutoff_frequency_hz=float(lowpass_hz)),
        NoiseGate(
            threshold_db=float(profile.gate_threshold_db),
            ratio=float(profile.gate_ratio),
            attack_ms=float(profile.gate_attack_ms),
            release_ms=float(profile.gate_release_ms),
        ),
    ])

    y_pb    = y[np.newaxis, :] if y.ndim == 1 else y.T
    y_clean = board(y_pb, sr)
    result  = y_clean[0] if y_clean.shape[0] == 1 else y_clean.T
    return result, True