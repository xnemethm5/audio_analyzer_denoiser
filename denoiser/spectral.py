"""
spectral.py – STFT spracovanie a multi-band MMSE-LSA denoising.

Obsahuje:
  _expint()              – aproximácia exponenciálneho integrálu E₁(x)
  mmse_lsa_gain()        – Ephraim-Malah (1984) MMSE-LSA gain estimátor
  BANDS                  – definícia frekvenčných pásiem
  apply_multiband_mmse() – aplikuje MMSE-LSA zvlášť pre každé pásmo
  spectral_pass()        – jeden kompletný STFT denoising prechod
"""

import numpy as np
import librosa
import scipy.ndimage

from .profiles import DenoiseProfile
from .noise_estimation import estimate_noise_profile, DIAG_MODE
from .filters import (
    harmonic_mask,
    psychoacoustic_floor,
    detect_transients,
    smooth_mask_time,
)


# ==============================================================================
# Pomocná funkcia: exponenciálny integrál E₁(x)
# ==============================================================================

def _expint(x: np.ndarray) -> np.ndarray:
    """
    Aproximácia E₁(x) potrebná pre MMSE-LSA gain:
      - pre x < 1: Taylorova rada
      - pre x ≥ 1: asymptotická expanzia
      - pre x > 500: limitne 0

    Numericky stabilná implementácia bez overflow.
    """
    x     = np.maximum(x, 1e-10)
    small = -np.log(x) - 0.5772 + x - x ** 2 / 4.0 + x ** 3 / 18.0
    safe_x = np.minimum(x, 500.0)
    with np.errstate(over="ignore", invalid="ignore"):
        large = np.exp(-safe_x) / safe_x * (
            1.0 - 1.0 / safe_x + 2.0 / safe_x ** 2 - 6.0 / safe_x ** 3
        )
    large  = np.where(np.isfinite(large), large, 0.0)
    result = np.where(x < 1.0, small, large)
    result = np.where(x > 500.0, 0.0, result)
    return np.maximum(result, -50.0)


# ==============================================================================
# MMSE-LSA gain estimátor (Ephraim-Malah 1984)
# ==============================================================================

def mmse_lsa_gain(
    power: np.ndarray,
    noise_power: np.ndarray,
    alpha_ns: float = 0.98,
) -> np.ndarray:
    """
    Vypočíta MMSE-LSA gain masku pre každý STFT bin a frame.

    Algoritmus iteruje frame po frame a udržuje odhad a priori SNR (ξ)
    cez Decision-Directed prístup (Ephraim & Malah 1984):
        ξₜ = α × Gₜ₋₁² × γₜ + (1−α) × max(γₜ−1, 0)

    kde γₜ = a posteriori SNR = power / noise_power

    Args:
        power:       spektrálny výkon (n_bins, n_frames)
        noise_power: odhadovaný šumový výkon (n_bins, n_frames)
        alpha_ns:    vyhladzovanie a priori SNR (0.95–0.99)

    Returns:
        gain maska v rozsahu [0, 1] rovnakého tvaru ako power
    """
    gain_prev = np.ones_like(power[:, 0:1])
    xi_frames = []

    for t in range(power.shape[1]):
        snr_post_t = power[:, t] / (noise_power[:, t] + 1e-10)
        xi_t = (
            alpha_ns * (gain_prev[:, 0] ** 2) * noise_power[:, t] / (noise_power[:, t] + 1e-10)
            + (1.0 - alpha_ns) * np.maximum(snr_post_t - 1.0, 0.0)
        )
        xi_frames.append(xi_t)
        nu        = np.maximum(xi_t / (1.0 + xi_t) * snr_post_t, 1e-10)
        gain_t    = xi_t / (1.0 + xi_t) * np.exp(0.5 * _expint(nu))
        gain_prev = gain_t[:, np.newaxis]

    xi_arr       = np.stack(xi_frames, axis=1)
    snr_post_all = power / (noise_power + 1e-10)
    nu_all       = np.maximum(xi_arr / (1.0 + xi_arr) * snr_post_all, 1e-10)
    gain         = xi_arr / (1.0 + xi_arr) * np.exp(0.5 * _expint(nu_all))
    return np.clip(gain, 0.0, 1.0)


# ==============================================================================
# Definícia frekvenčných pásiem
# ==============================================================================

# Každá trojica: (f_lo Hz, f_hi Hz, meno_pasma)
# "air" má hornú hranicu 96000 Hz – pokryje akékoľvek sr/2
BANDS: list[tuple[int, int, str]] = [
    (0,     250,   "sub_bass"),
    (250,   2000,  "bass_mid"),
    (2000,  6000,  "upper_mid"),
    (6000,  12000, "highs"),
    (12000, 96000, "air"),
]


# ==============================================================================
# Multi-band MMSE-LSA
# ==============================================================================

def apply_multiband_mmse(
    power: np.ndarray,
    noise_power: np.ndarray,
    freqs: np.ndarray,
    profile: DenoiseProfile,
    scale: float,
) -> np.ndarray:
    """
    Aplikuje MMSE-LSA zvlášť pre každé frekvenčné pásmo.

    Pre každé pásmo:
      1. Vypočíta MMSE-LSA gain s pásmovou silou filtrovania
      2. Per-bin SNR váha: kde dominuje signál, filtruj menej
      3. Zmieša gain a identitu podľa sily

    Hard cap 0.72 zabraňuje plošnému útlmu bez selektivity.

    Args:
        power:       spektrálny výkon (n_bins, n_frames)
        noise_power: odhadovaný šumový výkon (n_bins, n_frames)
        freqs:       frekvencie jednotlivých binov (Hz)
        profile:     žánrový denoising profil
        scale:       globálny SNR škálovací faktor z snr_scale()

    Returns:
        maska v rozsahu [0.05, 1.0] rovnakého tvaru ako power
    """
    # DIAGNOSTIC: aggressive – žiadny cap, žiadny blend, čistý MMSE-LSA gain
    if DIAG_MODE == "aggressive":
        mask = mmse_lsa_gain(power, noise_power * 3.0)
        return np.clip(mask, 0.0, 1.0)

    strengths = {
        "sub_bass":  profile.strength_low  * scale,
        "bass_mid":  profile.strength_low  * scale,
        "upper_mid": profile.strength_mid  * scale,
        "highs":     profile.strength_high * scale,
        "air":       profile.strength_high * scale * 0.6,  # vzduch – minimálne
    }
    # Hard cap: zabraňuje príliš agresívnemu filtrovaniu
    strengths = {k: min(v, 0.90) for k, v in strengths.items()}

    mask = np.ones_like(power)

    for f_lo, f_hi, band_name in BANDS:
        idx = np.where((freqs >= f_lo) & (freqs < f_hi))[0]
        if len(idx) == 0:
            continue

        p_b = power[idx, :]
        n_b = noise_power[idx, :]
        s_b = strengths[band_name]

        # MMSE-LSA gain pre toto pásmo
        lsa_gain = mmse_lsa_gain(p_b, n_b)

        # Jednoduchý lineárny blend:
        #   s_b = 1.0 → čistý MMSE-LSA (maximálne filtrovanie)
        #   s_b = 0.0 → maska = 1.0 (bez filtrovania)
        # Pôvodný vzorec násobil s_b aj snr_w, čím halvoval silu –
        # pri s_b=0.65, snr_w=0.5 maska nikdy neklesla pod 0.675.
        blended = s_b * lsa_gain + (1.0 - s_b)
        mask[idx, :] = np.clip(blended, 0.02, 1.0)

    return mask


# ==============================================================================
# Jeden spektrálny prechod
# ==============================================================================

def spectral_pass(
    y: np.ndarray,
    sr: int,
    profile: DenoiseProfile,
    snr_scale_factor: float,
    noise_ext: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Jeden kompletný STFT denoising prechod:
      1. STFT → magnitude + phase + power
      2. Odhad šumového profilu (percentil alebo externý)
      3. Multi-band MMSE-LSA maska
      4. Harmonická ochrana (pyin F0, protection=0.35)
      5. Psychoakustický floor (ISO 226 ATH + Bark maskovanie)
      6. Transient protection (spectral flux, 88. percentil)
      7. Jemné časové vyhladzovanie masky
      8. iSTFT rekonštrukcia

    Args:
        y:                mono audio pole
        sr:               vzorkovacia frekvencia
        profile:          žánrový denoising profil
        snr_scale_factor: výstup z snr_scale()
        noise_ext:        externý šumový profil (pre 2. prechod), alebo None

    Returns:
        (y_clean, residual_noise_estimate, noise_source_str)
    """
    n_fft = max(512, min(int(2 ** np.round(np.log2(sr * 0.025))), 4096))
    hop   = n_fft // 4

    original_len = len(y)
    y_pad = np.pad(y.astype(np.float32), (0, n_fft))

    D         = librosa.stft(y_pad, n_fft=n_fft, hop_length=hop)
    magnitude = np.abs(D)
    phase     = np.angle(D)
    power     = magnitude ** 2
    freqs     = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_frames  = power.shape[1]

    # Šumový profil
    if noise_ext is not None:
        noise_profile_1d, noise_src = noise_ext, "prev_pass"
    else:
        noise_profile_1d, noise_src = estimate_noise_profile(y, sr, n_fft, hop)

    noise_power = noise_profile_1d[:, np.newaxis] * np.ones((1, n_frames))

    # Multi-band MMSE-LSA
    mask = apply_multiband_mmse(power, noise_power, freqs, profile, snr_scale_factor)

    # Harmonická ochrana (mierna – protection=0.35)
    hmask = harmonic_mask(y, sr, n_fft, hop, protection=0.35)
    if hmask.shape[1] != n_frames:
        hmask = scipy.ndimage.zoom(
            hmask, (1, n_frames / max(hmask.shape[1], 1)), order=1
        )
        hmask = hmask[:power.shape[0], :n_frames]
    mask = np.minimum(mask * np.clip(hmask, 1.0, 1.35), 1.0)

    # Psychoakustický floor – znížený (0.06–0.12) pre efektívnejšiu redukciu šumu
    psych_floor = psychoacoustic_floor(freqs, power, floor_min=0.06, floor_max=0.12)
    mask        = np.clip(mask, psych_floor[:, np.newaxis], 1.0)

    # Transient protection – frekvenčne selektívna ochrana bicích
    # Problém starého prístupu: mask[:, is_tr] = max(mask, 0.78) chránilo
    # CELÉ spektrum počas transientu → šum v tichých binoch prežil.
    #
    # Nový prístup: chráni len biny kde je energia transienta výrazná
    # (relatívne k priemeru). Tiché biny počas úderu sa stále filtrujú.
    is_tr = detect_transients(y, sr, n_fft, hop)
    if len(is_tr) > n_frames:
        is_tr = is_tr[:n_frames]
    elif len(is_tr) < n_frames:
        is_tr = np.pad(is_tr, (0, n_frames - len(is_tr)))

    if np.any(is_tr):
        tr_frames  = power[:, is_tr]                         # energia počas transientov
        mean_power = np.mean(power, axis=1, keepdims=True) + 1e-10
        # Bin je "aktívny" počas transienta ak je jeho energia
        # aspoň 3× vyššia ako jeho priemerná energia
        active_bins = (tr_frames / mean_power) > 3.0         # (n_bins, n_tr_frames)
        # Len aktívne biny dostanú ochranu 0.95, ostatné sa normálne filtrujú
        tr_mask = mask[:, is_tr].copy()
        tr_mask[active_bins] = np.maximum(tr_mask[active_bins], 0.95)
        mask[:, is_tr] = tr_mask

    # Jemné časové vyhladzovanie (len v čase, nie 2D)
    smooth = smooth_mask_time(mask, size=5)
    mask   = np.where(is_tr[np.newaxis, :], mask, smooth)
    mask   = np.clip(mask, 0.0, 1.0)

    # Rekonštrukcia – amplitúdová maska, pôvodná fáza zachovaná
    D_clean = mask * magnitude * np.exp(1j * phase)
    y_clean = librosa.istft(D_clean, hop_length=hop, n_fft=n_fft, length=len(y_pad))
    y_clean = y_clean[:original_len].astype(np.float32)

    # Odhad reziduálneho šumu pre prípadný 2. prechod
    D_res   = librosa.stft(np.pad(y_clean, (0, n_fft)), n_fft=n_fft, hop_length=hop)
    res_est = np.percentile(np.abs(D_res) ** 2, 35, axis=1)

    return y_clean, res_est, noise_src