"""
spectral.py – STFT spracovanie a multi-band MMSE-LSA denoising.

Opravené oproti pôvodnej verzii:

1. Bug v decision-directed a-priori SNR odhade.
   Pôvodne:
       xi_t = alpha * G_prev² * noise/noise + (1-alpha) * max(gamma-1, 0)
   čo sa vykracovalo na `alpha * G_prev²` a rekurzívna pamäť DD bola
   pokazená. Správne má byť:
       xi_t = alpha * G_prev² * |Y_prev|² / N_t + (1-alpha) * max(gamma-1, 0)
   (Ephraim & Malah 1984). Toto je zdroj hlavného benefitu MMSE-LSA –
   supresie musical noise. Bez opravy pipeline degradovala na niečo
   ako obyčajný Wiener s prahom.

2. Lineárny blend masky nahradený exponentom.
   Pôvodne: `blended = s_b * lsa_gain + (1-s_b)`
            → maximálna supresia = 1-s_b, pre s_b=0.73 to je −11 dB
            → mask_floor (0.02 = −34 dB) sa nikdy neuplatnil
   Nové:    `blended = lsa_gain ** s_b`
            → plná supresia pre s_b ≥ 1, postupne jemnejšia pre s_b < 1
            → mask_floor reálne limituje dno masky

3. Odstránená duplicitná batched rekonštrukcia gainu po cykle
   (pôvodne sa gain počítal v cykle, zahadzoval a potom sa prepočítaval
   znovu cez xi_arr – ale xi_arr má inherentnú sekvenčnú závislosť,
   takže "vektorizácia" bola iluzórna).
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
    MMSE-LSA gain estimátor (Ephraim-Malah 1984) so SPRÁVNYM decision-directed
    a-priori SNR updatom.

    Args:
        power:       |Y|² spektrálneho výkonu signálu (n_bins, n_frames)
        noise_power: odhadnutý výkon šumu (n_bins, n_frames)
        alpha_ns:    DD vyhladzovací faktor (typicky 0.92–0.98)

    Returns:
        gain: (n_bins, n_frames) v [0, 1]
    """
    n_bins, n_frames = power.shape

    gain_out   = np.empty_like(power)
    gain_prev  = np.ones(n_bins)
    power_prev = power[:, 0].copy()

    for t in range(n_frames):
        n_t       = noise_power[:, t] + 1e-10
        gamma_t   = power[:, t] / n_t  # a-posteriori SNR

        # DECISION-DIRECTED a-priori SNR update (Ephraim-Malah 1984):
        #   xi = alpha * |Ŝ_prev|² / N  +  (1-alpha) * max(gamma-1, 0)
        # kde |Ŝ_prev|² = (G_prev * |Y_prev|)² = G_prev² * power_prev
        xi_dd = alpha_ns * (gain_prev ** 2) * power_prev / n_t
        xi_ml = (1.0 - alpha_ns) * np.maximum(gamma_t - 1.0, 0.0)
        xi_t  = np.maximum(xi_dd + xi_ml, 1e-10)

        # LSA gain (Ephraim-Malah log-spectral amplitude estimator)
        nu_t   = xi_t / (1.0 + xi_t) * gamma_t
        nu_t   = np.maximum(nu_t, 1e-10)
        gain_t = xi_t / (1.0 + xi_t) * np.exp(0.5 * _expint(nu_t))
        gain_t = np.clip(gain_t, 0.0, 1.0)

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
    Multi-band MMSE-LSA s exponent blendom.

    Pre každé pásmo sa spočíta štandardný MMSE-LSA gain a potom sa aplikuje
    exponent podľa sily pásma:
        mask = gain ** (strength * scale)
    Toto je zásadný rozdiel oproti pôvodnému lineárnemu blendu, ktorý
    shora limitoval supresiu.
    """
    if DIAG_MODE == "aggressive":
        mask = mmse_lsa_gain(power, noise_power * 3.0, alpha_ns=profile.alpha_ns)
        return np.clip(mask, 0.0, 1.0)

    strengths = {
        "sub_bass":  profile.strength_low  * scale,
        "bass_mid":  profile.strength_low  * scale,
        "upper_mid": profile.strength_mid  * scale,
        "highs":     profile.strength_high * scale,
        # "air" je vyššie pásmo – trocha jemnejšie ako highs (× 0.85)
        "air":       profile.strength_high * scale * 0.85,
    }
    # Ochranné limity: pod 0.2 nebude mať filter žiaden efekt,
    # nad 1.5 by výsledok mohol byť nestabilný
    strengths = {k: float(np.clip(v, 0.20, 1.50)) for k, v in strengths.items()}

    mask = np.ones_like(power)

    for f_lo, f_hi, band_name in BANDS:
        idx = np.where((freqs >= f_lo) & (freqs < f_hi))[0]
        if len(idx) == 0:
            continue
        p_b = power[idx, :]
        n_b = noise_power[idx, :]
        s_b = strengths[band_name]

        # Štandardný MMSE-LSA gain (správny DD update)
        lsa_gain = mmse_lsa_gain(p_b, n_b, alpha_ns=profile.alpha_ns)

        # EXPONENT BLEND: gain ** strength
        #   s=1.0 → čistý MMSE-LSA
        #   s<1.0 → jemnejšie (G=0.3 → G^0.8=0.38, menej supresie)
        #   s>1.0 → agresívnejšie
        # Mask floor reálne limituje spodok (predtým bol nedosiahnuteľný)
        effective = np.power(np.maximum(lsa_gain, 1e-10), s_b)
        mask[idx, :] = np.clip(effective, profile.mask_floor, 1.0)

    return mask


def spectral_pass(
    y: np.ndarray,
    sr: int,
    profile: DenoiseProfile,
    snr_scale_factor: float,
    noise_ext: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Jeden kompletný STFT denoising prechod.

    Tok:
      STFT → noise profile estimation → multi-band MMSE-LSA (exp blend) →
      harmonic protection (HPSS) → psychoacoustic floor → transient protection
      → temporal mask smoothing → ISTFT
    """
    # n_fft dynamicky podľa profilu (n_fft_ms v ms)
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

    # Noise profile odhad
    if noise_ext is not None:
        noise_profile_1d, noise_src = noise_ext, "prev_pass"
    else:
        noise_profile_1d, noise_src = estimate_noise_profile(
            y, sr, n_fft, hop,
            window_sec=profile.window_sec,
            bias=profile.bias,
        )

    noise_power = noise_profile_1d[:, np.newaxis] * np.ones((1, n_frames))

    # Multi-band MMSE-LSA maska
    mask = apply_multiband_mmse(power, noise_power, freqs, profile, snr_scale_factor)

    # Harmonic protection cez HPSS (polyfonné, na rozdiel od pôvodného YIN)
    hmask = harmonic_mask(y, sr, n_fft, hop, protection=0.35)
    if hmask.shape[1] != n_frames:
        hmask = scipy.ndimage.zoom(
            hmask, (1, n_frames / max(hmask.shape[1], 1)), order=1
        )
        hmask = hmask[:power.shape[0], :n_frames]
    # Harmonická ochrana môže masku zvýšiť max o (1+protection), nikdy nad 1
    mask = np.minimum(mask * np.clip(hmask, 1.0, 1.35), 1.0)

    # Psychoakustický floor
    psych_floor = psychoacoustic_floor(freqs, power, floor_min=0.06, floor_max=0.12)
    mask        = np.clip(mask, psych_floor[:, np.newaxis], 1.0)

    # Transient protection – na transient rámcoch nechaj aktívne biny takmer nedotknuté
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

    # Temporal smoothing – len mimo transientných rámcov
    smooth = smooth_mask_time(mask, size=5)
    mask   = np.where(is_tr[np.newaxis, :], mask, smooth)
    mask   = np.clip(mask, 0.0, 1.0)

    # ISTFT
    D_clean = mask * magnitude * np.exp(1j * phase)
    y_clean = librosa.istft(D_clean, hop_length=hop, n_fft=n_fft, length=len(y_pad))
    y_clean = y_clean[:original_len].astype(np.float32)

    # Reziduálny odhad pre prípadný druhý prechod (35. percentil výkonu)
    D_res   = librosa.stft(np.pad(y_clean, (0, n_fft)), n_fft=n_fft, hop_length=hop)
    res_est = np.percentile(np.abs(D_res) ** 2, 35, axis=1)

    return y_clean, res_est, noise_src
