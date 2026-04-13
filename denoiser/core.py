"""
core.py – Hlavná denoising pipeline.

Zmeny oproti pôvodnej verzii:
  - declick volanie ide cez declick_lsar_sr(y, sr), ktorý má správny sample rate
    (pôvodné remove_impulses používalo globálne std, ktoré nefungovalo)
  - snr_before/after sa teraz počíta zo skutočného SNR (minimum statistics),
    nie dynamického rozsahu → SNR_CLEAN_THRESHOLD_DB je reálne použiteľný
"""

import numpy as np
import soundfile as sf

from .profiles import get_profile, adapt_profile
from .noise_estimation import (
    estimate_snr,
    snr_scale,
    detect_noise_type,
    SNR_CLEAN_THRESHOLD_DB,
    DIAG_MODE,
)
from .spectral import spectral_pass
from .filters import declick_lsar_sr


def _dc_block(y: np.ndarray, sr: int, cutoff_hz: float = 20.0) -> np.ndarray:
    """
    Jednopólový IIR high-pass filter na odstránenie DC offsetu a sub-sonic rumble.

    Pink a hnedý šum majú veľa energie pod 40 Hz a po filtrovaní sub-bass
    pásma môže v signáli zostať asymetrický reziduál s non-zero priemerom,
    čo sa prejaví ako "posunutá" waveform.

    20 Hz cutoff je pod hranicou počuteľnosti, takže odstránenie je
    transparentné. Filter sa aplikuje per kanál pre stereo signály.

    Rovnica: y[n] = x[n] - x[n-1] + R * y[n-1]
    kde R = exp(-2π * cutoff / sr) ≈ 0.997 pre 20 Hz @ 44.1 kHz.
    """
    R = float(np.exp(-2.0 * np.pi * cutoff_hz / sr))

    def _filter_1d(x: np.ndarray) -> np.ndarray:
        # scipy.signal.lfilter forma: b = [1, -1], a = [1, -R]
        from scipy.signal import lfilter
        return lfilter([1.0, -1.0], [1.0, -R], x).astype(np.float32)

    if y.ndim == 1:
        return _filter_1d(y)
    return np.stack([_filter_1d(y[:, ch]) for ch in range(y.shape[1])], axis=1)


def _peak_normalize(y: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Peak normalizácia na -1 dBFS. Gain <= 1.0 – nikdy nezosilňuje."""
    peak          = float(np.max(np.abs(y))) + 1e-10
    target_linear = 10 ** (target_db / 20.0)
    gain          = min(target_linear / peak, 1.0)
    return (y * gain).astype(np.float32)


def _process_channel(
    y: np.ndarray,
    sr: int,
    profile,
    global_snr_db: float,
) -> tuple[np.ndarray, str, list[str]]:
    noise_type, scores = detect_noise_type(y, sr)
    applied: list[str] = []

    if DIAG_MODE == "bypass":
        return y, noise_type, ["DIAG:bypass"]

    if DIAG_MODE == "aggressive":
        from .profiles import DenoiseProfile
        profile = DenoiseProfile(
            gate_threshold_db=-30, gate_ratio=5.0,
            gate_attack_ms=2.0,    gate_release_ms=80.0,
            highpass_hz=40,
            strength_low=1.20, strength_mid=1.20, strength_high=1.15,
        )
    else:
        if global_snr_db >= SNR_CLEAN_THRESHOLD_DB:
            return y, noise_type, [f"skipped(clean-signal, snr={global_snr_db:.1f}dB)"]

    # Dynamická úprava všetkých parametrov podľa šumu aj žánru
    profile = adapt_profile(profile, scores)
    applied.append(
        f"profile-adapted("
        f"slope={scores.get('spectral_slope', 0):.2f},"
        f"imp={scores['impulsive']:.2f},"
        f"nonstat={scores['nonstationary']:.2f} | "
        f"s_lo={profile.strength_low},s_mid={profile.strength_mid},s_hi={profile.strength_high},"
        f"α={profile.alpha_ns},win={profile.window_sec}s,"
        f"bias={profile.bias},fft={profile.n_fft_ms}ms,"
        f"floor={profile.mask_floor})"
    )

    # --- Declick / decrackle FIRST ---
    # Odstránime impulzné šumy pred spektrálnym prechodom. Toto je dôležité
    # lebo clicks sú širokopásmové a inak by rozmazali odhad šumu a kazili
    # by sa aj transienty v MMSE-LSA.
    if scores["impulsive"] > 0.2:
        y, n_fixed = declick_lsar_sr(y, sr)
        applied.append(f"declick-lsar(fixed={n_fixed})")
        # Po declicku prepočítame SNR – bez clickov je signál zvyčajne
        # oveľa čistejší a pôvodný global_snr_db by pretlačil filter.
        effective_snr = estimate_snr(y, sr)
    else:
        effective_snr = global_snr_db

    # Scale pre MMSE-LSA strength – používame efektívne SNR (po declicku)
    if noise_type == "stationary":
        scale = 1.0
    else:
        scale = snr_scale(effective_snr)

    y, _, src = spectral_pass(y, sr, profile, snr_scale_factor=scale)
    applied.append(f"multiband-mmse[{src}]")

    return y, noise_type, applied


def denoise_audio(
    file_path: str,
    output_path: str,
    genres: list[dict] | None = None,
) -> dict:
    profile, profile_label = get_profile(genres)

    y, sr = sf.read(file_path, always_2d=False)
    y     = y.astype(np.float32)

    y_mono     = y if y.ndim == 1 else y.mean(axis=1)
    snr_before = estimate_snr(y_mono, sr)

    if y.ndim == 1:
        y_clean, noise_type, applied = _process_channel(y, sr, profile, snr_before)
    else:
        results    = [_process_channel(y[:, ch], sr, profile, snr_before) for ch in range(y.shape[1])]
        y_clean    = np.stack([r[0] for r in results], axis=1)
        noise_type = results[0][1]
        applied    = results[0][2]

    y_clean = _dc_block(y_clean, sr, cutoff_hz=20.0)
    y_clean = _peak_normalize(y_clean, target_db=-1.0)

    clean_mono = y_clean if y_clean.ndim == 1 else y_clean.mean(axis=1)
    snr_after  = estimate_snr(clean_mono, sr)

    sf.write(output_path, y_clean, sr)

    return {
        "snr_before":   round(float(snr_before), 2),
        "snr_after":    round(float(snr_after),  2),
        "profile_used": profile_label,
        "noise_type":   noise_type,
        "engine":       " -> ".join(applied),
        "output_path":  output_path,
    }