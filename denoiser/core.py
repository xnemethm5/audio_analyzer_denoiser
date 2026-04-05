"""
core.py – Hlavná denoising pipeline.

Vstupný bod pre celý denoiser modul.

Funkcie:
  _process_channel() – spracuje jeden audio kanál celou pipeline
  denoise_audio()    – načíta súbor, spracuje, uloží, vráti metriky

Zámerne odstránené (degradujú zvuk):
  - Kalmanov filter v časovej oblasti (správal sa ako LP filter, mazal výšky)
  - 2. STFT prechod pri SNR < 10 dB (reťazenie MMSE = efekt "pod vodou")
  - Pedalboard NoiseGate (pumping artefakty na konci zničeného signálu)
  - RMS normalizácia (nasilu zdvíhala artefakty, spôsobovala clipping)
"""

import numpy as np
import soundfile as sf

from .profiles import get_profile, adapt_profile
from .noise_estimation import estimate_snr, snr_scale, detect_noise_type, SNR_CLEAN_THRESHOLD_DB, DIAG_MODE
from .spectral import spectral_pass
from .filters import remove_impulses


# ==============================================================================
# Peak normalizácia na -1 dBFS
# ==============================================================================

def _peak_normalize(y: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """
    Normalizuje signál tak, aby peak dosiahol target_db (default -1 dBFS).

    Na rozdiel od RMS normalizácie:
      - Nezosilňuje artefakty v tichých úsekoch
      - Nezabráni digitálnemu clippingu násilným zosilnením
      - Zachová prirodzený dynamický rozsah po denoising

    Ak je peak už pod target_db, signál sa nezosilňuje (len utíši).
    """
    peak = float(np.max(np.abs(y))) + 1e-10
    target_linear = 10 ** (target_db / 20.0)
    gain = min(target_linear / peak, 1.0)
    return (y * gain).astype(np.float32)


# ==============================================================================
# Spracovanie jedného kanála
# ==============================================================================

def _process_channel(
    y: np.ndarray,
    sr: int,
    profile,
    global_snr_db: float,
) -> tuple[np.ndarray, str, list[str]]:
    """
    Aplikuje denoising pipeline na jeden audio kanál.

    Kroky:
      1. Early exit ak je signál čistý (SNR >= SNR_CLEAN_THRESHOLD_DB)
      2. Mediánový filter – len ak je detekovaný impulzný šum (skóre > 0.2)
      3. Jeden STFT prechod – MMSE-LSA + harmonická ochrana
                            + psychoakustický floor + transient protection

    Jeden kvalitný prechod znie hudobnejšie ako reťazenie viacerých.

    Args:
        y:             mono audio pole (float32)
        sr:            vzorkovacia frekvencia
        profile:       DenoiseProfile pre daný žáner
        global_snr_db: SNR vstupného signálu

    Returns:
        (y_clean, noise_type_str, applied_steps_list)
    """
    noise_type, scores = detect_noise_type(y, sr)
    applied: list[str] = []

    # BYPASS režim – vráti signál bez akejkoľvek zmeny
    if DIAG_MODE == "bypass":
        return y, noise_type, ["DIAG:bypass"]

    # AGGRESSIVE režim – ignoruje žánrový profil aj SNR threshold
    # maximálna sila na všetkých pásmach, pipeline sa spustí vždy
    if DIAG_MODE == "aggressive":
        from .profiles import DenoiseProfile
        profile = DenoiseProfile(
            gate_threshold_db=-30, gate_ratio=5.0,
            gate_attack_ms=2.0,    gate_release_ms=80.0,
            highpass_hz=40,
            strength_low=0.90, strength_mid=0.90, strength_high=0.85,
        )
    else:
        # Early exit – signál je čistý, nič nerobíme (len v normal režime)
        if global_snr_db >= SNR_CLEAN_THRESHOLD_DB:
            return y, noise_type, ["skipped(clean-signal)"]

    # Dynamická úprava profilu podľa detekovaného šumu
    profile = adapt_profile(profile, scores)
    applied.append(
        f"profile-adapted(slope={scores.get('spectral_slope', 0):.2f},"
        f"imp={scores['impulsive']:.2f},nonstat={scores['nonstationary']:.2f})"
    )

    # Mediánový filter – len pri impulznom šume (praskanie, clicky)
    # Prah 4.0σ namiesto 3.5σ – konzervatívnejší, chráni bicie
    if scores["impulsive"] > 0.2:
        y = remove_impulses(y, threshold_sigma=4.0)
        applied.append("impulse-filter")

    scale = snr_scale(global_snr_db)

    # Jeden STFT prechod – základ celého denoisingu
    y, _, src = spectral_pass(y, sr, profile, snr_scale_factor=scale)
    applied.append(f"multiband-mmse[{src}]")

    return y, noise_type, applied


# ==============================================================================
# Hlavná funkcia
# ==============================================================================

def denoise_audio(
    file_path: str,
    output_path: str,
    genres: list[dict] | None = None,
) -> dict:
    """
    Konzervatívny multi-band denoising so zachovaním kvality hudby.

    Pipeline:
      1. SNR-adaptívna sila         (globálny SNR moduluje agresivitu filtrovania)
      2. Early exit                 (ak SNR >= 20 dB – signál je čistý)
      3. Mediánový filter           (len pri impulznom šume, prah 4.0σ)
      4. Percentilový odhad šumu    (35. percentil alebo tiché úseky)
      5. Multi-band MMSE-LSA        (Ephraim-Malah, každé pásmo zvlášť)
      6. Harmonická ochrana         (pyin F0 – ochrana tónov, protection=0.35)
      7. Psychoakustický floor      (ISO 226 ATH + Bark, floor 0.12–0.20)
      8. Transient protection       (spectral flux – ochrana útokov bicích)
      9. Jemné časové vyhladzovanie (len v čase, nie 2D – redukcia musical noise)
      10. Peak normalizácia          (−1 dBFS, gain <= 1.0 – bez zosilňovania artefaktov)

    Args:
        file_path:   cesta k vstupnému audio súboru (mp3/wav)
        output_path: cesta kam sa uloží vyčistený súbor (wav)
        genres:      výstup z classify_genre()["genres"], alebo None pre default

    Returns:
        dict s kľúčmi:
          snr_before   – SNR pôvodného signálu (dB)
          snr_after    – SNR vyčisteného signálu (dB)
          profile_used – textový popis použitého profilu
          noise_type   – detekovaný typ šumu
          engine       – reťazec krokov pipeline
          output_path  – cesta k výstupnému súboru
    """
    profile, profile_label = get_profile(genres)

    y, sr = sf.read(file_path, always_2d=False)
    y = y.astype(np.float32)

    y_mono     = y if y.ndim == 1 else y.mean(axis=1)
    snr_before = estimate_snr(y_mono, sr)

    # Spracovanie kanálov
    if y.ndim == 1:
        y_clean, noise_type, applied = _process_channel(
            y, sr, profile, snr_before
        )
    else:
        results    = [
            _process_channel(y[:, ch], sr, profile, snr_before)
            for ch in range(y.shape[1])
        ]
        y_clean    = np.stack([r[0] for r in results], axis=1)
        noise_type = results[0][1]
        applied    = results[0][2]

    # Peak normalizácia na -1 dBFS (gain <= 1.0, nikdy nezosilňuje)
    y_clean = _peak_normalize(y_clean, target_db=-1.0)

    # Finálny SNR
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