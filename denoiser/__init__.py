"""
denoiser – Modul pre odstraňovanie šumu z audio nahrávok.

Verejné API:
  denoise_audio(file_path, output_path, genres=None) → dict

Interné moduly:
  profiles         – DenoiseProfile, GENRE_PROFILES, get_profile
  noise_estimation – estimate_snr, snr_scale, detect_noise_type, estimate_noise_profile
  spectral         – MMSE-LSA, multi-band processing, spectral_pass
  filters          – remove_impulses, harmonic_mask, psychoacoustic_floor,
                     detect_transients, smooth_mask_time, kalman_denoise, maybe_gate
  core             – _process_channel, denoise_audio
"""

from .core import denoise_audio

__all__ = ["denoise_audio"]