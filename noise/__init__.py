"""
audio.noise – Pridávanie testovacieho šumu do audio nahrávok.

Verejné API:
  add_noise(input_path, output_path, noise_type, noise_level) → dict
"""

from .add_noise import add_noise

__all__ = ["add_noise"]