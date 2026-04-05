"""
audio.classifier – Klasifikácia hudobného žánru.

Verejné API:
  classify_genre(file_path) → dict
"""

from .classifier import classify_genre

__all__ = ["classify_genre"]