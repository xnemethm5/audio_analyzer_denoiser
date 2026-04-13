"""
filters.py – DSP filtre pre denoising pipeline.

Obsahuje:
  declick_lsar_sr()      – AR detekcia + LSAR interpolácia (declick / decrackle)
  harmonic_mask()        – HPSS-based ochrana harmonických zložiek
  psychoacoustic_floor() – ATH + Bark maskovanie ako dolný limit STFT masky
  detect_transients()    – spectral flux detektor transientov
  smooth_mask_time()     – 1D časové vyhladzovanie STFT masky
"""

import numpy as np
import librosa
import scipy.ndimage
import scipy.signal


# ==============================================================================
# De-click / de-crackle – AR predikcia + LSAR interpolácia
# ==============================================================================
#
# Dvojkrokový postup:
#   1) Detekcia cez reziduál lokálneho AR (LPC) modelu. AR model popisuje
#      "predpovedateľnú" hudbu; click je impulz, ktorý model nevie predpovedať
#      → v reziduáli vyskočí. Prah cez MAD je robustný voči samotným clickom.
#   2) Interpolácia cez LSAR (Janssen 1986) – nájdené vzorky sa nahradia tak,
#      aby minimalizovali predikčnú chybu AR modelu odhadnutého z OKOLIA
#      kliknutia (nie z kontaminovaných dát).
#
# Referencia: Godsill & Rayner, "Digital Audio Restoration" (1998), kap. 5.

_LPC_ORDER        = 32     # rád AR modelu
_DETECT_BLOCK_SEC = 0.03   # 30 ms block pre LPC odhad pri detekcii
_MAX_GAP_SAMPLES  = 80     # dlhšie výpadky nerekonštruujeme (nespoľahlivé)
_CTX_MULT         = 6      # kontext pre LSAR = 6 × order vzoriek z každej strany


def _detect_clicks_ar(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Detekcia clickov cez reziduál lokálneho AR modelu.

    Postup:
      1. High-pass >1.5 kHz – impulzy sú širokopásmové, hudba má
         energiu dole → clicks vyniknú voči hudobnému pozadiu
      2. Po blokoch: LPC odhad → predikčná chyba → MAD-based prah
      3. Binárna dilatácia (click trvá 2–6 vzoriek okolo peaku)
    """
    block_len = int(_DETECT_BLOCK_SEC * sr)
    hop_len   = block_len // 2
    if block_len <= _LPC_ORDER * 2:
        return np.zeros(len(y), dtype=bool)

    nyq = sr / 2.0
    if nyq > 2000:
        b_hp, a_hp = scipy.signal.butter(4, 1500 / nyq, btype="high")
        y_hp = scipy.signal.filtfilt(b_hp, a_hp, y.astype(np.float64))
    else:
        y_hp = y.astype(np.float64)

    mask = np.zeros(len(y), dtype=bool)
    for start in range(0, len(y) - block_len + 1, hop_len):
        seg = y_hp[start:start + block_len]
        if float(np.std(seg)) < 1e-6:
            continue
        try:
            a = librosa.lpc(seg, order=_LPC_ORDER)
        except (FloatingPointError, np.linalg.LinAlgError):
            continue
        e = scipy.signal.lfilter(a, [1.0], seg)
        valid = e[_LPC_ORDER:]
        if len(valid) == 0:
            continue
        med   = np.median(valid)
        mad   = np.median(np.abs(valid - med)) + 1e-12
        sigma = 1.4826 * mad  # MAD → σ pre gaussovské rozdelenie
        mask[start:start + block_len] |= np.abs(e - med) > 4.0 * sigma

    return scipy.ndimage.binary_dilation(mask, iterations=2)


def _lsar_interpolate_gap(
    segment: np.ndarray,
    gap_start: int,
    gap_end:   int,
    a:         np.ndarray,
) -> np.ndarray:
    """
    LSAR interpolácia jedného súvislého výpadku.

    Minimalizuje ||A·x||² nad chýbajúcimi vzorkami, kde A je Toeplitzova
    matica predikčnej chyby AR modelu (Janssen 1986).
    """
    N = len(segment)
    p = len(a) - 1
    M = gap_end - gap_start

    if N - p <= 0 or M == 0:
        return segment

    # Konvolučná matica filtra [a_p, ..., a_1, 1] → rozmer (N-p) × N
    A = np.zeros((N - p, N), dtype=np.float64)
    rows = np.arange(N - p)
    for k in range(p + 1):
        A[rows, rows + (p - k)] = a[k]

    mask_u = np.zeros(N, dtype=bool)
    mask_u[gap_start:gap_end] = True
    idx_u = np.where(mask_u)[0]
    idx_k = np.where(~mask_u)[0]

    A_U = A[:, idx_u]
    A_K = A[:, idx_k]
    x_K = segment[idx_k].astype(np.float64)

    # Normálne rovnice + Tichonovova regularizácia
    lhs = A_U.T @ A_U
    rhs = -A_U.T @ (A_K @ x_K)
    lhs += (1e-8 * np.trace(lhs) / max(M, 1)) * np.eye(M)

    try:
        x_U = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return segment

    out = segment.copy()
    out[idx_u] = x_U.astype(segment.dtype)
    return out


def declick_lsar_sr(y: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """
    Odstráni clicks a crackle cez AR detekciu + LSAR interpoláciu.

    Returns:
        (y_out, n_gaps_fixed)
    """
    if len(y) < int(sr * 0.1):
        return y.astype(np.float32), 0

    mask = _detect_clicks_ar(y, sr)
    if not np.any(mask):
        return y.astype(np.float32), 0

    y_work = y.astype(np.float64).copy()

    # Súvislé behy True v maske
    edges  = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts = np.where(edges ==  1)[0]
    ends   = np.where(edges == -1)[0]

    context = _LPC_ORDER * _CTX_MULT
    n_fixed = 0

    for s, e in zip(starts, ends):
        gap_len = e - s
        if gap_len == 0 or gap_len > _MAX_GAP_SAMPLES:
            continue

        lo = max(0, s - context)
        hi = min(len(y_work), e + context)

        # AR model z ČISTÉHO kontextu – clicky ho nekontaminujú
        ctx_clean = np.concatenate([y_work[lo:s], y_work[e:hi]])
        if len(ctx_clean) < _LPC_ORDER * 4 or float(np.std(ctx_clean)) < 1e-6:
            continue

        try:
            a = librosa.lpc(ctx_clean, order=_LPC_ORDER)
        except (FloatingPointError, np.linalg.LinAlgError):
            continue

        local_seg = y_work[lo:hi].copy()
        fixed     = _lsar_interpolate_gap(local_seg, s - lo, e - lo, a)
        y_work[lo:hi] = fixed
        n_fixed += 1

    return y_work.astype(np.float32), n_fixed


# ==============================================================================
# Harmonic mask cez HPSS – polyfonná ochrana harmonických
# ==============================================================================

def harmonic_mask(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    protection: float = 0.35,
) -> np.ndarray:
    """
    2D ochranná maska pre harmonické (tonálne) zložky cez HPSS.

    Harmonic-Percussive Source Separation cez mediánové filtre v čase
    (horizontálne = harmonické) a vo frekvencii (vertikálne = perkusívne).
    Pomer `mag_H / (mag_H + mag_P)` dáva hodnotu v [0, 1], kde 1 = čisto
    tonálne, 0 = čisto perkusívne.

    Výstupná maska v rozsahu [1.0, 1.0 + protection] – tonálne biny budú
    chránené pred MMSE-LSA filtrovaním.
    """
    try:
        D    = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        H, P = librosa.decompose.hpss(D, margin=3.0)
        h_ratio = np.abs(H) / (np.abs(H) + np.abs(P) + 1e-10)
        return (1.0 + protection * h_ratio).astype(np.float32)
    except Exception:
        return np.ones((n_fft // 2 + 1, 1), dtype=np.float32)


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
    Psychoakustický dolný limit pre STFT masku.

    Kombinuje ATH (ISO 226 absolútny prah sluchu) a Bark simultánne
    maskovanie (±1.5 Bark okno). Kde je sluch necitlivý, môže byť floor
    vyšší (filter agresívnejší); kde je citlivý, floor je nízky.
    """
    f = np.maximum(freqs, 20.0)

    # ATH – ISO 226 aproximácia
    ath = (
        3.64 * (f / 1000.0) ** -0.8
        - 6.5 * np.exp(-0.6 * (f / 1000.0 - 3.3) ** 2)
        + 1e-3 * (f / 1000.0) ** 4
    )
    ath      = np.clip(ath, 0.0, 60.0)
    ath_norm = (ath - ath.min()) / (ath.max() - ath.min() + 1e-10)

    # Bark simultánne maskovanie
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

    combined = np.maximum(ath_norm, mask_norm)
    return (floor_max - combined * (floor_max - floor_min)).astype(np.float32)


# ==============================================================================
# Transient protection + mask smoothing
# ==============================================================================

def detect_transients(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    percentile: int = 88,
) -> np.ndarray:
    """Spectral flux detektor transientov (bicí, útoky)."""
    D    = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    flux = np.sum(np.maximum(np.diff(np.abs(D) ** 2, axis=1), 0), axis=0)
    flux = np.concatenate([[0], flux])
    thr  = np.percentile(flux, percentile)
    return scipy.ndimage.binary_dilation(flux > thr, iterations=2)


def smooth_mask_time(mask: np.ndarray, size: int = 5) -> np.ndarray:
    """Vyhladzuje STFT masku v časovej osi – redukuje musical noise."""
    return np.clip(
        scipy.ndimage.uniform_filter1d(mask, size=size, axis=1),
        0.0, 1.0,
    )
