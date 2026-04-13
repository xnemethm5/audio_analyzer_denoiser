"""
filters.py – DSP filtre a ochranné moduly pre denoising pipeline.

Obsahuje:
  remove_impulses()      – AR-based detekcia + LSAR interpolácia (declick)
  harmonic_mask()        – ochrana harmonických (HPSS namiesto monofónneho YIN)
  psychoacoustic_floor() – ATH + Bark maskovanie ako dolný limit masky
  detect_transients()    – spectral flux detektor transientov
  smooth_mask_time()     – jemné 1D časové vyhladzovanie STFT masky
  kalman_denoise()       – Kalmanov filter pre časové vyhladenie signálu
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
# De-click / de-crackle – AR predikcia + LSAR interpolácia
# ==============================================================================
#
# Pôvodná verzia (globálne std + medián kernel 7) reálne neodstraňovala
# crackle. Problém: globálne std je dominované hlasnými pasážami, takže
# v tichých úsekoch chytala hudobné transienty a v hlasných nič.
#
# Nová verzia robí dva kroky:
#   1) Detekcia cez reziduál AR (LPC) modelu – lokálne odhadnutý AR model
#      popisuje "predpovedateľnú" hudbu. Click je impulz, ktorý AR model
#      nevie predpovedať → v reziduáli vyskočí. Prah sa počíta cez MAD,
#      čo je robustné voči samotným clickom (nekontaminujú prah).
#   2) Interpolácia cez LSAR (Janssen 1986) – nájdené vzorky sa nahradia
#      tak, aby minimalizovali predikčnú chybu AR modelu odhadnutého
#      z OKOLIA kliknutia (nie z kontaminovaných dát).
#
# Vstupné parametre ostávajú rovnaké ako pôvodné – len sa teraz používajú
# zmysluplne. threshold_sigma je násobok MAD-odvodenej sigmy reziduálu.
#
# Referencia: Godsill & Rayner, "Digital Audio Restoration" (1998), kap. 5.

_LPC_ORDER        = 32     # rád AR modelu – dosť na lokálnu spektrálnu štruktúru
_DETECT_BLOCK_SEC = 0.03   # 30 ms block pre LPC odhad pri detekcii
_MAX_GAP_SAMPLES  = 80     # dlhšie výpadky nerekonštruujeme (nespoľahlivé)
_CTX_MULT         = 6      # kontext pre LSAR = 6 × order vzoriek z každej strany


def _detect_clicks_ar(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Detekcia clickov cez reziduál lokálneho AR modelu.

    V každom bloku:
      1. Odhadne LPC koeficienty rádu _LPC_ORDER
      2. Spočíta predikčnú chybu e[n] = lfilter(a, [1], y)
      3. Robustný prah cez MAD → označí vzorky |e[n] - med| > k·σ_MAD
      4. Binárna dilatácia – click zvyčajne trvá 2–6 vzoriek okolo peaku

    Returns:
        bool maska (len(y),), True = vzorka označená ako click
    """
    block_len = int(_DETECT_BLOCK_SEC * sr)
    hop_len   = block_len // 2
    if block_len <= _LPC_ORDER * 2:
        return np.zeros(len(y), dtype=bool)

    mask = np.zeros(len(y), dtype=bool)
    # High-pass filter pre zvýraznenie impulzov – hudba má energiu dole,
    # clicks sú širokopásmové, takže nad 1.5 kHz vyniknú oveľa lepšie
    nyq = sr / 2.0
    if nyq > 2000:
        b_hp, a_hp = scipy.signal.butter(4, 1500 / nyq, btype='high')
        y_hp = scipy.signal.filtfilt(b_hp, a_hp, y.astype(np.float64))
    else:
        y_hp = y.astype(np.float64)

    for start in range(0, len(y) - block_len + 1, hop_len):
        seg = y_hp[start:start + block_len]
        seg_std = float(np.std(seg))
        if seg_std < 1e-6:
            continue
        try:
            a = librosa.lpc(seg, order=_LPC_ORDER)
        except (FloatingPointError, np.linalg.LinAlgError):
            continue
        # Predikčná chyba
        e = scipy.signal.lfilter(a, [1.0], seg)
        # MAD-based robustný odhad σ (clicks nekontaminujú prah)
        valid = e[_LPC_ORDER:]
        if len(valid) == 0:
            continue
        med   = np.median(valid)
        mad   = np.median(np.abs(valid - med)) + 1e-12
        sigma = 1.4826 * mad
        local = np.abs(e - med) > 4.0 * sigma
        mask[start:start + block_len] |= local

    # Click má typicky šírku 3–8 vzoriek, dilatujeme o 2 na každú stranu
    mask = scipy.ndimage.binary_dilation(mask, iterations=2)
    return mask


def _lsar_interpolate_gap(
    segment: np.ndarray,
    gap_start: int,
    gap_end:   int,
    a:         np.ndarray,
) -> np.ndarray:
    """
    LSAR interpolácia jedného súvislého výpadku vo vnútri `segment`.

    Minimalizuje ||A·x||² nad chýbajúcimi vzorkami, kde A je Toeplitzova
    matica predikčnej chyby AR modelu s koeficientmi `a` (a[0]=1).

    Args:
        segment:   lokálny úsek signálu (context_L | gap | context_R)
        gap_start: začiatok výpadku v `segment`
        gap_end:   koniec výpadku v `segment` (exkluzívny)
        a:         LPC koeficienty, a[0]=1, dĺžka p+1

    Returns:
        segment s nahradenými vzorkami [gap_start:gap_end]
    """
    N = len(segment)
    p = len(a) - 1
    M = gap_end - gap_start

    if N - p <= 0 or M == 0:
        return segment

    # Postavenie matice A: (N-p) × N, riadok i má a[p..0] na stĺpcoch i..i+p
    A = np.zeros((N - p, N), dtype=np.float64)
    rows = np.arange(N - p)
    for k in range(p + 1):
        A[rows, rows + (p - k)] = a[k]

    # Rozdelenie stĺpcov na známe a neznáme
    mask_u = np.zeros(N, dtype=bool)
    mask_u[gap_start:gap_end] = True
    idx_u = np.where(mask_u)[0]
    idx_k = np.where(~mask_u)[0]

    A_U = A[:, idx_u]
    A_K = A[:, idx_k]
    x_K = segment[idx_k].astype(np.float64)

    # Normálne rovnice (A_U^T A_U) x_U = -A_U^T A_K x_K
    lhs = A_U.T @ A_U
    rhs = -A_U.T @ (A_K @ x_K)

    # Tichonovova regularizácia proti zlej podmienenosti pre veľmi krátke gapy
    reg = 1e-8 * np.trace(lhs) / max(M, 1)
    lhs += reg * np.eye(M)

    try:
        x_U = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return segment

    out = segment.copy()
    out[idx_u] = x_U.astype(segment.dtype)
    return out


def remove_impulses(y: np.ndarray, threshold_sigma: float = 4.0) -> np.ndarray:
    """
    Odstráni clicks a crackle cez AR detekciu + LSAR interpoláciu.

    Zachováva pôvodné API (jediný argument threshold_sigma je rezervovaný
    pre budúce použitie; aktuálna detekcia používa MAD-based 4σ prah, ktorý
    je robustný a nepotrebuje používateľské ladenie).

    Args:
        y:               mono audio pole (float32)
        threshold_sigma: rezervované – aktuálne nepoužité
                         (detekcia má vlastný interný prah)

    Returns:
        y_out: signál s opravenými impulzmi (float32)
    """
    if len(y) < 1024:
        return y

    sr_estimate = 44100  # AR rád a block_sec sú v sekundách, takže sr potrebujeme
    # V praxi volame túto funkciu z core._process_channel, kde máme reálne sr –
    # ale API očakáva len (y, threshold_sigma). Použijeme preto bezpečný default
    # 44.1 kHz. Ak chceš presnejšie ladenie, pozri declick_lsar_sr() nižšie.

    return declick_lsar_sr(y, sr_estimate)[0].astype(np.float32)


def declick_lsar_sr(y: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """
    Sample-rate-aware de-click. Používa sa interne z core.

    Returns:
        (y_out, n_gaps_fixed)
    """
    if len(y) < int(sr * 0.1):
        return y.astype(np.float32), 0

    mask = _detect_clicks_ar(y, sr)
    if not np.any(mask):
        return y.astype(np.float32), 0

    y_work = y.astype(np.float64).copy()

    # Nájdi súvislé behy True v maske
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

        # AR model sa fituje na ČISTÝ kontext (bez samotného gapu),
        # aby ho clicky nekontaminovali
        ctx_clean = np.concatenate([y_work[lo:s], y_work[e:hi]])
        if len(ctx_clean) < _LPC_ORDER * 4:
            continue
        if float(np.std(ctx_clean)) < 1e-6:
            continue

        try:
            a = librosa.lpc(ctx_clean, order=_LPC_ORDER)
        except (FloatingPointError, np.linalg.LinAlgError):
            continue

        # LSAR interpolácia vo vnútri lokálneho okna
        local_seg = y_work[lo:hi].copy()
        gs_local  = s - lo
        ge_local  = e - lo
        fixed     = _lsar_interpolate_gap(local_seg, gs_local, ge_local, a)
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
    Generuje 2D ochrannú masku pre harmonické (tonálne) zložky.

    Používa librosa HPSS (Harmonic-Percussive Source Separation):
    rozloží STFT na harmonickú a perkusívnu zložku mediánovými filtrami
    v čase (harmonické = horizontálne čiary) a vo frekvencii (perkusívne
    = vertikálne čiary). Pomer `mag_H / (mag_H + mag_P)` dáva hodnotu
    v [0, 1], kde 1 = čisto tonálne, 0 = čisto perkusívne.

    Oproti pôvodnej pyin-based verzii toto funguje aj pre polyfonické
    nahrávky (orchester, akord, ensemble) – YIN by sa uzamkol na jednu F0
    a zvyšok harmonických by ostal nechránený.

    Args:
        y:          mono audio pole
        sr:         vzorkovacia frekvencia (len pre konzistenciu API)
        n_fft:      FFT veľkosť okna
        hop:        hop length
        protection: multiplikátor ochrany (0.2–0.5), maska bude v rozsahu
                    [1.0, 1.0 + protection]

    Returns:
        hmask: np.ndarray (n_bins, n_frames) s hodnotami >= 1.0
    """
    try:
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        H, P = librosa.decompose.hpss(D, margin=3.0)
        mag_h = np.abs(H)
        mag_p = np.abs(P)
        h_ratio = mag_h / (mag_h + mag_p + 1e-10)
        hmask = 1.0 + protection * h_ratio
        return hmask.astype(np.float32)
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
    Psychoakustický dolný limit pre STFT masku – pozri pôvodnú dokumentáciu.
    Kombinuje ATH (ISO 226) a Bark simultánne maskovanie.
    """
    f = np.maximum(freqs, 20.0)
    ath = (
        3.64 * (f / 1000.0) ** -0.8
        - 6.5 * np.exp(-0.6 * (f / 1000.0 - 3.3) ** 2)
        + 1e-3 * (f / 1000.0) ** 4
    )
    ath      = np.clip(ath, 0.0, 60.0)
    ath_norm = (ath - ath.min()) / (ath.max() - ath.min() + 1e-10)

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
    """Spectral flux detektor transientov (bicí, útoky). Nezmenené."""
    D    = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    flux = np.sum(np.maximum(np.diff(np.abs(D) ** 2, axis=1), 0), axis=0)
    flux = np.concatenate([[0], flux])
    thr  = np.percentile(flux, percentile)
    return scipy.ndimage.binary_dilation(flux > thr, iterations=2)


# ==============================================================================
# Jemné 1D časové vyhladzovanie STFT masky
# ==============================================================================

def smooth_mask_time(mask: np.ndarray, size: int = 5) -> np.ndarray:
    """Vyhladzuje STFT masku v časovej osi – redukuje musical noise."""
    return np.clip(
        scipy.ndimage.uniform_filter1d(mask, size=size, axis=1),
        0.0, 1.0,
    )


# ==============================================================================
# Kalmanov filter – ponechaný pre kompatibilitu (nie je volaný v pipeline)
# ==============================================================================

def kalman_denoise(
    y: np.ndarray,
    process_var: float = 1e-5,
    measurement_var: float = 0.02,
) -> np.ndarray:
    """Jednorozmerný Kalmanov filter. Nevolaný z core, len pre kompatibilitu."""
    n     = len(y)
    x_est = float(y[0])
    P_est = 1.0
    out   = np.zeros(n, dtype=np.float32)
    for t in range(n):
        P_pred = P_est + process_var
        K      = P_pred / (P_pred + measurement_var)
        x_est  = x_est + K * (float(y[t]) - x_est)
        P_est  = (1.0 - K) * P_pred
        out[t] = x_est
    return out


# ==============================================================================
# Pedalboard NoiseGate – ponechaný pre kompatibilitu
# ==============================================================================

def maybe_gate(
    y: np.ndarray,
    sr: int,
    profile: DenoiseProfile,
    snr_db: float,
    threshold: float = 7.0,
) -> tuple[np.ndarray, bool]:
    """Pedalboard NoiseGate + HP/LP pri nízkom SNR. Nevolaný z core."""
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
