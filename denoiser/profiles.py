"""
profiles.py – Žánrové profily pre denoising.

Každý profil definuje parametre pre spektrálne filtrovanie pre tri
frekvenčné pásma + dynamické parametre pre MMSE-LSA a minimum statistics.

Parametre `strength_*` sa používajú ako EXPONENT na MMSE-LSA gain:
    effective_gain = mmse_gain ** strength
  strength = 1.0  → čistý MMSE-LSA (štandard)
  strength < 1.0  → mäkšie filtrovanie (menšia supresia)
  strength > 1.0  → agresívnejšie filtrovanie (väčšia supresia)

Toto je zásadná zmena oproti pôvodnému lineárnemu blendu `s·G + (1-s)`,
ktorý shora limitoval supresiu na max −10 dB pre bežné hodnoty s ~ 0.7
a efektívne robil `mask_floor` nepoužiteľný.
"""

from dataclasses import dataclass


@dataclass
class DenoiseProfile:
    # Noise gate parametre (rezervované – maybe_gate)
    gate_threshold_db: float
    gate_ratio:        float
    gate_attack_ms:    float
    gate_release_ms:   float
    highpass_hz:       float

    # Sily filtrovania per frekvenčné pásmo (EXPONENT na gain, nie lineárny blend)
    #   1.00 = čistý MMSE-LSA
    #   0.80 = mäkké (gitara, klavír, vokály – zachovaj sustain)
    #   1.10 = silnejšie (šumové pozadie v bicích, hustá produkcia)
    strength_low:      float   # 0–2 kHz
    strength_mid:      float   # 2–6 kHz
    strength_high:     float   # >6 kHz

    # Dynamické parametre – menia sa podľa žánru aj detekovaného šumu
    alpha_ns:    float = 0.96  # MMSE-LSA vyhladzovanie (0.92=rýchle, 0.98=hladké)
    window_sec:  float = 0.6   # minimum statistics okno v sekundách
    bias:        float = 1.6   # bias korekcia minimum statistics (nižší ako predtým –
                               # rekurzívne vyhladenie v MS už kompenzuje časť biasu)
    n_fft_ms:    float = 32.0  # FFT okno v ms (väčšie = lepšie frekv. rozlíšenie)
    mask_floor:  float = 0.05  # minimálna hodnota masky (0.05 ≈ −26 dB floor)


# Frekvenčné pásma v pipeline (spectral.py):
#   strength_low  → sub_bass (0–250 Hz) + bass_mid (250–2 kHz)
#   strength_mid  → upper_mid (2–6 kHz)
#   strength_high → highs (6–12 kHz) + air (12 kHz+, × 0.85)
#
# Rozsahy sú teraz diskriminujúce – medzi classical a metal je reálny
# počuteľný rozdiel, nie 0.72 vs 0.74.
#
#                          gate_db ratio att    rel     hp    s_lo s_mid s_hi  alpha  win  bias fft_ms floor
GENRE_PROFILES: dict[str, DenoiseProfile] = {

    # Classical: jemné tóny, dlhý sustain, pomalé zmeny
    # → jemné strengthy, pomalý MMSE (alpha 0.98), veľký FFT, nízky floor
    "classical":  DenoiseProfile(-45, 3.0,  5.0, 200.0,  40, 0.80, 0.78, 0.72, 0.98, 1.0, 1.5, 46.0, 0.04),

    # Jazz: podobný classical, činely definujúce, jemné brush na bicích
    "jazz":       DenoiseProfile(-44, 3.0,  5.0, 180.0,  40, 0.82, 0.78, 0.72, 0.98, 1.0, 1.5, 46.0, 0.04),

    # Blues: gitarové slides, harmonika, sustain – stred trochu silnejší
    "blues":      DenoiseProfile(-43, 3.0,  5.0, 180.0,  40, 0.85, 0.82, 0.78, 0.97, 0.8, 1.6, 40.0, 0.04),

    # Pop: hlas dominantný v 200–3 kHz, stredná dynamika
    "pop":        DenoiseProfile(-40, 3.5,  5.0, 150.0,  40, 0.92, 0.90, 0.85, 0.96, 0.6, 1.6, 32.0, 0.05),

    # Country: akustická gitara, husle, banjo – jemnejšie výšky kvôli harmonikám
    "country":    DenoiseProfile(-42, 3.0,  5.0, 160.0,  40, 0.88, 0.86, 0.82, 0.97, 0.8, 1.6, 40.0, 0.04),

    # Reggae: silný bas (40–120 Hz), skiffle drums
    "reggae":     DenoiseProfile(-40, 3.5,  4.0, 160.0,  30, 0.90, 0.90, 0.85, 0.96, 0.6, 1.6, 32.0, 0.05),

    # Rock: elektrická gitara, bicie – rýchlejší MMSE, menšie okno
    "rock":       DenoiseProfile(-38, 4.0,  3.0, 120.0,  50, 0.98, 0.95, 0.90, 0.94, 0.5, 1.8, 25.0, 0.06),

    # Hip-hop: sub-bas a 808 (40–80 Hz), transienty dôležité
    "hiphop":     DenoiseProfile(-40, 4.0,  3.0, 150.0,  30, 0.95, 0.98, 0.95, 0.94, 0.5, 1.8, 25.0, 0.06),

    # Disco: činely na každý takt, sláčiková sekcia v strede
    "disco":      DenoiseProfile(-38, 4.0,  3.0, 120.0,  40, 0.95, 0.92, 0.88, 0.94, 0.5, 1.8, 25.0, 0.06),

    # Metal: husté harmonické spektrum, rýchle transienty, robustný šum
    "metal":      DenoiseProfile(-35, 5.0,  2.0,  80.0,  60, 1.05, 1.00, 0.95, 0.92, 0.4, 2.0, 20.0, 0.07),

    # Electronic: úmyselný šum v padoch – konzervatívnejšie
    "electronic": DenoiseProfile(-36, 4.5,  2.0,  80.0,  30, 0.85, 0.85, 0.80, 0.94, 0.5, 1.6, 25.0, 0.05),

    # Default: vyvážený základ pre neznámy žáner
    "default":    DenoiseProfile(-40, 4.0,  4.0, 120.0,  40, 0.92, 0.90, 0.85, 0.96, 0.6, 1.7, 32.0, 0.05),
}


def get_profile(genres: list[dict] | None) -> tuple[DenoiseProfile, str]:
    """
    Vráti profil podľa výstupu classify_genre().

    Ak istota top žánru >= 60 %, použije sa priamo. Inak sa interpoluje
    medzi top-2 žánrami podľa pravdepodobností.
    """
    if not genres:
        return GENRE_PROFILES["default"], "default"

    top = genres[0]
    if top["probability"] >= 60.0:
        profile = GENRE_PROFILES.get(top["genre"].lower(), GENRE_PROFILES["default"])
        label   = f"{top['genre']} ({top['probability']}%)"
        return profile, label

    g1, g2 = genres[0], genres[1]
    p1 = GENRE_PROFILES.get(g1["genre"].lower(), GENRE_PROFILES["default"])
    p2 = GENRE_PROFILES.get(g2["genre"].lower(), GENRE_PROFILES["default"])
    w1 = g1["probability"] / (g1["probability"] + g2["probability"])
    w2 = 1.0 - w1

    mixed = DenoiseProfile(
        gate_threshold_db = p1.gate_threshold_db * w1 + p2.gate_threshold_db * w2,
        gate_ratio        = p1.gate_ratio        * w1 + p2.gate_ratio        * w2,
        gate_attack_ms    = p1.gate_attack_ms    * w1 + p2.gate_attack_ms    * w2,
        gate_release_ms   = p1.gate_release_ms   * w1 + p2.gate_release_ms   * w2,
        highpass_hz       = p1.highpass_hz       * w1 + p2.highpass_hz       * w2,
        strength_low      = p1.strength_low      * w1 + p2.strength_low      * w2,
        strength_mid      = p1.strength_mid      * w1 + p2.strength_mid      * w2,
        strength_high     = p1.strength_high     * w1 + p2.strength_high     * w2,
        alpha_ns          = p1.alpha_ns          * w1 + p2.alpha_ns          * w2,
        window_sec        = p1.window_sec        * w1 + p2.window_sec        * w2,
        bias              = p1.bias              * w1 + p2.bias              * w2,
        n_fft_ms          = p1.n_fft_ms          * w1 + p2.n_fft_ms          * w2,
        mask_floor        = p1.mask_floor        * w1 + p2.mask_floor        * w2,
    )
    label = f"{g1['genre']} + {g2['genre']} (mix {w1:.0%}/{w2:.0%})"
    return mixed, label


def adapt_profile(
    profile: DenoiseProfile,
    noise_scores: dict,
) -> DenoiseProfile:
    """
    Dynamicky upraví parametre profilu podľa detekovaného šumu.

    Rozsahy boli posunuté aby zodpovedali novým exponent-based strengthom.
    Strength ostane v [0.5, 1.3] – 1.3 už je veľmi agresívne.

    Biely šum   (slope > -0.5) → výšky+, bias+, window+
    Prechodná   (-0.5 až -0.8) → rovnomerný boost
    Ružový šum  (slope < -0.8) → bas+, výšky-
    Impulzný    (skóre > 0.3)  → strength-, vyšší floor
    Nestacionárny (skóre > 0.4)→ strength-, kratší window
    """
    s_lo       = profile.strength_low
    s_mid      = profile.strength_mid
    s_hi       = profile.strength_high
    alpha_ns   = profile.alpha_ns
    window_sec = profile.window_sec
    bias       = profile.bias
    n_fft_ms   = profile.n_fft_ms
    mask_floor = profile.mask_floor

    slope     = noise_scores.get("spectral_slope", 0.0)
    impulsive = noise_scores.get("impulsive",      0.0)
    nonstat   = noise_scores.get("nonstationary",  0.0)

    S_MAX = 1.30
    S_MIN = 0.50

    # --- Biely šum ---
    if slope > -0.5:
        hi_boost   = min(0.15, (0.5 - slope) * 0.15)
        s_hi       = min(s_hi  + hi_boost,       S_MAX)
        s_mid      = min(s_mid + hi_boost * 0.8, S_MAX)
        s_lo       = min(s_lo  + hi_boost * 0.6, S_MAX)
        bias       = min(bias  + 0.4,             3.0)
        window_sec = min(window_sec + 0.2,        1.2)
        n_fft_ms   = min(n_fft_ms   + 6.0,        50.0)

    # --- Prechodná zóna ---
    elif slope >= -0.8:
        boost    = min(0.12, abs(slope + 0.5) * 0.20 + 0.05)
        s_hi     = min(s_hi  + boost,       S_MAX)
        s_mid    = min(s_mid + boost,       S_MAX)
        s_lo     = min(s_lo  + boost * 0.5, S_MAX)
        bias     = min(bias  + 0.2,         2.8)

    # --- Ružový / hnedý šum ---
    else:
        bass_boost = min(0.15, abs(slope + 0.8) * 0.12)
        s_lo       = min(s_lo  + bass_boost,       S_MAX)
        s_mid      = min(s_mid + bass_boost * 0.5, S_MAX)
        s_hi       = max(s_hi  - 0.03,             S_MIN)
        n_fft_ms   = min(n_fft_ms + 6.0,           50.0)

    # --- Impulzný šum ---
    # Nezoslabujeme strength zbytočne – declick_lsar to vyrieši pred MMSE.
    # Zvýšime len floor aby sa nevyrobili artefakty na zvyškoch
    if impulsive > 0.3:
        mask_floor = min(mask_floor + 0.02, 0.10)
        window_sec = max(window_sec - 0.1,  0.3)

    # --- Nestacionárny šum ---
    if nonstat > 0.4:
        reduction  = min((nonstat - 0.4) * 0.15, 0.08)
        s_lo       = max(s_lo  - reduction, S_MIN)
        s_mid      = max(s_mid - reduction, S_MIN)
        s_hi       = max(s_hi  - reduction, S_MIN)
        window_sec = max(window_sec - 0.15, 0.3)
        bias       = max(bias - 0.3,        1.2)
        alpha_ns   = min(alpha_ns + 0.005,  0.99)

    return DenoiseProfile(
        gate_threshold_db = profile.gate_threshold_db,
        gate_ratio        = profile.gate_ratio,
        gate_attack_ms    = profile.gate_attack_ms,
        gate_release_ms   = profile.gate_release_ms,
        highpass_hz       = profile.highpass_hz,
        strength_low      = round(s_lo,       3),
        strength_mid      = round(s_mid,      3),
        strength_high     = round(s_hi,       3),
        alpha_ns          = round(alpha_ns,   3),
        window_sec        = round(window_sec, 2),
        bias              = round(bias,       2),
        n_fft_ms          = round(n_fft_ms,   1),
        mask_floor        = round(mask_floor, 3),
    )
