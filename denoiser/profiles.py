"""
profiles.py – Žánrové profily pre denoising.

Každý profil definuje parametre pre noise gate (pedalboard)
a silu filtrovania pre tri frekvenčné pásma.
Ak model nie je dostatočne istý (< 60 %), profil sa interpoluje
medzi dvomi najpravdepodobnejšími žánrami.
"""

from dataclasses import dataclass


@dataclass
class DenoiseProfile:
    # Noise gate parametre (rezervované)
    gate_threshold_db: float
    gate_ratio:        float
    gate_attack_ms:    float
    gate_release_ms:   float
    highpass_hz:       float

    # Sily filtrovania per frekvenčné pásmo
    strength_low:      float   # 0–2 kHz
    strength_mid:      float   # 2–6 kHz
    strength_high:     float   # >6 kHz

    # Dynamické parametre – menia sa podľa žánru aj detekovaného šumu
    alpha_ns:    float = 0.98  # MMSE-LSA vyhladzovanie (0.95=rýchle, 0.99=hladké)
    window_sec:  float = 0.4   # minimum statistics okno v sekundách
    bias:        float = 2.5   # bias korekcia minimum statistics
    n_fft_ms:    float = 25.0  # FFT okno v ms (väčšie = lepšie frekv. rozlíšenie)
    mask_floor:  float = 0.02  # minimálna hodnota masky


# Frekvenčné pásma v pipeline (spectral.py):
#   strength_low  → sub_bass (0–250 Hz) + bass_mid (250–2 kHz)
#   strength_mid  → upper_mid (2–6 kHz)
#   strength_high → highs (6–12 kHz) + air (12 kHz+, × 0.6)
#
# gate_db  ratio  att   rel     hp    s_lo  s_mid  s_hi  alpha  win   bias  fft_ms floor
GENRE_PROFILES: dict[str, DenoiseProfile] = {

    # Classical: jemné tóny, dlhý sustain, pomalé zmeny
    # → pomalší MMSE (alpha 0.99), dlhé okno, väčší FFT
    "classical":  DenoiseProfile(-45, 3.0,  5.0, 200.0,  40, 0.72, 0.70, 0.65, 0.99, 0.6, 2.5, 40.0, 0.01),

    # Jazz: podobný classical, činely (4–12 kHz) sú definujúce
    "jazz":       DenoiseProfile(-44, 3.0,  5.0, 180.0,  40, 0.70, 0.68, 0.60, 0.99, 0.6, 2.5, 40.0, 0.01),

    # Blues: gitarové slides, harmonika vo vysokom mid-pásme
    "blues":      DenoiseProfile(-43, 3.0,  5.0, 180.0,  40, 0.70, 0.68, 0.62, 0.99, 0.6, 2.5, 40.0, 0.01),

    # Pop: hlas v 200–3000 Hz kritický, stredné tempo
    "pop":        DenoiseProfile(-40, 3.5,  5.0, 150.0,  40, 0.73, 0.70, 0.66, 0.98, 0.4, 2.5, 32.0, 0.02),

    # Country: akustická gitara, husle, banjo – vysoké harmoniky dôležité
    "country":    DenoiseProfile(-42, 3.0,  5.0, 160.0,  40, 0.71, 0.69, 0.63, 0.98, 0.4, 2.5, 32.0, 0.02),

    # Reggae: basová linka (40–120 Hz) je melodickým základom
    "reggae":     DenoiseProfile(-40, 3.5,  4.0, 160.0,  30, 0.70, 0.72, 0.65, 0.98, 0.4, 2.5, 32.0, 0.02),

    # Rock: elektrická gitara, bicie – dynamické zmeny
    "rock":       DenoiseProfile(-38, 4.0,  3.0, 120.0,  50, 0.74, 0.70, 0.67, 0.96, 0.3, 2.8, 25.0, 0.01),

    # Hip-hop: sub-bas a 808 (40–80 Hz) definujúce – strength_low nižší
    "hiphop":     DenoiseProfile(-40, 4.0,  3.0, 150.0,  30, 0.70, 0.74, 0.68, 0.96, 0.3, 3.0, 25.0, 0.01),

    # Disco: činely na každý takt, sláčiková sekcia
    "disco":      DenoiseProfile(-38, 4.0,  3.0, 120.0,  40, 0.74, 0.72, 0.62, 0.96, 0.3, 2.8, 25.0, 0.01),

    # Metal: husté harmonické spektrum, rýchle transienty
    "metal":      DenoiseProfile(-35, 5.0,  2.0,  80.0,  60, 0.74, 0.72, 0.70, 0.94, 0.2, 3.0, 20.0, 0.02),

    # Electronic: úmyselný šum v padoch – konzervatívnejšie
    "electronic": DenoiseProfile(-36, 4.5,  2.0,  80.0,  30, 0.68, 0.68, 0.63, 0.94, 0.2, 3.0, 20.0, 0.02),

    # Default: vyvážená základňa pre neznámy žáner
    "default":    DenoiseProfile(-40, 4.0,  4.0, 120.0,  40, 0.72, 0.70, 0.65, 0.97, 0.4, 2.5, 30.0, 0.02),
}


def get_profile(genres: list[dict] | None) -> tuple[DenoiseProfile, str]:
    """
    Vráti profil a jeho textový popis na základe výstupu classify_genre().

    Ak je istota top žánru >= 60 %, použije sa priamo jeho profil.
    Inak sa interpoluje medzi top-2 žánrami podľa ich pravdepodobností.
    Interpolujú sa VŠETKY polia vrátane dynamických parametrov.
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

    # Interpolujeme VŠETKY polia – inak mixed profil dostane default hodnoty
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
    Dynamicky upraví VŠETKY parametre profilu podľa detekovaného šumu.

    Biely šum   (slope > -0.5) → výšky+, bias+, window+, n_fft+
    Prechodná   (-0.5 až -0.8) → rovnomerný boost, bias+
    Ružový šum  (slope < -0.8) → bas+, výšky-, n_fft+
    Impulzný    (skóre > 0.3)  → všetky strength-, kratší window, vyšší floor
    Nestacionárny (skóre > 0.4)→ všetky strength-, kratší window, nižší bias
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

    # --- Biely šum (slope > -0.5) ---
    if slope > -0.5:
        hi_boost   = min(0.30, (0.5 - slope) * 0.25)
        s_hi       = min(s_hi  + hi_boost,       0.90)
        s_mid      = min(s_mid + hi_boost * 0.8, 0.90)
        s_lo       = min(s_lo  + hi_boost * 0.7, 0.90)  # biely šum je aj v basoch!
        bias       = min(bias  + 1.2,             5.0)   # stacionárny → vyšší bias
        window_sec = min(window_sec + 0.2,        1.0)
        n_fft_ms   = min(n_fft_ms   + 10.0,       50.0)  # lepšie frekv. rozlíšenie

    # --- Prechodná zóna (-0.5 až -0.8) ---
    elif slope >= -0.8:
        boost    = min(0.20, abs(slope + 0.5) * 0.30 + 0.10)
        s_hi     = min(s_hi  + boost,       0.90)
        s_mid    = min(s_mid + boost,       0.90)
        s_lo     = min(s_lo  + boost * 0.5, 0.90)
        bias     = min(bias  + 0.5,         4.0)

    # --- Ružový / hnedý šum (slope < -0.8) ---
    else:
        bass_boost = min(0.25, abs(slope + 0.8) * 0.20)
        s_lo       = min(s_lo  + bass_boost,       0.90)
        s_mid      = min(s_mid + bass_boost * 0.5, 0.90)
        s_hi       = max(s_hi  - 0.05,             0.10)
        n_fft_ms   = min(n_fft_ms + 10.0,          50.0)

    # --- Impulzný šum ---
    if impulsive > 0.3:
        reduction  = min(impulsive * 0.3, 0.15)
        s_lo       = max(s_lo  - reduction, 0.05)
        s_mid      = max(s_mid - reduction, 0.05)
        s_hi       = max(s_hi  - reduction, 0.05)
        window_sec = max(window_sec - 0.1,  0.15)
        mask_floor = min(mask_floor + 0.02, 0.08)

    # --- Nestacionárny šum ---
    if nonstat > 0.4:
        reduction  = min((nonstat - 0.4) * 0.25, 0.12)
        s_lo       = max(s_lo  - reduction, 0.05)
        s_mid      = max(s_mid - reduction, 0.05)
        s_hi       = max(s_hi  - reduction, 0.05)
        window_sec = max(window_sec - 0.15, 0.15)
        bias       = max(bias - 0.5,        1.2)
        alpha_ns   = min(alpha_ns + 0.005,  0.995)

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