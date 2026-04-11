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
    gate_threshold_db: float
    gate_ratio:        float
    gate_attack_ms:    float
    gate_release_ms:   float
    highpass_hz:       float
    strength_low:      float   # 0–2 kHz  – konzervatívne
    strength_mid:      float   # 2–6 kHz
    strength_high:     float   # >6 kHz   – najnižšia vždy


# Pevné profily pre každý žáner.
# Poznámka: gate_* a highpass_hz parametre sú rezervované (NoiseGate je
# odstránený z aktívnej pipeline). Reálne vplývajú len strength_low/mid/high.
#
# Frekvenčné pásma v pipeline (spectral.py):
#   strength_low  → sub_bass (0–250 Hz) + bass_mid (250–2 kHz)
#   strength_mid  → upper_mid (2–6 kHz)
#   strength_high → highs (6–12 kHz) + air (12 kHz+, × 0.6)
#
#              gate_db  ratio  att   rel     hp    s_lo  s_mid  s_hi
GENRE_PROFILES: dict[str, DenoiseProfile] = {

    # Classical: ladiť konzervatívne — pianissimo pasáže, dlhé dozvuky,
    # harmonické bohatstvo sláčikov a dychových nástrojov.
    # Referenčné hodnoty z predchádzajúceho ladenia.
    "classical":  DenoiseProfile(-45, 3.0,  5.0, 200.0,  40, 0.72, 0.70, 0.65),

    # Jazz: podobný classical, ale činely (4–12 kHz) sú definujúce →
    # nižší strength_high. Dlhší release pre piánové dozvuky.
    "jazz":       DenoiseProfile(-44, 3.0,  5.0, 180.0,  40, 0.70, 0.68, 0.60),

    # Rock: elektrická gitara má bohaté harmoniky v mid-pásme (1–5 kHz) →
    # nižší strength_mid oproti pôvodnému 0.73. Basová gitara začína ~41 Hz.
    "rock":       DenoiseProfile(-38, 4.0,  3.0, 120.0,  50, 0.74, 0.70, 0.67),

    # Metal: extrémne husté harmonické spektrum (skreslené gitary do 8+ kHz) →
    # príliš agresívne filtrovanie ničí "wall of sound". Zníženie z 0.78/0.75.
    "metal":      DenoiseProfile(-35, 5.0,  2.0,  80.0,  60, 0.74, 0.72, 0.70),

    # Pop: spracované nahrávky, hlas v 200–3 000 Hz je kritický →
    # miernejší strength_mid pre zachovanie vokálnej zrozumiteľnosti.
    "pop":        DenoiseProfile(-40, 3.5,  5.0, 150.0,  40, 0.73, 0.70, 0.66),

    # Hip-hop: sub-bas a 808 (40–80 Hz) sú žánrovo definujúce →
    # strength_low znížený z 0.75 na 0.65, aby denoiser neoškriabal bas.
    # Dlhší release pre vokálne pasáže.
    "hiphop":     DenoiseProfile(-40, 4.0,  3.0, 150.0,  30, 0.65, 0.72, 0.67),

    # Electronic: syntetizátorové pady a textúry obsahujú úmyselný šum
    # (biely šum v padoch, vzduch v syntézach) — príliš agresívne filtrovanie
    # ničí zvukový dizajn. Zníženie všetkých pásiem.
    "electronic": DenoiseProfile(-36, 4.5,  2.0,  80.0,  30, 0.68, 0.68, 0.63),

    # Blues: gitarové slides a bends potrebujú sustain, harmonika má harmoniky
    # vo vysokom mid-pásme. Miernejší ako classical, menej ako rock.
    "blues":      DenoiseProfile(-43, 3.0,  5.0, 180.0,  40, 0.70, 0.68, 0.62),

    # Country: akustická gitara, husle, banjo (dôležité vysoké harmoniky) →
    # miernejší strength_high oproti classical pre zachovanie jasnosti tónov.
    "country":    DenoiseProfile(-42, 3.0,  5.0, 160.0,  40, 0.71, 0.69, 0.63),

    # Reggae: basová linka (40–120 Hz) je melodickým základom žánru →
    # strength_low znížený z 0.73 na 0.65. Dlhší release pre sustain nôt.
    "reggae":     DenoiseProfile(-40, 3.5,  4.0, 160.0,  30, 0.65, 0.70, 0.64),

    # Disco: štyri činely na každý takt (8–12 kHz) a sláčiková sekcia →
    # strength_high znížený z 0.68 na 0.62 pre zachovanie shimmer.
    "disco":      DenoiseProfile(-38, 4.0,  3.0, 120.0,  40, 0.74, 0.72, 0.62),

    # Default: vyvážená základňa pre neznámy žáner.
    "default":    DenoiseProfile(-40, 4.0,  4.0, 120.0,  40, 0.72, 0.70, 0.65),
}


def get_profile(genres: list[dict] | None) -> tuple[DenoiseProfile, str]:
    """
    Vráti profil a jeho textový popis na základe výstupu classify_genre().

    Ak je istota top žánru >= 60 %, použije sa priamo jeho profil.
    Inak sa interpoluje medzi top-2 žánrami podľa ich pravdepodobností.

    Args:
        genres: zoznam {"genre": str, "probability": float} z classify_genre()
                alebo None pre default profil.

    Returns:
        (DenoiseProfile, label_str)
    """
    if not genres:
        return GENRE_PROFILES["default"], "default"

    top = genres[0]
    if top["probability"] >= 60.0:
        profile = GENRE_PROFILES.get(top["genre"].lower(), GENRE_PROFILES["default"])
        label   = f"{top['genre']} ({top['probability']}%)"
        return profile, label

    # Interpolácia medzi top-2
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
    )
    label = f"{g1['genre']} + {g2['genre']} (mix {w1:.0%}/{w2:.0%})"
    return mixed, label


def adapt_profile(
    profile: DenoiseProfile,
    noise_scores: dict,
) -> DenoiseProfile:
    """
    Dynamicky upraví sily filtrovania profilu podľa detekovaného šumu.

    Logika:
      Biely šum   (slope ~0)    → rovnomerné filtrovanie, výšky silnejšie
      Ružový šum  (slope < -0.8)→ bas/stred silnejšie, výšky slabšie
      Impulzný    (skóre > 0.3) → nízke s_lo/s_mid (zachová tóny), mediánový
                                   filter rieši impulzy zvlášť
      Nestacionárny (skóre>0.4) → miernejšie celkovo, šum sa mení v čase

    Args:
        profile:      základný žánrový profil
        noise_scores: dict z detect_noise_type() vrátane "spectral_slope"

    Returns:
        upravený DenoiseProfile s prispôsobenými strength hodnotami
    """
    s_lo  = profile.strength_low
    s_mid = profile.strength_mid
    s_hi  = profile.strength_high

    slope       = noise_scores.get("spectral_slope", 0.0)
    impulsive   = noise_scores.get("impulsive",      0.0)
    nonstat     = noise_scores.get("nonstationary",  0.0)

    # --- Biely šum (plochý spektrálny sklon, slope > -0.5) ---
    # Energia šumu rovnomerná, ucho ho počuje najviac vo výškach
    if slope > -0.5:
        hi_boost = min(0.30, (0.5 - slope) * 0.25)
        s_hi  = min(s_hi  + hi_boost, 0.90)
        s_mid = min(s_mid + hi_boost * 0.6, 0.90)
        s_lo  = min(s_lo  + hi_boost * 0.3, 0.90)

    # --- Prechodná zóna (-0.5 až -0.8): zmiešaný charakter ---
    # slope -0.56 sem patrí – rovnomerný boost všetkých pásiem
    elif slope >= -0.8:
        boost = min(0.20, abs(slope + 0.5) * 0.30 + 0.10)
        s_hi  = min(s_hi  + boost, 0.90)
        s_mid = min(s_mid + boost, 0.90)
        s_lo  = min(s_lo  + boost * 0.5, 0.90)

    # --- Ružový / hnedý šum (slope < -0.8): dominujú basy ---
    else:
        bass_boost = min(0.25, abs(slope + 0.8) * 0.20)
        s_lo  = min(s_lo  + bass_boost, 0.90)
        s_mid = min(s_mid + bass_boost * 0.5, 0.90)
        s_hi  = max(s_hi  - 0.05, 0.10)

    # --- Impulzný šum (praskanie) ---
    # Mediánový filter rieši špičky, MMSE nemusí byť agresívny
    if impulsive > 0.3:
        reduction = min(impulsive * 0.3, 0.15)
        s_lo  = max(s_lo  - reduction, 0.05)
        s_mid = max(s_mid - reduction, 0.05)
        s_hi  = max(s_hi  - reduction, 0.05)

    # --- Nestacionárny šum (prostredie, vietor, kroky) ---
    # Šum sa mení v čase → príliš agresívne filtrovanie zanechá artefakty
    if nonstat > 0.4:
        reduction = min((nonstat - 0.4) * 0.25, 0.12)
        s_lo  = max(s_lo  - reduction, 0.05)
        s_mid = max(s_mid - reduction, 0.05)
        s_hi  = max(s_hi  - reduction, 0.05)

    return DenoiseProfile(
        gate_threshold_db = profile.gate_threshold_db,
        gate_ratio        = profile.gate_ratio,
        gate_attack_ms    = profile.gate_attack_ms,
        gate_release_ms   = profile.gate_release_ms,
        highpass_hz       = profile.highpass_hz,
        strength_low      = round(s_lo,  3),
        strength_mid      = round(s_mid, 3),
        strength_high     = round(s_hi,  3),
    )