"""
add_noise.py – pridanie rôznych typov šumu do audio súboru pre testovanie denoisingu.

Podporované typy šumu:
    white    – biely šum (plochý spektrálny výkon)
    pink     – ružový šum (1/f, prirodzenejší, viac basov)
    brown    – hnedý / červený šum (1/f², hlboký rumble)
    hiss     – tape-like hiss (vysokofrekvenčný, ako stará páska)
    hum      – 50 Hz sieťový hum + harmonické (európska sieť)
    crackle  – vinylové praskanie (multi-sample impulzy s exponenciálnym rozpadom)
    clicks   – zriedkavé, výrazné kliknutia (ako škriabance na platni)
    vinyl    – kombinácia pink + crackle + low rumble (realistická stará nahrávka)

SNR sa meria rovnakou metódou ako denoiser (minimum statistics),
takže čísla z add_noise a denoise_audio sú priamo porovnateľné.
"""

import numpy as np
import soundfile as sf
import argparse
import os

from denoiser.noise_estimation import estimate_snr as _estimate_snr


# ==============================================================================
# Generátory šumu
# ==============================================================================

def _white(n: int, level: float, rng: np.random.Generator) -> np.ndarray:
    """Biely šum – gaussovský, plochý výkon cez všetky frekvencie."""
    return rng.normal(0, level, n).astype(np.float32)


def _colored_noise(n: int, level: float, exponent: float,
                   rng: np.random.Generator,
                   sr: int = 44100,
                   hp_cutoff_hz: float = 0.0) -> np.ndarray:
    """
    Farebný šum s výkonovým spektrom 1/f^exponent.
        exponent = 0  → biely (plochý)
        exponent = 1  → ružový (-3 dB/oct)
        exponent = 2  → hnedý (-6 dB/oct)

    Implementácia:
      1. Biely šum → FFT
      2. Násobenie 1/f^(exponent/2) v amplitúdovom spektre
      3. **DC bin vynulovaný** – inak 1/0 = ∞
      4. Voliteľný sub-sonic HP (`hp_cutoff_hz`) – bins pod cutoffom vynulované.
         Dôležité pre brown šum: 1/f² dáva 98 % výkonu pod 50 Hz, takže
         bez HP by RMS normalizácia vložila takmer všetku energiu do
         nepočuteľného sub-basu a počuteľná časť by bola ticho.
      5. IRFFT → RMS normalizácia na požadovaný level
    """
    white = rng.normal(0, 1, n)
    spec  = np.fft.rfft(white)
    freqs_norm = np.fft.rfftfreq(n)          # normované (0..0.5)
    freqs_hz   = freqs_norm * sr             # v Hz

    # Škálovanie amplitúd – DC bin necháme nulový
    scale = np.zeros_like(freqs_norm)
    scale[1:] = 1.0 / (freqs_norm[1:] ** (exponent / 2.0))
    spec_colored = spec * scale
    spec_colored[0] = 0.0

    # Sub-sonic HP filter – vynulovať bins pod cutoffom
    if hp_cutoff_hz > 0.0:
        spec_colored[freqs_hz < hp_cutoff_hz] = 0.0

    noise = np.fft.irfft(spec_colored, n=n).astype(np.float32)
    noise -= float(np.mean(noise))
    rms = float(np.sqrt(np.mean(noise ** 2))) + 1e-12
    return noise * (level / rms)


def _pink(n: int, level: float, rng: np.random.Generator,
          sr: int = 44100) -> np.ndarray:
    """Ružový šum – 1/f PSD. Jemný 10 Hz HP ako ochrana pred rumble."""
    return _colored_noise(n, level, exponent=1.0, rng=rng,
                          sr=sr, hp_cutoff_hz=10.0)


def _brown(n: int, level: float, rng: np.random.Generator,
           sr: int = 44100) -> np.ndarray:
    """
    Hnedý / červený šum – 1/f² PSD, hlboký rumble (ako tečúca voda).

    HP 30 Hz je KRITICKÝ: bez neho by 98 % energie skončilo pod 50 Hz
    (hlboko v sub-basoch) a počuteľná časť šumu by bola prakticky
    neexistujúca. S HP 30 Hz je normalizovaný výkon v audio pásme
    skutočne počuť ako hlboké "hučanie".
    """
    return _colored_noise(n, level, exponent=2.0, rng=rng,
                          sr=sr, hp_cutoff_hz=30.0)


def _hiss(n: int, level: float, rng: np.random.Generator) -> np.ndarray:
    """
    Tape-like hiss – biely šum s jemným high-pass tvarovaním.
    Simuluje zvuk analógovej pásky alebo šum z predzosilňovača.
    Implementácia: biely šum × (1 + f^0.5), čím sa zvýraznia vysoké
    frekvencie o ~10 dB v hornom pásme.
    """
    white = rng.normal(0, 1, n)
    spec  = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    shaped = spec * (1.0 + np.sqrt(freqs) * 3.0)
    shaped[0] = 0.0
    noise = np.fft.irfft(shaped, n=n).astype(np.float32)
    noise -= float(np.mean(noise))
    rms = float(np.sqrt(np.mean(noise ** 2))) + 1e-12
    return noise * (level / rms)


def _hum(n: int, level: float, sr: int,
         fundamental_hz: float = 50.0) -> np.ndarray:
    """
    Sieťový hum – základná frekvencia + 3 harmonické s klesajúcou amplitúdou.
    Default 50 Hz (európska sieť); pre USA nastav fundamental_hz=60.

    Amplitúdy harmoník: 1.0, 0.5, 0.25, 0.15 (typické pre zlé uzemnenie).
    Pridaný je aj jemný amplitúdový drift, aby hum nebol úplne "čistý".
    """
    t = np.arange(n) / sr
    harmonics = [(1.0, 1.0), (2.0, 0.5), (3.0, 0.25), (4.0, 0.15)]
    hum = np.zeros(n, dtype=np.float32)
    for mult, amp in harmonics:
        f = fundamental_hz * mult
        if f >= sr / 2:
            break
        # Malý drift v amplitúde – reálny hum nie je dokonalá sínusovka
        drift = 1.0 + 0.05 * np.sin(2 * np.pi * 0.3 * t)
        hum += (amp * drift * np.sin(2 * np.pi * f * t)).astype(np.float32)
    rms = float(np.sqrt(np.mean(hum ** 2))) + 1e-12
    return hum * (level / rms)


def _crackle(n: int, level: float, sr: int,
             rng: np.random.Generator,
             density_per_sec: float = 50.0) -> np.ndarray:
    """
    Vinylové praskanie – husté multi-sample impulzy s exponenciálnym rozpadom.

    Každý crackle má:
      - šírku 3–10 vzoriek (pri 44.1 kHz = 0.07–0.23 ms)
      - exponenciálny rozpad (e^(-k/2)) – prirodzenejšie ako delta impulzy
      - náhodnú polaritu a amplitúdu

    density_per_sec riadi hustotu: 50 = jemné praskanie, 200 = silné
    praskanie starej zničenej platne.

    Oproti pôvodnej verzii (single-sample delta):
      - širšie impulzy → realistickejší zvuk a vyšší detect kurtosis
      - exp decay → menej "click-like", viac "crackle-like"
    """
    noise = np.zeros(n, dtype=np.float32)
    num_crackles = int(density_per_sec * (n / sr))
    if num_crackles == 0:
        return noise

    positions = rng.integers(0, n, num_crackles)
    for pos in positions:
        width = int(rng.integers(3, 11))
        end   = min(pos + width, n)
        amp   = float(rng.uniform(0.4, 1.0)) * rng.choice([-1.0, 1.0])
        decay = np.exp(-np.arange(end - pos) / 2.0)
        noise[pos:end] += (amp * decay).astype(np.float32)

    # RMS normalizácia – level je cielený RMS crackle zložky
    rms = float(np.sqrt(np.mean(noise ** 2))) + 1e-12
    return noise * (level / rms)


def _clicks(n: int, level: float, sr: int,
            rng: np.random.Generator,
            density_per_sec: float = 3.0) -> np.ndarray:
    """
    Zriedkavé výrazné kliknutia – ako škrabance na platni.
    Podobné crackle, ale oveľa menšia hustota a väčšia amplitúda.
    """
    noise = np.zeros(n, dtype=np.float32)
    num_clicks = int(density_per_sec * (n / sr))
    if num_clicks == 0:
        return noise

    positions = rng.integers(0, n, num_clicks)
    for pos in positions:
        width = int(rng.integers(8, 25))
        end   = min(pos + width, n)
        amp   = float(rng.uniform(0.7, 1.0)) * rng.choice([-1.0, 1.0])
        decay = np.exp(-np.arange(end - pos) / 4.0)
        noise[pos:end] += (amp * decay).astype(np.float32)

    rms = float(np.sqrt(np.mean(noise ** 2))) + 1e-12
    return noise * (level / rms)


def _vinyl(n: int, level: float, sr: int,
           rng: np.random.Generator) -> np.ndarray:
    """
    Kombinovaný "vinyl" šum – simulácia starej gramofónovej platne.

    Zloženie:
      - 50 % pink šum (podkladový hiss)
      - 15 % brown šum (low-frequency rumble z motora)
      - 35 % crackle (praskanie)

    Toto je najrealistickejší testovací prípad pre audio restoration.
    """
    pink_part    = _pink(n, level * 0.50, rng, sr=sr)
    brown_part   = _brown(n, level * 0.15, rng, sr=sr)
    crackle_part = _crackle(n, level * 0.35, sr, rng, density_per_sec=80.0)
    return (pink_part + brown_part + crackle_part).astype(np.float32)


# ==============================================================================
# Hlavná funkcia
# ==============================================================================

NOISE_TYPES = ["white", "pink", "brown", "hiss", "hum", "crackle", "clicks", "vinyl"]


def add_noise(input_path: str, output_path: str,
              noise_type: str = "white",
              noise_level: float = 0.02) -> dict:
    """
    Zašumí audio súbor pre testovanie denoisingu.

    Args:
        input_path:  vstupný súbor (mp3/wav/flac)
        output_path: výstupný súbor (wav)
        noise_type:  jeden z NOISE_TYPES
        noise_level: RMS amplitúda šumu
                     0.01 = jemný | 0.02 = stredný | 0.05 = silný

    Returns:
        dict so SNR, typom a cestou k výstupu
    """
    if noise_type not in NOISE_TYPES:
        raise ValueError(
            f"Neznámy typ šumu: {noise_type}. Použi: {', '.join(NOISE_TYPES)}"
        )

    y, sr = sf.read(input_path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    n = len(y)

    rng = np.random.default_rng(42)

    if   noise_type == "white":   noise = _white(n, noise_level, rng)
    elif noise_type == "pink":    noise = _pink(n, noise_level, rng, sr=sr)
    elif noise_type == "brown":   noise = _brown(n, noise_level, rng, sr=sr)
    elif noise_type == "hiss":    noise = _hiss(n, noise_level, rng)
    elif noise_type == "hum":     noise = _hum(n, noise_level, sr)
    elif noise_type == "crackle": noise = _crackle(n, noise_level, sr, rng)
    elif noise_type == "clicks":  noise = _clicks(n, noise_level, sr, rng)
    elif noise_type == "vinyl":   noise = _vinyl(n, noise_level, sr, rng)

    y_noisy = np.clip(y + noise, -1.0, 1.0).astype(np.float32)

    # SNR cez minimum statistics – zhodné s denoiserom, čísla porovnateľné
    snr = _estimate_snr(y_noisy, sr)

    sf.write(output_path, y_noisy, sr)

    return {
        "snr":         round(float(snr), 2),
        "noise_type":  noise_type,
        "noise_level": noise_level,
        "output_path": output_path,
    }


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pridaj šum do audio súboru")
    parser.add_argument("input",  help="Vstupný súbor (mp3/wav)")
    parser.add_argument("output", help="Výstupný súbor (wav)")
    parser.add_argument("--type",  default="white", choices=NOISE_TYPES,
                        help="Typ šumu (default: white)")
    parser.add_argument("--level", type=float, default=0.02,
                        help="Úroveň šumu 0.01–0.05 (default: 0.02)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Súbor nenájdený: {args.input}")
        exit(1)

    result = add_noise(args.input, args.output, args.type, args.level)
    print(f"Vstup:      {args.input}")
    print(f"Výstup:     {args.output}")
    print(f"Typ šumu:   {result['noise_type']}")
    print(f"Úroveň:     {result['noise_level']}")
    print(f"SNR:        {result['snr']:.2f} dB  (čím nižšie, tým viac šumu)")