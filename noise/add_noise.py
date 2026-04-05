import numpy as np
import soundfile as sf
import argparse
import os


def _estimate_snr(y, sr):
    """Rovnaka metoda ako v denoiser.py – najtichsi usek 0.5s."""
    window = int(sr * 0.5)
    if len(y) < window:
        window = max(1, len(y))
    step      = max(window // 2, 1)
    min_var   = float("inf")
    noise_var = float(np.var(y)) + 1e-10
    for start in range(0, max(1, len(y) - window), step):
        segment = y[start : start + window]
        v = float(np.var(segment))
        if v < min_var:
            min_var   = v
            noise_var = max(v, 1e-10)
    signal_var = float(np.var(y)) + 1e-10
    return 10.0 * np.log10(signal_var / noise_var)


def add_noise(input_path, output_path, noise_type="white", noise_level=0.02):
    """
    Zasumi audio subor pre testovanie denoisingu.

    noise_type:
        white    – biely sum (nahodne frekvencie)
        pink     – ruzovy sum (viac basov, prirodzenejsi)
        crackle  – praskanie ako stara vinylova platna

    noise_level:
        0.01 = jemny sum
        0.02 = stredny sum (odporucane pre testovanie)
        0.05 = silny sum
    """
    y, sr = sf.read(input_path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    rng = np.random.default_rng(42)

    if noise_type == "white":
        noise = rng.normal(0, noise_level, len(y)).astype(np.float32)

    elif noise_type == "pink":
        white = rng.normal(0, 1, len(y))
        fft   = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(len(y))
        freqs[0] = 1e-10
        fft_pink = fft * (1 / np.sqrt(freqs))
        noise    = np.fft.irfft(fft_pink, n=len(y)).astype(np.float32)
        noise    = noise / noise.std() * noise_level

    elif noise_type == "crackle":
        noise        = np.zeros(len(y), dtype=np.float32)
        num_crackles = int(len(y) * 0.001)
        positions    = rng.integers(0, len(y), num_crackles)
        amplitudes   = rng.uniform(-noise_level * 10, noise_level * 10, num_crackles)
        noise[positions] = amplitudes

    else:
        raise ValueError(f"Neznamy typ sumu: {noise_type}. Pouzi: white, pink, crackle")

    y_noisy = np.clip(y + noise, -1.0, 1.0).astype(np.float32)

    # SNR rovnakou metodou ako denoiser – vysledky su porovnatelne
    snr = _estimate_snr(y_noisy, sr)

    sf.write(output_path, y_noisy, sr)

    return {
        "snr":         round(float(snr), 2),
        "noise_type":  noise_type,
        "noise_level": noise_level,
        "output_path": output_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pridaj sum do audio suboru")
    parser.add_argument("input",  help="Vstupny subor (mp3/wav)")
    parser.add_argument("output", help="Vystupny subor (wav)")
    parser.add_argument("--type",  default="white",
                        choices=["white", "pink", "crackle"],
                        help="Typ sumu (default: white)")
    parser.add_argument("--level", type=float, default=0.02,
                        help="Uroven sumu 0.01-0.05 (default: 0.02)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Subor nenajdeny: {args.input}")
        exit(1)

    result = add_noise(args.input, args.output, args.type, args.level)
    print(f"Vstup:      {args.input}")
    print(f"Vystup:     {args.output}")
    print(f"Typ sumu:   {result['noise_type']}")
    print(f"Uroven:     {result['noise_level']}")
    print(f"SNR:        {result['snr']:.2f} dB  (cim nizssie, tym viac sumu)")