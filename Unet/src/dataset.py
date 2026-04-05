"""
dataset.py — Dataset a augmentácia pre tréning U-Net audio denoisera.

Zodpovednosti:
- Načítanie čistých nahrávok a šumových vzoriek
- Mixovanie čistého audia so šumom pri rôznych SNR úrovniach
- Konverzia na STFT spektrogramy
- Príprava tréningových párov (zašumený spektrogram, čistý spektrogram)
"""

import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDenoiserDataset(Dataset):
    """
    Dataset pre tréning audio denoisera.

    Pre každý sample:
    1. Načíta náhodný čistý WAV
    2. Načíta náhodný šum WAV
    3. Zmixuje ich pri náhodnom SNR
    4. Vráti STFT magnitúdy (zašumený, čistý) ako tréningový pár
    """

    def __init__(
        self,
        clean_dir: str,
        noise_dir: str,
        sample_rate: int = 44100,
        segment_length: int = 44100 * 8,  # 8 sekúnd
        n_fft: int = 4096,
        hop_length: int = 1024,
        snr_range: tuple = (0, 20),
        num_samples: int = 8000,
    ):
        """
        Args:
            clean_dir: cesta k priečinku s čistými WAV súbormi
            noise_dir: cesta k priečinku so šumovými WAV súbormi
            sample_rate: cieľový sample rate (všetko sa resampleuje)
            segment_length: dĺžka segmentu v samploch (sample_rate * sekundy)
            n_fft: veľkosť FFT okna
            hop_length: krok medzi FFT oknami
            snr_range: rozsah Signal-to-Noise Ratio v dB (min, max)
            num_samples: počet tréningových párov za epochu
        """
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.snr_range = snr_range
        self.num_samples = num_samples

        # Nájdi všetky WAV súbory
        self.clean_files = self._find_wav_files(clean_dir)
        self.noise_files = self._find_wav_files(noise_dir)

        if len(self.clean_files) == 0:
            raise ValueError(f"Žiadne WAV súbory v {clean_dir}")
        if len(self.noise_files) == 0:
            raise ValueError(f"Žiadne WAV súbory v {noise_dir}")

        print(f"[Dataset] Nájdených {len(self.clean_files)} čistých súborov")
        print(f"[Dataset] Nájdených {len(self.noise_files)} šumových súborov")

    def _find_wav_files(self, directory: str) -> list:
        """Rekurzívne nájde všetky WAV súbory v priečinku."""
        wav_files = []
        for root, _, files in os.walk(directory):
            for f in files:
                if f.lower().endswith(".wav"):
                    wav_files.append(os.path.join(root, f))
        return sorted(wav_files)

    def _load_audio(self, filepath: str) -> torch.Tensor:
        """
        Načíta WAV, skonvertuje na mono, resampleuje na cieľový sample rate.
        Vráti 1D tensor.
        """
        waveform, sr = torchaudio.load(filepath)

        # Konverzia na mono (priemer kanálov)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample ak treba
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        return waveform.squeeze(0)  # [samples]

    def _get_random_segment(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Vystrihne náhodný segment danej dĺžky.
        Ak je waveform kratší, opakuje ho (loop).
        """
        length = self.segment_length

        if waveform.shape[0] < length:
            # Loop: opakuj kým nie je dosť dlhý
            repeats = (length // waveform.shape[0]) + 1
            waveform = waveform.repeat(repeats)

        # Náhodný začiatok
        max_start = waveform.shape[0] - length
        start = random.randint(0, max_start)
        return waveform[start : start + length]

    def _mix_at_snr(
        self, clean: torch.Tensor, noise: torch.Tensor, snr_db: float
    ) -> torch.Tensor:
        """
        Zmixuje čistý signál so šumom pri danom SNR (v dB).

        SNR = 10 * log10(power_clean / power_noise)
        => power_noise_target = power_clean / (10 ^ (snr_db / 10))
        => scale = sqrt(power_noise_target / power_noise)
        """
        clean_power = clean.pow(2).mean()
        noise_power = noise.pow(2).mean()

        # Ochrana pred tichým signálom
        if noise_power < 1e-10 or clean_power < 1e-10:
            return clean

        # Škálovanie šumu na cieľový SNR
        target_noise_power = clean_power / (10 ** (snr_db / 10))
        scale = torch.sqrt(target_noise_power / noise_power)

        noisy = clean + scale * noise
        return noisy

    def _compute_stft_magnitude(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Vypočíta STFT a vráti magnitúdu spektrogramu.
        Výstup: [1, freq_bins, time_frames] — 1 kanál pre Conv2d.
        """
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft),
            return_complex=True,
        )

        # Magnitúda + channel dimenzia [1, F, T]
        magnitude = stft.abs().unsqueeze(0)
        return magnitude

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Vráti jeden tréningový pár:
        - noisy_mag: magnitúda zašumeného spektrogramu [1, F, T]
        - clean_mag: magnitúda čistého spektrogramu [1, F, T]
        """
        # 1. Načítaj náhodný čistý a šumový súbor
        clean_path = random.choice(self.clean_files)
        noise_path = random.choice(self.noise_files)

        clean_waveform = self._load_audio(clean_path)
        noise_waveform = self._load_audio(noise_path)

        # 2. Vystrihni náhodné segmenty rovnakej dĺžky
        clean_segment = self._get_random_segment(clean_waveform)
        noise_segment = self._get_random_segment(noise_waveform)

        # 3. Zmixuj pri náhodnom SNR
        snr_db = random.uniform(*self.snr_range)
        noisy_segment = self._mix_at_snr(clean_segment, noise_segment, snr_db)

        # 4. Normalizácia
        max_val = max(noisy_segment.abs().max(), clean_segment.abs().max(), 1e-8)
        clean_segment = clean_segment / max_val
        noisy_segment = noisy_segment / max_val

        # 5. Vypočítaj STFT magnitúdy
        clean_mag = self._compute_stft_magnitude(clean_segment)
        noisy_mag = self._compute_stft_magnitude(noisy_segment)

        return noisy_mag, clean_mag


def get_dataloaders(
    clean_dir: str,
    noise_dir: str,
    batch_size: int = 8,
    num_samples_train: int = 5000,
    num_samples_val: int = 500,
    num_workers: int = 0,
    **kwargs,
):
    """
    Vytvorí tréningový a validačný DataLoader.
    """
    train_dataset = AudioDenoiserDataset(
        clean_dir=clean_dir,
        noise_dir=noise_dir,
        num_samples=num_samples_train,
        **kwargs,
    )

    val_dataset = AudioDenoiserDataset(
        clean_dir=clean_dir,
        noise_dir=noise_dir,
        num_samples=num_samples_val,
        **kwargs,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ============================================================
# Test: spusti priamo tento súbor pre overenie funkčnosti
# python src/dataset.py
# ============================================================
if __name__ == "__main__":
    import sys

    CLEAN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "clean")
    NOISE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "noise")

    print("=" * 50)
    print("TEST: AudioDenoiserDataset")
    print("=" * 50)

    try:
        dataset = AudioDenoiserDataset(
            clean_dir=CLEAN_DIR,
            noise_dir=NOISE_DIR,
            num_samples=10,
        )

        noisy_mag, clean_mag = dataset[0]

        print(f"\nNoisy spektrogram shape: {noisy_mag.shape}")
        print(f"Clean spektrogram shape: {clean_mag.shape}")
        print(f"Noisy rozsah: [{noisy_mag.min():.4f}, {noisy_mag.max():.4f}]")
        print(f"Clean rozsah: [{clean_mag.min():.4f}, {clean_mag.max():.4f}]")
        print(f"\n>>> Dataset OK — {len(dataset)} samplov pripravených")

    except Exception as e:
        print(f"\nCHYBA: {e}")
        sys.exit(1)
