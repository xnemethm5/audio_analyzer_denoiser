"""
inference.py — Predikcia / denoise pre nové audio súbory.

Zodpovednosti:
- Načítanie natrénovaného modelu
- Spracovanie vstupného audia (STFT)
- Predikcia masky cez U-Net
- Rekonštrukcia vyčisteného audia (ISTFT)
- Export do WAV
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from model import UNet


class AudioDenoiser:
    """
    Trieda pre denoising audia pomocou natrénovaného U-Net modelu.

    Použitie:
        denoiser = AudioDenoiser("models/best_model.pth")
        denoiser.denoise_file("input.wav", "output.wav")
    """

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 44100,
        n_fft: int = 4096,
        hop_length: int = 1024,
        device: str = None,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Načítaj model
        self.model = UNet(in_channels=1).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"[Denoiser] Model načítaný z {model_path}")
        print(f"[Denoiser] Epoch: {checkpoint.get('epoch', '?')}, Val Loss: {checkpoint.get('val_loss', '?'):.6f}")
        print(f"[Denoiser] Zariadenie: {self.device}")

    def load_audio(self, filepath: str) -> tuple:
        """
        Načíta audio súbor (WAV, MP3, FLAC...), skonvertuje na mono.
        MP3 sa automaticky konvertuje na WAV cez pydub.
        Vracia (waveform ako numpy array, pôvodný sample_rate).
        """
        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".mp3":
            # MP3 → WAV konverzia cez pydub
            from pydub import AudioSegment
            import tempfile

            audio_seg = AudioSegment.from_mp3(filepath)
            sr = audio_seg.frame_rate

            # Dočasný WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            audio_seg.export(tmp_path, format="wav")
            data, sr = sf.read(tmp_path, dtype="float32")
            os.unlink(tmp_path)
        else:
            data, sr = sf.read(filepath, dtype="float32")

        # Stereo → mono
        if data.ndim == 2:
            data = data.mean(axis=1)

        return data, sr

    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resampleuje audio na cieľový sample rate."""
        if orig_sr == target_sr:
            return audio

        import torchaudio
        waveform = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        resampled = resampler(waveform)
        return resampled.squeeze(0).numpy()

    @torch.no_grad()
    def denoise_waveform(self, waveform: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Vyčistí audio waveform (numpy array, mono, správny sample rate).

        Args:
            waveform: 1D numpy array s audio dátami
            strength: sila denoisingu 0.0 (žiadny) až 1.0 (plný)

        Returns:
            Vyčistený waveform ako numpy array
        """
        # Normalizácia
        max_val = np.max(np.abs(waveform)) + 1e-8
        waveform_norm = waveform / max_val

        # Na tensor
        waveform_tensor = torch.from_numpy(waveform_norm).float()

        # STFT
        window = torch.hann_window(self.n_fft)
        stft = torch.stft(
            waveform_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )

        # Magnitúda a fáza
        magnitude = stft.abs().unsqueeze(0).unsqueeze(0)  # [1, 1, F, T]
        phase = stft.angle()

        # Predikcia masky
        magnitude = magnitude.to(self.device)
        mask = self.model(magnitude)
        mask = mask.cpu().squeeze(0).squeeze(0)  # [F, T]

        # Aplikuj strength (interpolácia medzi 1.0 maskou a predikovanou)
        if strength < 1.0:
            mask = strength * mask + (1.0 - strength) * torch.ones_like(mask)

        # Aplikuj masku na magnitúdu
        magnitude_clean = stft.abs() * mask

        # Rekonštrukcia komplexného spektra (čistá magnitúda + pôvodná fáza)
        stft_clean = magnitude_clean * torch.exp(1j * phase)

        # ISTFT → späť na waveform
        waveform_clean = torch.istft(
            stft_clean,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
        )

        # Denormalizácia
        result = waveform_clean.numpy() * max_val

        return result

    def denoise_file(
        self,
        input_path: str,
        output_path: str,
        strength: float = 1.0,
        chunk_seconds: float = 10.0,
    ) -> dict:
        """
        Vyčistí celý audio súbor. Dlhé súbory spracuje po častiach (chunks).

        Args:
            input_path: cesta k vstupnému súboru (WAV/MP3)
            output_path: cesta pre výstupný WAV
            strength: sila denoisingu 0.0–1.0
            chunk_seconds: dĺžka jedného chunka v sekundách

        Returns:
            dict s informáciami o spracovaní
        """
        print(f"\n[Denoise] Vstup: {input_path}")

        # Načítaj audio
        audio, orig_sr = self.load_audio(input_path)
        duration = len(audio) / orig_sr
        print(f"[Denoise] Dĺžka: {duration:.1f}s, Sample rate: {orig_sr}")

        # Resample na pracovný sample rate
        audio_resampled = self.resample(audio, orig_sr, self.sample_rate)

        # Spracovanie po chunkoch (pre dlhé súbory)
        chunk_size = int(chunk_seconds * self.sample_rate)
        overlap = int(0.5 * self.sample_rate)  # 0.5s overlap pre plynulé prechody

        total_length = len(audio_resampled)
        output = np.zeros(total_length, dtype=np.float32)
        weight = np.zeros(total_length, dtype=np.float32)

        pos = 0
        chunk_num = 0

        while pos < total_length:
            end = min(pos + chunk_size, total_length)
            chunk = audio_resampled[pos:end]

            # Denoise chunk
            clean_chunk = self.denoise_waveform(chunk, strength=strength)

            # Fade in/out pre overlap
            fade_len = min(overlap, len(clean_chunk))
            if pos > 0 and fade_len > 0:
                fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
                clean_chunk[:fade_len] *= fade_in

            if end < total_length and fade_len > 0:
                fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
                clean_chunk[-fade_len:] *= fade_out

            # Akumuluj
            out_end = pos + len(clean_chunk)
            output[pos:out_end] += clean_chunk
            weight[pos:out_end] += 1.0

            chunk_num += 1
            pos += chunk_size - overlap

        # Normalizuj podľa váh (kde sa overlappy sčítali)
        weight = np.maximum(weight, 1e-8)
        output = output / weight

        # Resample späť na pôvodný sample rate
        output_final = self.resample(output, self.sample_rate, orig_sr)

        # Zarovnaj dĺžku s originálom
        min_len = min(len(output_final), len(audio))
        output_final = output_final[:min_len]

        # Ulož
        sf.write(output_path, output_final, orig_sr)
        print(f"[Denoise] Výstup: {output_path}")
        print(f"[Denoise] Spracovaných chunkov: {chunk_num}")

        return {
            "input_path": input_path,
            "output_path": output_path,
            "duration": duration,
            "orig_sr": orig_sr,
            "chunks": chunk_num,
            "strength": strength,
        }


# ============================================================
# Test: python src/inference.py <input.wav> [output.wav]
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Použitie: python src/inference.py <vstup.wav> [výstup.wav]")
        print("Príklad:  python src/inference.py data/test/song.wav denoised.wav")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "denoised_output.wav"
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pth")

    if not os.path.exists(model_path):
        print(f"CHYBA: Model nenájdený: {model_path}")
        print("Najprv spusti tréning: python src/train.py")
        sys.exit(1)

    denoiser = AudioDenoiser(model_path)
    result = denoiser.denoise_file(input_file, output_file)
    print(f"\nHotovo! Vyčistené audio: {result['output_path']}")
