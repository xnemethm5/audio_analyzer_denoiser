"""
model.py — U-Net architektúra pre audio denoising.

Vstup:  magnitúda zašumeného spektrogramu [B, 1, F, T]
Výstup: maska [B, 1, F, T] s hodnotami 0–1

Princíp: model predikuje masku, ktorou sa vynásobí zašumený spektrogram.
         Tam kde je šum, maska bude blízko 0 (potlačí ho).
         Tam kde je užitočný signál, maska bude blízko 1 (zachová ho).

Architektúra:
    Encoder (4 úrovne) → Bottleneck → Decoder (4 úrovne) so skip connections
    Kanály: 1 → 16 → 32 → 64 → 128 → 128 (bottleneck) → späť
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Základný stavebný blok: 2x (Conv2d → BatchNorm → ReLU)
    Zachováva priestorové rozmery (padding='same').
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """
    Encoder: postupné zmenšovanie rozlíšenia a zvyšovanie kanálov.
    Každá úroveň: ConvBlock → MaxPool2d(2)
    Vracia features z každej úrovne (pre skip connections).
    """

    def __init__(self, channels: list):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.blocks.append(ConvBlock(channels[i], channels[i + 1]))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        skip_connections = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            skip_connections.append(x)  # Ulož pre decoder
            x = pool(x)
        return x, skip_connections


class Decoder(nn.Module):
    """
    Decoder: postupné zväčšovanie rozlíšenia a znižovanie kanálov.
    Každá úroveň: Upsample → Concat so skip → ConvBlock
    """

    def __init__(self, channels: list):
        super().__init__()
        self.upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for i in range(len(channels) - 1):
            # ConvTranspose2d na upsampling (naučí sa ako zvýšiť rozlíšenie)
            self.upsamples.append(
                nn.ConvTranspose2d(
                    channels[i], channels[i + 1],
                    kernel_size=2, stride=2
                )
            )
            # Po concat so skip connection: channels[i+1]*2 vstupov
            self.blocks.append(ConvBlock(channels[i + 1] * 2, channels[i + 1]))

    def forward(self, x, skip_connections):
        for i, (upsample, block) in enumerate(zip(self.upsamples, self.blocks)):
            x = upsample(x)
            skip = skip_connections[i]

            # Ak sa rozmery líšia (kvôli nepárnym rozmerom), orezaj skip
            if x.shape != skip.shape:
                x = self._match_size(x, skip)

            x = torch.cat([x, skip], dim=1)  # Concat pozdĺž kanálov
            x = block(x)
        return x

    @staticmethod
    def _match_size(x, target):
        """Orezaj/padni x aby sedel s target rozmermi."""
        diff_h = target.shape[2] - x.shape[2]
        diff_w = target.shape[3] - x.shape[3]
        # Padding ak je x menší
        x = nn.functional.pad(x, [0, diff_w, 0, diff_h])
        return x


class UNet(nn.Module):
    """
    Kompletný U-Net pre audio denoising.

    Vstup:  [B, 1, F, T]  — magnitúda zašumeného spektrogramu
    Výstup: [B, 1, F, T]  — maska (0–1)

    Použitie:
        mask = model(noisy_spectrogram)
        clean_estimate = noisy_spectrogram * mask
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        # Kanály pre encoder: 1 → 16 → 32 → 64 → 128
        enc_channels = [in_channels, 16, 32, 64, 128]
        # Bottleneck: 128 → 256
        bottleneck_channels = 256
        # Kanály pre decoder: 256 → 128 → 64 → 32 → 16
        dec_channels = [bottleneck_channels, 128, 64, 32, 16]

        self.encoder = Encoder(enc_channels)
        self.bottleneck = ConvBlock(enc_channels[-1], bottleneck_channels)
        self.decoder = Decoder(dec_channels)

        # Výstupná vrstva: 1x1 konvolúcia → Sigmoid pre masku
        self.output_conv = nn.Sequential(
            nn.Conv2d(dec_channels[-1], in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, F, T] zašumený spektrogram

        Returns:
            mask: [B, 1, F, T] maska s hodnotami 0–1
        """
        # Zapamätaj si pôvodné rozmery
        original_h, original_w = x.shape[2], x.shape[3]

        # Padding na rozmery deliteľné 16 (4 úrovne poolingu po 2)
        pad_h = (16 - original_h % 16) % 16
        pad_w = (16 - original_w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, [0, pad_w, 0, pad_h])

        # Encoder
        x, skip_connections = self.encoder(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder (skip connections v obrátenom poradí)
        x = self.decoder(x, skip_connections[::-1])

        # Výstupná maska
        mask = self.output_conv(x)

        # Orezaj späť na pôvodné rozmery
        mask = mask[:, :, :original_h, :original_w]

        return mask


# ============================================================
# Test: spusti priamo tento súbor pre overenie
# python src/model.py
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("TEST: U-Net Model")
    print("=" * 50)

    model = UNet(in_channels=1)

    # Simulovaný vstup — rovnaký tvar ako z datasetu [B, 1, 513, 188]
    dummy_input = torch.randn(2, 1, 513, 188)

    print(f"\nVstup:  {dummy_input.shape}")

    mask = model(dummy_input)

    print(f"Výstup (maska): {mask.shape}")
    print(f"Rozsah masky: [{mask.min():.4f}, {mask.max():.4f}]")

    # Aplikácia masky
    clean_estimate = dummy_input * mask
    print(f"Odhad čistého: {clean_estimate.shape}")

    # Počet parametrov
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParametre: {total_params:,} (trénovateľných: {trainable:,})")
    print(f"\n>>> Model OK")
