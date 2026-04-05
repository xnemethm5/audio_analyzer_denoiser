"""
train.py — Trénovacia slučka pre U-Net audio denoiser.

Zodpovednosti:
- Načítanie datasetu
- Inicializácia modelu, optimizéra, loss funkcie
- Tréningový loop s validáciou
- Ukladanie checkpointov (best model + posledný)
- Logovanie metrík (loss) + vizualizácia priebehu tréningu
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# Importy z nášho projektu
sys.path.insert(0, os.path.dirname(__file__))
from dataset import AudioDenoiserDataset, get_dataloaders
from model import UNet


# ============================================================
# Konfigurácia — všetky hyperparametre na jednom mieste
# ============================================================
CONFIG = {
    # Cesty
    "clean_dir": os.path.join(os.path.dirname(__file__), "..", "data", "clean"),
    "noise_dir": os.path.join(os.path.dirname(__file__), "..", "data", "noise"),
    "model_dir": os.path.join(os.path.dirname(__file__), "..", "models"),

    # Audio parametre
    "sample_rate": 44100,
    "segment_length": 44100 * 8,  # 8 sekúnd
    "n_fft": 4096,
    "hop_length": 1024,
    "snr_range": (0, 20),

    # Tréning
    "epochs": 100,
    "batch_size": 4,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "num_samples_train": 8000,
    "num_samples_val": 800,
    "num_workers": 0,  # Na Windows nechaj 0

    # Scheduler
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "min_lr": 1e-6,

    # Early stopping
    "early_stopping_patience": 15,
}


class EarlyStopping:
    """Zastaví tréning ak sa validačná loss nezlepšuje."""

    def __init__(self, patience: int = 7):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")

    def should_stop(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\n[Early Stopping] Žiadne zlepšenie {self.patience} epoch. Zastavujem.")
                return True
            return False


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Jeden tréningový epoch."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc="  Train", leave=False)
    for noisy_mag, clean_mag in progress:
        noisy_mag = noisy_mag.to(device)
        clean_mag = clean_mag.to(device)

        # Forward: model predikuje masku
        mask = model(noisy_mag)
        clean_estimate = noisy_mag * mask

        # Loss: porovnaj odhad s čistým spektrogramom
        loss = criterion(clean_estimate, clean_mag)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        progress.set_postfix(loss=f"{loss.item():.6f}")

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validácia — bez gradientov."""
    model.eval()
    running_loss = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc="  Val  ", leave=False)
    for noisy_mag, clean_mag in progress:
        noisy_mag = noisy_mag.to(device)
        clean_mag = clean_mag.to(device)

        mask = model(noisy_mag)
        clean_estimate = noisy_mag * mask

        loss = criterion(clean_estimate, clean_mag)

        running_loss += loss.item()
        num_batches += 1
        progress.set_postfix(loss=f"{loss.item():.6f}")

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path):
    """Uloží checkpoint modelu."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, path)


def plot_training_history(history, save_path):
    """Vykreslí graf tréningovej a validačnej loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (L1)")
    plt.title("Priebeh tréningu — Audio Denoiser U-Net")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Graf] Uložený: {save_path}")


def train():
    """Hlavná tréningová funkcia."""
    print("=" * 60)
    print("  TRÉNING: U-Net Audio Denoiser")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nZariadenie: {device}")

    # Vytvor priečinok pre modely
    os.makedirs(CONFIG["model_dir"], exist_ok=True)

    # DataLoadery
    print("\nNačítavam dáta...")
    train_loader, val_loader = get_dataloaders(
        clean_dir=CONFIG["clean_dir"],
        noise_dir=CONFIG["noise_dir"],
        batch_size=CONFIG["batch_size"],
        num_samples_train=CONFIG["num_samples_train"],
        num_samples_val=CONFIG["num_samples_val"],
        num_workers=CONFIG["num_workers"],
        sample_rate=CONFIG["sample_rate"],
        segment_length=CONFIG["segment_length"],
        n_fft=CONFIG["n_fft"],
        hop_length=CONFIG["hop_length"],
        snr_range=CONFIG["snr_range"],
    )

    # Model
    model = UNet(in_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametrov: {total_params:,}")

    # Optimizer — Adam s weight decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Scheduler — zníži LR keď sa val_loss prestane zlepšovať
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=CONFIG["scheduler_patience"],
        factor=CONFIG["scheduler_factor"],
        min_lr=CONFIG["min_lr"],
        verbose=True,
    )

    # Loss — L1 (Mean Absolute Error) — lepšia pre audio ako MSE
    criterion = nn.L1Loss()

    # Early stopping
    early_stopping = EarlyStopping(patience=CONFIG["early_stopping_patience"])

    # História pre graf
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")

    # Ulož konfiguráciu
    config_path = os.path.join(CONFIG["model_dir"], "config.json")
    with open(config_path, "w") as f:
        json.dump({k: str(v) if isinstance(v, tuple) else v for k, v in CONFIG.items()}, f, indent=2)

    print(f"\nŠtart tréningu — {CONFIG['epochs']} epoch")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch}/{CONFIG['epochs']}  (LR: {current_lr:.2e})")

        # Tréning
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validácia
        val_loss = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Zaznamenaj históriu
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Čas: {epoch_time:.1f}s")

        # Ulož best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CONFIG["model_dir"], "best_model.pth")
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, best_path)
            print(f"  >>> Nový najlepší model! (val_loss: {val_loss:.6f})")

        # Ulož posledný model
        last_path = os.path.join(CONFIG["model_dir"], "last_model.pth")
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, last_path)

        # Early stopping
        if early_stopping.should_stop(val_loss):
            break

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"  TRÉNING DOKONČENÝ")
    print(f"  Celkový čas: {total_time / 60:.1f} minút")
    print(f"  Najlepšia val loss: {best_val_loss:.6f}")
    print(f"  Model uložený: {os.path.abspath(CONFIG['model_dir'])}")
    print("=" * 60)

    # Vykresli graf
    plot_path = os.path.join(CONFIG["model_dir"], "training_history.png")
    plot_training_history(history, plot_path)

    # Ulož históriu ako JSON
    history_path = os.path.join(CONFIG["model_dir"], "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    train()
