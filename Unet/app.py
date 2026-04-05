"""
app.py — Streamlit frontend pre Audio Denoiser.

Spustenie: streamlit run app.py
"""

import os
import sys
import tempfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import soundfile as sf

# Pridaj src do path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.inference import AudioDenoiser


# ============================================================
# Konfigurácia stránky
# ============================================================
st.set_page_config(
    page_title="Audio Denoiser — U-Net",
    page_icon="🎵",
    layout="wide",
)


# ============================================================
# Cache: načítaj model iba raz
# ============================================================
@st.cache_resource
def load_model():
    """Načíta natrénovaný model (cached — raz za session)."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.pth")
    if not os.path.exists(model_path):
        return None
    return AudioDenoiser(model_path)


def plot_spectrogram(audio: np.ndarray, sr: int, title: str):
    """Vykreslí spektrogram pre audio."""
    fig, ax = plt.subplots(figsize=(10, 3))

    # Výpočet spektrogramu cez matplotlib
    ax.specgram(audio, NFFT=1024, Fs=sr, noverlap=512, cmap="magma")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Čas (s)")
    ax.set_ylabel("Frekvencia (Hz)")
    ax.set_ylim(0, sr // 2)

    plt.tight_layout()
    return fig


def plot_waveform(audio: np.ndarray, sr: int, title: str):
    """Vykreslí waveform."""
    fig, ax = plt.subplots(figsize=(10, 2))

    time_axis = np.arange(len(audio)) / sr
    ax.plot(time_axis, audio, linewidth=0.3, color="#1f77b4")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Čas (s)")
    ax.set_ylabel("Amplitúda")
    ax.set_xlim(0, time_axis[-1])

    plt.tight_layout()
    return fig


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Odhadne SNR medzi dvoma signálmi."""
    min_len = min(len(clean), len(noisy))
    clean = clean[:min_len]
    noisy = noisy[:min_len]
    noise = noisy - clean
    power_signal = np.mean(clean ** 2)
    power_noise = np.mean(noise ** 2)
    if power_noise < 1e-10:
        return float("inf")
    return 10 * np.log10(power_signal / power_noise)


# ============================================================
# UI
# ============================================================
st.title("🎵 Audio Denoiser — U-Net")
st.markdown("Odstránenie šumu z audio nahrávok pomocou spektrogramového U-Net modelu.")

# Sidebar
st.sidebar.header("⚙️ Nastavenia")

strength = st.sidebar.slider(
    "Sila denoisingu",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.05,
    help="0.0 = žiadny efekt, 1.0 = maximálne odstránenie šumu",
)

chunk_seconds = st.sidebar.slider(
    "Veľkosť chunku (s)",
    min_value=3.0,
    max_value=30.0,
    value=10.0,
    step=1.0,
    help="Dlhšie chunky = lepšia kvalita, viac pamäte",
)

show_waveform = st.sidebar.checkbox("Zobraziť waveform", value=True)
show_spectrogram = st.sidebar.checkbox("Zobraziť spektrogram", value=True)

# Načítaj model
denoiser = load_model()

if denoiser is None:
    st.error(
        "⚠️ Model nenájdený! Najprv spusti tréning:\n\n"
        "```\npython src/train.py\n```\n\n"
        "Model sa uloží do `models/best_model.pth`."
    )
    st.stop()

st.sidebar.success(f"✅ Model načítaný")
st.sidebar.info(f"Zariadenie: {denoiser.device}")

# Upload
st.header("📤 Nahrať audio")
uploaded_file = st.file_uploader(
    "Vyber WAV alebo MP3 súbor",
    type=["wav", "mp3"],
    help="Nahraj WAV alebo MP3 súbor, ktorý chceš vyčistiť od šumu.",
)

if uploaded_file is not None:
    # Ulož uploadnutý súbor dočasne (zachovaj príponu)
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_in:
        tmp_in.write(uploaded_file.read())
        input_path = tmp_in.name

    # Načítaj originál (inference.py zvládne WAV aj MP3)
    audio_original, sr = denoiser.load_audio(input_path)
    if audio_original.ndim == 2:
        audio_original = audio_original.mean(axis=1)

    duration = len(audio_original) / sr

    st.markdown(f"**Súbor:** {uploaded_file.name} | **Dĺžka:** {duration:.1f}s | **Sample rate:** {sr} Hz")

    # Prehrávanie originálu
    st.subheader("🔊 Originál (zašumený)")
    st.audio(input_path, format="audio/wav")

    # Denoise tlačidlo
    if st.button("🧹 Odstrániť šum", type="primary", use_container_width=True):

        # Progress bar
        progress = st.progress(0, text="Spracovávam audio...")

        # Dočasný výstupný súbor
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            output_path = tmp_out.name

        progress.progress(10, text="Načítavam audio...")

        # Denoise
        try:
            progress.progress(30, text="Odstraňujem šum (U-Net inferencia)...")

            result = denoiser.denoise_file(
                input_path,
                output_path,
                strength=strength,
                chunk_seconds=chunk_seconds,
            )

            progress.progress(80, text="Generujem vizualizácie...")

            # Načítaj výsledok
            audio_denoised, sr_out = sf.read(output_path, dtype="float32")
            if audio_denoised.ndim == 2:
                audio_denoised = audio_denoised.mean(axis=1)

            progress.progress(100, text="Hotovo!")

            # ============================================================
            # Výsledky
            # ============================================================
            st.subheader("🎧 Vyčistené audio")
            st.audio(output_path, format="audio/wav")

            # Porovnanie vedľa seba
            st.subheader("📊 Porovnanie")

            # Waveformy
            if show_waveform:
                col1, col2 = st.columns(2)
                with col1:
                    fig = plot_waveform(audio_original, sr, "Originál — Waveform")
                    st.pyplot(fig)
                    plt.close(fig)
                with col2:
                    fig = plot_waveform(audio_denoised, sr_out, "Vyčistené — Waveform")
                    st.pyplot(fig)
                    plt.close(fig)

            # Spektrogramy
            if show_spectrogram:
                col1, col2 = st.columns(2)
                with col1:
                    fig = plot_spectrogram(audio_original, sr, "Originál — Spektrogram")
                    st.pyplot(fig)
                    plt.close(fig)
                with col2:
                    fig = plot_spectrogram(audio_denoised, sr_out, "Vyčistené — Spektrogram")
                    st.pyplot(fig)
                    plt.close(fig)

            # Štatistiky
            st.subheader("📈 Štatistiky")
            col1, col2, col3 = st.columns(3)

            with col1:
                orig_rms = np.sqrt(np.mean(audio_original ** 2))
                st.metric("RMS Originál", f"{orig_rms:.4f}")

            with col2:
                clean_rms = np.sqrt(np.mean(audio_denoised ** 2))
                st.metric("RMS Vyčistené", f"{clean_rms:.4f}")

            with col3:
                reduction = (1 - clean_rms / max(orig_rms, 1e-8)) * 100
                st.metric("Redukcia energie", f"{reduction:.1f}%")

            # Download
            st.subheader("💾 Stiahnuť")
            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ Stiahnuť vyčistené audio (WAV)",
                    data=f.read(),
                    file_name=f"denoised_{uploaded_file.name}",
                    mime="audio/wav",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Chyba pri spracovaní: {e}")
            import traceback
            st.code(traceback.format_exc())

        finally:
            # Cleanup dočasných súborov
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass

else:
    # Placeholder keď nie je nahraný súbor
    st.info("👆 Nahraj WAV súbor pre začatie denoisingu.")

    # Info sekcia
    with st.expander("ℹ️ Ako to funguje?"):
        st.markdown("""
        **Princíp fungovania:**

        1. Audio sa prevedie na **STFT spektrogram** (vizuálna reprezentácia frekvencií v čase)
        2. **U-Net neurónová sieť** analyzuje spektrogram a predikuje masku šumu
        3. Maska sa aplikuje na pôvodný spektrogram — šum sa potlačí, signál sa zachová
        4. Vyčistený spektrogram sa prevedie späť na audio cez **ISTFT**

        **Nastavenia:**
        - **Sila denoisingu** — 0.0 (žiadny efekt) až 1.0 (maximálne čistenie)
        - **Veľkosť chunku** — dlhšie = lepšia kvalita pri dlhých nahrávkach
        """)

    with st.expander("🏗️ Architektúra modelu"):
        st.markdown("""
        - **Model:** U-Net (encoder-decoder so skip connections)
        - **Vstup:** Magnitúda STFT spektrogramu [1, 513, T]
        - **Výstup:** Maska [0–1] rovnakého tvaru
        - **Parametre:** ~1.9M
        - **Loss:** L1 (Mean Absolute Error)
        - **Optimizer:** Adam s ReduceLROnPlateau schedulerom
        """)
