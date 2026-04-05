import streamlit as st
import os
import uuid
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from classifier import classify_genre
from denoiser import denoise_audio
from noise import add_noise

st.set_page_config(page_title="Audio Analyzer", page_icon="🎵", layout="centered")

# Adaptive CSS - funguje v svetlom aj tmavom mode
st.markdown("""
<style>
@media (prefers-color-scheme: dark) {
    :root {
        --text-primary:   #ffffff;
        --text-secondary: #bbbbbb;
        --text-muted:     #888888;
        --bg-chip:        #2a2a2a;
        --border-chip:    #444444;
        --bg-bar:         #333333;
        --border-divider: #2a2a2a;
        --border-col:     #444444;
    }
}
@media (prefers-color-scheme: light) {
    :root {
        --text-primary:   #111111;
        --text-secondary: #333333;
        --text-muted:     #555555;
        --bg-chip:        #f0f0f0;
        --border-chip:    #cccccc;
        --bg-bar:         #e0e0e0;
        --border-divider: #dddddd;
        --border-col:     #cccccc;
    }
}
.az-text    { color: var(--text-primary) !important; }
.az-sub     { color: var(--text-secondary) !important; }
.az-muted   { color: var(--text-muted) !important; }

.az-genre-wrap  { margin-bottom: 0.75rem; }
.az-genre-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.92rem;
    font-weight: 600;
    margin-bottom: 5px;
    color: var(--text-primary);
}
.az-bar-bg {
    background: var(--bg-bar);
    border-radius: 6px;
    height: 12px;
    overflow: hidden;
}
.az-bar-fill { height: 12px; border-radius: 6px; }

.az-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 14px 0; }
.az-chip  {
    background: var(--bg-chip);
    border: 1px solid var(--border-chip);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.82rem;
    color: var(--text-primary);
}
.az-chip-label { color: var(--text-muted); margin-right: 4px; }
.az-chip-value { color: var(--text-primary); font-weight: 600; }

.az-section {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 1.6rem 0 0.7rem 0;
}
.az-col-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.6rem;
    border-bottom: 1px solid var(--border-col);
    padding-bottom: 4px;
}
.az-divider {
    border: none;
    border-top: 1px solid var(--border-divider);
    margin: 1.4rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Komponenty ─────────────────────────────────────────────────────────────────

def genre_bars(genres):
    colors = ["#1db954", "#17a349", "#128a3d"]
    for i, g in enumerate(genres):
        color = colors[min(i, len(colors) - 1)]
        pct   = min(float(g["probability"]), 100)
        st.markdown(f"""
        <div class="az-genre-wrap">
            <div class="az-genre-label">
                <span>{g["genre"].capitalize()}</span>
                <span>{pct}%</span>
            </div>
            <div class="az-bar-bg">
                <div class="az-bar-fill" style="width:{pct}%; background:{color};"></div>
            </div>
        </div>""", unsafe_allow_html=True)


def chips(**kwargs):
    inner = "".join(
        f"<div class='az-chip'><span class='az-chip-label'>{k}:</span><span class='az-chip-value'>{v}</span></div>"
        for k, v in kwargs.items()
    )
    st.markdown(f"<div class='az-chips'>{inner}</div>", unsafe_allow_html=True)


def section(icon, title):
    st.markdown(f"<div class='az-section'>{icon}&nbsp; {title}</div>", unsafe_allow_html=True)


def divider():
    st.markdown("<hr class='az-divider'>", unsafe_allow_html=True)


def col_title(text):
    st.markdown(f"<div class='az-col-title'>{text}</div>", unsafe_allow_html=True)


def note(text):
    st.markdown(f"<p class='az-sub' style='font-size:0.85rem; margin-top:4px;'>{text}</p>",
                unsafe_allow_html=True)




# ── Diagnostika ────────────────────────────────────────────────────────────────

def show_diagnostics(input_path, output_path, result_denoise):
    """Zobrazí diagnostický panel so spektrogramami a textovými metrikami."""
    try:
        y_orig,  sr1 = sf.read(input_path,  always_2d=False)
        y_clean, sr2 = sf.read(output_path, always_2d=False)
    except Exception:
        st.warning("Diagnostika: súbory nie sú dostupné (už boli zmazané).")
        return

    if y_orig.ndim  > 1: y_orig  = y_orig.mean(axis=1)
    if y_clean.ndim > 1: y_clean = y_clean.mean(axis=1)
    sr = sr1

    min_len = min(len(y_orig), len(y_clean))
    y_orig  = y_orig[:min_len].astype(np.float32)
    y_clean = y_clean[:min_len].astype(np.float32)

    # --- Výpočet spektrogramov ---
    n_fft = 2048
    hop   = 512

    def stft_db(y):
        D   = np.abs(np.fft.rfft(
            np.array([y[i:i+n_fft] for i in range(0, len(y)-n_fft, hop)
                      if i+n_fft <= len(y)]), axis=1
        ).T)
        return 20 * np.log10(D + 1e-10)

    spec_orig  = stft_db(y_orig)
    spec_clean = stft_db(y_clean)
    spec_diff  = spec_orig - spec_clean   # čo bolo odfiltrované

    # --- Frekvenčná os ---
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    times = np.arange(spec_orig.shape[1]) * hop / sr

    # --- Priemerné spektrum (pred vs po) ---
    avg_orig  = spec_orig.mean(axis=1)
    avg_clean = spec_clean.mean(axis=1)
    avg_diff  = avg_orig - avg_clean

    # --- Figure ---
    fig = plt.figure(figsize=(12, 9), facecolor="none")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    cmap_spec = "magma"
    cmap_diff = "RdYlGn_r"
    vmin, vmax = -90, 0

    def _spec_ax(ax, data, title, cmap=cmap_spec):
        im = ax.imshow(
            data, aspect="auto", origin="lower", cmap=cmap,
            extent=[times[0], times[-1], freqs[0]/1000, freqs[-1]/1000],
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel("čas (s)", fontsize=7)
        ax.set_ylabel("freq (kHz)", fontsize=7)
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax, pad=0.02).ax.tick_params(labelsize=6)

    # Spektrogramy
    _spec_ax(fig.add_subplot(gs[0, 0]), spec_orig,  "Spektrogram – originál")
    _spec_ax(fig.add_subplot(gs[0, 1]), spec_clean, "Spektrogram – po denoising")
    _spec_ax(fig.add_subplot(gs[1, 0]), spec_diff,
             "Odfiltrovaná energia (pred − po)", cmap=cmap_diff)

    # Priemerné spektrum
    ax_avg = fig.add_subplot(gs[1, 1])
    ax_avg.plot(freqs / 1000, avg_orig,  label="originál",   linewidth=1.0, color="#5b8dd9")
    ax_avg.plot(freqs / 1000, avg_clean, label="po denoising", linewidth=1.0, color="#57c785")
    ax_avg.fill_between(freqs / 1000, avg_orig, avg_clean,
                        where=(avg_orig > avg_clean), alpha=0.25,
                        color="#e05c5c", label="odfiltrované")
    ax_avg.set_title("Priemerné frekvenčné spektrum", fontsize=9, pad=4)
    ax_avg.set_xlabel("freq (kHz)", fontsize=7)
    ax_avg.set_ylabel("dB", fontsize=7)
    ax_avg.legend(fontsize=6)
    ax_avg.tick_params(labelsize=7)
    ax_avg.set_xlim(0, sr / 2000)

    # Waveformy
    t = np.arange(min_len) / sr
    ax_wave = fig.add_subplot(gs[2, :])
    ax_wave.plot(t, y_orig,  alpha=0.55, linewidth=0.4, label="originál",   color="#5b8dd9")
    ax_wave.plot(t, y_clean, alpha=0.80, linewidth=0.4, label="po denoising", color="#57c785")
    ax_wave.set_title("Waveform porovnanie", fontsize=9, pad=4)
    ax_wave.set_xlabel("čas (s)", fontsize=7)
    ax_wave.set_ylabel("amplitúda", fontsize=7)
    ax_wave.legend(fontsize=6, loc="upper right")
    ax_wave.tick_params(labelsize=7)

    st.pyplot(fig)
    plt.close(fig)

    # --- Textové metriky ---
    noise_est    = y_orig - y_clean
    noise_rms    = float(np.sqrt(np.mean(noise_est ** 2)))
    signal_rms   = float(np.sqrt(np.mean(y_clean ** 2)))
    residual_snr = 20 * np.log10(signal_rms / (noise_rms + 1e-10))

    # Per-band redukcia
    bands = [(0,250,"Sub-bas"), (250,2000,"Bas/stred"),
             (2000,6000,"Výšky stred"), (6000,20000,"Výšky")]
    band_data = []
    for lo, hi, name in bands:
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if len(idx) == 0:
            continue
        red = float(np.mean(avg_diff[idx]))
        band_data.append({"Pásmo": name, "Redukcia (dB)": f"{red:+.2f}"})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Per-band redukcia šumu**")
        st.table(band_data)
    with col2:
        st.markdown("**Detailné metriky**")
        st.markdown(f"""
| Metrika | Hodnota |
|---|---|
| Odfiltrovaná energia (RMS) | `{noise_rms:.5f}` |
| Signál po denoising (RMS) | `{signal_rms:.4f}` |
| Odhadovaný reziduálny SNR | `{residual_snr:.1f} dB` |
| Noise type | `{result_denoise.get("noise_type", "—")}` |
| Spectral slope | `{result_denoise.get("slope", "—")}` |
| Engine | `{result_denoise["engine"]}` |
        """)


# ── Hlavička ───────────────────────────────────────────────────────────────────
st.markdown("## 🎵 Audio Analyzer")
note("Nahraj MP3 alebo WAV súbor – zistíme žáner a odfiltrujeme šum")

os.makedirs("uploads", exist_ok=True)

uploaded_file = st.file_uploader("Vyber audio súbor", type=["mp3", "wav"],
                                  label_visibility="collapsed")

if uploaded_file is not None:
    uid = str(uuid.uuid4())
    ext = os.path.splitext(uploaded_file.name)[1]
    input_path  = f"uploads/{uid}_input{ext}"
    output_path = f"uploads/{uid}_clean.wav"
    noisy_path  = f"uploads/{uid}_noisy.wav"

    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(input_path)
    divider()

    mode = st.radio(
        "Režim",
        ["🔍 Analyzovať žáner a odstrániť šum", "🔊 Pridať šum"],
        horizontal=True,
        label_visibility="collapsed"
    )

    divider()

    # ══════════════════════════════════════════════════════════════════════
    # REŽIM 1: Analýza + Denoising
    # ══════════════════════════════════════════════════════════════════════
    if mode == "🔍 Analyzovať žáner a odstrániť šum":
        if st.button("🔍 Spustiť analýzu", use_container_width=True, type="primary"):

            with st.spinner("Analyzujem žáner..."):
                try:
                    result_before = classify_genre(input_path)
                except Exception as e:
                    st.error(f"Chyba pri klasifikácii: {e}")
                    st.stop()

            section("🎸", "Klasifikácia žánru – pôvodná nahrávka")
            chips(
                Úsekov=result_before["segments_analyzed"],
                Trvanie=f"{result_before['total_duration']} s"
            )
            genre_bars(result_before["genres"])
            divider()

            with st.spinner("Odstraňujem šum..."):
                try:
                    result_denoise = denoise_audio(
                        input_path, output_path,
                        genres=result_before["genres"]
                    )
                except Exception as e:
                    st.error(f"Chyba pri odstraňovaní šumu: {e}")
                    st.stop()

            section("🔇", "Redukcia šumu")
            chips(
                Profil=result_denoise["profile_used"],
                Noise=result_denoise.get("noise_type", "—").capitalize(),
                Engine=result_denoise["engine"]
            )

            snr_diff = round(result_denoise["snr_after"] - result_denoise["snr_before"], 2)
            col1, col2, col3 = st.columns(3)
            col1.metric("SNR pred",  f"{result_denoise['snr_before']} dB")
            col2.metric("SNR po",    f"{result_denoise['snr_after']} dB")
            col3.metric("Rozdiel",   f"{snr_diff:+.2f} dB",
                        delta=f"{snr_diff:+.2f} dB", delta_color="normal")

            if abs(snr_diff) < 0.1:
                st.info("Nahrávka neobsahuje merateľný šum – denoising nebol potrebný.")
            elif snr_diff > 0:
                st.success(f"Šum úspešne redukovaný o {snr_diff} dB")
            else:
                st.warning("Denoising mierne znížil kvalitu – odporúčame pôvodný súbor.")

            st.audio(output_path, format="audio/wav")
            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇ Stiahnuť vyčistené audio",
                    data=f, file_name="clean_audio.wav",
                    mime="audio/wav", use_container_width=True,
                )

            with st.expander("🔬 Diagnostika – spektrogramy a metriky"):
                show_diagnostics(input_path, output_path, result_denoise)
            divider()

            with st.spinner("Klasifikujem vyčistené audio..."):
                try:
                    result_after = classify_genre(output_path)
                except Exception as e:
                    st.error(f"Chyba pri klasifikácii po denoising: {e}")
                    st.stop()

            section("📊", "Porovnanie klasifikácie")
            col1, col2 = st.columns(2)
            with col1:
                col_title("Pred denoisingom")
                genre_bars(result_before["genres"])
            with col2:
                col_title("Po denoising")
                genre_bars(result_after["genres"])

            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.remove(path)

    # ══════════════════════════════════════════════════════════════════════
    # REŽIM 2: Pridanie šumu
    # ══════════════════════════════════════════════════════════════════════
    else:
        section("🔊", "Nastavenia šumu")

        noise_type = st.selectbox(
            "Typ šumu",
            options=["white", "pink", "crackle"],
            format_func=lambda x: {
                "white":   "⬜ Biely šum – rovnomerné náhodné frekvencie",
                "pink":    "🌸 Ružový šum – viac basov, prirodzenejší",
                "crackle": "📻 Praskanie – ako stará vinylová platňa"
            }[x]
        )

        noise_level = st.slider(
            "Intenzita šumu",
            min_value=0.01, max_value=0.10,
            value=0.02, step=0.01,
            help="0.01 = jemný šum | 0.05 = silný šum"
        )

        chips(
            Typ=noise_type.capitalize(),
            Intenzita=f"{noise_level:.2f}",
        )
        divider()

        if st.button("🔊 Zašumiť nahrávku", use_container_width=True, type="primary"):
            with st.spinner("Pridávam šum..."):
                try:
                    result_noise = add_noise(input_path, noisy_path,
                                             noise_type=noise_type,
                                             noise_level=noise_level)
                except Exception as e:
                    st.error(f"Chyba pri pridávaní šumu: {e}")
                    st.stop()

            st.success("Šum bol úspešne pridaný!")
            chips(
                Typ=result_noise["noise_type"].capitalize(),
                Intenzita=result_noise["noise_level"],
                SNR=f"{result_noise['snr']:.2f} dB"
            )
            note("Čím nižšie SNR, tým viac šumu v nahrávke.")
            st.audio(noisy_path, format="audio/wav")

            with open(noisy_path, "rb") as f:
                st.download_button(
                    label="⬇ Stiahnuť zašumenú nahrávku",
                    data=f, file_name=f"noisy_{noise_type}.wav",
                    mime="audio/wav", use_container_width=True,
                )

            for path in [input_path, noisy_path]:
                if os.path.exists(path):
                    os.remove(path)