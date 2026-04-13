import streamlit as st
import os
import sys
import uuid
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# U-Net path setup
_unet_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Unet", "src")
sys.path.insert(0, _unet_src)

try:
    from inference import AudioDenoiser
    _UNET_AVAILABLE = True
except Exception:
    _UNET_AVAILABLE = False

from classifier import classify_genre
from denoiser import denoise_audio
from denoiser.noise_estimation import estimate_snr
from noise import add_noise

st.set_page_config(page_title="Audio Analyzer", page_icon="🎵", layout="wide")

st.markdown("""
<style>
/*
 * Všetky farby sú zámerne vynechané — texty dedia farbu od Streamlit témy
 * (color: inherit). Takto fungujú správne v light aj dark móde bez ohľadu
 * na OS preferenciu. Semi-transparentné rgba() pozadia a opacity pre
 * stlmený text sa automaticky prispôsobujú tmavému aj svetlému pozadiu.
 */

.az-genre-wrap  { margin-bottom: 0.75rem; }
.az-genre-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.92rem;
    font-weight: 600;
    margin-bottom: 5px;
    /* farba: inherit zo Streamlit témy */
}
.az-bar-bg {
    background: rgba(128, 128, 128, 0.18);
    border-radius: 6px;
    height: 12px;
    overflow: hidden;
}
.az-bar-fill { height: 12px; border-radius: 6px; }

.az-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 14px 0; }
.az-chip  {
    background: rgba(128, 128, 128, 0.10);
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.82rem;
    /* farba: inherit */
}
.az-chip-label { opacity: 0.58; margin-right: 4px; }
.az-chip-value { font-weight: 600; }

.az-section {
    font-size: 1.05rem;
    font-weight: 700;
    margin: 1.6rem 0 0.7rem 0;
    /* farba: inherit */
}
.az-col-title {
    font-size: 0.8rem;
    font-weight: 700;
    opacity: 0.60;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.6rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.28);
    padding-bottom: 4px;
}
.az-divider {
    border: none;
    border-top: 1px solid rgba(128, 128, 128, 0.18);
    margin: 1.4rem 0;
}
.az-sub {
    font-size: 0.85rem;
    margin-top: 4px;
    opacity: 0.72;
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
        f"<div class='az-chip'><span class='az-chip-label'>{k}:</span>"
        f"<span class='az-chip-value'>{v}</span></div>"
        for k, v in kwargs.items()
    )
    st.markdown(f"<div class='az-chips'>{inner}</div>", unsafe_allow_html=True)


def section(icon, title):
    st.markdown(f"<div class='az-section'>{icon}&nbsp; {title}</div>",
                unsafe_allow_html=True)


def divider():
    st.markdown("<hr class='az-divider'>", unsafe_allow_html=True)


def col_title(text):
    st.markdown(f"<div class='az-col-title'>{text}</div>", unsafe_allow_html=True)


def note(text):
    st.markdown(f"<p class='az-sub'>{text}</p>", unsafe_allow_html=True)


# ── Analytický panel – rovnaký formát pre oba denoisery ───────────────────────

def show_analysis_panel(input_path: str, output_path: str,
                        snr_before: float, snr_after: float) -> None:
    """
    Zobrazí spektrogramy, waveform a metriky.
    Rovnaký formát pre klasický aj U-Net denoiser.
    """
    try:
        y_orig,  sr = sf.read(input_path,  always_2d=False)
        y_clean, _  = sf.read(output_path, always_2d=False)
    except Exception:
        st.warning("Analytika: súbory nie sú dostupné.")
        return

    if y_orig.ndim  > 1: y_orig  = y_orig.mean(axis=1)
    if y_clean.ndim > 1: y_clean = y_clean.mean(axis=1)

    min_len = min(len(y_orig), len(y_clean))
    y_orig  = y_orig[:min_len].astype(np.float32)
    y_clean = y_clean[:min_len].astype(np.float32)

    # ── STFT pomocou librosa (Hann okno, správny výsledok) ────────────────
    n_fft, hop = 2048, 512

    D_orig  = librosa.stft(y_orig,  n_fft=n_fft, hop_length=hop)
    D_clean = librosa.stft(y_clean, n_fft=n_fft, hop_length=hop)

    # Spoločná referencia → škály sú porovnateľné
    ref_val    = np.max(np.abs(D_orig)) + 1e-10
    spec_orig  = librosa.amplitude_to_db(np.abs(D_orig),  ref=ref_val)
    spec_clean = librosa.amplitude_to_db(np.abs(D_clean), ref=ref_val)
    spec_diff  = spec_orig - spec_clean   # kladné = odfiltrované

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(
        np.arange(spec_orig.shape[1]), sr=sr, hop_length=hop
    )

    avg_orig  = spec_orig.mean(axis=1)
    avg_clean = spec_clean.mean(axis=1)
    avg_diff  = avg_orig - avg_clean

    # ── Figura: spektrogramy + waveform ──────────────────────────────────
    # facecolor="white" — matplotlib renderuje ako PNG obrázok, nie CSS element.
    # Biele pozadie zaručuje čitateľnosť textu v light aj dark Streamlit móde.
    fig = plt.figure(figsize=(16, 11), facecolor="white")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.30)

    vmin, vmax   = -80, 0
    diff_abs_max = max(float(np.percentile(np.abs(spec_diff), 99)), 1.0)

    def _spec_ax(ax, data, title, cmap="magma", v0=vmin, v1=vmax):
        im = ax.imshow(
            data, aspect="auto", origin="lower", cmap=cmap,
            extent=[times[0], times[-1], freqs[0] / 1000, freqs[-1] / 1000],
            vmin=v0, vmax=v1,
        )
        ax.set_title(title, fontsize=10, pad=5)
        ax.set_xlabel("čas (s)", fontsize=8)
        ax.set_ylabel("freq (kHz)", fontsize=8)
        ax.tick_params(labelsize=8)
        plt.colorbar(im, ax=ax, pad=0.02).ax.tick_params(labelsize=7)

    _spec_ax(fig.add_subplot(gs[0, 0]), spec_orig,  "Spektrogram – originál")
    _spec_ax(fig.add_subplot(gs[0, 1]), spec_clean, "Spektrogram – po denoising")
    _spec_ax(
        fig.add_subplot(gs[1, 0]), spec_diff,
        "Odfiltrovaná energia (originál − čistý) [dB]",
        cmap="RdYlGn_r", v0=0, v1=diff_abs_max,
    )

    # Priemerné frekvenčné spektrum
    ax_avg = fig.add_subplot(gs[1, 1])
    ax_avg.plot(freqs / 1000, avg_orig,  label="originál",    lw=1.2, color="#5b8dd9")
    ax_avg.plot(freqs / 1000, avg_clean, label="po denoising", lw=1.2, color="#57c785")
    ax_avg.fill_between(freqs / 1000, avg_orig, avg_clean,
                        where=(avg_orig > avg_clean),
                        alpha=0.25, color="#e05c5c", label="odfiltrované")
    ax_avg.set_title("Priemerné frekvenčné spektrum", fontsize=10, pad=5)
    ax_avg.set_xlabel("freq (kHz)", fontsize=8)
    ax_avg.set_ylabel("dB", fontsize=8)
    ax_avg.legend(fontsize=8)
    ax_avg.tick_params(labelsize=8)
    ax_avg.set_xlim(0, sr / 2000)

    # Waveform (celá šírka)
    t = np.arange(min_len) / sr
    ax_w = fig.add_subplot(gs[2, :])
    ax_w.plot(t, y_orig,  alpha=0.55, lw=0.35, label="originál",    color="#5b8dd9")
    ax_w.plot(t, y_clean, alpha=0.85, lw=0.35, label="po denoising", color="#57c785")
    ax_w.set_title("Waveform – porovnanie", fontsize=10, pad=5)
    ax_w.set_xlabel("čas (s)", fontsize=8)
    ax_w.set_ylabel("amplitúda", fontsize=8)
    ax_w.legend(fontsize=8, loc="upper right")
    ax_w.tick_params(labelsize=8)
    ax_w.set_xlim(t[0], t[-1])

    st.pyplot(fig)
    plt.close(fig)

    # ── Textové metriky ───────────────────────────────────────────────────
    noise_est    = y_orig - y_clean
    noise_rms    = float(np.sqrt(np.mean(noise_est ** 2)))
    signal_rms   = float(np.sqrt(np.mean(y_clean ** 2)))
    residual_snr = 20 * np.log10(signal_rms / (noise_rms + 1e-10))

    bands = [
        (0,     250,   "Sub-bas"),
        (250,   2000,  "Bas / stred"),
        (2000,  6000,  "Výšky stred"),
        (6000,  20000, "Výšky"),
    ]
    band_data = []
    for lo, hi, name in bands:
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if len(idx) == 0:
            continue
        band_data.append({
            "Pásmo": name,
            "Redukcia (dB)": f"{float(np.mean(avg_diff[idx])):+.2f}",
        })

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Per-band redukcia šumu**")
        st.table(band_data)
    with c2:
        st.markdown("**Detailné metriky**")
        st.markdown(f"""
| Metrika | Hodnota |
|---|---|
| SNR pred denoising | `{snr_before:.1f} dB` |
| SNR po denoising | `{snr_after:.1f} dB` |
| Δ SNR | `{snr_after - snr_before:+.2f} dB` |
| Odfiltrovaná energia (RMS) | `{noise_rms:.5f}` |
| Signál po denoising (RMS) | `{signal_rms:.4f}` |
| Reziduálny SNR (odhadovaný) | `{residual_snr:.1f} dB` |
        """)


# ── Textový súhrn výsledkov ────────────────────────────────────────────────────

def build_summary_text(
    filename:             str,
    result_before:        dict,
    result_classic:       dict,
    result_after_classic: dict,
    result_after_unet:    dict | None,
    unet_snr_after:       float | None,
    unet_strength:        float | None,
    unet_chunk:           float | None,
    unet_device:          str   | None,
) -> str:
    """
    Zostaví plain-text súhrn všetkých výstupných parametrov jedného behu.
    Formát je určený na jednoduché kopírovanie a posielanie na ďalšiu analýzu.
    """
    lines = []
    sep   = "=" * 52

    # ── Hlavička ──────────────────────────────────────────
    lines += [sep, "  AUDIO ANALYZER – výsledky", sep]
    lines.append(f"Súbor    : {filename}")
    lines.append(f"Trvanie  : {result_before['total_duration']} s")
    lines.append(f"Úseky    : {result_before['segments_analyzed']}")
    lines.append("")

    # ── Klasifikácia pred denoising ───────────────────────
    lines.append("── Klasifikácia pred denoising ─────────────────────")
    for i, g in enumerate(result_before["genres"], 1):
        lines.append(f"  {i}. {g['genre'].capitalize():<14} {g['probability']:.1f} %")
    lines.append("")

    # ── Klasický denoiser ─────────────────────────────────
    snr_b  = result_classic["snr_before"]
    snr_c  = result_classic["snr_after"]
    diff_c = round(snr_c - snr_b, 2)

    lines.append("── Klasický denoiser ───────────────────────────────")
    lines.append(f"  Profil     : {result_classic['profile_used']}")
    lines.append(f"  Noise type : {result_classic.get('noise_type', '—')}")
    lines.append(f"  Engine     : {result_classic['engine']}")
    lines.append(f"  SNR pred   : {snr_b:.2f} dB")
    lines.append(f"  SNR po     : {snr_c:.2f} dB")
    lines.append(f"  Δ SNR      : {diff_c:+.2f} dB")
    lines.append("")

    # ── U-Net denoiser ────────────────────────────────────
    lines.append("── U-Net denoiser ──────────────────────────────────")
    if unet_snr_after is not None:
        diff_u = round(unet_snr_after - snr_b, 2)
        lines.append(f"  Sila       : {unet_strength:.2f}")
        lines.append(f"  Chunk      : {unet_chunk:.0f} s")
        lines.append(f"  Device     : {unet_device}")
        lines.append(f"  SNR pred   : {snr_b:.2f} dB")
        lines.append(f"  SNR po     : {unet_snr_after:.2f} dB")
        lines.append(f"  Δ SNR      : {diff_u:+.2f} dB")
    else:
        lines.append("  (nedostupný)")
    lines.append("")

    # ── Klasifikácia po klasickom denoising ───────────────
    lines.append("── Klasifikácia po klasickom denoising ─────────────")
    for i, g in enumerate(result_after_classic["genres"], 1):
        lines.append(f"  {i}. {g['genre'].capitalize():<14} {g['probability']:.1f} %")
    lines.append("")

    # ── Klasifikácia po U-Net denoising ───────────────────
    lines.append("── Klasifikácia po U-Net denoising ─────────────────")
    if result_after_unet is not None:
        for i, g in enumerate(result_after_unet["genres"], 1):
            lines.append(f"  {i}. {g['genre'].capitalize():<14} {g['probability']:.1f} %")
    else:
        lines.append("  (U-Net nedostupný)")
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


# ── U-Net model (cached) ────────────────────────────────────────────────────────

@st.cache_resource
def load_unet_model():
    if not _UNET_AVAILABLE:
        return None
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Unet", "models", "best_model.pth"
    )
    if not os.path.exists(model_path):
        return None
    try:
        return AudioDenoiser(model_path)
    except Exception:
        return None


# ── Sidebar – U-Net nastavenia ─────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ U-Net nastavenia")
    unet_strength = st.slider(
        "Sila denoisingu", min_value=0.0, max_value=1.0, value=0.8, step=0.05,
        help="0.0 = žiadny efekt, 1.0 = maximálne odstránenie šumu",
    )
    unet_chunk_seconds = st.slider(
        "Veľkosť chunku (s)", min_value=3.0, max_value=30.0, value=10.0, step=1.0,
        help="Dlhšie chunky = lepšia kvalita, viac pamäte",
    )

unet_denoiser = load_unet_model()

with st.sidebar:
    if unet_denoiser is not None:
        st.success("✅ U-Net model načítaný")
        st.info(f"Zariadenie: {unet_denoiser.device}")
    else:
        st.warning("⚠️ U-Net model nedostupný")


# ── Hlavička ───────────────────────────────────────────────────────────────────

st.markdown("## 🎵 Audio Analyzer")
note("Nahraj MP3 alebo WAV súbor – zistíme žáner a odfiltrujeme šum")

os.makedirs("uploads", exist_ok=True)

uploaded_file = st.file_uploader("Vyber audio súbor", type=["mp3", "wav"],
                                  label_visibility="collapsed")

if uploaded_file is not None:
    uid         = str(uuid.uuid4())
    ext         = os.path.splitext(uploaded_file.name)[1]
    input_path  = f"uploads/{uid}_input{ext}"
    output_path = f"uploads/{uid}_clean.wav"
    unet_path   = f"uploads/{uid}_unet.wav"
    noisy_path  = f"uploads/{uid}_noisy.wav"

    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(input_path)
    divider()

    mode = st.radio(
        "Režim",
        ["🔍 Analyzovať žáner a odstrániť šum", "🔊 Pridať šum"],
        horizontal=True,
        label_visibility="collapsed",
    )

    divider()

    # ══════════════════════════════════════════════════════════════════════
    # REŽIM 1: Analýza + Denoising
    # ══════════════════════════════════════════════════════════════════════
    if mode == "🔍 Analyzovať žáner a odstrániť šum":
        if st.button("🔍 Spustiť analýzu", use_container_width=True, type="primary"):

            # 1. Klasifikácia žánru
            with st.spinner("Analyzujem žáner..."):
                try:
                    result_before = classify_genre(input_path)
                except Exception as e:
                    st.error(f"Chyba pri klasifikácii: {e}")
                    st.stop()

            section("🎸", "Klasifikácia žánru – pôvodná nahrávka")
            chips(
                Úsekov=result_before["segments_analyzed"],
                Trvanie=f"{result_before['total_duration']} s",
            )
            genre_bars(result_before["genres"])
            divider()

            # 2. Klasický denoiser
            with st.spinner("Klasický denoiser – odstraňujem šum..."):
                try:
                    result_classic = denoise_audio(
                        input_path, output_path,
                        genres=result_before["genres"],
                    )
                except Exception as e:
                    st.error(f"Chyba pri klasickom denoising: {e}")
                    st.stop()

            # 3. U-Net denoiser
            unet_snr_after = None
            unet_error     = None

            if unet_denoiser is not None:
                with st.spinner("U-Net denoiser – odstraňujem šum..."):
                    try:
                        unet_denoiser.denoise_file(
                            input_path, unet_path,
                            unet_strength, unet_chunk_seconds,
                        )
                        y_u, sr_u      = sf.read(unet_path, always_2d=False)
                        y_u_mono       = y_u if y_u.ndim == 1 else y_u.mean(axis=1)
                        unet_snr_after = round(float(estimate_snr(y_u_mono, sr_u)), 2)
                    except Exception as e:
                        unet_error = str(e)

            # ── Súhrnné porovnanie vedľa seba ─────────────────────────────
            section("🔇", "Porovnanie denoisingu")

            snr_before_val = result_classic["snr_before"]
            snr_classic    = result_classic["snr_after"]
            snr_diff_c     = round(snr_classic - snr_before_val, 2)

            col_c, col_u = st.columns(2)

            with col_c:
                col_title("Klasický denoiser")
                chips(
                    Profil=result_classic["profile_used"],
                    Noise=result_classic.get("noise_type", "—").capitalize(),
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("SNR pred", f"{snr_before_val} dB")
                m2.metric("SNR po",   f"{snr_classic} dB")
                m3.metric("Δ SNR",    f"{snr_diff_c:+.2f} dB",
                          delta=f"{snr_diff_c:+.2f} dB", delta_color="normal")

                if abs(snr_diff_c) < 0.1:
                    st.info("Neobsahuje merateľný šum.")
                elif snr_diff_c > 0:
                    st.success(f"Redukcia o {snr_diff_c} dB")
                else:
                    st.warning("Mierne zníženie kvality.")

                st.audio(output_path, format="audio/wav")
                with open(output_path, "rb") as f:
                    st.download_button(
                        "⬇ Stiahnuť (klasický)", data=f,
                        file_name="clean_classic.wav", mime="audio/wav",
                        use_container_width=True,
                    )

            with col_u:
                col_title("U-Net denoiser")
                if unet_denoiser is None:
                    st.warning("U-Net model nie je dostupný.\n"
                               "Skontroluj `Unet/models/best_model.pth`.")
                elif unet_error:
                    st.error(f"Chyba U-Net: {unet_error}")
                else:
                    snr_diff_u = round(unet_snr_after - snr_before_val, 2)
                    chips(
                        Sila=f"{unet_strength:.2f}",
                        Chunk=f"{unet_chunk_seconds:.0f} s",
                        Device=str(unet_denoiser.device),
                    )
                    m1, m2, m3 = st.columns(3)
                    m1.metric("SNR pred", f"{snr_before_val} dB")
                    m2.metric("SNR po",   f"{unet_snr_after} dB")
                    m3.metric("Δ SNR",    f"{snr_diff_u:+.2f} dB",
                              delta=f"{snr_diff_u:+.2f} dB", delta_color="normal")

                    if abs(snr_diff_u) < 0.1:
                        st.info("Neobsahuje merateľný šum.")
                    elif snr_diff_u > 0:
                        st.success(f"Redukcia o {snr_diff_u} dB")
                    else:
                        st.warning("Mierne zníženie kvality.")

                    st.audio(unet_path, format="audio/wav")
                    with open(unet_path, "rb") as f:
                        st.download_button(
                            "⬇ Stiahnuť (U-Net)", data=f,
                            file_name="clean_unet.wav", mime="audio/wav",
                            use_container_width=True,
                        )

            # ── Analytika – plná šírka, pod porovnaním ────────────────────
            divider()
            section("🔬", "Analytika – spektrogramy a metriky")

            unet_ok = unet_denoiser is not None and unet_error is None and unet_snr_after is not None

            if unet_ok:
                tab_c, tab_u = st.tabs(["📊 Klasický denoiser", "🤖 U-Net denoiser"])
                with tab_c:
                    show_analysis_panel(input_path, output_path,
                                        snr_before_val, snr_classic)
                with tab_u:
                    show_analysis_panel(input_path, unet_path,
                                        snr_before_val, unet_snr_after)
            else:
                show_analysis_panel(input_path, output_path,
                                    snr_before_val, snr_classic)

            divider()

            # 4. Klasifikácia po denoising – zvlášť pre každý denoiser
            with st.spinner("Klasifikujem výstupy..."):
                try:
                    result_after_classic = classify_genre(output_path)
                except Exception as e:
                    st.error(f"Chyba pri klasifikácii (klasický): {e}")
                    st.stop()

                result_after_unet = None
                if unet_ok:
                    try:
                        result_after_unet = classify_genre(unet_path)
                    except Exception as e:
                        st.warning(f"Klasifikácia U-Net zlyhala: {e}")

            section("📊", "Porovnanie klasifikácie")

            if unet_ok and result_after_unet is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    col_title("Pred denoisingom")
                    genre_bars(result_before["genres"])
                with col2:
                    col_title("Po klasickom denoising")
                    genre_bars(result_after_classic["genres"])
                with col3:
                    col_title("Po U-Net denoising")
                    genre_bars(result_after_unet["genres"])
            else:
                col1, col2 = st.columns(2)
                with col1:
                    col_title("Pred denoisingom")
                    genre_bars(result_before["genres"])
                with col2:
                    col_title("Po klasickom denoising")
                    genre_bars(result_after_classic["genres"])

            # ── Textový súhrn parametrov ───────────────────────────────────
            divider()
            section("📋", "Súhrn parametrov")
            note("Skopíruj a pošli na analýzu:")

            summary = build_summary_text(
                filename             = uploaded_file.name,
                result_before        = result_before,
                result_classic       = result_classic,
                result_after_classic = result_after_classic,
                result_after_unet    = result_after_unet,
                unet_snr_after       = unet_snr_after if unet_ok else None,
                unet_strength        = unet_strength  if unet_ok else None,
                unet_chunk           = unet_chunk_seconds if unet_ok else None,
                unet_device          = str(unet_denoiser.device) if unet_ok else None,
            )
            st.code(summary, language=None)

            for path in [input_path, output_path, unet_path]:
                if os.path.exists(path):
                    os.remove(path)

    # ══════════════════════════════════════════════════════════════════════
    # REŽIM 2: Pridanie šumu
    # ══════════════════════════════════════════════════════════════════════
    else:
        section("🔊", "Nastavenia šumu")

        noise_type = st.selectbox(
            "Typ šumu",
            options=["white", "pink", "brown", "hiss", "hum",
                     "crackle", "clicks", "vinyl"],
            format_func=lambda x: {
                "white":   "⬜ Biely šum – rovnomerné náhodné frekvencie",
                "pink":    "🌸 Ružový šum – 1/f, prirodzenejší",
                "brown":   "🟤 Hnedý šum – hlboký rumble (1/f²)",
                "hiss":    "🎞️ Tape hiss – vysokofrekvenčný, ako stará páska",
                "hum":     "⚡ Sieťový hum – 50 Hz + harmonické",
                "crackle": "📻 Praskanie – husté drobné impulzy",
                "clicks":  "💥 Kliknutia – zriedkavé výrazné škrabance",
                "vinyl":   "🎵 Vinyl – pink + rumble + crackle (realistické)",
            }[x],
        )

        noise_level = st.slider(
            "Intenzita šumu",
            min_value=0.01, max_value=0.10, value=0.02, step=0.01,
            help="0.01 = jemný šum | 0.05 = silný šum",
        )

        chips(Typ=noise_type.capitalize(), Intenzita=f"{noise_level:.2f}")
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
                SNR=f"{result_noise['snr']:.2f} dB",
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