"""
classifier.py – Klasifikácia žánru z audio súboru.

Feature extraction musí byť IDENTICKÁ s train.py (v5, 344 featúr).
Ak sa nezhoduje, model bude dávať nezmyselné výsledky.
"""

import librosa
import numpy as np
import pickle

with open("models/genre_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model       = bundle["model"]
scaler      = bundle["scaler"]
le          = bundle["le"]
SEGMENT_SEC = bundle.get("segment_sec", 15)

# Kontrola verzie featúr – ochrana pred nekompatibilným modelom
_expected_version = "v5"
_model_version = bundle.get("feature_version", "unknown")
if _model_version != _expected_version:
    import warnings
    warnings.warn(
        f"Model bol trénovaný s feature_version='{_model_version}', "
        f"ale classifier očakáva '{_expected_version}'. "
        f"Pretrénovaj model cez train.py."
    )


def extract_features(y, sr):
    """
    Extrahuje 344 featúr z audio segmentu.

    Musí byť IDENTICKÁ s extract_features_from_audio() v train.py.
    Ak zmeníš niečo tu, zmeň aj tam (a pretrénovaj model).
    """
    # MFCC + delta + delta-delta
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_feat = np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)])

    delta = librosa.feature.delta(mfcc, order=1)
    delta_feat = np.hstack([delta.mean(axis=1), delta.std(axis=1)])

    delta2 = librosa.feature.delta(mfcc, order=2)
    delta2_feat = np.hstack([delta2.mean(axis=1), delta2.std(axis=1)])

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_feat = np.hstack([chroma.mean(axis=1), chroma.std(axis=1)])

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_feat = np.array([centroid.mean(), centroid.std()])

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_feat = np.array([zcr.mean(), zcr.std()])

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_feat = np.hstack([contrast.mean(axis=1), contrast.std(axis=1)])

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_feat = np.array([rolloff.mean(), rolloff.std()])

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_feat = np.array([bandwidth.mean(), bandwidth.std()])

    # RMS energy
    rms = librosa.feature.rms(y=y)
    rms_feat = np.array([rms.mean(), rms.std()])

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_feat = np.array([float(np.squeeze(tempo))])

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_feat = np.hstack([mel_db.mean(axis=1), mel_db.std(axis=1)])

    # Tonnetz
    y_harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    tonnetz_feat = np.hstack([tonnetz.mean(axis=1), tonnetz.std(axis=1)])

    # Onset rate
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration = len(y) / sr
    onset_feat = np.array([
        onset_env.mean(),
        onset_env.std(),
        float(len(onsets)) / duration,
    ])

    return np.hstack([
        mfcc_feat,        # 80
        delta_feat,       # 80
        delta2_feat,      # 80
        chroma_feat,      # 24
        centroid_feat,    #  2
        zcr_feat,         #  2
        contrast_feat,    # 14
        rolloff_feat,     #  2
        bandwidth_feat,   #  2
        rms_feat,         #  2
        tempo_feat,       #  1
        mel_feat,         # 40
        tonnetz_feat,     # 12
        onset_feat,       #  3
    ])                    # SPOLU: 344


def classify_genre(file_path):
    """
    Klasifikuje žáner audio súboru.

    Rozdelí skladbu na 15s segmenty, klasifikuje každý zvlášť
    a vráti vážený priemer pravdepodobností (istejšie segmenty
    majú väčší vplyv).
    """
    y, sr = librosa.load(file_path, mono=True)
    total_duration = len(y) / sr
    segment_len = int(SEGMENT_SEC * sr)

    segments = []
    for start in range(0, len(y), segment_len):
        segment = y[start:start + segment_len]
        if len(segment) < sr * 5:
            continue
        segments.append(segment)

    if not segments:
        segments = [y]

    all_probas = []
    weights = []
    for segment in segments:
        features = extract_features(segment, sr).reshape(1, -1)
        features_scaled = scaler.transform(features)
        proba = model.predict_proba(features_scaled)[0]
        all_probas.append(proba)
        weights.append(np.max(proba))

    weights = np.array(weights)
    weights = weights / weights.sum()
    avg_proba = np.average(all_probas, axis=0, weights=weights)

    top3_idx = np.argsort(avg_proba)[::-1][:3]
    return {
        "genres": [
            {
                "genre": str(le.classes_[i]),
                "probability": round(float(avg_proba[i]) * 100, 1)
            }
            for i in top3_idx
        ],
        "segments_analyzed": len(segments),
        "total_duration": round(total_duration, 1)
    }