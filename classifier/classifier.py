import librosa
import numpy as np
import pickle

with open("models/genre_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model       = bundle["model"]
scaler      = bundle["scaler"]
le          = bundle["le"]
SEGMENT_SEC = bundle.get("segment_sec", 15)  # načítame z modelu, fallback 15


def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_feat = np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)])

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_feat = np.hstack([chroma.mean(axis=1), chroma.std(axis=1)])

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_feat = np.array([centroid.mean(), centroid.std()])

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_feat = np.array([zcr.mean(), zcr.std()])

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_feat = np.hstack([contrast.mean(axis=1), contrast.std(axis=1)])

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_feat = np.array([rolloff.mean(), rolloff.std()])

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_feat = np.array([bandwidth.mean(), bandwidth.std()])

    rms = librosa.feature.rms(y=y)
    rms_feat = np.array([rms.mean(), rms.std()])

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_feat = np.array([float(np.squeeze(tempo))])

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_feat = np.hstack([mel_db.mean(axis=1), mel_db.std(axis=1)])

    return np.hstack([
        mfcc_feat, chroma_feat, centroid_feat, zcr_feat,
        contrast_feat, rolloff_feat, bandwidth_feat,
        rms_feat, tempo_feat, mel_feat
    ])


def classify_genre(file_path):
    y, sr = librosa.load(file_path, mono=True)  # načíta CELÚ skladbu
    total_duration = len(y) / sr
    segment_len = int(SEGMENT_SEC * sr)          # 15s segmenty

    # Rozdeľ celú skladbu na 15s úseky
    segments = []
    for start in range(0, len(y), segment_len):
        segment = y[start:start + segment_len]
        if len(segment) < sr * 5:  # preskočí úseky kratšie ako 5s
            continue
        segments.append(segment)

    if not segments:
        segments = [y]

    # Klasifikuj každý úsek s váženým priemerom
    all_probas = []
    weights = []
    for segment in segments:
        features = extract_features(segment, sr).reshape(1, -1)
        features_scaled = scaler.transform(features)
        proba = model.predict_proba(features_scaled)[0]
        all_probas.append(proba)
        weights.append(np.max(proba))  # váha = istota modelu

    # Vážený priemer – istejšie segmenty majú väčší vplyv
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