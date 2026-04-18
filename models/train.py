"""
train.py – Trénovanie žánrového klasifikátora na GTZAN datasete.

Zmeny oproti pôvodnej verzii:
  - Delta a delta-delta MFCC (+160 featúr) – zachytávajú dynamiku spektra
  - Tonnetz (+12) – harmonická štruktúra (kvintový/terciový kruh)
  - Onset rate (+3) – hustota úderov za sekundu
  - 50% overlap segmentov – takmer 2× viac trénovacích dát
  - Data augmentácia (pitch shift, time stretch, šum) – 4× viac dát
  - Stratified 5-Fold krížová validácia – spoľahlivejší odhad accuracy
  - Feature count: ~169 → ~344
"""

import os
import time
import numpy as np
import pickle
import librosa
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']
DATA_PATH = os.path.join("gtzan", "genres_original")
SEGMENT_SEC = 15


# ==============================================================================
# Feature extraction – musí byť IDENTICKÁ s classifier.py
# ==============================================================================

def extract_features_from_audio(y, sr):
    """
    Extrahuje 344 featúr z audio segmentu.

    Oproti pôvodnej verzii (169 featúr) pridané:
      - Delta MFCC (1. derivácia)    – 80 hodnôt (dynamika spektra)
      - Delta-delta MFCC (2. deriv.) – 80 hodnôt (zrýchlenie zmien)
      - Tonnetz                      – 12 hodnôt (harmonická štruktúra)
      - Onset rate + onset stats     –  3 hodnoty (rytmická hustota)
    """
    if len(y) < sr * 2:
        raise ValueError("Segment je príliš krátky (menej ako 2s)")

    # MFCC + delta + delta-delta
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_feat = np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)])           # 80

    delta = librosa.feature.delta(mfcc, order=1)
    delta_feat = np.hstack([delta.mean(axis=1), delta.std(axis=1)])         # 80

    delta2 = librosa.feature.delta(mfcc, order=2)
    delta2_feat = np.hstack([delta2.mean(axis=1), delta2.std(axis=1)])      # 80

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_feat = np.hstack([chroma.mean(axis=1), chroma.std(axis=1)])      # 24

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_feat = np.array([centroid.mean(), centroid.std()])              # 2

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_feat = np.array([zcr.mean(), zcr.std()])                            # 2

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_feat = np.hstack([contrast.mean(axis=1), contrast.std(axis=1)]) # 14

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_feat = np.array([rolloff.mean(), rolloff.std()])                # 2

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_feat = np.array([bandwidth.mean(), bandwidth.std()])          # 2

    # RMS energy
    rms = librosa.feature.rms(y=y)
    rms_feat = np.array([rms.mean(), rms.std()])                            # 2

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_feat = np.array([float(np.squeeze(tempo))])                       # 1

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_feat = np.hstack([mel_db.mean(axis=1), mel_db.std(axis=1)])         # 40

    # Tonnetz – harmonická štruktúra (kvintový a terciový kruh)
    # Výborne rozlišuje classical/jazz (bohaté harmónie) od hiphop (jednoduché)
    y_harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    tonnetz_feat = np.hstack([tonnetz.mean(axis=1), tonnetz.std(axis=1)])    # 12

    # Onset rate – hustota úderov za sekundu
    # Metal má veľa onsetov, reggae málo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration = len(y) / sr
    onset_feat = np.array([
        onset_env.mean(),
        onset_env.std(),
        float(len(onsets)) / duration,  # onset rate (onsets/sec)
    ])                                                                       # 3

    return np.hstack([
        mfcc_feat,        # 80
        delta_feat,       # 80  ← NOVÉ
        delta2_feat,      # 80  ← NOVÉ
        chroma_feat,      # 24
        centroid_feat,    #  2
        zcr_feat,         #  2
        contrast_feat,    # 14
        rolloff_feat,     #  2
        bandwidth_feat,   #  2
        rms_feat,         #  2
        tempo_feat,       #  1
        mel_feat,         # 40
        tonnetz_feat,     # 12  ← NOVÉ
        onset_feat,       #  3  ← NOVÉ
    ])                    # SPOLU: 344


# ==============================================================================
# Segmentácia s overlapom
# ==============================================================================

def extract_segments(file_path, segment_sec=SEGMENT_SEC):
    """
    Rozdelí audio na segmenty s 50% overlapom.

    Pôvodne bez overlapu → ~2 segmenty per 30s skladbu.
    S 50% overlapom   → ~3 segmenty per 30s skladbu.
    Viac trénovacích dát bez augmentácie.
    """
    y, sr = librosa.load(file_path, mono=True)
    segment_len = sr * segment_sec
    hop = segment_len // 2  # 50% overlap
    features = []

    if len(y) < segment_len:
        try:
            features.append(extract_features_from_audio(y, sr))
        except Exception:
            pass
    else:
        for start in range(0, len(y) - segment_len, hop):
            seg = y[start:start + segment_len]
            try:
                features.append(extract_features_from_audio(seg, sr))
            except Exception:
                pass

    return features, y, sr


# ==============================================================================
# Data augmentácia
# ==============================================================================

def augment_and_extract(y, sr, segment_sec, rng):
    """
    Augmentuje audio a extrahuje featúry z augmentovaných verzií.

    Augmentácie (len pri trénovaní, NIE pri inferencii):
      - Pitch shift ±2 polotóny (variácia v ladení)
      - Time stretch 0.85–1.15× (variácia v tempe)
      - Aditívny šum σ=0.005 (variácia v kvalite nahrávky)

    Efektívne 4× viac dát (originál + 3 augmentácie).
    """
    segment_len = sr * segment_sec
    augmented_features = []

    # 1. Pitch shift
    try:
        n_steps = rng.uniform(-2, 2)
        y_ps = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        for start in range(0, len(y_ps) - segment_len, segment_len // 2):
            seg = y_ps[start:start + segment_len]
            try:
                augmented_features.append(extract_features_from_audio(seg, sr))
            except Exception:
                pass
    except Exception:
        pass

    # 2. Time stretch
    try:
        rate = rng.uniform(0.85, 1.15)
        y_ts = librosa.effects.time_stretch(y, rate=rate)
        ts_seg_len = int(sr * segment_sec)
        for start in range(0, len(y_ts) - ts_seg_len, ts_seg_len // 2):
            seg = y_ts[start:start + ts_seg_len]
            try:
                augmented_features.append(extract_features_from_audio(seg, sr))
            except Exception:
                pass
    except Exception:
        pass

    # 3. Aditívny šum
    try:
        noise = rng.normal(0, 0.005, len(y)).astype(np.float32)
        y_noisy = y + noise
        for start in range(0, len(y_noisy) - segment_len, segment_len // 2):
            seg = y_noisy[start:start + segment_len]
            try:
                augmented_features.append(extract_features_from_audio(seg, sr))
            except Exception:
                pass
    except Exception:
        pass

    return augmented_features


# ==============================================================================
# Hlavný trénovací cyklus
# ==============================================================================

print("Extrahujem príznaky (s overlapom + augmentáciou)...")
rng = np.random.default_rng(42)
X, y_labels = [], []
skipped = 0

# Spočítaj celkový počet súborov pre progress
all_files = []
for genre in GENRES:
    folder = os.path.join(DATA_PATH, genre)
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.wav'):
            all_files.append((genre, os.path.join(folder, fname), fname))

total = len(all_files)
print(f"Nájdených {total} súborov v {len(GENRES)} žánroch\n")

t_start = time.time()
for i, (genre, fpath, fname) in enumerate(all_files, 1):
    try:
        # Originálne segmenty (s overlapom)
        segs, y_audio, sr = extract_segments(fpath)
        for feat in segs:
            X.append(feat)
            y_labels.append(genre)

        # Augmentované segmenty
        aug_feats = augment_and_extract(y_audio, sr, SEGMENT_SEC, rng)
        for feat in aug_feats:
            X.append(feat)
            y_labels.append(genre)

        n_segs = len(segs) + len(aug_feats)
        elapsed = time.time() - t_start
        per_file = elapsed / i
        remaining = per_file * (total - i)
        print(f"  [{i:3d}/{total}] {genre:12s} {fname:30s} → {n_segs:2d} seg  "
              f"({len(X):5d} celkovo)  ETA {remaining/60:.0f}m")

    except Exception as e:
        print(f"  [{i:3d}/{total}] SKIP {fname}: {e}")
        skipped += 1

elapsed_total = time.time() - t_start
print(f"\nExtrakcia hotová – {len(X)} segmentov ({skipped} preskočených) za {elapsed_total/60:.1f} min")

X = np.array(X)
le = LabelEncoder()
y_enc = le.fit_transform(y_labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================================================================
# Trénovanie + krížová validácia
# ==============================================================================

print(f"\nFeature dimension: {X.shape[1]}")
print(f"Vzoriek celkom:   {X.shape[0]}")
print(f"Vzoriek per žáner: ~{X.shape[0] // len(GENRES)}")

# Jednotlivé modely
svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True)
rf  = RandomForestClassifier(n_estimators=300, max_features='sqrt',
                              min_samples_leaf=2, random_state=42, n_jobs=-1)
gb  = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                      max_depth=6, random_state=42)

# 5-fold krížová validácia – spoľahlivejší odhad než jeden split
print("\n--- 5-Fold krížová validácia ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, clf in [("SVM", svm), ("Random Forest", rf), ("Gradient Boosting", gb)]:
    scores = cross_val_score(clf, X_scaled, y_enc, cv=cv, scoring='accuracy')
    print(f"  {name}: {scores.mean():.2%} ± {scores.std():.2%}")

# Finálny tréning na celých dátach pre ensemble
print("\nTrénujem finálny Ensemble (SVM + RF + GB) na celých dátach...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Najprv natrénuj jednotlivé modely + ensemble
results = {}
for name, clf in [("SVM", svm), ("Random Forest", rf), ("Gradient Boosting", gb)]:
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    results[name] = (clf, acc)
    print(f"  {name}: {acc:.2%}")

ensemble = VotingClassifier(
    estimators=[('svm', svm), ('rf', rf), ('gb', gb)],
    voting='soft'
)
ensemble.fit(X_train, y_train)
ensemble_acc = ensemble.score(X_test, y_test)
results["Ensemble"] = (ensemble, ensemble_acc)
print(f"  Ensemble: {ensemble_acc:.2%}")

best_name = max(results, key=lambda k: results[k][1])
best_model, best_acc = results[best_name]
print(f"\nNajlepší model: {best_name} ({best_acc:.2%})")

print("\n--- Finálne výsledky ---")
print(classification_report(y_test, best_model.predict(X_test), target_names=le.classes_))

# ==============================================================================
# Uloženie modelu
# ==============================================================================

os.makedirs("models", exist_ok=True)
with open("models/genre_model.pkl", "wb") as f:
    pickle.dump({
        "model": best_model,
        "scaler": scaler,
        "le": le,
        "genres": GENRES,
        "segment_sec": SEGMENT_SEC,
        "feature_version": "v5",  # v4 → v5 (delta MFCC, tonnetz, onset)
    }, f)

print("Model uložený do models/genre_model.pkl")
print(f"Feature version: v5 ({X.shape[1]} features)")