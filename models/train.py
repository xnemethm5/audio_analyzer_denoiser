import os
import numpy as np
import pickle
import librosa
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']
DATA_PATH = os.path.join("gtzan", "genres_original")
SEGMENT_SEC = 15  # dĺžka jedného segmentu v sekundách


# --- ZMENA 1: extract_features teraz pracuje s raw audiom ---
def extract_features_from_audio(y, sr):
    if len(y) < sr * 2:
        raise ValueError("Segment je príliš krátky (menej ako 2s)")

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
        mfcc_feat,        # 80 hodnôt
        chroma_feat,      # 24 hodnôt
        centroid_feat,    #  2 hodnoty
        zcr_feat,         #  2 hodnoty
        contrast_feat,    # 14 hodnôt
        rolloff_feat,     #  2 hodnoty
        bandwidth_feat,   #  2 hodnoty
        rms_feat,         #  2 hodnoty
        tempo_feat,       #  1 hodnota
        mel_feat,         # 40 hodnôt
    ])                    # SPOLU: ~169 hodnôt


# --- ZMENA 2: nová funkcia ktorá súbor rozdelí na segmenty ---
def extract_segments(file_path, segment_sec=SEGMENT_SEC):
    y, sr = librosa.load(file_path, mono=True)
    segment_len = sr * segment_sec
    features = []

    if len(y) < segment_len:
        # Skladba kratšia ako segment – použi celú
        try:
            features.append(extract_features_from_audio(y, sr))
        except Exception:
            pass
    else:
        for start in range(0, len(y) - segment_len, segment_len):
            seg = y[start:start + segment_len]
            try:
                features.append(extract_features_from_audio(seg, sr))
            except Exception:
                pass

    return features


# --- ZMENA 3: hlavný cyklus používa extract_segments ---
print("Extrahujem príznaky...")
X, y = [], []
for genre in GENRES:
    folder = os.path.join(DATA_PATH, genre)
    for fname in os.listdir(folder):
        if fname.endswith('.wav'):
            try:
                segs = extract_segments(os.path.join(folder, fname))
                for feat in segs:
                    X.append(feat)
                    y.append(genre)
            except Exception as e:
                print(f"Preskočený súbor {fname}: {e}")

print(f"Extrakcia hotová – {len(X)} segmentov spracovaných")

X = np.array(X)
le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# --- Jednotlivé modely ---
svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True)
rf  = RandomForestClassifier(n_estimators=300, max_features='sqrt',
                              min_samples_leaf=2, random_state=42, n_jobs=-1)
gb  = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                      max_depth=6, random_state=42)

print("\nTrénujem a porovnávam modely...")
results = {}
for name, clf in [("SVM", svm), ("Random Forest", rf), ("Gradient Boosting", gb)]:
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    results[name] = (clf, acc)
    print(f"  {name}: {acc:.2%}")

print("\nTrénujem Ensemble (SVM + RF + GB)...")
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

with open("models/genre_model.pkl", "wb") as f:
    pickle.dump({
        "model": best_model,
        "scaler": scaler,
        "le": le,
        "genres": GENRES,
        "segment_sec": SEGMENT_SEC,
        "feature_version": "v4"
    }, f)

print("Model uložený do models/genre_model.pkl")