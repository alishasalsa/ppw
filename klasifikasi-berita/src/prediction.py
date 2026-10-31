
import re
import pickle
import joblib
import numpy as np


class PredictionError(Exception):
    """Error khusus untuk proses prediksi."""
    pass


def load_models(
    vectors_path,
    vocab_path,
    nb_path,
    config_path
):
    """Memuat seluruh model yang dibutuhkan untuk prediksi."""

    try:
        # Load embedding Skip-gram
        W = np.load(vectors_path)

        # Load vocabulary
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        # Load Naive Bayes
        nb_model = joblib.load(nb_path)

        # Load preprocessing configuration
        with open(config_path, "rb") as f:
            preprocessing_config = pickle.load(f)

        return W, vocab, nb_model, preprocessing_config

    except Exception as e:
        raise PredictionError(
            f"Gagal memuat model: {e}"
        )


def preprocess_text(text, config):
    """Preprocessing teks sesuai konfigurasi model."""

    if not isinstance(text, str):
        raise PredictionError("Teks berita harus berupa string.")

    # Lowercase
    if config.get("lowercase", True):
        text = text.lower()

    # Hapus URL
    if config.get("hapus_url", True):
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Hapus simbol/tanda baca
    # Angka tetap dipertahankan.
    if config.get("hapus_simbol", True):
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Hapus angka hanya jika konfigurasi meminta.
    # Model kamu: hapus_angka = False
    if config.get("hapus_angka", False):
        text = re.sub(r"\d+", " ", text)

    # Rapikan spasi
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenisasi
    tokens = text.split()

    # Stopword
    if config.get("hapus_stopword", False):
        stopwords = set(config.get("stopwords", []))
        tokens = [
            token for token in tokens
            if token not in stopwords
        ]

    return tokens


def document_vector(tokens, W, word2id):
    """
    Mengubah kumpulan token menjadi satu vektor dokumen
    dengan cara menghitung rata-rata vektor Skip-gram.
    """

    vectors = []

    for token in tokens:
        if token in word2id:
            word_id = word2id[token]

            if 0 <= word_id < len(W):
                vectors.append(W[word_id])

    if not vectors:
        raise PredictionError(
            "Tidak ada kata dari berita yang ditemukan "
            "di vocabulary Skip-gram."
        )

    return np.mean(vectors, axis=0)


def predict_category(
    text,
    W,
    vocab,
    nb_model,
    preprocessing_config
):
    """Melakukan preprocessing, embedding, dan klasifikasi."""

    try:
        # =========================
        # PREPROCESSING
        # =========================

        tokens = preprocess_text(
            text,
            preprocessing_config
        )

        if not tokens:
            raise PredictionError(
                "Tidak ada token setelah preprocessing."
            )

        # =========================
        # VOCABULARY
        # =========================

        word2id = vocab["word2id"]

        # =========================
        # DOCUMENT VECTOR
        # =========================

        vector = document_vector(
            tokens,
            W,
            word2id
        )

        # Bentuk: (1, 100)
        X = vector.reshape(1, -1)

        # =========================
        # NAIVE BAYES
        # =========================

        prediction = nb_model.predict(X)[0]

        # Confidence
        confidence = None

        if hasattr(nb_model, "predict_proba"):
            probabilities = nb_model.predict_proba(X)[0]
            confidence = float(np.max(probabilities))

        return {
            "label": prediction,
            "confidence": confidence,
            "tokens": tokens,
            "vector": vector
        }

    except PredictionError:
        raise

    except Exception as e:
        raise PredictionError(
            f"Gagal melakukan prediksi: {e}"
        )
