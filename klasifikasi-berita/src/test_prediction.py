from src.scraper import scrape_article, ScrapingError
from src.prediction import (
    load_models,
    predict_category,
    PredictionError
)

# =========================
# PATH MODEL
# =========================

VECTORS_PATH = "models/skipgram_vectors.npy"
VOCAB_PATH = "models/skipgram_vocab.pkl"
NB_PATH = "models/naive_bayes.pkl"
CONFIG_PATH = "models/preprocessing_config.pkl"


# =========================
# URL BERITA UNTUK TES
# =========================

URL = "https://sport.detik.com/moto-gp/d-8652587/motogp-san-marino-antusiasme-bezzecchi-balapan-di-rumah-sendiri"


# =========================
# LOAD MODEL
# =========================

print("Loading model...")

W, vocab, nb_model, preprocessing_config = load_models(
    VECTORS_PATH,
    VOCAB_PATH,
    NB_PATH,
    CONFIG_PATH
)

print("Model berhasil dimuat.")
print("Ukuran embedding:", W.shape)
print("Jumlah vocabulary:", len(vocab["word2id"]))


# =========================
# SCRAPING
# =========================

print("\nMengambil berita...")

try:
    article = scrape_article(URL)

    print("URL   :", URL)
    print("Panjang isi:", len(article))

except ScrapingError as e:
    print("SCRAPING ERROR:", e)
    raise SystemExit


# =========================
# PREDIKSI
# =========================

print("\nMelakukan prediksi...")

try:
    hasil = predict_category(
        article,
        W,
        vocab,
        nb_model,
        preprocessing_config
    )

    print("\n==============================")
    print("HASIL KLASIFIKASI")
    print("==============================")
    print("Kategori   :", hasil["label"].upper())

    if hasil["confidence"] is not None:
        print("Confidence :", f"{hasil['confidence'] * 100:.2f}%")

    print("Jumlah token:", len(hasil["tokens"]))

except PredictionError as e:
    print("PREDICTION ERROR:", e)