
import streamlit as st
from pathlib import Path

from src.scraper import scrape_article, ScrapingError
from src.prediction import (
    load_models,
    predict_category,
    PredictionError
)

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Klasifikasi Berita",
    page_icon="📰",
    layout="centered"
)

# ==========================================
# PATH MODEL
# ==========================================

# Mengambil lokasi folder app.py
BASE_DIR = Path(__file__).resolve().parent

# Lokasi folder models
MODEL_DIR = BASE_DIR / "models"

# Lokasi masing-masing file model
VECTORS_PATH = MODEL_DIR / "skipgram_vectors.npy"
VOCAB_PATH = MODEL_DIR / "skipgram_vocab.pkl"
NB_PATH = MODEL_DIR / "naive_bayes.pkl"
CONFIG_PATH = MODEL_DIR / "preprocessing_config.pkl"

# ==========================================
# JUDUL APLIKASI
# ==========================================

st.title("📰 Klasifikasi Berita")

st.write(
    "Masukkan URL berita untuk mengetahui apakah berita "
    "termasuk kategori **Sport** atau **Finance**."
)

# ==========================================
# CEK KETERSEDIAAN FILE MODEL
# ==========================================

model_files = [
    VECTORS_PATH,
    VOCAB_PATH,
    NB_PATH,
    CONFIG_PATH
]

missing_files = [
    path.name for path in model_files
    if not path.is_file()
]

if missing_files:
    st.error(
        "Beberapa file model tidak ditemukan di folder models:"
    )

    for filename in missing_files:
        st.write(f"- {filename}")

    st.write("Lokasi folder model:", str(MODEL_DIR))
    st.stop()

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_all_models():
    return load_models(
        VECTORS_PATH,
        VOCAB_PATH,
        NB_PATH,
        CONFIG_PATH
    )

try:
    W, vocab, nb_model, preprocessing_config = load_all_models()

except FileNotFoundError as e:
    st.error(f"File model tidak ditemukan: {e}")
    st.stop()

except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ==========================================
# INFORMASI MODEL
# ==========================================

with st.expander("Informasi Model"):
    st.write("Ukuran embedding:", W.shape)
    st.write("Jumlah vocabulary:", len(vocab["word2id"]))

# ==========================================
# INPUT URL
# ==========================================

url = st.text_input(
    "Masukkan URL berita:",
    placeholder="https://contoh.com/berita..."
)

# ==========================================
# TOMBOL KLASIFIKASI
# ==========================================

if st.button("🔍 Klasifikasikan Berita", use_container_width=True):

    if not url.strip():
        st.warning("Silakan masukkan URL berita terlebih dahulu.")

    else:

        # --------------------------------------
        # SCRAPING BERITA
        # --------------------------------------

        with st.spinner("Mengambil isi berita..."):
            try:
                article = scrape_article(url)

            except ScrapingError as e:
                st.error(f"Gagal mengambil berita: {e}")
                st.stop()

            except Exception as e:
                st.error(f"Terjadi kesalahan saat scraping: {e}")
                st.stop()

        # --------------------------------------
        # TAMPILKAN BERITA
        # --------------------------------------

        st.subheader("📄 Berita Ditemukan")

        st.markdown(f"**Judul:** {article['title']}")
        st.markdown(f"**URL:** {article['url']}")

        with st.expander("Lihat isi berita"):
            st.write(article["content"])

        # --------------------------------------
        # PREDIKSI KATEGORI
        # --------------------------------------

        with st.spinner("Melakukan klasifikasi..."):
            try:
                hasil = predict_category(
                    article["content"],
                    W,
                    vocab,
                    nb_model,
                    preprocessing_config
                )

            except PredictionError as e:
                st.error(f"Gagal melakukan prediksi: {e}")
                st.stop()

            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
                st.stop()

        # --------------------------------------
        # HASIL KLASIFIKASI
        # --------------------------------------

        st.divider()
        st.subheader("📊 Hasil Klasifikasi")

        label = hasil["label"].upper()
        confidence = hasil["confidence"]

        if label == "SPORT":
            st.success(f"Kategori: {label}")

        elif label == "FINANCE":
            st.info(f"Kategori: {label}")

        else:
            st.warning(f"Kategori: {label}")

        # --------------------------------------
        # CONFIDENCE
        # --------------------------------------

        if confidence is not None:
            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.caption(
                "Confidence adalah probabilitas prediksi model, "
                "bukan tingkat kebenaran klasifikasi."
            )

        # --------------------------------------
        # JUMLAH TOKEN
        # --------------------------------------

        st.write(
            "Jumlah token setelah preprocessing:",
            len(hasil["tokens"])
        )
