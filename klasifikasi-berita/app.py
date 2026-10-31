import streamlit as st

from src.scraper import scrape_article, ScrapingError
from src.prediction import (
    load_models,
    predict_category,
    PredictionError
)

# ==========================================
# KONFIGURASI
# ==========================================

VECTORS_PATH = "models/skipgram_vectors.npy"
VOCAB_PATH = "models/skipgram_vocab.pkl"
NB_PATH = "models/naive_bayes.pkl"
CONFIG_PATH = "models/preprocessing_config.pkl"

# ==========================================
# HALAMAN
# ==========================================

st.set_page_config(
    page_title="Klasifikasi Berita",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Klasifikasi Berita")

st.write(
    "Masukkan URL berita untuk mengetahui apakah berita "
    "termasuk kategori **Sport** atau **Finance**."
)

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

except FileNotFoundError:
    st.error(
        "File model tidak ditemukan di folder `models/`. "
        "Pastikan semua file model sudah dibuat dari notebook training."
    )
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
# (semua proses di bawah ini HARUS di dalam blok if st.button)
# ==========================================

if st.button("🔍 Klasifikasikan Berita", use_container_width=True):

    if not url.strip():
        st.warning("Silakan masukkan URL berita terlebih dahulu.")
        st.stop()

    # --------------------------------------
    # SCRAPING
    # --------------------------------------
    with st.spinner("Mengambil isi berita..."):
        try:
            article = scrape_article(url)   # dict: title, content, url
        except ScrapingError as e:
            st.error(f"Gagal mengambil berita: {e}")
            st.stop()

    # --------------------------------------
    # TAMPILKAN BERITA (judul, URL, isi)
    # --------------------------------------
    st.subheader("📄 Berita Ditemukan")
    st.markdown(f"**Judul:** {article['title']}")
    st.markdown(f"**URL:** {article['url']}")

    with st.expander("Lihat isi berita"):
        st.write(article["content"])

    # --------------------------------------
    # PREDIKSI (yang dikirim hanya ISI berita, berupa teks)
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

    # --------------------------------------
    # HASIL
    # --------------------------------------
    st.divider()
    st.subheader("📊 Hasil Klasifikasi")

    label = hasil["label"].upper()
    confidence = hasil["confidence"]

    if label == "SPORT":
        st.success(f"Kategori: {label}")
    else:
        st.info(f"Kategori: {label}")

    if confidence is not None:
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.caption(
            "Confidence adalah probabilitas prediksi model, "
            "bukan tingkat kebenaran klasifikasi."
        )

    st.write(
        "Jumlah token setelah preprocessing:",
        len(hasil["tokens"])
    )