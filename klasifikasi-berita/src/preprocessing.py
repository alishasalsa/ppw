import re

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")


def bersihkan(teks):
    teks = str(teks).lower()                       # huruf kecil
    teks = URL_PATTERN.sub(" ", teks)              # hapus URL
    teks = re.sub(r"[^a-z0-9\s]", " ", teks)       # buang simbol (angka tetap)
    teks = re.sub(r"\s+", " ", teks).strip()       # rapikan spasi
    return teks


def preprocess(teks, stopwords):
    """Bersihkan teks, pecah jadi kata, lalu buang stopword."""
    return [w for w in bersihkan(teks).split() if w not in stopwords]
