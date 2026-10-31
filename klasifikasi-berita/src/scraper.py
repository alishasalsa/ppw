import re
import requests
import trafilatura


class ScrapingError(Exception):
    """Error khusus untuk proses scraping artikel."""
    pass


def scrape_article(url):
    """
    Mengambil judul dan isi utama artikel dari URL menggunakan
    requests + trafilatura.

    Mengembalikan dict: {"title": ..., "content": ..., "url": ...}
    """

    # Validasi URL (perubahan: supaya pesan "URL tidak valid" muncul)
    if not re.match(r"^https?://", url.strip()):
        raise ScrapingError("URL tidak valid")

    for percobaan in range(3):
        try:
            response = requests.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/154.0.0.0 Safari/537.36"
                    )
                },
                timeout=30
            )

            if response.status_code != 200:
                raise ScrapingError(
                    f"Gagal mengakses URL. "
                    f"Status code: {response.status_code}"
                )

            text = trafilatura.extract(
                response.text,
                url=url
            )

            if not text:
                raise ScrapingError(
                    "Isi artikel tidak berhasil diekstrak."
                )

            # --- Perubahan: ambil JUDUL ---
            title = None
            metadata = trafilatura.extract_metadata(response.text)
            if metadata and metadata.title:
                title = metadata.title

            if not title:  # cadangan: tag <title> pada HTML
                match = re.search(
                    r"<title>(.*?)</title>",
                    response.text,
                    re.IGNORECASE | re.DOTALL
                )
                title = match.group(1).strip() if match else "(judul tidak ditemukan)"

            # --- Perubahan: kembalikan dict, bukan teks saja ---
            return {
                "title": title,
                "content": text,
                "url": url
            }

        except requests.exceptions.Timeout:
            print(
                f"Percobaan {percobaan + 1}/3: "
                "request timeout."
            )

            if percobaan == 2:
                raise ScrapingError(
                    "Koneksi ke website terlalu lama "
                    "dan waktu tunggu habis."
                )

        except requests.RequestException as e:
            raise ScrapingError(
                f"Gagal mengambil artikel: {e}"
            )

        except Exception as e:
            if isinstance(e, ScrapingError):
                raise

            raise ScrapingError(
                f"Terjadi kesalahan saat scraping: {e}"
            )