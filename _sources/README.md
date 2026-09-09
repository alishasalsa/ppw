<div align="center">

# 📰 PPW - Web Search & Mining

**Klasifikasi Berita Indonesia: Sport vs Finance**

Tugas mata kuliah Pencarian dan Penambangan Web — disusun dengan metodologi CRISP-DM

[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=github)](https://haninhammoud01.github.io/ppw/)
[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter Book](https://img.shields.io/badge/built%20with-Jupyter%20Book-orange?logo=jupyter)](https://jupyterbook.org/)

</div>

---

## 📖 Tentang Proyek

Proyek ini mengumpulkan dan menyiapkan dataset **200 artikel berita** berbahasa Indonesia dari [detik.com](https://detik.com), untuk keperluan klasifikasi teks berita ke dalam dua kategori: **Sport** dan **Finance**.

**📚 Baca dokumentasi lengkap →** [haninhammoud01.github.io/ppw](https://haninhammoud01.github.io/ppw/)

## 🗂️ Struktur Proyek

```
ppw/
├── CRISP-DM/
│   ├── Business.ipynb      → Business Understanding
│   ├── EDA.ipynb           → Data Understanding
│   ├── Input.ipynb         → Data Preparation
│   ├── Modeling.md         → Modeling (segera hadir)
│   ├── Output.md           → Evaluation (segera hadir)
│   └── Production.md       → Deployment (segera hadir)
├── data/
│   └── dataset_berita_200.xlsx
├── _config.yml
├── _toc.yml
└── intro.md
```

## 📊 Ringkasan Dataset

| Kategori | Jumlah Artikel | ID Range |
|----------|:---:|:---:|
| ⚽ Sport | 100 | 1–100 |
| 💰 Finance | 100 | 101–200 |

**Kolom:** `id` · `isi_berita` · `label` · `url`

## 🛠️ Tools

`Python` · `Trafilatura` · `Pandas` · `Seaborn` · `Jupyter Book`

---

<div align="center">

*Hanin Hammoud*

</div>
