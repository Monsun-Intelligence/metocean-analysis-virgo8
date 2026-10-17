# Analisis metocean KMP Virgo Transport 8 (Masalembo / Laut Jawa)

Konten analisis kondisi metocean sekitar hilang kontak KMP Virgo Transport 8
(13 Sep 2026 ~02:00 WITA / 12 Sep 2026 18:00 UTC) di perairan Masalembo, Laut Jawa.

Sumber berita: [ANTARA — kronologi](https://www.antaranews.com/berita/5740341/kronologi-kmp-virgo-transport-8-terbalik-di-laut-jawa) ·
[Basarnas koordinat pencarian](https://www.antaranews.com/berita/5739801/basarnas-cari-129-korban-km-virgo-hingga-2600-mil-laut-dari-koordinat)

## Yang diplot

| Mode | Isi |
|------|-----|
| **Spasial** | SWH, primary swell, angin 10 m, arus permukaan — domain 4°×4° sekitar kejadian |
| **Titik** | Time series di titik hilang kontak (+ estimasi drift bangkai), **±3 hari**, timestep native sumber |

Densifikasi display default **0.01° (~1.1 km)** — interpolasi + Gaussian ringan di atas grid native (GFS 0.25°). Ini **bukan** resolusi fisik baru.

## Dataset

| Variabel | Primary | Optional |
|----------|---------|----------|
| SWH, swell | GFS Wave (AWS archive, 3 h) | ERA5 (CDS, 1 h) |
| Angin 10 m | GFS Atmos (AWS, 3 h) | ERA5 |
| Arus permukaan | HYCOM ESPC archive (3 h) | Marine Copernicus |

Konfigurasi: [`event.yaml`](event.yaml)

## Setup

```bash
cd blog/02-virgo-transport
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Opsional: `~/.cdsapirc` + `cdsapi` untuk ERA5; `copernicusmarine login` untuk CMEMS.

## Pipeline

```bash
# 1) Unduh primary (±3 hari)
python scripts/download_gfs.py
python scripts/download_hycom.py

# Opsional
python scripts/download_era5.py
python scripts/download_cmems.py

# 2) Densifikasi spasial + ekstrak titik
python scripts/refine_field.py
python scripts/extract_point.py

# 3) Plot (pakai --every 4 untuk draft cepat)
python scripts/plot_spatial.py
python scripts/plot_point.py
```

Notebook: [`notebooks/analyze_virgo_transport.ipynb`](notebooks/analyze_virgo_transport.ipynb)

Output:

- `figures/spatial/{gfs,hycom}/…`
- `figures/point/…`

## Catatan metodologi

- GFS Wave/Atmos historis dari `noaa-gfs-bdp-pds`.
- HYCOM memakai **NCSS** day-chunk ke `ESPC-D-V02/{u,v}3z/{year}` (`scripts/download_hycom.py`). OpenDAP agregat tahunan sering hang; jika NCSS/TDS down, jalankan ulang skript nanti — raw harian di `data/hycom/raw/` di-resume otomatis.
- Marker peta: titik hilang kontak Basarnas + estimasi drift ~1.6 NM barat (bukan fix salvage).
- Upsampling hanya untuk readability peta; sinyal asli tetap skala model.


## Status data (Oct 2026)

- **Domain plot:** 4°×4° (`event.yaml` bbox `[112.035, 116.035, -6.531, -2.531]`).
- **GFS Wave/Atmos:** lengkap 10 Sep 00Z – 15 Sep 21Z.
- **HYCOM:** NCSS day-chunk 10–14 Sep 2026 (40 frame) pada domain 4°×4°; viewer arus berakhir 14 Sep 21Z.
## GitHub Pages

Static report: [`index.html`](index.html) (metogram, spatial viewer, metadata model + caveat).

1. Push repo (pastikan `figures/` ikut ter-commit — dibutuhkan halaman).
2. Settings → Pages → Source: **Deploy from a branch** → branch `main` → folder **`/` (root)**.
3. `.nojekyll` sudah ada agar GitHub tidak memproses lewat Jekyll.

Preview lokal:

```bash
cd blog/02-virgo-transport
python3 -m http.server 8080
# buka http://127.0.0.1:8080/
```

Slot **INACAWO ~3 km / 1 jam** disiapkan di halaman (`#inacawo`); integrasi data menyusul bila paket tersedia.
