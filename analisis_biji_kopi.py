"""
Tugas Pengolahan Citra Digital
Segmentasi & Penghitungan Biji Kopi: Otsu Thresholding vs Watershed
Library: OpenCV, NumPy, Matplotlib

Tiga kasus (case1.png, case2.png, case3.png) diproses terpisah dan
setiap kasus menyimpan hasilnya di folder sendiri.
"""

import os
import time
import cv2
import numpy as np
import matplotlib

# Set ke False jika ingin jendela matplotlib muncul satu per satu
TAMPILKAN_JENDELA = False
if not TAMPILKAN_JENDELA:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================================================================
# 0. KONFIGURASI
# ======================================================================
# Hitungan manual tiap kasus (cek ulang dengan mata sendiri, ubah jika perlu)
KASUS = {
    "case1": {"file": "case1.png", "manual": 10, "folder": "hasil_case1"},
    "case2": {"file": "case2.png", "manual": 11, "folder": "hasil_case2"},
    "case3": {"file": "case3.png", "manual": 12, "folder": "hasil_case3"},
}

MIN_AREA = 300      # area minimum (piksel) agar bukan dianggap noise
AMBANG_BESAR = 1.4  # komponen > 1.4x luas biji tunggal dianggap gabungan
RASIO_DIST = 0.8    # threshold distance map = RASIO * jarak maks LOKAL komponen


def baca_citra(path):
    """Membaca citra. Jika PNG punya alpha (transparan), digabung ke latar putih."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(
            f"File '{path}' tidak ditemukan / gagal dibaca. "
            "Letakkan di folder yang sama dengan script."
        )
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        img = (bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    print(f"[INFO] Citra dibaca: {path} | ukuran: {img.shape}")
    return img


# ======================================================================
# 1. PREPROCESSING
# ======================================================================
def preprocessing(img_bgr):
    """Grayscale + Gaussian blur ringan untuk mengurangi noise."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blur


# ======================================================================
# 2. OTSU THRESHOLDING
# ======================================================================
def metode_otsu(blur):
    """Otsu (INV: objek gelap -> putih), morfologi, lalu hitung komponen."""
    t0 = time.perf_counter()

    nilai_t, biner = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bersih = cv2.morphologyEx(biner, cv2.MORPH_OPEN, kernel, iterations=1)
    bersih = cv2.morphologyEx(bersih, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Isi lubang di dalam objek (highlight/kilap biji yang lebih terang dari T).
    # Tanpa ini distance transform punya banyak puncak palsu -> over-segmentation.
    kontur, _ = cv2.findContours(bersih, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    terisi = np.zeros_like(bersih)
    cv2.drawContours(terisi, kontur, -1, 255, thickness=cv2.FILLED)
    bersih = terisi

    n, label, stats, _ = cv2.connectedComponentsWithStats(bersih, connectivity=8)
    valid = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA]
    jumlah = len(valid)

    waktu = time.perf_counter() - t0
    return nilai_t, bersih, label, valid, jumlah, waktu


def estimasi_luas(label, valid, bersih):
    """Estimasi jumlah biji = total luas foreground / luas biji tunggal (median).
    Hanya pembanding kasar untuk memeriksa hitungan manual."""
    luas = np.array([np.sum(label == i) for i in valid])
    return bersih.astype(bool).sum() / np.median(luas), np.median(luas)


# ======================================================================
# 3. WATERSHED SEGMENTATION
# ======================================================================
def metode_watershed(img_bgr, bersih):
    """Distance transform -> marker -> cv2.watershed."""
    t0 = time.perf_counter()

    # a. Sure background
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(bersih, kernel, iterations=3)

    # b. Distance transform
    dist = cv2.distanceTransform(bersih, cv2.DIST_L2, 5)

    # c. Sure foreground: ambang RELATIF per komponen
    n_k, lbl_k = cv2.connectedComponents(bersih)
    sure_fg = np.zeros_like(bersih)
    luas_tunggal = np.median(np.bincount(lbl_k.ravel())[1:])  # luas biji tunggal
    for i in range(1, n_k):
        wilayah = lbl_k == i
        fg_i = wilayah & (dist >= RASIO_DIST * dist[wilayah].max())
        sure_fg[fg_i] = 255

        # Koreksi biji menempel/bertumpuk tanpa 'leher': jika marker kurang dari
        # perkiraan jumlah biji (dari luas), tambah marker dengan K-Means.
        harapan = int(round(wilayah.sum() / luas_tunggal))
        ada = cv2.connectedComponents(np.uint8(fg_i))[0] - 1
        if harapan >= 2 and ada < harapan and wilayah.sum() > AMBANG_BESAR * luas_tunggal:
            ys, xs = np.nonzero(wilayah)
            pts = np.float32(np.column_stack([xs, ys]))
            kriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
            _, _, pusat = cv2.kmeans(pts, harapan, None, kriteria, 5,
                                     cv2.KMEANS_PP_CENTERS)
            tmp = np.zeros_like(bersih)
            r = int(0.5 * dist[wilayah].max())
            for cx, cy in pusat:
                cv2.circle(tmp, (int(cx), int(cy)), r, 255, -1)
            # ganti marker komponen ini, jangan sampai keluar dari komponennya
            sure_fg[wilayah] = tmp[wilayah]

    # d. Daerah tak pasti
    unknown = cv2.subtract(sure_bg, sure_fg)

    # e. Marker
    n_marker, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # f. Watershed
    hasil = markers.copy()
    cv2.watershed(img_bgr.copy(), hasil)

    # g. Hitung objek unik (-1 = batas, 1 = background)
    label_unik = [l for l in np.unique(hasil) if l > 1]
    valid = [l for l in label_unik if np.sum(hasil == l) >= MIN_AREA]
    jumlah = len(valid)

    waktu = time.perf_counter() - t0

    overlay = img_bgr.copy()
    overlay[hasil == -1] = (0, 0, 255)

    return dist, sure_fg, sure_bg, unknown, markers, hasil, overlay, jumlah, waktu


# ======================================================================
# 4. VISUALISASI & PENYIMPANAN (per kasus)
# ======================================================================
def tampilkan(nama, folder, img, gray, blur, nilai_t, otsu_mask, otsu_label,
              otsu_valid, dist, sure_fg, unknown, markers, overlay, n_otsu, n_ws):
    warna = np.zeros((*otsu_label.shape, 3), np.uint8)
    rng = np.random.default_rng(1)
    for i in otsu_valid:
        warna[otsu_label == i] = rng.integers(60, 255, 3)

    fig, ax = plt.subplots(3, 3, figsize=(16, 10))
    ax = ax.ravel()
    tampil = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "1. Citra Asli", None),
        (gray, "2. Grayscale", "gray"),
        (blur, "3. Gaussian Blur (Preprocessing)", "gray"),
        (otsu_mask, f"4. Otsu + Morfologi (T={nilai_t:.0f})", "gray"),
        (warna, f"5. Komponen Otsu: {n_otsu} objek", None),
        (dist, "6. Distance Transform", "jet"),
        (sure_fg, "7. Sure Foreground (Marker)", "gray"),
        (markers, "8. Marker Watershed", "nipy_spectral"),
        (cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
         f"9. Hasil Watershed: {n_ws} objek", None),
    ]
    for a, (im, judul, cmap) in zip(ax, tampil):
        a.imshow(im, cmap=cmap)
        a.set_title(judul, fontsize=10)
        a.axis("off")
    fig.suptitle(f"{nama.upper()} - Otsu vs Watershed", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "00_ringkasan_semua_tahap.png"), dpi=150)
    if TAMPILKAN_JENDELA:
        plt.show()
    plt.close(fig)


def simpan_hasil(folder, img, gray, blur, otsu_mask, otsu_label, otsu_valid,
                 dist, sure_fg, markers, overlay, n_otsu, n_ws):
    os.makedirs(folder, exist_ok=True)

    warna = np.full(img.shape, 255, np.uint8)
    rng = np.random.default_rng(1)
    for i in otsu_valid:
        warna[otsu_label == i] = rng.integers(60, 230, 3)
    otsu_info = cv2.putText(warna.copy(), f"Otsu: {n_otsu} objek", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)

    dist_vis = cv2.applyColorMap(
        cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET)

    mk_vis = np.full(img.shape, 255, np.uint8)
    for l in np.unique(markers):
        if l > 1:
            mk_vis[markers == l] = rng.integers(60, 230, 3)

    ws_info = cv2.putText(overlay.copy(), f"Watershed: {n_ws} objek", (30, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)

    daftar = {
        "01_citra_asli.png": img,
        "02_grayscale.png": gray,
        "03_preprocessing_blur.png": blur,
        "04_otsu_mask_morfologi.png": otsu_mask,
        "05_otsu_komponen_berwarna.png": otsu_info,
        "06_distance_transform.png": dist_vis,
        "07_sure_foreground.png": sure_fg,
        "08_marker_watershed.png": mk_vis,
        "09_hasil_watershed.png": ws_info,
    }
    for nama, gambar in daftar.items():
        cv2.imwrite(os.path.join(folder, nama), gambar)
    print(f"[INFO] {len(daftar)} gambar + ringkasan disimpan di folder '{folder}/'")


# ======================================================================
# 5. DRAF ANALISIS LAPORAN (dinamis per kasus)
# ======================================================================
def status(n, acuan):
    if n < acuan:
        return "under-segmentation (objek menempel terhitung satu)"
    if n > acuan:
        return "over-segmentation (satu objek terpecah)"
    return "sesuai dengan acuan jumlah objek"


def buat_analisis(nama, manual, n_otsu, n_ws, t_otsu, t_ws, nilai_t, est):
    sel_otsu = n_otsu - manual
    sel_ws = n_ws - manual
    err_otsu, err_ws = abs(sel_otsu), abs(sel_ws)

    if err_ws < err_otsu:
        kesimpulan = (
            f"Pada {nama}, Watershed menghasilkan jumlah objek yang lebih dekat "
            f"dengan hitungan manual ({manual}) dibanding Otsu "
            f"(selisih Watershed {sel_ws:+d} vs Otsu {sel_otsu:+d}). Biji yang "
            "saling menempel berhasil dipisahkan lewat distance transform dan "
            "marker, dengan konsekuensi komputasi dan parameter yang lebih kompleks."
        )
    elif err_ws == err_otsu:
        kesimpulan = (
            f"Pada {nama}, Otsu dan Watershed memberi selisih yang sama terhadap "
            f"hitungan manual ({manual}): Otsu {sel_otsu:+d}, Watershed {sel_ws:+d}. "
            "Karena hasilnya sama, Otsu lebih unggul dari sisi kesederhanaan dan "
            "kecepatan, sedangkan Watershed tidak memberi keuntungan pada kasus ini."
        )
    else:
        kesimpulan = (
            f"Pada {nama}, Otsu lebih dekat ke hitungan manual ({manual}) "
            f"(selisih Otsu {sel_otsu:+d} vs Watershed {sel_ws:+d}). Watershed "
            "kemungkinan mengalami over-segmentation atau marker terlewat; "
            "parameter RASIO_DIST / AMBANG_BESAR perlu disesuaikan."
        )

    return f"""
======================================================================
DRAF ANALISIS LAPORAN - {nama.upper()}
======================================================================

1. KARAKTERISTIK GAMBAR
-----------------------
- Objek: biji kopi sangrai cokelat gelap, oval dengan alur tengah, orientasi acak.
- Latar putih bersih, kontras objek-latar tinggi (histogram cenderung bimodal,
  cocok untuk Otsu).
- Jumlah objek hasil hitung manual: {manual} biji.
- Estimasi dari total luas foreground / luas biji tunggal: ~{est:.1f}
  (pembanding kasar; biji yang menumpuk menutupi sebagian luasnya).
- Tekstur permukaan (kilap dan bayangan alur) dapat menimbulkan variasi
  intensitas di dalam objek.

2. PERFORMA OTSU THRESHOLDING
-----------------------------
- Threshold otomatis Otsu: {nilai_t:.0f}.
- Jumlah objek terdeteksi: {n_otsu} (selisih {sel_otsu:+d} terhadap manual)
  -> {status(n_otsu, manual)}.
- Kelebihan: sederhana, cepat ({t_otsu*1000:.2f} ms), tanpa threshold manual,
  sangat baik memisahkan objek dari latar putih.
- Kekurangan: hanya melihat intensitas, tidak mengenal bentuk. Biji yang
  bersentuhan menjadi satu komponen terhubung sehingga terhitung satu objek.

3. PERFORMA WATERSHED SEGMENTATION
----------------------------------
- Jumlah objek terdeteksi: {n_ws} (selisih {sel_ws:+d} terhadap manual)
  -> {status(n_ws, manual)}.
- Watershed memakai distance transform: tiap biji (oval, cembung) memberi
  puncak jarak sendiri sebagai marker, lalu garis batas dibuat pada lekukan
  di antara biji yang menempel. Komponen yang luasnya jauh di atas biji tunggal
  diberi marker tambahan dengan K-Means.
- Kekurangan: sensitif terhadap parameter (rasio threshold distance map).
  Rasio terlalu rendah -> marker menyatu; terlalu tinggi -> biji oval panjang
  bisa terpecah atau marker hilang.
- Waktu komputasi: {t_ws*1000:.2f} ms.

4. TABEL PERBANDINGAN
---------------------
| Kriteria                  | Otsu Thresholding | Watershed         |
|---------------------------|-------------------|-------------------|
| Jumlah objek manual       | {manual:<17} | {manual:<17} |
| Jumlah objek terdeteksi   | {n_otsu:<17} | {n_ws:<17} |
| Selisih thd manual        | {sel_otsu:<+17d} | {sel_ws:<+17d} |
| Waktu komputasi           | {t_otsu*1000:<14.2f} ms | {t_ws*1000:<14.2f} ms |
| Kompleksitas metode       | Rendah            | Sedang-tinggi     |

5. KESIMPULAN
-------------
{kesimpulan}
"""


# ======================================================================
# 6. PROSES SATU KASUS
# ======================================================================
def proses_kasus(nama, cfg):
    print("\n" + "#" * 70)
    print(f"# {nama.upper()}  ->  {cfg['file']}  ->  folder '{cfg['folder']}/'")
    print("#" * 70)

    folder = cfg["folder"]
    manual = cfg["manual"]
    os.makedirs(folder, exist_ok=True)

    img = baca_citra(cfg["file"])
    gray, blur = preprocessing(img)

    nilai_t, otsu_mask, otsu_label, otsu_valid, n_otsu, t_otsu = metode_otsu(blur)
    (dist, sure_fg, sure_bg, unknown, markers,
     hasil_ws, overlay, n_ws, t_ws) = metode_watershed(img, otsu_mask)
    est, luas_med = estimasi_luas(otsu_label, otsu_valid, otsu_mask)

    print(f"Jumlah manual   : {manual}")
    print(f"Hasil Otsu      : {n_otsu}  (selisih {n_otsu - manual:+d})")
    print(f"Hasil Watershed : {n_ws}  (selisih {n_ws - manual:+d})")
    print(f"Estimasi luas   : ~{est:.1f} (luas biji tunggal ~{luas_med:.0f} px)")
    print(f"Waktu Otsu      : {t_otsu*1000:.2f} ms | Watershed: {t_ws*1000:.2f} ms")

    simpan_hasil(folder, img, gray, blur, otsu_mask, otsu_label, otsu_valid,
                 dist, sure_fg, markers, overlay, n_otsu, n_ws)
    tampilkan(nama, folder, img, gray, blur, nilai_t, otsu_mask, otsu_label,
              otsu_valid, dist, sure_fg, unknown, markers, overlay, n_otsu, n_ws)

    teks = buat_analisis(nama, manual, n_otsu, n_ws, t_otsu, t_ws, nilai_t, est)
    with open(os.path.join(folder, "analisis.txt"), "w", encoding="utf-8") as f:
        f.write(teks)
    print(teks)

    return {"kasus": nama, "manual": manual, "otsu": n_otsu, "watershed": n_ws,
            "t_otsu": t_otsu * 1000, "t_ws": t_ws * 1000}


# ======================================================================
# 7. MAIN
# ======================================================================
def main():
    rekap = [proses_kasus(nama, cfg) for nama, cfg in KASUS.items()]

    print("\n" + "=" * 70)
    print("REKAP SEMUA KASUS")
    print("=" * 70)
    print(f"{'Kasus':<8}{'Manual':>8}{'Otsu':>8}{'Selisih':>9}"
          f"{'Watershed':>11}{'Selisih':>9}")
    for r in rekap:
        print(f"{r['kasus']:<8}{r['manual']:>8}{r['otsu']:>8}"
              f"{r['otsu'] - r['manual']:>+9d}{r['watershed']:>11}"
              f"{r['watershed'] - r['manual']:>+9d}")
    print("=" * 70)


if __name__ == "__main__":
    main()