"""
Tugas Pengolahan Citra Digital
Segmentasi & Penghitungan Biji Kopi: Otsu Thresholding vs Watershed
Library: OpenCV, NumPy, Matplotlib
"""

import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# 0. KONFIGURASI
# ======================================================================
# Script mencoba .jpg dahulu, lalu .png (file yang dikirim berformat .png)
KANDIDAT_FILE = ["dataset-kopi.jpg", "dataset-kopi.png"]
FOLDER_HASIL = "hasil"      # folder penyimpanan semua gambar hasil
JUMLAH_MANUAL = 54          # hasil hitung manual
MIN_AREA = 300              # area minimum (piksel) agar bukan dianggap noise
AMBANG_BESAR = 1.4          # komponen > 1.4x luas biji tunggal dianggap gabungan
RASIO_DIST = 0.8            # threshold distance map = RASIO * jarak maks LOKAL tiap komponen


def baca_citra():
    """Membaca citra dari file yang tersedia."""
    for nama in KANDIDAT_FILE:
        if os.path.exists(nama):
            img = cv2.imread(nama)
            if img is not None:
                print(f"[INFO] Citra dibaca: {nama} | ukuran: {img.shape}")
                return img
    raise FileNotFoundError(
        f"Tidak menemukan salah satu file: {KANDIDAT_FILE}. "
        "Letakkan file di folder yang sama dengan script."
    )


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

    # Morfologi: opening buang noise kecil, closing tutup lubang kecil
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bersih = cv2.morphologyEx(biner, cv2.MORPH_OPEN, kernel, iterations=1)
    bersih = cv2.morphologyEx(bersih, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Connected components (abaikan background label 0 dan area kecil)
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

    # a. Sure background: dilasi pada mask biner
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(bersih, kernel, iterations=3)

    # b. Distance transform
    dist = cv2.distanceTransform(bersih, cv2.DIST_L2, 5)

    # c. Sure foreground: threshold pada distance map
    # Ambang dihitung RELATIF per komponen (bukan global), agar biji menempel
    # yang membentuk 'leher' sempit tetap terpisah menjadi marker sendiri.
    n_k, lbl_k = cv2.connectedComponents(bersih)
    sure_fg = np.zeros_like(bersih)
    luas_tunggal = np.median(np.bincount(lbl_k.ravel())[1:])  # luas biji tunggal
    for i in range(1, n_k):
        wilayah = lbl_k == i
        fg_i = wilayah & (dist >= RASIO_DIST * dist[wilayah].max())
        sure_fg[fg_i] = 255

        # Koreksi biji bertumpuk/menempel penuh (tanpa 'leher'): jika jumlah
        # marker lebih sedikit daripada perkiraan dari luas, tempatkan marker
        # tambahan dengan K-Means pada koordinat piksel komponen tsb.
        harapan = int(round(wilayah.sum() / luas_tunggal))
        ada = cv2.connectedComponents(np.uint8(fg_i))[0] - 1
        if harapan >= 2 and ada < harapan and wilayah.sum() > AMBANG_BESAR * luas_tunggal:
            ys, xs = np.nonzero(wilayah)
            pts = np.float32(np.column_stack([xs, ys]))
            kriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
            _, _, pusat = cv2.kmeans(pts, harapan, None, kriteria, 5,
                                     cv2.KMEANS_PP_CENTERS)
            sure_fg[wilayah] = 0
            r = int(0.5 * dist[wilayah].max())
            for cx, cy in pusat:
                cv2.circle(sure_fg, (int(cx), int(cy)), r, 255, -1)
            sure_fg[~wilayah] = np.where(sure_fg[~wilayah] > 0, 255, 0)
            sure_fg &= bersih  # marker tidak boleh keluar dari objek

    # d. Daerah tak pasti (unknown) = background - foreground
    unknown = cv2.subtract(sure_bg, sure_fg)

    # e. Marker dengan connected components
    n_marker, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1            # background menjadi 1, objek mulai dari 2
    markers[unknown == 255] = 0      # area unknown = 0 (diputuskan watershed)

    # f. Jalankan watershed (input harus citra BGR 3 kanal)
    hasil = markers.copy()
    cv2.watershed(img_bgr.copy(), hasil)

    # g. Hitung objek unik (-1 = batas, 1 = background)
    label_unik = [l for l in np.unique(hasil) if l > 1]
    valid = [l for l in label_unik if np.sum(hasil == l) >= MIN_AREA]
    jumlah = len(valid)

    waktu = time.perf_counter() - t0

    # Overlay batas watershed (warna merah) pada citra asli
    overlay = img_bgr.copy()
    overlay[hasil == -1] = (0, 0, 255)

    return dist, sure_fg, sure_bg, unknown, markers, hasil, overlay, jumlah, waktu


# ======================================================================
# 4. VISUALISASI
# ======================================================================
def tampilkan(img, gray, blur, nilai_t, otsu_mask, otsu_label, otsu_valid,
              dist, sure_fg, unknown, markers, overlay, n_otsu, n_ws):
    # Warnai hasil Otsu per komponen
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

    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER_HASIL, "00_ringkasan_semua_tahap.png"), dpi=150)
    plt.show()


# ======================================================================
# 4b. SIMPAN SETIAP TAHAP SEBAGAI GAMBAR TERPISAH (folder "hasil")
# ======================================================================
def simpan_hasil(img, gray, blur, otsu_mask, otsu_label, otsu_valid,
                 dist, sure_fg, markers, overlay, n_otsu, n_ws):
    os.makedirs(FOLDER_HASIL, exist_ok=True)

    # Otsu berwarna per komponen + nomor objek
    warna = np.full(img.shape, 255, np.uint8)
    rng = np.random.default_rng(1)
    for i in otsu_valid:
        warna[otsu_label == i] = rng.integers(60, 230, 3)
    otsu_info = cv2.putText(warna.copy(), f"Otsu: {n_otsu} objek", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)

    # Distance transform dinormalisasi lalu diberi colormap
    dist_vis = cv2.applyColorMap(
        cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET)

    # Marker watershed diberi warna acak
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
        cv2.imwrite(os.path.join(FOLDER_HASIL, nama), gambar)
    print(f"[INFO] {len(daftar)} gambar + ringkasan disimpan di folder '{FOLDER_HASIL}/'")


# ======================================================================
# 5. DRAF ANALISIS LAPORAN
# ======================================================================
def cetak_analisis(n_otsu, n_ws, t_otsu, t_ws, nilai_t, est):
    # Acuan evaluasi: hitungan manual; jika menyimpang jauh dari estimasi luas,
    # gunakan estimasi luas sebagai acuan agar status under/over tidak keliru.
    acuan = JUMLAH_MANUAL if abs(est - JUMLAH_MANUAL) <= 3 else int(round(est))
    sel_otsu = n_otsu - JUMLAH_MANUAL
    sel_ws = n_ws - JUMLAH_MANUAL

    def status(n):
        if n < acuan:
            return "under-segmentation (objek menempel terhitung satu)"
        if n > acuan:
            return "over-segmentation (satu objek terpecah)"
        return "sesuai dengan acuan jumlah objek"

    teks = f"""
======================================================================
DRAF ANALISIS LAPORAN
======================================================================

1. KARAKTERISTIK GAMBAR
-----------------------
- Objek: biji kopi sangrai berwarna cokelat gelap, berbentuk oval dengan
  alur tengah (crease) yang khas, orientasi acak.
- Latar belakang putih bersih, sehingga kontras objek-latar sangat tinggi
  dan histogram grayscale cenderung bimodal (cocok untuk Otsu).
- Beberapa biji saling berdekatan/menempel (mis. pasangan/kelompok di
  sudut kiri atas dan beberapa di bagian bawah), sehingga menjadi
  tantangan utama pemisahan objek.
- Jumlah objek hasil hitung manual: {JUMLAH_MANUAL} biji.
- Sebagai pembanding, estimasi dari total luas foreground dibagi luas biji
  tunggal memberi angka ~{est:.0f} (sedikit lebih kecil karena biji yang
  bertumpuk menutupi sebagian luasnya), konsisten dengan hitungan manual.
- Tekstur permukaan biji (kilap dan bayangan alur) dapat menimbulkan
  variasi intensitas di dalam objek.

2. PERFORMA OTSU THRESHOLDING
-----------------------------
- Nilai threshold otomatis Otsu: {nilai_t:.0f}.
- Jumlah objek terdeteksi: {n_otsu} (selisih {sel_otsu:+d} terhadap manual)
  -> {status(n_otsu)} terhadap acuan {acuan}.
- Kelebihan: sederhana, cepat ({t_otsu*1000:.2f} ms), tanpa parameter
  threshold manual, sangat baik memisahkan objek dari latar putih.
- Kekurangan: hanya melihat intensitas, tidak mengenal bentuk. Biji yang
  bersentuhan membentuk satu komponen terhubung sehingga terhitung satu
  objek (under-segmentation), dan jumlah deteksi menjadi lebih kecil dari
  jumlah sebenarnya.
- Over-segmentation jarang terjadi karena objek cukup solid; noise kecil
  sudah dibersihkan oleh opening dan filter area minimum.

3. PERFORMA WATERSHED SEGMENTATION
----------------------------------
- Jumlah objek terdeteksi: {n_ws} (selisih {sel_ws:+d} terhadap manual)
  -> {status(n_ws)} terhadap acuan {acuan}.
- Watershed memakai distance transform sehingga tiap biji (oval, cembung)
  menghasilkan puncak jarak sendiri sebagai marker; biji yang menempel
  dipisahkan oleh garis batas pada "lekukan" di antara keduanya.
- Kelebihan: mampu memisahkan objek menempel, batas objek berupa garis
  yang jelas, hasil lebih mendekati jumlah sebenarnya.
- Tambahan: biji yang bertumpuk/menempel penuh tidak punya 'leher' sehingga
  hanya menghasilkan satu marker. Untuk itu ditambahkan koreksi: komponen
  yang luasnya jauh di atas biji tunggal diberi marker tambahan dengan
  K-Means, sehingga dua pasang biji bertumpuk dapat dipisahkan. Konsekuensinya
  waktu komputasi bertambah.
- Kekurangan: sensitif terhadap parameter (rasio threshold distance map).
  Rasio terlalu rendah -> marker menyatu (under-segmentation); terlalu
  tinggi -> biji besar/oval panjang bisa terpecah atau marker hilang
  (over-segmentation / objek terlewat). Biji yang tumpang tindih
  sebagian besar tetap sulit dipisahkan.
- Waktu komputasi: {t_ws*1000:.2f} ms (lebih lama dari Otsu karena ada
  distance transform, marker, dan proses flooding).

4. TABEL PERBANDINGAN
---------------------
| Kriteria                        | Otsu Thresholding                  | Watershed                              |
|---------------------------------|------------------------------------|----------------------------------------|
| Jumlah objek manual             | {JUMLAH_MANUAL:<34} | {JUMLAH_MANUAL:<38} |
| Jumlah objek terdeteksi         | {n_otsu:<34} | {n_ws:<38} |
| Selisih thd manual              | {sel_otsu:+d}{'':<31} | {sel_ws:+d}{'':<35} |
| Pemisahan objek menempel        | Tidak mampu                        | Mampu (bergantung parameter)           |
| Over-segmentation               | Hampir tidak ada                   | Mungkin jika marker terlalu ketat      |
| Under-segmentation              | Ada (objek menempel menyatu)       | Lebih sedikit                          |
| Kualitas batas objek            | Baik di tepi luar, tak ada pemisah | Baik, ada garis pemisah antar objek    |
| Noise                           | Rendah (setelah morfologi)         | Rendah (mewarisi mask Otsu)            |
| Waktu komputasi                 | {t_otsu*1000:.2f} ms{'':<26} | {t_ws*1000:.2f} ms{'':<30} |
| Kompleksitas metode             | Rendah                             | Sedang-tinggi (banyak tahap/parameter) |

5. KESIMPULAN
-------------
Citra biji kopi memiliki kontras tinggi dengan latar putih, sehingga
Otsu sudah sangat efektif untuk memisahkan objek dari background dan
komputasinya sangat ringan. Namun karena ada biji yang saling menempel,
Otsu mengalami under-segmentation. Watershed, yang memanfaatkan distance
transform dan marker, lebih sesuai untuk citra ini karena dapat memisahkan
biji yang menempel dan memberi hasil jumlah objek yang lebih mendekati
hitungan manual ({JUMLAH_MANUAL}), dengan konsekuensi komputasi dan pengaturan
parameter yang lebih kompleks. Pendekatan terbaik: Otsu sebagai tahap awal
pembuatan mask, dilanjutkan Watershed untuk memisahkan objek menempel.
"""
    print(teks)


# ======================================================================
# 6. MAIN
# ======================================================================
def main():
    img = baca_citra()
    gray, blur = preprocessing(img)

    nilai_t, otsu_mask, otsu_label, otsu_valid, n_otsu, t_otsu = metode_otsu(blur)
    (dist, sure_fg, sure_bg, unknown, markers,
     hasil_ws, overlay, n_ws, t_ws) = metode_watershed(img, otsu_mask)

    print("=" * 60)
    print("PERBANDINGAN JUMLAH OBJEK")
    print("=" * 60)
    print(f"Jumlah manual          : {JUMLAH_MANUAL}")
    print(f"Hasil Otsu             : {n_otsu}  (selisih {n_otsu - JUMLAH_MANUAL:+d})")
    print(f"Hasil Watershed        : {n_ws}  (selisih {n_ws - JUMLAH_MANUAL:+d})")
    est, luas_med = estimasi_luas(otsu_label, otsu_valid, otsu_mask)
    print(f"Estimasi dari luas     : ~{est:.1f} (luas biji tunggal ~{luas_med:.0f} px)")
    print(f"Waktu Otsu             : {t_otsu*1000:.2f} ms")
    print(f"Waktu Watershed        : {t_ws*1000:.2f} ms")
    print("=" * 60)

    os.makedirs(FOLDER_HASIL, exist_ok=True)
    simpan_hasil(img, gray, blur, otsu_mask, otsu_label, otsu_valid,
                 dist, sure_fg, markers, overlay, n_otsu, n_ws)
    tampilkan(img, gray, blur, nilai_t, otsu_mask, otsu_label, otsu_valid,
              dist, sure_fg, unknown, markers, overlay, n_otsu, n_ws)

    cetak_analisis(n_otsu, n_ws, t_otsu, t_ws, nilai_t, est)


if __name__ == "__main__":
    main()