![image](https://github.com/user-attachments/assets/2e056b95-b4de-4fbd-8bd4-e6916c9d44f7)# Laporan Proyek Machine Learning -Febri Isthifa Adha
## Domain Proyek

### Latar Belakang
![foto Gamstone](https://i.ibb.co.com/q7xrMGH/Gamstone.png)

Batu permata (gemstone) merupakan komoditas berharga yang terus mengalami pertumbuhan signifikan di pasar global. Menurut laporan Grand View Research, pasar global batu permata mencapai nilai USD 29.89 miliar pada tahun 2021 dan diproyeksikan tumbuh dengan CAGR 8.9% hingga 2030 [1]. Namun, penentuan harga batu permata masih menghadapi tantangan karena melibatkan berbagai faktor kompleks seperti berat (carat), kejernihan (clarity), warna (color), dan potongan (cut) [2].

The Gemological Institute of America (GIA) melaporkan bahwa sekitar 30% transaksi batu permata mengalami ketidaksesuaian harga akibat penilaian yang subjektif [3]. Hal ini menunjukkan kebutuhan akan sistem prediksi harga yang lebih akurat dan objektif. Perkembangan teknologi machine learning membuka peluang baru dalam memprediksi harga batu permata dengan lebih presisi, mempertimbangkan berbagai faktor penilaian secara simultan untuk menghasilkan estimasi yang lebih objektif [4].

Dalam konteks Indonesia, industri perhiasan dan batu mulia mencatat pertumbuhan yang signifikan dengan nilai ekspor mencapai USD 2.3 miliar pada tahun 2022 [5]. Pengembangan sistem prediksi harga yang akurat dapat mendukung pertumbuhan industri ini dengan memberikan transparansi harga yang lebih baik dan meningkatkan kepercayaan konsumen.

## Business Understandings

### Problem Statements
Berdasarkan latar belakang yang telah diuraikan, berikut adalah rumusan masalah yang akan diselesaikan dalam proyek ini:
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap harga gemstone?
- Berapa harga pasar gemstone dengan karakteristik atau fitur tertentu?

### Goals
Tujuan dari proyek ini adalah:
- Mengetahui fitur yang paling berkorelasi dengan harga gamstone.
- Membuat model machine learning yang dapat memprediksi harga gamstone seakurat mungkin berdasarkan fitur-fitur yang ada.

### Solution statements
Untuk mencapai goals yang telah ditetapkan, berikut adalah solusi yang akan diterapkan:
- Membangun model regresi dengan harga gamstone sebagai target.
- Membuat 3 model, lalu memilih 1 model terbaik yang memiliki tingkat akurasi terbaik berdasarkan metrik Mean Squared Error (MSE).

## Data Understanding
### Informasi Dataset
Dataset Gemstone yang digunakan dalam proyek ini berasal dari platform Kaggle dengan judul "Gemstone Price". Dataset ini dapat diakses melalui link berikut:
(https://www.kaggle.com/datasets/dhanrajcodes/gemstone-price)

### Variabel-variabel pada Dataset:
Variabel | Keterangan | Nilai
----------|----------|----------
id | Identifier unik untuk setiap batu permata |  1, 2, 3 
carat | Berat batu permata dalam satuan karat |  0.2 - 5.01
cut | Kualitas potongan batu permata | Fair, Good, Very Good, Premium, Ideal
color | Warna batu permata | D, E, F, G, H, I, J (D adalah yang terbaik)
clarity | Tingkat kejernihan batu permata | I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF (IF adalah yang terbaik)
depth | Total kedalaman dalam persentase | 43.0 - 79.0
table | Lebar bagian atas batu permata relatif terhadap titik terlebar | 43.0 - 95.0
x | Panjang batu permata dalam mm |  0.0 - 10.74
y | Lebar batu permata dalam mm | 0.0 - 58.9
price | Harga batu permata dalam (USD) | 326 - 18,823

### EDA - Univariate Analysis

![Univariate Fitur cut](https://github.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/blob/main/images/Fitur%20Cut.png)

Gambar 1a. Univariat Analysis (Fitur cut)

![Univariate Fitur color](https://github.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/blob/main/images/Fitur%20color.png)

Gambar 1b. Univariate Analysis (Fitur color)

![Univariate Fitur clarity](https://github.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/blob/main/images/Fitur%20clarity.png)

Gambar 1c. Univariate Analysis (Fitur clarity)

![Univariate Analysis (Data Numerik)](https://github.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/blob/main/images/Univariate%20Analysis%20(Data%20Numerik).png)

Gambar 1d. Univariat Analysis (Data Numerik) 

### EDA - Multivariate Analysis

![Multivariat Analysis](https://github.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/blob/main/images/Multivariate%20Analysis%20(Data%20Categori).jpeg)

Gambar 2a. Multivariate Analysis (Data Categorical)

![Multivariat Analysis](https://github.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/blob/main/images/Multivariate%20Analysis%20(numerik).png)

Gambar 2b. Multivariate Analysis (Data Numerical)

![Multivariat Analysis](https://github.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/blob/main/images/heatmap%20fitur%20numerik.png)

Gambar 2c. Analisis Matriks Korelasi

## Data Preparation

Data preparation merupakan salah satu tahapan yang penting dalam proses pengembangan model machine learning. Pada tahapan ini akan dilakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan.

Pada proyek ini tahap Data Preparation yang dilakukan diantaranya sebagai berikut :

**A. Encoding Fitur Kategori.**

Proses encoding fitur kategori dilakukan dengan menggunakan teknik one-hot-encoding dari library scikit-learn. Teknik ini berfungsi untuk mendapatkan fitur baru yang sesuai sehingga dapat mewakili setiap fitur kategori. Pada proyek ini terdapat 3 fitur kategori, yaitu "cut", "color", "clarity". Proses encoding dilakukan dengan fitur get_dummies. Berikut output nya :

**B. Reduksi Dimensi dengan PCA.**

**C. Train Test Split**

Teknik ini dilakukan untuk membagi dataset menjadi dua bagian, yaitu data latih dan data uji. data latih akan digunakan untuk melatih model sedangkan data uji akan digunakan untuk evaluasi model. Hal tersebut perlu diterapkan agar model yang telah dilatih dapat diuji menggunakan data yang belum pernah dianalisa oleh model. Langkah-langkah yang dilakukan dalam menerapkan teknik ini adalah sebagai berikut.

- Membagi dataset terlebih dahulu menjadi data y sebagai data target dan data X sebagai data fitur.
- Membagi y dan X menjadi data latih dan data uji dengan rasio 90 : 10. Rasio tersebut dilakukan mengingat jumlah dataset yang besar setelah data cleaning yaitu sekitar 90.000. Pembagian dataset dilakukan dengan memanfaatkan library train_test_split.
- Terakhir, mengecek masing-masing ukuran keseluruhan dataset, X, dan y untuk memastikan pembagian dataset berhasil diterapkan. Dari 193563 baris keseluruhan dataset setelah melalui tahap data cleaning, terdapat 174206 baris merupakan data latih dan 19357 baris merupakan data uji.

![]

Dapat dilihat bahwa setelah proses standarisasi sekarang nilai mean = 0 dan standar deviasi = 1.

## Modeling

Algoritma pada proyek ini melakukan pemodelan dengan 3 algoritma, yaitu:

**1. K-Nearest Neighbors (KNN)**

K-Nearest Neighbors (KNN) adalah algoritma machine learning yang sederhana dan mudah dipahami untuk klasifikasi dan regresi. Algoritma ini bekerja dengan menemukan k tetangga terdekat dari data baru dan kemudian menggunakan kategori atau nilai rata-rata dari tetangga tersebut untuk memprediksi kategori atau nilai data baru. Adapun parameter yang digunakan pada proyek ini adalah:
- `n_neighbors` jumlah tetangga terdekat.

Keunggulan KNN :
- Dapat digunakan untuk klasifikasi dan regresi.
- Sederhana dan mudah dipahami.

Kerugian KNN :
- Sensitif terhadap outlier. 
- Membutuhkan banyak memori dan waktu komputasi untuk dataset besar. 
- Sulit untuk memilih nilai K yang optimal.

**2. Random Forest**

Random Forest adalah algoritma machine learning ensemble yang menggabungkan beberapa decision tree untuk meningkatkan akurasi prediksi. Algoritma ini bekerja dengan membuat banyak decision tree secara acak dan kemudian menggunakan voting untuk memprediksi kategori atau nilai data baru. Adapun parameter yang digunakan pada proyek ini adalah:
- `n_estimators` jumlah tetangga terdekat.
- `max_depth` kedalaman maksimum.
- `max_depth` Kedalaman maksimum pohon keputusan individual.
- `n_jobs` mempercepat pelatihan pada sistem dengan beberapa core CPU.

Keunggulan Random Forest :
- Memiliki akurasi prediksi yang tinggi.
- Mampu menangani dataset dengan dimensi tinggi.
- Tidak sensitif terhadap outlier.

Kerugian Random Forest :
- Cenderung overfit pada dataset kecil. 
- Membutuhkan banyak waktu komputasi untuk pelatihan. 
- Sulit untuk diinterpretasikan.

**3. Boosting Algorithm**


## Evaluation

## Referensi
[1] Grand View Research. (2022). Gemstone Market Size & Trends Analysis Report, 2022-2030. https://www.grandviewresearch.com/industry-analysis/gemstone-market-report

[2] International Gem Society (IGS). (2023). Gemstone Value and Pricing Factors. https://www.gemsociety.org/article/gemstone-value-factors/

[3] The Gemological Institute of America (GIA). (2023). Global Gemstone Market Analysis Report. https://www.gia.edu/gems-gemology/market-report-2023

[4] ResearchGate. (2021). Application of Machine Learning in Gemstone Price Prediction: A Review. https://www.researchgate.net/publication/machine-learning-gemstone-price-prediction

[5] Kementerian Perdagangan RI. (2023). Statistik Ekspor Perhiasan dan Batu Mulia Indonesia. https://statistik.kemendag.go.id/export-gems-jewelry

