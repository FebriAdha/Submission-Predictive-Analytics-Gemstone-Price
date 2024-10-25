# Laporan Proyek Machine Learning -Febri Isthifa Adha
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

### EDA - Multivariate Analysis

## Data Preparation

## Modeling

## Evaluation

## Referensi
[1] Grand View Research. (2022). Gemstone Market Size & Trends Analysis Report, 2022-2030. https://www.grandviewresearch.com/industry-analysis/gemstone-market-report

[2] International Gem Society (IGS). (2023). Gemstone Value and Pricing Factors. https://www.gemsociety.org/article/gemstone-value-factors/

[3] The Gemological Institute of America (GIA). (2023). Global Gemstone Market Analysis Report. https://www.gia.edu/gems-gemology/market-report-2023

[4] ResearchGate. (2021). Application of Machine Learning in Gemstone Price Prediction: A Review. https://www.researchgate.net/publication/machine-learning-gemstone-price-prediction

[5] Kementerian Perdagangan RI. (2023). Statistik Ekspor Perhiasan dan Batu Mulia Indonesia. https://statistik.kemendag.go.id/export-gems-jewelry

