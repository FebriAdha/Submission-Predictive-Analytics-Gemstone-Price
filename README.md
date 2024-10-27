# Laporan Proyek Machine Learning - Febri Isthifa Adha
## Domain Proyek

### Latar Belakang
![foto Gamstone](https://github.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/blob/main/images/Gemstone.png)

Batu permata (gemstone) merupakan komoditas berharga yang terus mengalami pertumbuhan signifikan di pasar global. Menurut laporan Grand View Research, pasar global batu permata mencapai nilai USD 29.89 miliar pada tahun 2021 dan diproyeksikan tumbuh dengan CAGR 8.9% hingga 2030 [[1]](https://www.grandviewresearch.com/industry-analysis/gemstone-market-report). Namun, penentuan harga batu permata masih menghadapi tantangan karena melibatkan berbagai faktor kompleks seperti berat (carat), kejernihan (clarity), warna (color), dan potongan (cut) [[2]](https://www.gemsociety.org/article/gemstone-value-factors/).

The Gemological Institute of America (GIA) melaporkan bahwa sekitar 30% transaksi batu permata mengalami ketidaksesuaian harga akibat penilaian yang subjektif [[3]](https://www.gia.edu/gems-gemology/market-report-2023). Hal ini menunjukkan kebutuhan akan sistem prediksi harga yang lebih akurat dan objektif. Perkembangan teknologi machine learning membuka peluang baru dalam memprediksi harga batu permata dengan lebih presisi, mempertimbangkan berbagai faktor penilaian secara simultan untuk menghasilkan estimasi yang lebih objektif [[4]](https://www.researchgate.net/publication/machine-learning-gemstone-price-prediction).

Dalam konteks Indonesia, industri perhiasan dan batu mulia mencatat pertumbuhan yang signifikan dengan nilai ekspor mencapai USD 2.3 miliar pada tahun 2022 [[5]](https://statistik.kemendag.go.id/export-gems-jewelry). Pengembangan sistem prediksi harga yang akurat dapat mendukung pertumbuhan industri ini dengan memberikan transparansi harga yang lebih baik dan meningkatkan kepercayaan konsumen.

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

<p align='center'><img src ="https://github.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/blob/main/images/Fitur%20cut.png"  width="500"></p>
<p align='center'>Gambar 1a. Univariat Analysis (Fitur cut)</p> 

Pada Gambar 1a terdapat 5 kategori pada fitur Cut, secara berurutan dari jumlahnya yang paling banyak yaitu: Ideal, Premium, Very Good, Good, dan Fair. Dari data persentase dapat kita simpulkan bahwa lebih dari 70% sampel merupakan diamonds tipe grade tinggi, yaitu grade Ideal dan Premium.

<p align='center'><img src ="https://github.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/blob/main/images/Fitur%20color.png"  width="500"></p>
<p align='center'>Gambar 1b. Univariate Analysis (Fitur color)</p>

Pada Gambar 1b terdapat urutan kategori warna dari yang paling buruk hingga yang paling bagus adalah J, I, H, G, F, E, dan D. Dari grafik di atas, dapat disimpulkan bahwa sebagian besar grade berada pada grade menengah, yaitu G, F, H.

<p align='center'><img src ="https://github.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/blob/main/images/Fitur%20clarity.png"  width="500"></p>
<p align='center'>Gambar 1c. Univariate Analysis (Fitur clarity)</p>

Pada Gambar 1c fitur Clarity terdiri dari 8 kategori dari yang paling buruk ke yang paling baik, yaitu: I1, SI2, SI1, VS2, VS1, VVS2, VVS1, dan IF.

1. `IF` - Internally Flawless 
2. `VVS2` - Very Very Slight Inclusions 
3. `VVS1` - Very Very Slight Inclusions 
4. `VS1` - Very Slight Inclusions
5. `VS2` - Very Slight Inclusions
6. `SI2` - Slight Inclusions
7. `SI1` - Slight Inclusions
8. `I1` - Imperfect

Dari grafik dapat disimpulkan bahwa sebagian besar fitur merupakan grade rendah, yaitu SI1, SI2, dan VS2.

<p align='center'><img src ="https://github.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/blob/main/images/Univariate%20Analysis%20(Data%20Numerik).png"  width="800"></p>
<p align='center'>Gambar 1d. Univariat Analysis (Data Numerik) </p>

Berdasarkan Gambar 1d, dapat diamati histogram untuk variabel "price" yang merupakan fitur target (label). Dari histogram "price", kita bisa memperoleh beberapa informasi, antara lain:

- Sebagian besar gemstone memiliki harga di bawah $4000, dengan puncak frekuensi pada kisaran harga yang lebih rendah.
- Rentang harga gemstone cukup luas, mulai dari ratusan dolar Amerika hingga sekitar $12000.
- Distribusi harga berlian miring ke kanan (right-skewed), menunjukkan bahwa meskipun sebagian besar berlian dijual dengan harga lebih rendah, terdapat sejumlah berlian dengan harga yang jauh lebih tinggi.
- Lebih dari setengah gemstone memiliki harga di bawah $2500, menunjukkan adanya kecenderungan harga yang lebih terjangkau pada sebagian besar data.

### EDA - Multivariate Analysis

<p align='center'><img src ="https://github.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/blob/main/images/Multivariate%20Analysis%20(Data%20Categori).jpeg"  width="800"></p>
<p align='center'>Gambar 2a. Multivariate Analysis (Data Categorical)</p>

Dengan mengamati Gambar 2a, memiliki rata-rata harga relatif terhadap data kategori, kita memperoleh _insight_ sebagai berikut:

- Pada fitur 'cut', rata-rata harga berlian berada dalam rentang yang mirip, yaitu antara 2500 hingga 4000. Grade tertinggi seperti "Ideal" justru memiliki harga rata-rata lebih rendah dibandingkan dengan grade lainnya seperti "Fair." Hal ini menunjukkan bahwa fitur "cut" memiliki pengaruh yang kecil terhadap variasi harga gemstone.
- Pada fitur 'color', terdapat kecenderungan bahwa harga rata-rata berlian lebih tinggi pada grade warna yang lebih rendah, seperti "I" dan "J," sementara grade warna yang lebih tinggi seperti "E" memiliki harga yang lebih rendah. Ini menunjukkan bahwa pengaruh warna terhadap harga berlian juga relatif rendah.
- Pada fitur 'clarity', gemstone dengan grade clarity lebih rendah, seperti "SI2" dan "I1," cenderung memiliki harga yang lebih tinggi dibandingkan grade clarity yang lebih tinggi seperti "IF." Ini mengindikasikan bahwa kejernihan berlian tidak selalu berkorelasi positif dengan harga, sehingga fitur "clarity" memiliki pengaruh yang rendah terhadap harga gemstone.
- Kesimpulan akhir, fitur kategori seperti "cut," "color," dan "clarity" memiliki pengaruh yang rendah terhadap harga gemstone.

<p align='center'><img src ="https://github.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/blob/main/images/Multivariate%20Analysis%20(Data%20Numerik).png"  width="800"></p>
<p align='center'>Gambar 2b. Multivariate Analysis (Data Numerical)</p>

<p align='center'><img src ="https://github.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/blob/main/images/Analysis%20Matrik%20Korelasi.png"  width="800"></p>
<p align='center'>Gambar 2c. Analisis Matriks Korelasi</p>

Berdasarkan Gambar 2c, fitur 'carat', 'x', 'y', dan 'z' memiliki skor korelasi yang besar (diatas 0.9) dengan fitur target 'price'. Dimana, fitur 'price' berkolerasi tinggi dengan keempat fitur tersebut. Sementara fitur 'depth' memiliki korelasi yang sangat kecil (0.05). Sehingga fitur 'depth' dapat dihapus. 

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
- Terakhir, mengecek masing-masing ukuran keseluruhan dataset, X, dan y untuk memastikan pembagian dataset berhasil diterapkan. Dari 168755 baris keseluruhan dataset setelah melalui tahap data cleaning, terdapat 151879 baris merupakan data latih dan 16876 baris merupakan data uji.

![Standarisasi](https://github.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/blob/main/images/Standarisasi.png)

Dapat dilihat bahwa setelah proses standarisasi sekarang nilai mean = 0 dan standar deviasi = 1.

## Modeling

Algoritma pada proyek ini melakukan pemodelan dengan 3 algoritma, yaitu:

**1. K-Nearest Neighbors (KNN)**

K-Nearest Neighbors (KNN) adalah algoritma machine learning yang sederhana dan mudah dipahami untuk klasifikasi dan regresi. Algoritma ini bekerja dengan menemukan k tetangga terdekat dari data baru dan kemudian menggunakan kategori atau nilai rata-rata dari tetangga tersebut untuk memprediksi kategori atau nilai data baru. Adapun parameter yang digunakan pada proyek ini adalah:
- `n_neighbors` jumlah tetangga terdekat.

Keunggulan KNN :
- Dapat digunakan untuk klasifikasi dan regresi.
- Sederhana dan mudah dipahami.

Kekurangan KNN :
- Sensitif terhadap outlier. 
- Membutuhkan banyak memori dan waktu komputasi untuk dataset besar. 
- Sulit untuk memilih nilai K yang optimal.

**2. Random Forest**

Random Forest adalah algoritma machine learning ensemble yang menggabungkan beberapa decision tree untuk meningkatkan akurasi prediksi. Algoritma ini bekerja dengan membuat banyak decision tree secara acak dan kemudian menggunakan voting untuk memprediksi kategori atau nilai data baru. Adapun parameter yang digunakan pada proyek ini adalah:
- `n_estimators` jumlah tetangga terdekat.
- `max_depth` Kedalaman maksimum pohon keputusan individual.
- `n_jobs` mempercepat pelatihan pada sistem dengan beberapa core CPU.

Keunggulan Random Forest :
- Memiliki akurasi prediksi yang tinggi.
- Mampu menangani dataset dengan dimensi tinggi.
- Tidak sensitif terhadap outlier.

Kekurangan Random Forest :
- Cenderung overfit pada dataset kecil. 
- Membutuhkan banyak waktu komputasi untuk pelatihan. 
- Sulit untuk diinterpretasikan.

**3. Gradient Boosting**

Gradient Boosting adalah algoritma machine learning yang menggunakan teknik ensembel learning dari decision tree untuk memprediksi nilai. Gradient Boosting sangat mampu menangani pattern yang kompleks dan data ketika linear model tidak dapat menangani. Untuk parameter yang digunakan pada model ini ada 3 yaitu :
- `learning_rate`  menghitung nilai koreksi bobot pada waktu proses training. 
- `n_estimators` jumlah tetangga terdekat.
- `max_depth` Kedalaman maksimum pohon keputusan individual.
- `n_jobs` mempercepat pelatihan pada sistem dengan beberapa core CPU.

Kelebihan Gradient Boosting :
- Hasil pemodelan yang lebih akurat
- Model yang stabil dan lebih kuat (robust)
- Dapat digunakan untuk menangkap hubungan linear maupun non linear pada data

Kekurangan Gradient Boosting :
- Pengurangan kemampuan interpretasi model
- Waktu komputasi dan desain tinggi
- Tingkat kesulitan yang tinggi dalam pemilihan model

## Evaluation

Proses evaluasi model pada proyek ini menggunakan metrik Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi [[6]](https://www.dicoding.com/academies/319/tutorials/18595). MSE dipilih karena memberikan penalti yang lebih besar untuk kesalahan prediksi yang besar, sehingga membantu dalam mengidentifikasi model yang mampu memberikan prediksi lebih akurat.

![image](https://github.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/blob/main/images/Rumus%20MSE.jpeg)

_Keterangan:  
N = jumlah dataset  
yi = nilai sebenarnya  
y_pred = nilai prediksi_

**Cara kerja :**

Cara kerja Metrik MSE adalah dengan menghitung selisih hasil prediksi dengan nilai fitur target (PE). Nilai selisih tersebut, disebut juga sebagai nilai eror yang kemudian di kuadratkan untuk menangani nilai selisih negatif. Selanjutnya hasil pengkuadratan setiap nilai selisih dijumlahkan dan terakhir dibagi dengan banyak data point (n) untuk memperoleh nilai rata-ratanya. Rata-rata inilah yang disebut Mean Squared Error (MSE).

Berikut adalah tabel nilai MSE pada setiap model dengan data latih dan data uji :



Untuk memudahkan dalam mengevaluasi model kita akan melakukan visualisasi hasil menggunakan bar chart sebagai berikut.

![image]()

## Referensi
[1] Grand View Research. (2022). Gemstone Market Size & Trends Analysis Report, 2022-2030. https://www.grandviewresearch.com/industry-analysis/gemstone-market-report

[2] International Gem Society (IGS). (2023). Gemstone Value and Pricing Factors. https://www.gemsociety.org/article/gemstone-value-factors/

[3] The Gemological Institute of America (GIA). (2023). Global Gemstone Market Analysis Report. https://www.gia.edu/gems-gemology/market-report-2023

[4] ResearchGate. (2021). Application of Machine Learning in Gemstone Price Prediction: A Review. https://www.researchgate.net/publication/machine-learning-gemstone-price-prediction

[5] Kementerian Perdagangan RI. (2023). Statistik Ekspor Perhiasan dan Batu Mulia Indonesia. https://statistik.kemendag.go.id/export-gems-jewelry

[6] https://www.dicoding.com/academies/319/tutorials/18595
