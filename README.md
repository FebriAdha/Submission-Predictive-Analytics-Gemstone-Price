# Laporan Proyek Machine Learning - Febri Isthifa Adha

![foto Gamstone](https://raw.githubusercontent.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/refs/heads/main/images/Gemstone.png)

## Domain Proyek

Domain yang dipilih untuk proyek machine learning ini adalah **Ekonomi dan Bisnis**, dengan judul _"Predictive Analytics Gemstone Price"._

### Latar Belakang

Batu permata adalah sebuah mineral yang mengendap selama bertahun-tahun yang membuatnya mempunyai nilai harga yang tinggi. Ada beberapa jenis mineral/batuan yang termasuk golongan batu permata (gemstone) contohnya: berlian, rubi, sapphire, giok (jade), dan emerlad (zamrud). Jenis batu mulia dapat dikenali berdasarkan tekstur, motif, dan warnanya [[1]]( https://e-journal.stmik-tegal.ac.id/index.php/batirsi/article/view/41/31). Menurut laporan Data Bridge, ukuran pasar batu permata global dinilai sebesar USD 32,38 miliar pada tahun 2023 dan diproyeksikan mencapai USD 47,48 miliar pada tahun 2031, dengan CAGR sebesar 4,90% selama periode perkiraan tahun 2024 hingga 2031 [[2]](https://www.databridgemarketresearch.com/reports/global-gemstones-market).

Penentuan harga batu permata masih menghadapi tantangan karena melibatkan berbagai faktor kompleks seperti berat (carat), kejernihan (clarity), warna (color), dan potongan (cut) [[3]](https://www.gemsociety.org/article/gemstone-value-factors/). Namun yang ditakutkan dari permasalahan ini banyak pengemar yang membeli batu mulia dengan harga yang overprice dan bila mana ingin dijual kembali harganya terlalu lowerprice. Menyebabkan kerugian bagi pengemar batu maupun penjual batu permata [[1]]( https://e-journal.stmik-tegal.ac.id/index.php/batirsi/article/view/41/31).

Berdasarkan permasalahan di atas, maka pada proyek ini akan dibangun suatu model machine learning untuk memprediksi harga pasar Gemstone di masa depan. Dengan adanya model machine learning ini, dapat mendukung pertumbuhan industri ini dengan memberikan transparansi harga yang lebih baik dan meningkatkan kepercayaan konsumen.

## Business Understandings

### Problem Statements

Berdasarkan latar belakang yang telah diuraikan, berikut adalah rumusan masalah yang akan diselesaikan dalam proyek ini:
- Fitur apa saja yang paling berpengaruh terhadap harga batu permata (gemstone)?
- Bagaimana model dapat memprediksi harga pasar batu permata berdasarkan fitur-fitur yang ada?

### Goals

Tujuan dari proyek ini adalah:
- Mengidentifikasi fitur-fitur yang paling berkorelasi dengan harga batu permata.
- Membangun model machine learning yang dapat memprediksi harga batu permata secara akurat berdasarkan fitur-fitur yang ada.

### Solution statements

Untuk mencapai goals yang telah ditetapkan, berikut adalah solusi yang akan diterapkan:
1. Menggunakan analisis regresi untuk mengetahui fitur-fitur yang paling berpengaruh terhadap harga batu permata.
2. Membuat model _machine learning_ untuk mendapatkan model yang paling baik dari 3 algoritma yang berbeda dan kemudian akan dilakukan evaluasi model untuk membandingkan performa model yang terbaik. Algoritma yang akan digunakan, yaitu Algoritma K-Nearest Neighbor, Algoritma Random Forest, dan Boosting Algorithm.
    - **Algoritma K-Nearest Neighbor**

      Algoritma K-Nearest Neighbor (KNN) adalah algoritma sederhana yang mengklasifikasikan data atau kasus baru berdasarkan ukuran kesamaan. Hal ini sebagian besar digunakan untuk mengklasifikasikan titik data berdasarkan tetangga terdekatnya sebagai acuan [[4]](https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e).

    - **Algoritma Random Forest**

      Algoritma Random Forest adalah algoritma machine learning yang kuat yang dapat digunakan untuk berbagai tugas termasuk regresi dan klasifikasi. Ini adalah metode ensemble, yang berarti bahwa model random forest terdiri dari banyak decision tree kecil, yang disebut estimator, yang masing-masing menghasilkan prediksi mereka sendiri. Random forest menggabungkan prediksi estimator untuk menghasilkan prediksi yang lebih akurat [[5]](https://deepai.org/machine-learning-glossary-and-terms/random-forest). 
      
    - **Algoritma Gradient Boosting**

      Algoritma Gradient Boosting adalah sebuah teknik yang menggabungkan beberapa model yang lemah (weak model) menjadi sebuah model yang kuat. Model-model lemah ini sering disebut dengan weak learners, dan dapat berupa model regresi atau klasifikasi sederhana seperti Decision Tree. Algoritma ini menggunakan pendekatan iteratif, di mana setiap iterasi bertujuan untuk meningkatkan model sebelumnya dengan menambahkan model baru [[6]](https://www.trivusi.web.id/2023/03/algoritma-gradient-boosting.html).

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah dataset [Gemstone Price](https://www.kaggle.com/datasets/dhanrajcodes/gemstone-price) yang diambil dari platfrom Kaggle. File yang digunakan berupa file csv, yaitu `gemstone.csv` dengan ukuran 10.46 MB. Dataset tersebut terdiri dari 193573 baris dan 11 columns dengan berbagai karakteristik dan harga. Karaktristik berupa fitur non-numerik seperti _cut, color, dan clarity,_ serta fitur numerik seperti _carat, x, y, z, table, dan depth._ Sedangan harga (price) merupakan fitur target. 

Dari dataset tersebut, dilakukan penghapusan kolom pertama yaitu id yang berisikan nomor masing-masing data.

Kemudian dilakukan proses Exploratory Data Analysis (EDA) yang merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data.

1. **Deskripsi Variabel**

   Pada deskripsi variabel dilakukan pengecekan informasi variabel dari dataset yaitu jumlah kolom, nama kolom, jumlah data per kolom dan tipe datanya.

   Berikut adalah informasi variabel dari dataset Gemstone Price:

   Tabel 1. Deskripsi Variabel
   | # | Column  | Non-Null Count  | Dtype   |
   |---|---------|-----------------|---------|  
   | 0 | carat   | 193573 non-null | float64 |
   | 1 | cut     | 193573 non-null | object  |
   | 2 | color   | 193573 non-null | object  |
   | 3 | clarity | 193573 non-null | object  | 
   | 4 | depth   | 193573 non-null | float64 |
   | 5 | table   | 193573 non-null | float64 |
   | 6 | x       | 193573 non-null | float64 |
   | 7 | y       | 193573 non-null | float64 |
   | 8 | z       | 193573 non-null | float64 |
   | 9 | price   | 193573 non-null | int64   |

   Dari hasil di atas, terlihat bahwa kolom `cut`, `color`, `clarity` bertipe object, kolom `carat`, `depth`, `table`, `x`, `y`, `z` bertipe float64, dan kolom `price` bertipe int64.

   Berikut merupakan arti dari masing-masing variabel beserta nilai-nilainya.

   Tabel 2. Variabel beserta nilainya
   | Variabel | Keterangan                                                     | Nilai                                            |
   |----------|----------------------------------------------------------------|--------------------------------------------------|
   | carat    | Berat batu permata dalam satuan karat                          |  0.2 - 5.01                                      |
   | cut      | Kualitas potongan batu permata                                 | Fair, Good, Very Good, Premium, Ideal            |
   | color    | Warna batu permata                                             | D, E, F, G, H, I, J (D best)                     |
   | clarity  | Tingkat kejernihan batu permata                                | I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF (IF best) |
   | depth    | Total kedalaman dalam persentase                               | 43.0 - 79.0                                      |
   | table    | Lebar bagian atas batu permata relatif terhadap titik terlebar | 43.0 - 95.0                                      |
   | x        | Panjang batu permata dalam mm                                  |  0.0 - 10.74                                     |
   | y        | Lebar batu permata dalam mm                                    | 0.0 - 58.9                                       |
   | z	      | Kedalaman berlian dalam mm	                                   | 0-31.8                                           |
   | price    | Harga batu permata dalam (USD)                                 | 326 - 18,823                                     |

3. **Deskripsi Statistik Data**
   
   Selanjutnya, kita akan melihat deskripsi statistik dari data yang dimiliki.

   Tabel 3. Deskripsi Statistik Data
   |       | carat         | depth         | table         | x             | y             | z             | price         |
   |-------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
   | count | 193573.000000 | 193573.000000 | 193573.000000 | 193573.000000 | 193573.000000 | 193573.000000 | 193573.000000 |
   | mean  | 0.790688	   | 61.820574	   | 57.227675	   | 5.715312	   | 5.720094	   | 3.534246	   | 3969.155414   |
   | std   | 0.462688	   | 1.081704	   | 1.918844	   | 1.109422	   | 1.102333	   | 0.688922	   | 4034.374138   |
   | min   | 0.200000	   | 52.100000	   | 49.000000	   | 0.000000	   | 0.000000	   | 0.000000	   | 326.000000    |
   | 25%   | 0.400000	   | 61.300000	   | 56.000000	   | 4.700000	   | 4.710000	   | 2.900000	   | 951.000000    |
   | 50%   | 0.700000	   | 61.900000	   | 57.000000	   | 5.700000	   | 5.720000	   | 3.530000	   | 2401.000000   |
   | 75%   | 1.030000	   | 62.400000	   | 58.000000	   | 6.510000	   | 6.510000	   | 4.030000	   | 5408.000000   |
   | max   | 3.500000	   | 71.600000	   | 79.000000	   | 9.650000	   | 10.010000	   | 31.300000	   | 18818.000000  |

   Fungsi `describe()` memberikan informasi statistik pada masing-masing kolom, antara lain:
   - `Count` adalah jumlah sampel pada data.
   - `Mean` adalah nilai rata-rata.
   - `Std` adalah standar deviasi.
   - `Min` yaitu nilai minimum setiap kolom.
   - `25%` adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
   - `50%` adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
   - `75%` adalah kuartil ketiga.
   - `Max` adalah nilai maksimum.

4. **Analisis Missing Values**

   Berdasarkan hasil Fungsi `describe()`, pada nilai minimum untuk kolom x, y, dan z adalah 0. Diketahui bahwa x, y, dan z adalah ukuran panjang, lebar, dan kedalaman gemstone sehingga tidak mungkin ada gemstone dengan dimensi x, y, atau z itu bernilai 0. Oleh karena itu, perlu dilakukan pemeriksaan dan penghitungan jumlah nilai 0 pada kolom x, y, dan z sebagai langkah awal untuk mengetahui jumlah nilai yang dianggap sebagai missing values. Setelah mengetahui jumlahnya, langkah-langkah penanganan yang sesuai akan diterapkan di bagian **Menangani Missing Values**.

5. **Analisis Outliers**

   Outliers merupakan nilai-nilai yang jauh berada di luar rentang normal dari distribusi data dimensi atau harga gemstone. Keberadaan outliers dapat mengindikasikan data yang tidak normal, entri yang salah, atau pengamatan ekstrim yang mungkin berpengaruh terhadap analisis. Selanjutnya, untuk mendeteksi _outlier_ dengan menggunakan teknik visualisasi dengan data (boxplot). Setelah outliers terdeteksi, penanganannya akan dilakukan menggunakan metode IQR (Interquartile Range).

   IQR atau Interquartile Range adalah rentang antara kuartil pertama (Q1) dan kuartil ketiga (Q3) dalam suatu distribusi data. Untuk memahami konsep IQR, perlu diingat kembali mengenai konsep kuartil. Kuartil membagi data menjadi empat bagian yang sama besar, di mana:

   - **Kuartil pertama (Q1)** adalah nilai di mana 25% data berada di bawahnya,
   - **Kuartil kedua (Q2)** adalah median atau titik tengah di mana 50% data berada di bawahnya,
   - **Kuartil ketiga (Q3)** adalah nilai di mana 75% data berada di bawahnya.

   Dengan demikian, **Interquartile Range (IQR)** adalah selisih antara Q3 dan Q1 atau IQR = Q3 - Q1.

6. **Data Duplikat**

   Data duplikat adalah data yang muncul lebih dari satu kali dalam dataset dan dapat menyebabkan hasil analisis menjadi bias atau tidak akurat. Dalam kasus data gemstone, duplikat bisa terjadi ketika informasi mengenai gemstone tertentu tercatat lebih dari satu kali, baik secara tidak sengaja maupun karena proses penggabungan data yang tidak sempurna.

   Pentingnya mengidentifikasi dan menangani data duplikat, antara lain:

   - **Akurasi Analisis:** Data duplikat dapat mengganggu hasil statistik dan membuat perhitungan seperti rata-rata atau distribusi data menjadi tidak akurat.
   - **Efisiensi Pengolahan Data:** Data yang bersih dan bebas dari duplikat membuat proses analisis lebih cepat dan menghemat sumber daya komputasi.
   - **Reliabilitas Model:** Dalam konteks pembelajaran mesin, data duplikat dapat menyebabkan model belajar pola yang salah atau bias, sehingga berdampak negatif pada performa model.
  
7. **Univariate Analysis**

   Analisis univariat adalah metode dalam analisis data yang fokus pada pengamatan dan penjelasan satu variabel data saja. Kata "univariate" berasal dari "uni" yang berarti satu dan "variat" yang berarti variabel. Dalam praktiknya, analisis ini tidak membahas tentang penyebab atau hubungan antar variabel, namun lebih kepada mendeskripsikan dan menemukan pola dalam satu variabel tersebut.[[7]](https://revou.co/kosakata/analisis-univariat)

8. **Multivariate Analysis**

   Analisis multivariat adalah teknik mengumpulkan beberapa kelompok data dan menganalisis hubungan antara lebih dari dua variabel yang terkait dengan data tersebut. Analisis multivariat digunakan ketika berhadapan dengan data yang memiliki setidaknya tiga variabel yang berbeda. Bisa meliputi 2 variabel independen dan 1 variabel dependen, atau sebaliknya. Variabel dependen biasa disebut dengan variabel Y, variabel yang dipengaruhi, variabel respons, atau variabel terikat. Sedangkan variabel independen disebut juga dengan variabel X, variabel bebas, variabel prediktor, atau variabel yang memengaruhi.[[8]](https://revou.co/kosakata/analisis-multivariat)
  
## Data Preparation

Pada tahap persiapan data atau *data preparation* dilakukan beberapa proses, yaitu Menangani Missing Values, Menangani Outliers, Univariate Analysis, Multivariate Analysis, Correlation Matrix, *encoding* pada fitur kategori, reduksi dimensi dengan menggunakan Principal Component Analysis (PCA), Train Test Split, dan proses standarisasi data.

1. **Menangani Missing Values**

   Dilakukan pengecekan nilai yang hilang atau missing valie pada kolom x, y, dan z yang bernilai 0. Terdapat missing value pada kolom x sebanyak 3, y sebanyak 2, dan z sebanyak 10. 

   Tabel 4. Missing Value Kolom x, y, z
   |        | carat | cut       | color | clarity |	depth | table |	x    | y    | z   |	price |
   |--------|-------|-----------|-------|---------|-------|-------|------|------|-----|-------|
   | 8750   | 1.02	| Premium   | H	    | SI2	  | 59.4  | 61.0  | 6.57 | 6.53 | 0.0 |	4144  |
   | 39413  | 2.18	| Premium	| H	    | SI2	  | 59.4  | 60.0  | 8.46 | 8.41 | 0.0 | 15842 |
   | 92703  | 0.71	| Good	    | F	    | SI1	  | 64.1  | 60.0  | 0.00 | 0.00 | 0.0 | 2130  |
   | 98719  | 2.17	| Premium	| H	    | SI2	  | 60.3  | 57.0  | 8.42 | 8.36	| 0.0 | 15923 |
   | 99624  | 2.20	| Premium	| I	    | SI2	  | 60.1  | 60.0  | 8.45 | 8.41	| 0.0 | 11221 |
   | 117161 | 2.20	| Premium	| F	    | SI2	  | 60.3  | 58.0  | 8.49 | 8.45	| 0.0 | 15188 |
   | 151690 | 2.18	| Premium	| I	    | VS2	  | 61.2  | 62.0  | 8.45 | 8.37	| 0.0 | 15701 |
   | 159429 | 2.18	| Premium	| H	    | SI2	  | 60.8  | 59.0  | 8.42 | 8.38	| 0.0 | 13938 |
   | 170318 | 0.71	| Good	    | D	    | VS2	  | 64.1  | 60.0  | 0.00 | 0.00	| 0.0 | 910   |
   | 178000 | 0.71	| Very Good | F	    | SI2	  | 62.0  | 60.0  | 0.00 | 6.71	| 0.0 | 2130  |

   Terlihat bahwa pada untuk z bernilai 0, ternyata juga terdapat seluruh nilai 0 pada kolom x dan y. Oleh karena itu, baris-baris ini akan dihapus. Data setelah dihapus menjadi `193563` yang sebelumnya `193573`.

2. **Menangani Outliers**

   Outliers merupakan sampel yang nilainya sangat jauh dari cakupan umum data utama, dengan itu kita akan memeriksa apakah terdapat outlier pada kolom-kolom numerik.

   1. Fitur Carat
   
   ![image carat](https://raw.githubusercontent.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/refs/heads/main/images/Outlier%20carat.png)

   2. Fitur Table

   ![image table](https://raw.githubusercontent.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/refs/heads/main/images/Outlier%20table.png)

   3. Fitur X

   ![image x](https://raw.githubusercontent.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/refs/heads/main/images/Outlier%20x.png)

   Terlihat bahwa terdapat beberapa outlier pada kolom-kolom di atas bahwa ketiga fitur dataset, yakni `carat`, `table`, dan `x` memiliki outliers. Untuk menangani outliers akan digunakan metode IQR (_Inter Quartile Range_). IQR dihitung dengan mengurangkan kuartil ketiga (Q3) dari kuartil pertama (Q1) sebagaimana rumus berikut.

   $$IQR = Inter Quartile Range$$

   $$IQR = Q3 - Q1$$

   Setelah menggunakan metode IQR untuk menghilangkan outlier pada dataset jumlah dataset menjadi 168755 yang awalnya adalah 193573.

3. **Memeriksa Data Duplikat**

   ```python
   # Menghitung jumlah baris yang duplikat dalam Dataset
   jumlah_duplikat = gemstone.duplicated().sum()
   print("\nJumlah Duplikat:", jumlah_duplikat)
   ```
   Jumlah Duplikat: 0

   Terlihat bahwa tidak ada data duplikat pada dataset.

5. **Univariate Analysis**

   Pada tahap ini akan dilakukan analisis data dengan Univariate Analysis pada semua fitur, baik fitur kategorik maupun fitur numerik.

   - Categorical Features

     ![Univariate Analysis)](https://raw.githubusercontent.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/refs/heads/main/images/Fitur%20cut.png)
     
     Gambar 1. Univariate Analysis (Fitur cut)

     Pada Gambar 1 terdapat 5 kategori pada fitur Cut, secara berurutan dari jumlahnya yang paling banyak yaitu: Ideal, Premium, Very Good, Good, dan Fair. Dari data persentase dapat kita simpulkan bahwa lebih dari 70% sampel merupakan gemstone tipe grade tinggi, yaitu grade Ideal dan Premium.

     ![Univariate Analysis](https://raw.githubusercontent.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/refs/heads/main/images/Fitur%20color.png)

     Gambar 2. Grafik Univariate Distribusi (Fitur color)

     Pada Gambar 2 terdapat urutan kategori warna dari yang paling buruk hingga yang paling bagus adalah J, I, H, G, F, E, dan D. Dari grafik di atas, dapat disimpulkan bahwa sebagian besar grade berada pada grade menengah, yaitu G, F, H.

     ![Univariate Analysis](https://raw.githubusercontent.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/refs/heads/main/images/Fitur%20clarity.png)

     Gambar 3. Grafik Univariate Distribusi (Fitur clarity)

     Pada Gambar 3 fitur Clarity terdiri dari 8 kategori dari yang paling buruk ke yang paling baik, yaitu: I1, SI2, SI1, VS2, VS1, VVS2, VVS1, dan IF. Dari grafik dapat disimpulkan bahwa sebagian besar fitur merupakan grade rendah, yaitu SI1, SI2, dan VS2.

   - Numerical Features 

     ![Univariate Analysis](https://raw.githubusercontent.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/refs/heads/main/images/Univariate%20Analysis%20(Data%20Numerik).png)

     Gambar 4. Grafik Univariate Distribusi (Fitur Numerikal)

     Berdasarkan Gambar 4, dapat diamati histogram untuk variabel "price" yang merupakan fitur target (label). Dari histogram "price", kita bisa memperoleh beberapa informasi, antara lain:

     - Sebagian besar gemstone memiliki harga di bawah $4000, dengan puncak frekuensi pada kisaran harga yang lebih rendah.
     - Rentang harga gemstone cukup luas, mulai dari ratusan dolar Amerika hingga sekitar $12000.
     - Distribusi harga berlian miring ke kanan (right-skewed), menunjukkan bahwa meskipun sebagian besar berlian dijual dengan harga lebih rendah, terdapat sejumlah berlian dengan harga yang jauh lebih tinggi.
     - Lebih dari setengah gemstone memiliki harga di bawah $2500, menunjukkan adanya kecenderungan harga yang lebih terjangkau pada sebagian besar data.

6. **Multivariate Analysis**

   Pada tahap ini akan dilakukan pengecekan rata-rata probabilitas Price terhadap masing-masing fitur untuk mengetahui pengaruh fitur kategori dan numerik terhadap Price.

   - Categorical Features

     Melakukan pengecekan rata-rata harga terhadap masing-masing fitur kategori, yaitu cut, color, dan clarity untuk mengetahui pengaruh fitur tersebut terhadap harga.

     ![Multivariate Analysis](https://raw.githubusercontent.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/refs/heads/main/images/Multivariate%20Analysis%20(Data%20Categori).jpeg)

     Gambar 5. Grafik Multivariate (Fitur Kategorikal)

     Dengan mengamati Gambar 5, memiliki rata-rata harga relatif terhadap data kategori, kita memperoleh _insight_ sebagai berikut:

     - Pada fitur 'cut', rata-rata harga berlian berada dalam rentang yang mirip, yaitu antara 2500 hingga 4000. Grade tertinggi seperti "Ideal" justru memiliki harga rata-rata lebih rendah dibandingkan dengan grade lainnya seperti "Fair." Hal ini menunjukkan bahwa fitur "cut" memiliki pengaruh yang kecil terhadap variasi harga gemstone.
     - Pada fitur 'color', terdapat kecenderungan bahwa harga rata-rata berlian lebih tinggi pada grade warna yang lebih rendah, seperti "I" dan "J," sementara grade warna yang lebih tinggi seperti "E" memiliki harga yang lebih rendah. Ini menunjukkan bahwa pengaruh warna terhadap harga berlian juga relatif rendah.
     - Pada fitur 'clarity', gemstone dengan grade clarity lebih rendah, seperti "SI2" dan "I1," cenderung memiliki harga yang lebih tinggi dibandingkan grade clarity yang lebih tinggi seperti "IF." Ini mengindikasikan bahwa kejernihan berlian tidak selalu berkorelasi positif dengan harga, sehingga fitur "clarity" memiliki pengaruh yang rendah terhadap harga gemstone.
     - Kesimpulan akhir, fitur kategori seperti "cut," "color," dan "clarity" memiliki pengaruh yang rendah terhadap harga gemstone.
    
   - Numerical Features
  
     Melakukan pengecekan rata-rata harga terhadap masing-masing fitur numerik, yaitu carat, depth, table, x, y, dan z untuk mengetahui pengaruh fitur tersebut terhadap harga.

     ![Multivariate Aanlysis](https://raw.githubusercontent.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/refs/heads/main/images/Multivariate%20Analysis%20(Data%20Numerik).png)

     Gambar 6. Grafik Multivariate (Fitur Numerikal)

     Berdasarkan Gambar 6, fokus pada relasi antara semua fitur numerik dengan fitur target yaitu ‘price’. Pada pola sebaran data grafik pairplot terlihat ‘carat’, ‘x’, ‘y’, dan ‘z’ memiliki korelasi yang tinggi dengan fitur "price". Sedangkan kedua fitur lainnya yaitu 'depth' dan 'table' terlihat memiliki korelasi yang lemah karena sebarannya tidak membentuk pola.

7. **Correlation Matrix**

   ![Multivariate Analysis](https://raw.githubusercontent.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/refs/heads/main/images/Analysis%20Matrik%20Korelasi.png)

   Gambar 7. Diagram Heatmap Correlation Matrik (Fitur Numerikal)

   Berdasarkan Gambar 7, fitur 'carat', 'x', 'y', dan 'z' memiliki skor korelasi yang besar (diatas 0.9) dengan fitur target 'price'. Dimana, fitur 'price' berkolerasi tinggi dengan keempat fitur tersebut. Sementara fitur 'depth' memiliki korelasi yang sangat kecil (0.05). Sehingga fitur 'depth' dapat dihapus.

   Menghapus fitur depth pada dataset karena memiliki korelasi yang rendah terhadap fitur price.

   Tabel 5. Pengecekan dataset setelah menghapus fitur depth
   |   | carat | cut       | color | clarity | table | x    | y    | z    | price |
   |---|-------|-----------|-------|---------|-------|------|------|------|-------|
   | 2 | 0.70  | Ideal     |	G    | VS1  	 | 57.0	 | 5.69 | 5.73 | 3.50 | 2772  |
   | 3 | 0.32  | Ideal     |	G    | VS1	   | 56.0	 | 4.38 | 4.41 | 2.71 | 666   |
   | 5 | 1.51  | Very Good |	J    | SI1	   | 58.0	 | 7.34 | 7.29 | 4.59 | 7506  |
   | 6 | 0.74  | Ideal     |	E    | VS2	   | 57.0	 | 5.76 | 5.79 | 3.57 | 3229  |
   | 7 | 1.34  | Premium   |	G    | SI2	   | 57.0	 | 7.00 | 7.05 | 4.38 | 6224  |

8. **Encoding Fitur Kategori.**

   Proses encoding fitur kategori dilakukan dengan menggunakan teknik one-hot-encoding dari library scikit-learn. Teknik ini berfungsi untuk mendapatkan fitur baru yang sesuai sehingga dapat mewakili setiap fitur kategori. Pada proyek ini terdapat 3 fitur kategori, yaitu "cut", "color", "clarity". Proses encoding dilakukan dengan fitur get_dummies. Berikut output nya :

   ![image](https://raw.githubusercontent.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/refs/heads/main/images/encoding%20fitur%20categori.png)

   Gambar 8. Enconding Fitur Kategori (cut, color, clarity)

9. **Reduksi Dimensi dengan PCA.**

   Proses persiapan data atau _data preparation_ dengan teknik reduksi dimensi atau _dimension reduction_ merupakan teknik mengurangi jumlah fitur dengan tetap mempertahankan informasi pada data. Teknik pengurangan dimensi yang digunakan dalam kasus ini adalah Principal Component Analysis (PCA) untuk mereduksi dimensi, mengekstraksi fitur, dan mentransformasi data dari "n-dimensional space" ke dalam sistem berkoordinat baru dengan dimensi m, di mana m lebih kecil dari n.

   ![image](https://raw.githubusercontent.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/refs/heads/main/images/Reduksi%20Dimensi%20PCA.png)

   Gambar 9. Grafik Fitur x, y, dan z

   Hasil proporsi informasi dari fitur x, y, dan z dengan menggunakan Principal Component Analysis (PCA), yaitu
   `array([0.999, 0.001, 0.   ])`

10. **Train Test Split**

    Teknik ini dilakukan untuk membagi dataset menjadi dua bagian, yaitu data latih dan data uji. data latih akan digunakan untuk melatih model sedangkan data uji akan digunakan untuk evaluasi model. Hal tersebut perlu diterapkan agar model yang telah dilatih dapat diuji menggunakan data yang belum pernah dianalisa oleh model. Proses pembagian dataset menjadi data latih dan data uji dengan rasio perbandingan data latih dan data uji, yaitu 90 : 10. Terdapat 168755 total sampel data dalam dataset, sedangkan untuk total sampel data latih sebanyak 151879 data dan total sampel data uji sebanyak 16876 data.

11. **Standarisasi**

    Proses standarisasi fitur numerik, yaitu carat dan dimension menggunakan StandardScaler sehingga fitur data menjadi bentuk yang lebih mudah diolah oleh model machine learning.
   
    Tabel 6. Standarisasi Fitur Numerik
    |       | carat       | table       | dimension   |
    |-------|-------------|-------------|-------------|
    | count | 151879.0000 | 151879.0000 | 151879.0000 |
    | mean  | -0.0000	  | 0.0000      | 0.0000      |
    | std   | 1.0000	  | 1.0000	    | 1.0000      |
    | min   | -1.3734	  | -2.3493	    | -1.8553     |
    | 25%   | -0.9367	  | -0.5833	    | -0.9664     |
    | 50%   | -0.3089	  | 0.0054	    | -0.1388     |
    | 75%   | 0.8375	  | 0.5941	    | 0.9036      |
    | max   | 3.4033	  | 2.3601	    | 2.6412      |

    Dapat dilihat bahwa setelah proses standarisasi sekarang nilai mean = 0 dan standar deviasi = 1.

## Modeling

Pada tahap modeling, dilakukan pemilihan algoritma yang akan digunakan dalam membuat model *machine learning*, serta pengembangan dan pelatihan model *machine learning* agar dapat digunakan untuk melakukan analisis prediksi. Algoritma pada proyek ini melakukan pemodelan dengan 3 algoritma, yaitu:

1. **K-Nearest Neighbors (KNN)**

   K-Nearest Neighbors (KNN) adalah algoritma machine learning yang sederhana dan mudah dipahami untuk klasifikasi dan regresi. Algoritma ini bekerja dengan menemukan k tetangga terdekat dari data baru dan kemudian menggunakan kategori atau nilai rata-rata dari tetangga tersebut untuk memprediksi kategori atau nilai data baru. Pada algoritma K-Nearest Neighbor menggunakan parameter `n-neighbors` dengan nilai k = 10.

   ```python
   knn = KNeighborsRegressor(n_neighbors=10)
   ```

   Keunggulan KNN :
   - Dapat digunakan untuk klasifikasi dan regresi.
   - Sederhana dan mudah dipahami.
   
   Kekurangan KNN :
   - Sensitif terhadap outlier. 
   - Membutuhkan banyak memori dan waktu komputasi untuk dataset besar. 
   - Sulit untuk memilih nilai K yang optimal.

2. **Random Forest**

   Random Forest adalah algoritma machine learning ensemble yang menggabungkan beberapa decision tree untuk meningkatkan akurasi prediksi. Algoritma ini bekerja dengan membuat banyak decision tree secara acak dan kemudian menggunakan voting untuk memprediksi kategori atau nilai data baru. Pada algoritma Random Forest menggunakan parameter `n-estimator` dengan jumlah 50 trees (pohon), `max-depth` dengan nilai kedalaman atau panjang pohon sebesar 10, `random-state` dengan nilai 55, dan `n-jobs` yang bernilai -1 yang berarti pekerjaan dilakukan secara paralel.

   ```python
   RF = RandomForestRegressor(n_estimators = 50, max_depth = 10, random_state = 55, n_jobs = -1)
   ```
   Keunggulan Random Forest :
   - Memiliki akurasi prediksi yang tinggi.
   - Mampu menangani dataset dengan dimensi tinggi.
   - Tidak sensitif terhadap outlier.
   
   Kekurangan Random Forest :
   - Cenderung overfit pada dataset kecil. 
   - Membutuhkan banyak waktu komputasi untuk pelatihan. 
   - Sulit untuk diinterpretasikan.

3. **Gradient Boosting**

   Gradient Boosting adalah algoritma machine learning yang menggunakan teknik ensembel learning dari decision tree untuk memprediksi nilai. Gradient Boosting sangat mampu menangani pattern yang kompleks dan data ketika linear model tidak dapat menangani. Pada algoritma Gradient Boosting menggunakan parameter  `max-depth` dengan nilai kedalaman atau panjang pohon sebesar 7, dan `random-state` dengan nilai 55.

   ```python
   boosting = GradientBoostingRegressor(max_depth=7, random_state=55)
   ```

   Kelebihan Gradient Boosting :
   - Hasil pemodelan yang lebih akurat
   - Model yang stabil dan lebih kuat (robust)
   - Dapat digunakan untuk menangkap hubungan linear maupun non linear pada data
   
   Kekurangan Gradient Boosting :
   - Pengurangan kemampuan interpretasi model
   - Waktu komputasi dan desain tinggi
   - Tingkat kesulitan yang tinggi dalam pemilihan model

## Evaluation

Pada tahap evaluasi, dilakukan pengujian model dengan ketiga algoritma yang telah dibuat pada tahap modeling. Sebelum melakukan evaluasi, dilakukan proses scaling pada data latih untuk menghindari kebocoran data.

```python
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])
```

Proses evaluasi model pada proyek ini menggunakan metrik Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi [[7]](https://www.dicoding.com/academies/319/tutorials/18595). MSE dipilih karena memberikan penalti yang lebih besar untuk kesalahan prediksi yang besar, sehingga membantu dalam mengidentifikasi model yang mampu memberikan prediksi lebih akurat.

$$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

Keterangan:  
$N$ = Jumlah data    
$y_i$ = Nilai sebenarnya   
$\hat{y_i}$ = Nilai prediksi

**Cara kerja :**

Cara kerja Metrik MSE adalah dengan menghitung selisih hasil prediksi dengan nilai fitur target (PE). Nilai selisih tersebut, disebut juga sebagai nilai eror yang kemudian di kuadratkan untuk menangani nilai selisih negatif. Selanjutnya hasil pengkuadratan setiap nilai selisih dijumlahkan dan terakhir dibagi dengan banyak data point (n) untuk memperoleh nilai rata-ratanya. Rata-rata inilah yang disebut Mean Squared Error (MSE).

Berikut adalah tabel nilai MSE pada setiap model dengan data latih dan data uji.

Tabel 7. Nilai Evaluasi Model _Machine Learning_
|	         | train      | test       |
|----------|------------|------------|
| KNN	     | 158.060099 | 188.027968 |
| RF       | 184.881839 | 189.772913 |
| Boosting | 133.044841 | 146.840974 |

Untuk memudahkan dalam mengevaluasi model kita akan melakukan visualisasi hasil menggunakan bar chart sebagai berikut.

![image](https://raw.githubusercontent.com/FebriAdha/Submission-Predictive-Analytics-Gemstone-Price/refs/heads/main/images/barplot%20MSE.png)

Gambar 10. Grafik Evaluasi Model Machine Learning

Berdasarkan grafik diatas, dapat disimpulkan sebagai berikut:
- Model dengan algoritma Gradient Boosting memberikan nilai _error_ yang paling kecil yaitu _train_ sebesar 133.044841 dan _test_ sebesar 146.840974.
- Model dengan algoritma KKN memberikan nilai _error_ _train_ sebesar 158.060099 dan _test_ sebesar 188.027968.
- Model dengan algoritma Random Forest memberikan nilai _error_ yang paling besar yaitu _train_ sebesar 184.881839 dan *test* sebesar 189.772913.

Berikut hasil uji prediksi menggunakan beberapa harga dari data test.

Tabel 8. Hasil Pengujian Prediksi Model
|       | y_true | prediksi_KNN | prediksi_RF | prediksi_Boosting |
|-------|--------|--------------|-------------|-------------------|
| 65114 | 868    | 944.0        | 701.4	      | 915.8             |

Terlihat bahwa prediksi model KNN, prediksi model Random Forest, dan prediksi model Gradient Boosting adalah 994, 701, dan 915 dari 868. Dari ketiga model, model yang memiliki nilai prediksi meleset sangat kecil adalah model Boosting dan model yang memiliki nilai prediksi meleset sangat besar adalah RF.

### Kesimpulan

Berdasarkan proyek yang telah dilakukan, maka dapat disimpulkan beberapa hal berikut.

- Berdasarkan _Exploratory Data Analysis (EDA)_ menggunakan Correlation Matrix, fitur yang paling berkolerasi dengan harga gemstone adalah x (panjang), y (lebar), z (kedalam).
- Berdasarkan hasil analisis dan pemodelan _Machine Learning_ untuk kasus ini adalah model yang digunakan untuk melakukan analisis harga batu permata menghasilkan tingkat _error_ yang paling rendah menggunakan algoritma Gradient Boosting dan memberikan hasil prediksi yang paling mendekati dengan data sebenarnya jika dibandingkan dengan algoritma lainnya, yaitu K-Nearest Neighbor dan Random Forest.


## Referensi

[1] Firlydani Syifana Putra, "Prediksi Harga Batu Mulia/Gemstone Berdasarkan Karakteristiknya Menggunakan Linear Regression", Jurnal BATIRSI, Vol.6, No.2, Januari 2023. https://e-journal.stmik-tegal.ac.id/index.php/batirsi/article/view/41/31

[2] Data Bridge Market Research. (2024). Global Gemstones Market – Industry Trends and Forecast to 2031. https://www.databridgemarketresearch.com/reports/global-gemstones-market

[3] International Gem Society (IGS). (2023). Gemstone Value and Pricing Factors. https://www.gemsociety.org/article/gemstone-value-factors/

[4] Subramanian, D. (2019). A Simple Introduction to K-Nearest Neighbors Algorithm. Towards Data Science. https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e

[5] Wood, T. -.What is a Random Forest?. DeepAI. https://deepai.org/machine-learning-glossary-and-terms/random-forest

[6] Trivusi, "Gradient Boosting: Pengertian, Cara Kerja, dan Kegunaannya", *Trivusi*, 2023. https://www.trivusi.web.id/2023/03/algoritma-gradient-boosting.html

[7] Revou, "Analisis Univariat". https://revou.co/kosakata/analisis-univariat

[8] Revou, "Analisis Multivariat". https://revou.co/kosakata/analisis-multivariat

[7] https://www.dicoding.com/academies/319/tutorials/18595


