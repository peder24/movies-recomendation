# Laporan Proyek Machine Learning - Sistem Rekomendasi Film

## Domain Proyek

Industri hiburan, khususnya film, telah mengalami pertumbuhan yang signifikan dalam beberapa dekade terakhir. Dengan ribuan film yang tersedia, pengguna sering kali kesulitan menemukan konten yang sesuai dengan preferensi mereka. Fenomena ini dikenal sebagai "information overload" yang dapat menurunkan pengalaman pengguna dan engagement pada platform streaming atau database film [1].

Sistem rekomendasi film menjadi solusi penting untuk mengatasi masalah ini dengan menyaring konten yang relevan dan mempersonalisasi pengalaman pengguna. Menurut penelitian yang dilakukan oleh Netflix, sekitar 80% film yang ditonton di platform mereka berasal dari rekomendasi [2]. Ini menunjukkan bahwa sistem rekomendasi yang efektif tidak hanya meningkatkan pengalaman pengguna tetapi juga memberikan dampak bisnis yang signifikan.

Proyek ini mengembangkan sistem rekomendasi film dengan menggunakan dataset MovieLens untuk membantu pengguna menemukan film yang sesuai dengan preferensi mereka. Dengan mengimplementasikan pendekatan content-based filtering dan collaborative filtering, sistem ini diharapkan dapat memberikan rekomendasi yang relevan dan personal.

*Referensi:*

1. F. Ricci, L. Rokach, and B. Shapira, "Recommender Systems Handbook," Springer, 2015.
2. C. A. Gomez-Uribe and N. Hunt, "The Netflix Recommender System: Algorithms, Business Value, and Innovation," ACM Transactions on Management Information Systems, vol. 6, no. 4, pp. 1-19, 2015.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, berikut adalah rumusan masalah dalam proyek ini:

1. Bagaimana cara mengembangkan sistem rekomendasi film yang dapat menyarankan film berdasarkan kesamaan konten (genre) dengan film yang disukai pengguna?
2. Bagaimana cara mengembangkan sistem rekomendasi film yang dapat memprediksi preferensi pengguna berdasarkan pola rating dari pengguna lain yang memiliki selera serupa?
3. Bagaimana perbandingan performa antara pendekatan berbasis konten dan pendekatan kolaboratif dalam merekomendasikan film yang relevan bagi pengguna?

### Goals

Tujuan dari proyek ini adalah:

1. Mengembangkan model content-based filtering yang dapat merekomendasikan film berdasarkan kesamaan genre dengan film yang disukai pengguna.
2. Mengembangkan model collaborative filtering menggunakan deep learning yang dapat memprediksi rating pengguna terhadap film yang belum ditonton dan memberikan rekomendasi berdasarkan prediksi tersebut.
3. Membandingkan performa kedua pendekatan menggunakan metrik evaluasi yang sesuai untuk menentukan pendekatan yang lebih efektif dalam merekomendasikan film yang relevan.

### Solution Approach

Untuk mencapai tujuan tersebut, proyek ini mengusulkan dua pendekatan sistem rekomendasi:

1. Content-Based Filtering:
- Menggunakan TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengekstrak fitur dari genre film
- Menghitung cosine similarity antara film berdasarkan representasi TF-IDF
- Merekomendasikan film dengan similarity score tertinggi

2. Collaborative Filtering dengan Deep Learning:
- Mengimplementasikan model neural network (RecommenderNet) dengan embedding layer
- Mempelajari representasi latent dari pengguna dan film
- Memprediksi rating untuk film yang belum ditonton
- Merekomendasikan film dengan prediksi rating tertinggi

## Data Understanding

Dataset MovieLens Small terdiri dari beberapa file, namun dalam proyek ini saya fokus pada empat file utama:

Proyek ini menggunakan dataset MovieLens 100K yang berisi 100.000 rating dari 943 pengguna pada 1.682 film. Dataset ini dikembangkan oleh GroupLens Research dan tersedia untuk diunduh di [situs resmi GroupLens](https://grouplens.org/datasets/movielens/100k/).

Dataset terdiri dari beberapa file, namun dalam proyek ini kami menggunakan dua file utama:

1. ratings.csv: Berisi rating film dari pengguna dengan skala 0.5-5.0
- userId: ID unik untuk setiap pengguna
- movieId: ID unik untuk setiap film
- rating: Rating yang diberikan pengguna (0.5-5.0)
- timestamp: Waktu rating diberikan (format Unix timestamp)

2. movies.csv: Berisi informasi film
- movieId: ID unik untuk setiap film
- title: Judul film termasuk tahun rilis
- genres: Genre film (dipisahkan dengan karakter '|')

### Kondisi Data

#### Dataset Ratings (ratings.csv)

Informasi Umum:
- Jumlah data: 100,836 rating
- Jumlah pengguna: 610 pengguna unik
- Jumlah film: 9,724 film unik

Kualitas Data:
- Missing Values: Tidak ada missing values dalam dataset
- Data Duplikat: Tidak ditemukan data duplikat
- Outliers: Ditemukan 4,181 outliers menggunakan metode IQR (Interquartile Range)
- Tipe Data: Semua kolom memiliki tipe data yang sesuai (int64 untuk ID, float64 untuk rating)

Statistik Deskriptif Rating:
- Minimum: 0.5
- Maksimum: 5.0
- Rata-rata: 3.5
- Median: 3.5

#### Informasi Dataset Movies

Informasi Umum:
- Jumlah data: 9,742 film

Kualitas Data:
- Missing Values: Tidak ada missing values dalam dataset
- Data Duplikat: Tidak ditemukan data duplikat berdasarkan movieId
- Genre Kosong: Tidak ditemukan film dengan genre kosong
- Tipe Data: movieId (int64), title dan genres (object)

### Exploratory Data Analysis

#### 1. Distribusi Rating

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1747844200/output_othhkv.png)

Grafik di atas menunjukkan distribusi rating film yang diberikan oleh pengguna. Kita dapat melihat bahwa rating 4.0 merupakan rating yang paling banyak diberikan, diikuti oleh rating 3.0 dan 5.0. Hal ini menunjukkan kecenderungan pengguna untuk memberikan rating positif terhadap film yang mereka tonton.

#### 2. Distribusi Jumlah Rating per Film

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1747844327/output1_dw5eqi.png)

Histogram ini menunjukkan distribusi jumlah rating yang diterima oleh setiap film. Terlihat pola "long tail" di mana sebagian kecil film populer menerima banyak rating, sementara sebagian besar film hanya menerima sedikit rating. Fenomena ini umum terjadi pada dataset rekomendasi dan menunjukkan bahwa perhatian pengguna terkonsentrasi pada film-film populer.

#### 3. Genre Film Terpopuler

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1747844430/output2_cjrcqr.png)

Bar chart ini menunjukkan 15 genre film yang paling banyak muncul dalam dataset. Drama, Comedy, dan Action merupakan tiga genre teratas, yang mencerminkan dominasi genre-genre tersebut dalam industri film. Informasi ini penting untuk memahami distribusi genre dalam dataset dan potensi bias dalam rekomendasi.

## Data Preparation

Tahap data preparation merupakan langkah krusial dalam pengembangan sistem rekomendasi yang melibatkan berbagai teknik pemrosesan data untuk mempersiapkan dataset yang berkualitas untuk modeling. Berikut adalah tahapan-tahapan yang dilakukan:

### 1. Konversi Timestamp

Mengubah format Unix timestamp menjadi datetime untuk memudahkan analisis temporal jika diperlukan. Konversi ini mengubah format Unix timestamp menjadi format datetime yang lebih mudah dibaca dan dianalisis. Meskipun timestamp tidak digunakan dalam model akhir, konversi ini memfasilitasi analisis temporal jika diperlukan.

### 2. Penanganan Outliers

Berdasarkan hasil deteksi outliers pada tahap Data Understanding yang menemukan 4,181 outliers, dilakukan penanganan outliers dengan strategi domain-specific:

Strategi Penanganan:
- Validasi domain: Memeriksa apakah rating sesuai dengan skala MovieLens yang valid (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
- Jika semua rating valid, menggunakan metode IQR untuk menghapus outliers statistik

Hasil Penanganan:
- Data berhasil dibersihkan dari outliers yang dapat mengganggu performa model
- Kualitas data meningkat untuk menghasilkan rekomendasi yang lebih akurat

### 3. Data Filtering dan Pengurangan Data

Untuk mengatasi masalah sparsity yang umum terjadi pada dataset rekomendasi, dilakukan filtering berdasarkan popularitas film dan aktivitas pengguna:

**Parameter filtering**:
- Minimum rating per film: 3 rating
- Minimum rating per pengguna: 3 rating

**Hasil filtering**:
- Jumlah rating berkurang dari 96,655 menjadi 90,710
- Jumlah film berkurang menjadi 4,815 film
- Jumlah pengguna berkurang menjadi 610 pengguna

**Filtering ini penting untuk**:
- Mengurangi sparsity matrix user-item
- Menghilangkan noise dari film/pengguna dengan data terlalu sedikit
- Meningkatkan kualitas pembelajaran model

### 4. Penggabungan Dataset

Menggabungkan dataset ratings dengan informasi film untuk mendapatkan dataset lengkap:

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1747844783/Screenshot_2025-05-21_232609_lisi3p.png)

Penggabungan ini menghasilkan dataset yang lengkap dengan informasi rating dan film, yang diperlukan untuk analisis dan modeling.

### 5. Data Preparation untuk Content-Based Filtering

#### Pembuatan Dataset Khusus Content-Based

Membuat struktur data yang optimal untuk content-based filtering:

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1748071394/Screenshot_2025-05-24_142255_c6qyv7.png)

Membuat struktur data yang optimal untuk content-based filtering dengan fokus pada informasi film (movieId, title, genres). Dataset ini dirancang khusus untuk menganalisis kesamaan konten antar film berdasarkan genre.

#### Ekstraksi Fitur dengan TF-IDF

Mengubah informasi genre menjadi representasi numerik menggunakan TF-IDF:

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1747845017/Screenshot_2025-05-21_233000_nn7e0i.png)

**Proses TF-IDF**:

- Tokenization: Memisahkan string genre dengan delimiter '|'
- Term Frequency: Menghitung frekuensi kemunculan setiap genre dalam film
- Inverse Document Frequency: Memberikan bobot lebih tinggi pada genre yang jarang
- TF-IDF Matrix: Menghasilkan matriks sparse dengan dimensi (n_movies × n_genres)

**Keunggulan TF-IDF untuk genre**:
- Memberikan bobot lebih tinggi pada genre unik (seperti Film-Noir)
- Mengurangi dominasi genre umum (seperti Drama)
- Menghasilkan representasi numerik yang dapat diproses algoritma ML

#### Mengapa TF-IDF Dipilih:

1. Representasi Numerik: TF-IDF mengubah data kategorikal (genre) menjadi vektor numerik yang dapat diproses oleh algoritma machine learning.

2. Pembobotan yang Tepat: Genre yang lebih jarang (seperti "Film-Noir" atau "Documentary") mendapat bobot lebih tinggi dibandingkan genre yang umum (seperti "Drama" atau "Comedy"), sehingga film dengan genre langka yang sama akan memiliki similarity score yang lebih tinggi.

3. Penanganan Multi-genre: TF-IDF dapat menangani film dengan multiple genre dengan baik, memberi representasi yang tepat untuk kombinasi genre.

4. Efisiensi Komputasi: Dibandingkan dengan metode representasi teks lain seperti word embeddings, TF-IDF lebih sederhana dan efisien secara komputasi, terutama untuk dataset dengan jumlah fitur terbatas seperti genre film.

Implementasi TF-IDF dalam kode menggunakan TfidfVectorizer dari scikit-learn, yang secara otomatis menghitung matriks TF-IDF. Parameter tokenizer=lambda x: x.split('|') digunakan untuk memisahkan string genre yang dipisahkan oleh karakter '|' menjadi token individual.

Hasil dari proses ini adalah matriks sparse dengan dimensi (jumlah_film × jumlah_genre_unik), di mana setiap baris merepresentasikan sebuah film dan setiap kolom merepresentasikan sebuah genre. Nilai dalam matriks adalah skor TF-IDF yang menunjukkan pentingnya genre tersebut untuk film tertentu.

#### Perhitungan Cosine Similarity Matrix

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1748071500/Screenshot_2025-05-24_142438_epznja.png)

Menghitung similarity matrix untuk semua pasangan film berdasarkan representasi TF-IDF. Matrix cosine similarity menjadi dasar untuk memberikan rekomendasi berdasarkan kesamaan konten, menghasilkan nilai similarity antara 0-1 untuk setiap pasangan film.

### 6. Data Preparation untuk Collaborative Filtering

#### Encoding User dan Movie ID

Mengubah ID pengguna dan film menjadi indeks integer berurutan untuk keperluan neural network:

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1747844898/Screenshot_2025-05-21_232801_nod1jz.png)

**Proses encoding ini meliputi**:
- Membuat mapping dari user_id asli ke user_encoded (0 hingga n_users-1)
- Membuat mapping dari movie_id asli ke movie_encoded (0 hingga n_movies-1)
- Membuat reverse mapping untuk konversi balik
- Menambahkan kolom encoded ke dataset

#### Normalisasi Rating

Normalisasi ini mengubah rating ke skala 0-1 untuk mempermudah proses pembelajaran model. Rating yang dinormalisasi lebih sesuai untuk fungsi aktivasi sigmoid di output model dan membantu konvergensi training.

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1747845269/Screenshot_2025-05-21_233351_nwetxu.png)

Normalisasi ini penting karena:
- Fungsi aktivasi sigmoid menghasilkan output dalam range [0,1]
- Mempercepat konvergensi training
- Menghindari saturasi pada fungsi aktivasi

#### Pembagian Data Training dan Validasi

Membagi data menjadi training dan validation set dengan rasio 80:20:

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1747845333/Screenshot_2025-05-21_233520_eunh8m.png)

Langkah-langkah:
- Mengacak dataset untuk menghindari bias urutan
- Membagi fitur (user, movie) dan target (normalized_rating)
- Menggunakan 80% data untuk training dan 20% untuk validasi

### Rangkuman Data Preparation

Tahap data preparation berhasil menghasilkan:

1. Dataset bersih tanpa missing value dan outliers
- Semua nilai NaN telah dipastikan tidak ada.
- Outliers pada kolom rating berhasil dideteksi dan dibersihkan menggunakan metode IQR.

2. Data terfilter berdasarkan aktivitas pengguna dan popularitas film
- Hanya menyertakan film dan pengguna dengan minimal 3 interaksi untuk mengurangi sparsity.

3. Encoding ID pengguna dan film
- userId dan movieId dikonversi menjadi user_encoded dan movie_encoded sebagai integer index untuk input model neural network.

4. Normalisasi rating ke skala 0–1
- Rating dinormalisasi agar sesuai dengan fungsi aktivasi (sigmoid) dalam model rekomendasi berbasis neural network.

5. TF-IDF matrix dari informasi genre film
- Genre film dikonversi ke bentuk numerik menggunakan teknik TF-IDF untuk content-based filtering.

6. Cosine similarity matrix antar film
- Dihitung berdasarkan TF-IDF genre, digunakan untuk mengukur kemiripan antar film dalam sistem rekomendasi berbasis konten.

7. Pembagian data menjadi training dan validation set (80:20)
- Untuk memastikan evaluasi model dilakukan secara adil dan terpisah dari data training.

Semua tahapan ini memastikan bahwa data yang digunakan untuk modeling memiliki kualitas tinggi dan format yang sesuai dengan kebutuhan masing-masing algoritma rekomendasi.

## Modeling

Dalam proyek ini, dua pendekatan model rekomendasi film dikembangkan dengan karakteristik dan cara kerja yang berbeda:

### 1. Content-Based Filtering

Content-based filtering adalah pendekatan sistem rekomendasi yang merekomendasikan item berdasarkan kesamaan karakteristik atau fitur konten dengan item yang disukai pengguna sebelumnya.

#### Definisi dan Cara Kerja

**Prinsip Dasar:**

Content-based filtering bekerja dengan menganalisis fitur-fitur intrinsik dari item (dalam hal ini film) dan merekomendasikan item lain yang memiliki fitur serupa. Pendekatan ini mengasumsikan bahwa jika pengguna menyukai suatu item dengan karakteristik tertentu, mereka akan menyukai item lain dengan karakteristik serupa.

**Algoritma yang Digunakan:**

1. TF-IDF Vectorization: Mengubah fitur kategorikal (genre) menjadi representasi numerik
2. Cosine Similarity: Mengukur kesamaan antar film berdasarkan sudut antara vektor fitur

**Formula Cosine Similarity:**

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1748071937/Screenshot_2025-05-24_143158_qcptig.png)

Dimana:
- A dan B adalah vektor TF-IDF dari dua film
- A · B adalah dot product dari kedua vektor
- ||A|| dan ||B|| adalah magnitude (norm) dari masing-masing vektor

**Hasil Rekomendasi:**

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1748354528/Screenshot_2025-05-27_210002_yarsip.png)

Dari hasil di atas, kita dapat melihat bahwa model content-based filtering berhasil merekomendasikan film-film dengan genre serupa. Misalnya, "Toy Story" yang merupakan film Adventure, Animation, Children, Comedy, Fantasy mendapatkan rekomendasi film-film dengan genre serupa seperti "Shrek the Third", "The Good Dinosaur", dan "Antz". Similarity score yang tinggi (mendekati 1.0) menunjukkan kesamaan genre yang kuat.

**Keunggulan Content-Based Filtering:**
- Tidak memerlukan data dari pengguna lain (mengatasi cold-start problem)
- Dapat memberikan rekomendasi untuk item baru yang belum pernah dirating
- Transparan dalam hal mengapa suatu item direkomendasikan
- Tidak terpengaruh oleh sparsity data pengguna

**Kelemahan Content-Based Filtering:**
- Terbatas pada fitur yang tersedia (dalam kasus ini hanya genre)
- Cenderung memberikan rekomendasi yang monoton (over-specialization)
- Tidak dapat menangkap preferensi kompleks yang tidak tercermin dalam fitur konten
- Sulit menemukan item yang tidak jelas hubungannya tetapi menarik bagi pengguna

### 2. Collaborative Filtering

Model collaborative filtering merekomendasikan film berdasarkan pola rating dari pengguna lain yang memiliki selera serupa.

#### Definisi dan Cara Kerja

**Prinsip Dasar:**

Collaborative filtering bekerja dengan menganalisis pola rating atau interaksi pengguna dengan item, kemudian mengidentifikasi pengguna atau item yang serupa untuk memberikan rekomendasi. Pendekatan ini mengasumsikan bahwa pengguna yang memiliki preferensi serupa di masa lalu akan memiliki preferensi serupa di masa depan.

**Jenis Collaborative Filtering:**

Proyek ini mengimplementasikan Model-Based Collaborative Filtering menggunakan Matrix Factorization dengan neural network.

#### Implementasi Model:

1. Arsitektur Model Neural Network (RecommenderNet):

Model ini mengimplementasikan matrix factorization dengan neural network, dengan komponen utama:
   
- **Embedding Layers**: Layer ini memetakan ID pengguna dan film ke ruang vektor latent dengan dimensi embedding_size (30). Embedding ini dapat diinterpretasikan sebagai representasi fitur tersembunyi yang dipelajari model.
   
- **He Initialization**: Inisialisasi bobot menggunakan metode He Normal yang cocok untuk model deep learning modern, membantu konvergensi yang lebih baik.
   
- **L2 Regularization**: Regularisasi L2 dengan faktor 1e-5 diterapkan untuk mencegah overfitting dengan menghukum bobot yang terlalu besar.
   
- **Bias Terms**: Model mempelajari bias untuk setiap pengguna dan film, menangkap kecenderungan pengguna memberikan rating tinggi/rendah dan film menerima rating tinggi/rendah.
   
- **Forward Pass**: 
    - Mengambil embedding untuk pengguna dan film
    - Menghitung dot product (interaksi) antara embedding pengguna dan film
    - Menambahkan bias pengguna dan film
    - Menerapkan fungsi aktivasi sigmoid untuk mendapatkan prediksi rating dalam range [0,1]

2. Parameter Training:

- Loss Function: Binary Cross Entropy, cocok untuk nilai target dalam range [0,1]
- Optimizer: Adam dengan learning rate 0.001, algoritma optimasi yang adaptif dan efisien
- Metrics: Root Mean Squared Error (RMSE) untuk mengukur akurasi prediksi
- Batch Size: 32, menyeimbangkan antara kecepatan training dan stabilitas
- Early Stopping: Menghentikan training jika tidak ada perbaikan RMSE pada validation set selama 5 epochs berturut-turut
- Epochs: Maksimum 50, meskipun early stopping biasanya akan menghentikan training lebih awal

3. Proses Training:

- Loss Function: Binary Cross Entropy
- Optimizer: Adam dengan learning rate 0.001
- Regularization: L2 regularization (λ=1e-5) untuk mencegah overfitting
- Early Stopping: Menghentikan training jika validation RMSE tidak membaik selama 5 epochs

4. Algoritma Rekomendasi:

Algoritma rekomendasi melakukan langkah-langkah:
- Mengidentifikasi film yang sudah ditonton oleh pengguna
- Mengidentifikasi film yang belum ditonton untuk direkomendasikan
- Memprediksi rating untuk semua film yang belum ditonton menggunakan model
- Mengembalikan rating ke skala asli (0.5-5.0)
- Mengurutkan film berdasarkan prediksi rating dan mengambil top-k
- Membuat dataframe hasil rekomendasi dengan informasi film dan prediksi rating

**Kelebihan**:
- Dapat menemukan pola kompleks yang tidak jelas dari fitur konten
- Mampu memberikan rekomendasi yang tidak terduga tetapi relevan (serendipity)
- Tidak memerlukan analisis konten yang mendalam
- Dapat menangkap preferensi yang berubah seiring waktu

**Kekurangan**:
- Cold-start problem untuk pengguna atau item baru
- Memerlukan data rating yang cukup padat
- Komputasi intensif untuk dataset besar
- Rentan terhadap sparsity data

#### Hasil Rekomendasi:

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1748355802/Screenshot_2025-05-27_212259_iogabk.png)

Dari hasil di atas, kita dapat melihat bahwa model collaborative filtering berhasil merekomendasikan film-film dengan rating prediksi yang tinggi. Model ini merekomendasikan film-film populer dan highly-rated seperti "The Shawshank Redemption", "The Godfather", dan "Schindler's List". Menariknya, rekomendasi ini mencakup berbagai genre (Drama, Crime, Action, Thriller), yang menunjukkan bahwa model dapat menangkap preferensi pengguna yang lebih kompleks dibandingkan hanya berdasarkan kesamaan genre.

### Perbandingan Hasil Rekomendasi

Untuk memberikan perbandingan langsung antara kedua model, berikut adalah hasil rekomendasi untuk pengguna yang sama (ID 1) dan film referensi "Toy Story (1995)":

#### Content-Based Filtering (berdasarkan "Toy Story"):

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1747845806/Screenshot_2025-05-21_234257_cwkdwd.png)

#### Collaborative Filtering (untuk pengguna ID 1):

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1748355802/Screenshot_2025-05-27_212259_iogabk.png)

Dari perbandingan ini, kita dapat melihat perbedaan mendasar antara kedua pendekatan:

1. Content-Based Filtering fokus pada kesamaan konten, sehingga merekomendasikan film-film dengan genre serupa (Animation, Adventure, Children, Comedy). Rekomendasi ini konsisten dan dapat diprediksi, tetapi mungkin kurang bervariasi.

2. Collaborative Filtering memberikan rekomendasi berdasarkan pola rating pengguna lain yang memiliki selera serupa, menghasilkan rekomendasi yang lebih beragam dalam hal genre (Drama, Crime, War, Thriller). Pendekatan ini dapat menghasilkan rekomendasi yang tidak terduga tetapi relevan dengan preferensi pengguna.

## Evaluation

Evaluasi sistem rekomendasi memerlukan metrik yang tepat untuk mengukur kualitas rekomendasi yang dihasilkan. Dalam proyek ini, digunakan dua metrik evaluasi yang berbeda untuk masing-masing pendekatan.

### Metrik Evaluasi

1. **Precision untuk Content-Based Filtering**

Untuk model Content-Based Filtering, saya menggunakan metrik Precision@K (dalam hal ini Precision@10). Pemilihan Precision didasarkan pada beberapa pertimbangan penting:

- Relevansi Berbasis Konten: Precision secara langsung mengukur proporsi item yang relevan dari K rekomendasi teratas berdasarkan kesamaan konten (genre film).
- Mengatasi Cold Start Problem: Precision dapat dievaluasi tanpa memerlukan data historis interaksi pengguna yang ekstensif.
- Tidak Bias terhadap Popularitas: Precision fokus pada relevansi konten, bukan pada frekuensi interaksi, sehingga tidak bias terhadap item populer.

**Formula Precision:**

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1748073474/Screenshot_2025-05-24_145736_ftm54i.png)

1. **RMSE untuk Collaborative Filtering**

Untuk model Collaborative Filtering, saya menggunakan Root Mean Square Error (RMSE) yang mengukur akurasi prediksi rating:

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1748088749/Screenshot_2025-05-24_191131_iwkbax.png)

RMSE sangat cocok untuk Collaborative Filtering karena:

- Model ini pada dasarnya adalah model prediksi rating
- RMSE memberikan nilai error dalam skala yang sama dengan rating asli
- RMSE memberikan bobot lebih pada error besar, mendorong model untuk menghindari kesalahan prediksi yang signifikan

### Hasil Evaluasi

#### Content-Based Filtering

Model Content-Based Filtering mencapai Precision@10 = 1.0000 (100%) yang berarti semua film yang direkomendasikan memiliki relevansi genre dengan film referensi.

Contoh rekomendasi untuk film "MAS*H (1970)" (Comedy|Drama|War):

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1748355095/Screenshot_2025-05-27_211109_fyjzgi.png)

Hasil ini menunjukkan bahwa model TF-IDF dengan Cosine Similarity sangat efektif dalam mengidentifikasi film dengan genre serupa. enam rekomendasi teratas memiliki genre yang identik dengan film referensi (Comedy|Drama|War), dan empat lainnya memiliki dua dari tiga genre yang sama (Comedy|War).

#### Collaborative Filtering

Model Collaborative Filtering mencapai RMSE = 0.7229 pada skala rating asli (0.5-5.0), yang setara dengan error 20.65% dari range rating. Ini menunjukkan akurasi prediksi yang tinggi.

Grafik training dan validation RMSE menunjukkan:

- Training RMSE: 0.1523 (normalized)
- Validation RMSE: 0.2093 (normalized)
- Test RMSE: 0.7229 (skala asli 0.5-5.0)

Gap yang kecil antara training dan validation RMSE menunjukkan model tidak mengalami overfitting yang signifikan.

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1748355193/Screenshot_2025-05-27_211258_y4yrrc.png)

Contoh rekomendasi untuk User ID 1 menunjukkan film-film dengan predicted rating yang sangat tinggi (berkisar 4.73-4.87 dari skala 5.0), yang mengindikasikan kualitas prediksi model yang excellent. Rekomendasi didominasi oleh film-film klasik dan highly-acclaimed seperti "The Shawshank Redemption" (4.87), "The Godfather" (4.80), dan trilogi "Lord of the Rings" (4.75-4.78). Menariknya, model collaborative filtering menunjukkan kemampuan untuk merekomendasikan film lintas berbagai genre - dari Crime|Drama ("The Shawshank Redemption", "The Godfather"), Adventure|Fantasy ("Lord of the Rings"), Mystery|Thriller ("Memento"), hingga Comedy|Romance ("Amelie"). Hal ini menunjukkan bahwa model tidak terbatas pada satu genre tertentu, melainkan dapat menangkap pola preferensi pengguna yang kompleks berdasarkan similarity dengan pengguna lain yang memiliki selera serupa.

### Visualisasi Hasil Evaluasi

![alt_text](https://res.cloudinary.com/dk2tex4to/image/upload/v1748356757/output_kn4emq.png)

Visualisasi di atas menunjukkan perbandingan metrik evaluasi untuk kedua model:

**Panel Kiri (Content-Based Filtering):**
- Bar chart menunjukkan Precision@10 = 1.0000 (100%), yang merupakan skor maksimal.
- Tinggi bar mencapai batas atas skala, mengindikasikan performa optimal.
- Warna biru muda memberikan kesan stabilitas dan reliability.

**Panel Kanan (Collaborative Filtering):**
- Bar chart menunjukkan RMSE untuk training (0.1523) dan validation (0.2093) dalam skala normalized.
- Gap yang kecil antara kedua nilai menunjukkan model yang well-tuned tanpa overfitting signifikan.
- Warna merah muda untuk training dan hijau muda untuk validation memudahkan perbandingan.

Visualisasi ini memperkuat kesimpulan bahwa kedua model memiliki performa yang sangat baik dalam konteks metrik yang relevan untuk masing-masing pendekatan.

###  Dampak terhadap Business Understanding

#### Menjawab Problem Statement

1. **Pengembangan sistem rekomendasi berbasis konten:**
- Solusi: Berhasil mengimplementasikan sistem rekomendasi berbasis konten menggunakan TF-IDF dan Cosine Similarity.
- Dampak: Precision@10 = 100% menunjukkan kemampuan sistem untuk merekomendasikan film dengan genre yang sangat relevan dengan preferensi pengguna.
- Business Impact: Meningkatkan pengalaman pengguna dengan rekomendasi yang sangat relevan, terutama untuk pengguna baru (mengatasi cold-start problem).

2. **Pengembangan sistem rekomendasi berbasis kolaboratif:**
- Solusi: Berhasil mengimplementasikan sistem rekomendasi kolaboratif menggunakan Neural Network.
- Dampak: RMSE = 0.7229 menunjukkan kemampuan sistem untuk memprediksi rating dengan akurasi tinggi.
- Business Impact: Meningkatkan personalisasi rekomendasi berdasarkan preferensi pengguna serupa, yang dapat meningkatkan engagement dan retensi pengguna.

3. **Perbandingan performa kedua pendekatan:**
- Solusi: Berhasil membandingkan kedua pendekatan menggunakan metrik yang relevan.
- Dampak: Mengidentifikasi kelebihan masing-masing pendekatan: Content-Based unggul dalam relevansi konten, Collaborative unggul dalam diversitas dan personalisasi.
- Business Impact: Memberikan dasar untuk implementasi sistem hybrid yang mengkombinasikan kelebihan kedua pendekatan.

#### Dampak Solusi Statement

1. **Implementasi TF-IDF dan Cosine Similarity untuk Content-Based Filtering:**
- Dampak: Rekomendasi sangat relevan (Precision@10 = 100%) yang meningkatkan kepuasan pengguna.
- Business Value: 
  
    - Meningkatkan engagement pengguna baru yang belum memiliki history rating
  
    - Mendukung fitur "More Like This" yang meningkatkan discovery content

    - Meningkatkan waktu yang dihabiskan pengguna di platform

2. **Implementasi Neural Network untuk Collaborative Filtering:**
- Dampak: Prediksi rating akurat (RMSE = 0.7229) yang meningkatkan personalisasi.
- Business Value:
    
    - Meningkatkan retensi pengguna melalui rekomendasi yang dipersonalisasi

    - Mendukung fitur "Recommended For You" yang meningkatkan engagement

    - Meningkatkan konversi dan user satisfaction melalui rekomendasi yang tepat

1. **Evaluasi dengan metrik yang tepat:**
- Dampak: Pemahaman yang lebih baik tentang performa model dan area untuk improvement.
- Business Value:
    
    - Pengambilan keputusan berbasis data untuk pengembangan sistem
    
    - Optimalisasi berkelanjutan dari sistem rekomendasi
    
    - Alokasi sumber daya yang lebih efisien untuk pengembangan fitur

### Kesimpulan

Evaluasi menggunakan metrik yang tepat (Precision untuk Content-Based dan RMSE untuk Collaborative Filtering) menunjukkan bahwa kedua model sistem rekomendasi berhasil mencapai performa yang sangat baik:

1. Content-Based Filtering dengan Precision@10 = 100% sangat efektif dalam merekomendasikan film berdasarkan kesamaan genre, menjadikannya solusi ideal untuk mengatasi cold-start problem dan memberikan rekomendasi yang sangat relevan. Konsistensi sempurna pada 20 film uji menunjukkan robustness model ini.
   
2. Collaborative Filtering dengan RMSE = 0.7229 sangat akurat dalam memprediksi preferensi pengguna berdasarkan pola rating, menjadikannya solusi ideal untuk personalisasi dan discovery konten yang beragam. Model ini mampu merekomendasikan film-film berkualitas tinggi yang mungkin berada di luar genre favorit pengguna, memperluas pengalaman menonton mereka.

Visualisasi hasil evaluasi memperkuat kesimpulan ini, dengan bar chart Precision menunjukkan skor sempurna 1.0000 untuk Content-Based Filtering, dan bar chart RMSE menunjukkan gap yang kecil antara training (0.1523) dan validation (0.2093), mengindikasikan model Collaborative Filtering yang well-tuned.

Kedua model berhasil menjawab semua problem statement dan mencapai goals yang ditetapkan, dengan dampak bisnis yang signifikan dalam meningkatkan engagement, retensi, dan kepuasan pengguna. Untuk pengembangan lebih lanjut, pendekatan hybrid yang mengkombinasikan kedua model dapat memberikan hasil yang lebih optimal dengan menggabungkan kelebihan masing-masing pendekatan.