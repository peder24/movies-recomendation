# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import os
import traceback

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Data Loading dan Understanding
ratings_df = pd.read_csv('ratings.csv')
movies_df = pd.read_csv('movies.csv')

print(f"Ratings dataset: {ratings_df.shape[0]} ratings from {ratings_df['userId'].nunique()} users on {ratings_df['movieId'].nunique()} movies")
print(f"Movies dataset: {movies_df.shape[0]} movies")

# Eksplorasi Struktur Data
print("\nMenampilkan 5 baris pertama dari dataset ratings:")
print(ratings_df.head())

print("\nMenampilkan 5 baris pertama dari dataset movies:")
print(movies_df.head())

# Analisis Informasi Dataset
print("\nInformasi dataset ratings:")
print(ratings_df.info())

print("\nStatistik deskriptif ratings:")
print(ratings_df.describe())

# Memeriksa duplikasi pada ratings.csv
duplicate_ratings = ratings_df.duplicated().sum()
print(f"\nJumlah duplikat di ratings: {duplicate_ratings}")

# Memeriksa outliers pada ratings.csv
Q1 = ratings_df['rating'].quantile(0.25)
Q3 = ratings_df['rating'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = ratings_df[(ratings_df['rating'] < lower_bound) | (ratings_df['rating'] > upper_bound)]
print(f"\nJumlah outlier rating (IQR method): {outliers.shape[0]}")

print("Dataset movies.csv:")
movies_df.info()

# Memeriksa duplikasi pada movies.csv
duplicate_movies = movies_df.duplicated(subset='movieId').sum()
print(f"Jumlah duplikat movieId di movies: {duplicate_movies}")

# Univariate Exploratory Data Analysis
plt.figure(figsize=(10, 6))
sns.histplot(ratings_df['rating'], bins=10, kde=True)
plt.title('Distribusi Rating Film')
plt.xlabel('Rating')
plt.ylabel('Jumlah Rating')
plt.tight_layout()
plt.show()

movie_rating_counts = ratings_df.groupby('movieId')['rating'].count().reset_index(name='count')
plt.figure(figsize=(12, 6))
sns.histplot(movie_rating_counts['count'], bins=30)
plt.title('Distribusi Jumlah Rating per Film')
plt.xlabel('Jumlah Rating')
plt.ylabel('Jumlah Film')
plt.tight_layout()
plt.show()

all_genres = []
for genres in movies_df['genres']:
    all_genres.extend(genres.split('|'))
genre_counts = pd.Series(all_genres).value_counts()

plt.figure(figsize=(14, 8))
sns.barplot(x=genre_counts.values[:15], y=genre_counts.index[:15])
plt.title('15 Genre Film Terpopuler')
plt.xlabel('Jumlah Film')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()

# Data Preprocessing
ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')

# Menangani Outliers pada ratings.csv
valid_ratings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
invalid_ratings = ratings_df[~ratings_df['rating'].isin(valid_ratings)]

if len(invalid_ratings) > 0:
    print(f"Ditemukan {len(invalid_ratings)} rating tidak valid")
    ratings_df_cleaned = ratings_df[ratings_df['rating'].isin(valid_ratings)]
    print("Strategi: Hapus rating yang tidak sesuai skala MovieLens")
else:
    print("Semua rating sudah valid untuk skala MovieLens")
    ratings_df_cleaned = ratings_df[(ratings_df['rating'] >= lower_bound) & 
                                   (ratings_df['rating'] <= upper_bound)]
    print("Strategi: Hapus outliers statistik dengan metode IQR")

print(f"Data sebelum: {len(ratings_df)} rating")
print(f"Data sesudah: {len(ratings_df_cleaned)} rating")
print(f"Outliers yang dihapus: {len(ratings_df) - len(ratings_df_cleaned)} rating")

ratings_df = ratings_df_cleaned

# Filtering Data Berdasarkan Popularitas
min_movie_ratings = 3  
min_user_ratings = 3   

movie_count = ratings_df.groupby('movieId')['rating'].count()
user_count = ratings_df.groupby('userId')['rating'].count()

popular_movies = movie_count[movie_count >= min_movie_ratings].index
active_users = user_count[user_count >= min_user_ratings].index

filtered_ratings = ratings_df[
    (ratings_df['movieId'].isin(popular_movies)) & 
    (ratings_df['userId'].isin(active_users))
]

print(f"Jumlah rating sebelum filtering: {ratings_df.shape[0]}")
print(f"Jumlah rating setelah filtering: {filtered_ratings.shape[0]}")
print(f"Jumlah film setelah filtering: {filtered_ratings['movieId'].nunique()}")
print(f"Jumlah pengguna setelah filtering: {filtered_ratings['userId'].nunique()}")

# Penggabungan Data dan Pengecekan Missing Values
filtered_movies = movies_df[movies_df['movieId'].isin(filtered_ratings['movieId'].unique())]
movie_with_rating = pd.merge(filtered_ratings, filtered_movies, on='movieId')

print(movie_with_rating.isnull().sum())

# Data Preparation
# Persiapan Data untuk Content-Based Filtering
movie_with_genre = filtered_movies.copy()
movie_with_genre = movie_with_genre.dropna(subset=['genres'])

print("Genre yang unik dalam dataset:")
unique_genres = set()
for genres in movie_with_genre['genres']:
    unique_genres.update(genres.split('|'))
print(unique_genres)

# Pembuatan Dataset untuk Content-Based Filtering
movies_data = pd.DataFrame({
    'movieId': movie_with_genre['movieId'],
    'title': movie_with_genre['title'],
    'genres': movie_with_genre['genres']
})

print("\nData film untuk content-based filtering:")
print(movies_data.head())
print(f"Jumlah film: {len(movies_data)}")

# Model Development dengan Content Based Filtering
# Ekstraksi Fitur dengan TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies_data['genres'])

print("Ukuran matrix TF-IDF:", tfidf_matrix.shape)

# Perhitungan Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix)

cosine_sim_df = pd.DataFrame(
    cosine_sim, 
    index=movies_data['title'],
    columns=movies_data['title']
)

# Implementasi Fungsi Rekomendasi Content-Based
def movie_recommendations(title, similarity_df=cosine_sim_df, movies=movies_data, k=10):
    if title not in similarity_df.index:
        print(f"Film '{title}' tidak ditemukan dalam dataset")
        return pd.DataFrame()
    
    try:
        indices = similarity_df.loc[title].values.argsort()[-k-1:][::-1]
        similar_movies = similarity_df.index[indices].tolist()
        
        if title in similar_movies:
            similar_movies.remove(title)
        
        recommended_movies = []
        for movie in similar_movies[:k]:
            movie_info = movies[movies['title'] == movie]
            if not movie_info.empty:
                movie_info = movie_info.iloc[0]
                recommended_movies.append({
                    'movieId': movie_info['movieId'],
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'similarity_score': similarity_df.loc[title, movie]
                })
        
        return pd.DataFrame(recommended_movies)
    except Exception as e:
        print(f"Error in movie_recommendations for movie '{title}': {e}")
        traceback.print_exc()
        return pd.DataFrame()

# Testing Content-Based Filtering
sample_movie = movies_data['title'].iloc[0]
print(f"\nFilm referensi: {sample_movie}")
print(f"Genre: {movies_data[movies_data['title'] == sample_movie].iloc[0]['genres']}")

print("\nRekomendasi film berdasarkan Content-Based Filtering:")
content_recommendations = movie_recommendations(sample_movie)
print(content_recommendations[['title', 'genres', 'similarity_score']])

# Model Development dengan Collaborative Filtering
# Encoding User dan Movie ID
user_ids = filtered_ratings['userId'].unique().tolist()
movie_ids = filtered_ratings['movieId'].unique().tolist()

user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}

user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
movie_encoded_to_movie = {i: x for i, x in enumerate(movie_ids)}

# Preprocessing Data untuk Neural Network
ratings_data = filtered_ratings.copy()
ratings_data['user'] = ratings_data['userId'].map(user_to_user_encoded)
ratings_data['movie'] = ratings_data['movieId'].map(movie_to_movie_encoded)

min_rating = ratings_data['rating'].min()
max_rating = ratings_data['rating'].max()
ratings_data['normalized_rating'] = (ratings_data['rating'] - min_rating) / (max_rating - min_rating)

ratings_data = ratings_data.sample(frac=1, random_state=42)

x = ratings_data[['user', 'movie']].values
y = ratings_data['normalized_rating'].values

train_indices = int(0.8 * len(ratings_data))
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

# Implementasi Model RecommenderNet
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-5),
            name="user_embedding"
        )
        self.user_bias = layers.Embedding(
            num_users, 
            1, 
            embeddings_initializer="zeros", 
            name="user_bias"
        )
        
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-5),
            name="movie_embedding"
        )
        self.movie_bias = layers.Embedding(
            num_movies, 
            1, 
            embeddings_initializer="zeros", 
            name="movie_bias"
        )

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        
        dot_user_movie = tf.reduce_sum(user_vector * movie_vector, axis=1, keepdims=True)
        
        x = dot_user_movie + user_bias + movie_bias
        
        return tf.nn.sigmoid(x)

# Training Model Collaborative Filtering
num_users = len(user_ids)
num_movies = len(movie_ids)
embedding_size = 30  

model = RecommenderNet(num_users, num_movies, embedding_size)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_root_mean_squared_error',
    patience=5,
    restore_best_weights=True
)

print("\nTraining Collaborative Filtering model...")
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=32,
    epochs=50,  
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Visualisasi Performa Training
plt.figure(figsize=(10, 6))
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Model Metrics')
plt.ylabel('Root Mean Squared Error')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.show()

# Implementasi Fungsi Rekomendasi Collaborative Filtering
def collaborative_recommendations(user_id, model=model, movies_df=filtered_movies, 
                                 ratings_df=filtered_ratings,
                                 user_to_user_encoded=user_to_user_encoded,
                                 movie_encoded_to_movie=movie_encoded_to_movie,
                                 k=10):
    user_encoded = user_to_user_encoded.get(user_id)
    if user_encoded is None:
        print(f"User ID {user_id} tidak ditemukan")
        return pd.DataFrame()
    
    try:
        movies_watched_by_user = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
        
        movies_not_watched = movies_df[~movies_df['movieId'].isin(movies_watched_by_user)]['movieId'].values
        
        movies_not_watched = list(
            set(movies_not_watched).intersection(set(movie_to_movie_encoded.keys()))
        )
        
        if len(movies_not_watched) == 0:
            print(f"Tidak ada film yang belum ditonton oleh user {user_id}")
            return pd.DataFrame()
        
        movies_not_watched_encoded = [movie_to_movie_encoded.get(x) for x in movies_not_watched]
        user_movie_array = np.array([[user_encoded, movie_encoded] for movie_encoded in movies_not_watched_encoded])
        
        ratings = model.predict(user_movie_array).flatten()
        
        ratings = min_rating + ratings * (max_rating - min_rating)
        
        top_ratings_indices = ratings.argsort()[-k:][::-1]
        recommended_movie_ids = [movies_not_watched[x] for x in top_ratings_indices]
        
        recommended_movies = []
        for movie_id in recommended_movie_ids:
            movie_info = movies_df[movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                movie_info = movie_info.iloc[0]
                recommended_movies.append({
                    'movieId': movie_id,
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'predicted_rating': ratings[np.where(np.array(movies_not_watched) == movie_id)[0][0]]
                })
        
        return pd.DataFrame(recommended_movies)
    except Exception as e:
        print(f"Error in collaborative_recommendations for user {user_id}: {e}")
        traceback.print_exc()
        return pd.DataFrame()

# Testing Collaborative Filtering
if len(user_ids) > 0:
    sample_user_id = user_ids[0]
    print(f"\nRekomendasi film berdasarkan Collaborative Filtering untuk user_id = {sample_user_id}:")
    
    movies_watched = filtered_ratings[filtered_ratings['userId'] == sample_user_id]
    movies_watched_df = pd.merge(movies_watched, filtered_movies, on='movieId')
    movies_watched_df = movies_watched_df.sort_values('rating', ascending=False)
    
    print("\nFilm yang sudah ditonton (dengan rating tertinggi):")
    for _, row in movies_watched_df.head(5).iterrows():
        print(f"{row['title']} - Rating: {row['rating']} - Genres: {row['genres']}")
    
    collaborative_recs = collaborative_recommendations(sample_user_id)
    if not collaborative_recs.empty:
        print("\nRekomendasi film berdasarkan Collaborative Filtering:")
        print(collaborative_recs[['title', 'genres', 'predicted_rating']])

# Model Evaluation
# Implementasi Precision untuk Content-Based Filtering
def get_movie_genres(movie_title, movies_df=movies_data):
    movie_info = movies_df[movies_df['title'] == movie_title]
    if not movie_info.empty:
        return movie_info.iloc[0]['genres']
    return ""

def evaluate_precision_content_based(test_movies, k=10):
    precision_scores = []
    
    print(f"Evaluating Precision@{k} for Content-Based Filtering...")
    print("="*60)
    
    for i, movie_title in enumerate(test_movies):
        try:
            reference_genres = get_movie_genres(movie_title)
            
            if not reference_genres:
                print(f"Film '{movie_title}' tidak ditemukan, skip...")
                continue
                
            recommendations = movie_recommendations(movie_title, k=k)
            
            if recommendations.empty:
                print(f"Tidak ada rekomendasi untuk '{movie_title}', skip...")
                continue
            
            relevant_count = 0
            ref_genres = set(reference_genres.split('|'))
            
            for _, rec in recommendations.iterrows():
                rec_genres = set(rec['genres'].split('|'))
                
                if len(rec_genres.intersection(ref_genres)) > 0:
                    relevant_count += 1
            
            precision = relevant_count / k
            precision_scores.append(precision)
            
            print(f"Film {i+1}: '{movie_title}' - Precision@{k}: {precision:.4f} ({relevant_count}/{k} relevan)")
            
        except Exception as e:
            print(f"Error evaluating '{movie_title}': {e}")
            continue
    
    avg_precision = np.mean(precision_scores) if precision_scores else 0
    
    print("="*60)
    print(f"Average Precision@{k}: {avg_precision:.4f}")
    print(f"Jumlah film yang dievaluasi: {len(precision_scores)}")
    
    return avg_precision

# Implementasi RMSE untuk Collaborative Filtering
def evaluate_rmse_collaborative_filtering():
    train_rmse = history.history['root_mean_squared_error'][-1]
    val_rmse = history.history['val_root_mean_squared_error'][-1]
    
    test_predictions = model.predict(x_val).flatten()
    
    y_val_original = min_rating + y_val * (max_rating - min_rating)
    test_predictions_original = min_rating + test_predictions * (max_rating - min_rating)
    
    rmse_original_scale = np.sqrt(np.mean((y_val_original - test_predictions_original) ** 2))
    
    print("RMSE Evaluation for Collaborative Filtering")
    print("="*50)
    print(f"Training RMSE (normalized): {train_rmse:.4f}")
    print(f"Validation RMSE (normalized): {val_rmse:.4f}")
    print(f"Test RMSE (original scale {min_rating}-{max_rating}): {rmse_original_scale:.4f}")
    print(f"Rating scale: {min_rating} - {max_rating}")
    
    rating_range = max_rating - min_rating
    rmse_percentage = (rmse_original_scale / rating_range) * 100
    print(f"RMSE as percentage of rating range: {rmse_percentage:.2f}%")
    
    if rmse_original_scale < 1.0:
        print("RMSE < 1.0: Model memiliki akurasi prediksi yang baik")
    elif rmse_original_scale < 1.5:
        print("RMSE 1.0-1.5: Model memiliki akurasi prediksi yang cukup")
    else:
        print("RMSE > 1.5: Model memerlukan perbaikan")
    
    return {
        'train_rmse_normalized': train_rmse,
        'val_rmse_normalized': val_rmse,
        'test_rmse_original': rmse_original_scale,
        'rmse_percentage': rmse_percentage
    }

# Evaluasi Performa Model dengan Metrik yang Tepat
print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

print("\n1. CONTENT-BASED FILTERING EVALUATION")
print("-" * 50)

sample_movies_for_precision = movies_data['title'].sample(n=min(20, len(movies_data)), random_state=42).tolist()

content_based_precision = evaluate_precision_content_based(sample_movies_for_precision, k=10)

print("\n2. COLLABORATIVE FILTERING EVALUATION")
print("-" * 50)

collaborative_rmse_results = evaluate_rmse_collaborative_filtering()

# Visualisasi Hasil Evaluasi
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.bar(['Content-Based\nPrecision@10'], [content_based_precision], color='skyblue', alpha=0.7)
ax1.set_ylabel('Precision Score')
ax1.set_title('Content-Based Filtering\nPrecision Evaluation')
ax1.set_ylim(0, 1.1)
ax1.grid(True, alpha=0.3)
for i, v in enumerate([content_based_precision]):
    ax1.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

rmse_values = [
    collaborative_rmse_results['train_rmse_normalized'],
    collaborative_rmse_results['val_rmse_normalized']
]
rmse_labels = ['Training\nRMSE', 'Validation\nRMSE']

ax2.bar(rmse_labels, rmse_values, color=['lightcoral', 'lightgreen'], alpha=0.7)
ax2.set_ylabel('RMSE (Normalized)')
ax2.set_title('Collaborative Filtering\nRMSE Evaluation')
ax2.grid(True, alpha=0.3)
for i, v in enumerate(rmse_values):
    ax2.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Hasil Rekomendasi Final
print("\n" + "="*80)
print("TOP 10 REKOMENDASI DARI SETIAP MODEL")
print("="*80)

if len(user_ids) > 0:
    sample_user_id = user_ids[0]
    movies_watched = filtered_ratings[filtered_ratings['userId'] == sample_user_id]
    
    if len(movies_watched) > 0:
        favorite_movie = movies_watched.sort_values('rating', ascending=False).iloc[0]
        favorite_movie_id = favorite_movie['movieId']
        favorite_movie_info = filtered_movies[filtered_movies['movieId'] == favorite_movie_id].iloc[0]
        
        print(f"\nPengguna: User ID {sample_user_id}")
        print(f"Film Referensi: {favorite_movie_info['title']} (Rating: {favorite_movie['rating']})")
        print(f"Genre: {favorite_movie_info['genres']}")
        
        cb_recs = movie_recommendations(favorite_movie_info['title'], k=10)
        cf_recs = collaborative_recommendations(sample_user_id, k=10)
        
        print("\n1. Top 10 Rekomendasi Content-Based Filtering:")
        print("-" * 80)
        print(f"{'No':3} {'Judul Film':50} {'Genre':25} {'Similarity Score':15}")
        print("-" * 80)
        for i, (_, row) in enumerate(cb_recs.iterrows(), 1):
            print(f"{i:3} {row['title'][:47]:50} {row['genres'][:22]:25} {row['similarity_score']:.4f}")
        
        print("\n2. Top 10 Rekomendasi Collaborative Filtering:")
        print("-" * 80)
        print(f"{'No':3} {'Judul Film':50} {'Genre':25} {'Predicted Rating':15}")
        print("-" * 80)
        for i, (_, row) in enumerate(cf_recs.iterrows(), 1):
            print(f"{i:3} {row['title'][:47]:50} {row['genres'][:22]:25} {row['predicted_rating']:.2f}")

# Kesimpulan dan Evaluasi Final
print("\n" + "="*80)
print("KESIMPULAN DAN EVALUASI FINAL")
print("="*80)

print("\n1. SISTEM REKOMENDASI YANG DIBANGUN:")
print("   Content-Based Filtering: Menggunakan kesamaan genre dengan TF-IDF dan Cosine Similarity")
print("   Collaborative Filtering: Menggunakan Neural Network Matrix Factorization")

print("\n2. METRIK EVALUASI YANG DIGUNAKAN:")
print(f"   Content-Based Filtering - Precision@10: {content_based_precision:.4f} ({content_based_precision*100:.1f}%)")
print(f"     - Mengukur relevansi rekomendasi berdasarkan kesamaan genre")
print(f"     - Tidak bias terhadap popularitas item")
print(f"   Collaborative Filtering - RMSE: {collaborative_rmse_results['test_rmse_original']:.4f}")
print(f"     - Mengukur akurasi prediksi rating pada skala {min_rating}-{max_rating}")
print(f"     - Error sebesar {collaborative_rmse_results['rmse_percentage']:.1f}% dari range rating")

print("\n3. INTERPRETASI HASIL:")
if content_based_precision >= 0.8:
    print("   Content-Based: Precision tinggi menunjukkan rekomendasi yang sangat relevan")
elif content_based_precision >= 0.6:
    print("   Content-Based: Precision cukup baik untuk sistem rekomendasi")
else:
    print("   Content-Based: Precision rendah, perlu perbaikan algoritma")

if collaborative_rmse_results['test_rmse_original'] < 1.0:
    print("   Collaborative: RMSE rendah menunjukkan prediksi rating yang akurat")
elif collaborative_rmse_results['test_rmse_original'] < 1.5:
    print("   Collaborative: RMSE cukup untuk prediksi rating")
else:
    print("   Collaborative: RMSE tinggi, model perlu diperbaiki")