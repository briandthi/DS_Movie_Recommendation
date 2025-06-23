import numpy as np
import pandas as pd
from utils import create_affinity_features_optimized, find_similar_movies_from_model
from load_data import data


def predict_movie_rating(user_id, movie_identifier):
    """
    Prédit la note qu'un utilisateur donnerait à un film.

    Args:
        user_id (int): ID de l'utilisateur
        movie_identifier (str/int): Peut être movieId, tconst ou primaryTitle

    Returns:
        float: Note prédite
    """
    movie_df = data["movie_df"]
    user_features = data["user_features"]
    ratings = data["ratings"]
    model = data["model"]

    if isinstance(movie_identifier, int):
        movie = movie_df[movie_df["movieId"] == movie_identifier]
    elif movie_identifier.startswith("tt"):
        movie = movie_df[movie_df["tconst"] == movie_identifier]
    else:
        movie = movie_df[movie_df["primaryTitle"] == movie_identifier]

    if len(movie) == 0:
        raise ValueError("Film non trouvé")

    user = user_features[user_features["userId"] == user_id]
    if len(user) == 0:
        raise ValueError("Utilisateur non trouvé")

    user_ratings = ratings[ratings["userId"] == user_id]

    if movie.iloc[0]["movieId"] not in user_ratings["movieId"].values:
        new_rating = pd.DataFrame(
            {
                "userId": [user_id],
                "movieId": [movie.iloc[0]["movieId"]],
                "rating": [0],
                "timestamp": [0],
            }
        )
        user_ratings = pd.concat([user_ratings, new_rating], ignore_index=True)

    prep_df = user.merge(user_ratings, on="userId", how="inner").merge(
        movie_df, on="movieId", how="inner"
    )
    full_df = create_affinity_features_optimized(prep_df)
    full_df_filtered = full_df[full_df["movieId"] == movie.iloc[0]["movieId"]]
    pred_df = pd.DataFrame(
        {
            "averageRating": full_df_filtered["averageRating"],
            "mean_rating": full_df_filtered["mean_rating"],
            "rating_bias": full_df_filtered["rating_bias"],
            "director_affinity": full_df_filtered["director_affinity"],
            "actor1_affinity": full_df_filtered["actor1_affinity"],
            "actor2_affinity": full_df_filtered["actor2_affinity"],
            "writer_affinity": full_df_filtered["writer_affinity"],
        }
    )

    prediction = (model.predict(pred_df)[0] + 1) / 2

    return prediction


def display_similar_movies_via_model(movie_id, model, item_features_df, movies_df, 
                          item_scaler=None, top_n=10):
    """
    Find and display movies similar to a given movie based on model embeddings.
    
    Parameters:
    -----------
    movie_id : int
        ID of the reference movie
    model : tf.keras.Model
        Trained recommendation model
    item_features_df : DataFrame
        DataFrame with movieId and movie features
    movies_df : DataFrame
        DataFrame with movie metadata
    item_scaler : sklearn.preprocessing.StandardScaler, optional
        Scaler used for item features during training
    top_n : int, default=10
        Number of similar movies to return
    """
    # Get similar movies
    similar_movies = find_similar_movies_from_model(
        movie_id=movie_id,
        model=model,
        item_features_df=item_features_df,
        movies_df=movies_df,
        item_scaler=item_scaler,
        top_n=top_n
    )

    if similar_movies.empty:
        return

    # Get reference movie details
    ref_movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]

    # Determine which title to use for reference movie
    if 'primaryTitle' in ref_movie and pd.notna(ref_movie['primaryTitle']):
        ref_title = ref_movie['primaryTitle']
    elif 'title' in ref_movie and pd.notna(ref_movie['title']):
        ref_title = ref_movie['title']
    else:
        ref_title = f"Movie {movie_id}"

    # Display year if available
    year_str = f" ({ref_movie['startYear']})" if 'startYear' in ref_movie and pd.notna(ref_movie['startYear']) else ""

    # Display genres if available
    genres_str = f"\nGenres: {ref_movie['genres']}" if 'genres' in ref_movie and pd.notna(ref_movie['genres']) else ""

    # Display rating if available
    rating_str = f"\nRating: {ref_movie['averageRating']:.1f}" if 'averageRating' in ref_movie and pd.notna(ref_movie['averageRating']) else ""

    # Print header
    print(f"\n===== Movies Similar to: {ref_title}{year_str} =====")
    print(f"Movie ID: {movie_id}{genres_str}{rating_str}")
    print(f"Based on Neural Network Item Embeddings")
    print("\nTop Similar Movies:")

    # Print each similar movie
    for i, (_, movie) in enumerate(similar_movies.iterrows(), 1):
        # Determine which title to use
        if 'primaryTitle' in movie and pd.notna(movie['primaryTitle']):
            movie_title = movie['primaryTitle']
        elif 'title' in movie and pd.notna(movie['title']):
            movie_title = movie['title']
        else:
            movie_title = f"Movie {movie['movieId']}"

        # Display year if available
        year_str = f" ({movie['startYear']})" if 'startYear' in movie and pd.notna(movie['startYear']) else ""

        # Format similarity score
        similarity_str = f"{movie['similarity']:.4f}"

        print(f"{i}. {movie_title}{year_str} - Similarity: {similarity_str}")

        # Display genres if available
        if 'genres' in movie and pd.notna(movie['genres']):
            print(f"   Genres: {movie['genres']}")

        # Display rating if available
        if 'averageRating' in movie and pd.notna(movie['averageRating']):
            print(f"   Rating: {movie['averageRating']:.1f}")

        print()

    # Return the DataFrame for further analysis
    return similar_movies

# check_similar_movies_1408 = display_similar_movies_via_model(
#     movie_id=1408,
#     model=model_basis,
#     item_features_df=item_features_df,
#     movies_df=df_movieB,
#     top_n=10,
# )
#


def compare_ratings_with_recommendations_vectorized(
    user_id,
    model=data["model_DNN"],
    user_features_df=data["user_features_df"],
    item_features_df=data["item_features_df"],
    user_scaler=data["user_scaler"],
    item_scaler=data["movie_scaler"],
    y_scaler=data["scalerTarget"],
    ratings_df=data["ratings_df"],
    movies_df=data["movie_df"],
    top_n=10,
):
    """
    Compare a user's actual ratings with model recommendations using vectorized operations.

    This helps evaluate how well the model captures the user's preferences.

    Parameters:
    -----------
    user_id : int
        ID of the user
    model : tf.keras.Model
        Trained recommendation model
    user_features_df : DataFrame
        DataFrame with userId and user features
    item_features_df : DataFrame
        DataFrame with movieId and movie features
    user_scaler : sklearn.preprocessing.StandardScaler
        Scaler used to normalize user features during training
    item_scaler : sklearn.preprocessing.StandardScaler
        Scaler used to normalize movie features during training
    y_scaler : sklearn.preprocessing.StandardScaler, optional
        Scaler used to normalize target values during training
    ratings_df : DataFrame
        DataFrame containing user ratings
    movies_df : DataFrame
        DataFrame containing movie information
    top_n : int, default=10
        Number of recommendations to analyze
    """
    # Get the user's top-rated movies (vectorized)
    user_ratings = ratings_df[ratings_df["userId"] == user_id]

    if user_ratings.empty:
        print(f"User {user_id} has not rated any movies in the dataset.")
        return

    # Get highly rated movies (4+ stars)
    user_top_movies = user_ratings[user_ratings["rating"] >= 4.0].sort_values(
        "rating", ascending=False
    )

    # Merge with movie information - include primaryTitle if available
    movie_cols = ["movieId"]
    for col in ["title", "primaryTitle", "genres", "startYear", "averageRating"]:
        if col in movies_df.columns:
            movie_cols.append(col)

    user_top_movies = user_top_movies.merge(
        movies_df[movie_cols], on="movieId", how="left"
    )

    # Limit to top 10 for display
    user_top_movies = user_top_movies.head(10)

    # Get model recommendations (vectorized)
    # Get user features
    user_row = user_features_df[user_features_df["userId"] == user_id]
    user_features = user_row.drop("userId", axis=1).values

    # Scale user features
    user_features_scaled = user_scaler.transform(user_features)

    # Get movie IDs and features
    movie_ids = item_features_df["movieId"].values
    movie_features = item_features_df.drop("movieId", axis=1).values

    # Scale movie features
    movie_features_scaled = item_scaler.transform(movie_features)

    # Prepare for batch prediction
    user_features_batch = np.repeat(user_features_scaled, len(movie_ids), axis=0)

    # Predict scores
    scores_scaled = model.predict(
        [user_features_batch, movie_features_scaled], verbose=0
    )

    # Inverse transform scores if a target scaler was used
    if y_scaler is not None:
        scores = y_scaler.inverse_transform(scores_scaled)
    else:
        scores = scores_scaled

    # Create recommendations DataFrame
    recommendations = pd.DataFrame({"movieId": movie_ids, "score": scores.flatten()})

    # Exclude movies the user has already rated
    rated_movies = user_ratings["movieId"].values
    recommendations = recommendations[~recommendations["movieId"].isin(rated_movies)]

    # Sort by score and get top N
    recommendations = recommendations.sort_values("score", ascending=False).head(top_n)

    # Add movie metadata
    recommendations = recommendations.merge(
        movies_df[movie_cols], on="movieId", how="left"
    )

    # Display user's top-rated movies
    print("\n===== USER'S TOP-RATED MOVIES =====")
    if not user_top_movies.empty:
        for i, (_, movie) in enumerate(user_top_movies.iterrows(), 1):
            # Determine which title to use (primaryTitle preferred if available)
            if "primaryTitle" in movie and pd.notna(movie["primaryTitle"]):
                movie_title = movie["primaryTitle"]
            elif "title" in movie and pd.notna(movie["title"]):
                movie_title = movie["title"]
            else:
                movie_title = f"Movie {movie['movieId']}"

            # Display year if available
            year_str = (
                f" ({movie['startYear']})"
                if "startYear" in movie and pd.notna(movie["startYear"])
                else ""
            )

            print(f"{i}. {movie_title}{year_str} - Rating: {movie['rating']:.1f}")

            # Display genres if available
            if "genres" in movie and pd.notna(movie["genres"]):
                print(f"   Genres: {movie['genres']}")
    else:
        print("No highly-rated movies found for this user.")

    # Display model recommendations
    print("\n===== MODEL RECOMMENDATIONS =====")
    for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
        # Determine which title to use (primaryTitle preferred if available)
        if "primaryTitle" in movie and pd.notna(movie["primaryTitle"]):
            movie_title = movie["primaryTitle"]
        elif "title" in movie and pd.notna(movie["title"]):
            movie_title = movie["title"]
        else:
            movie_title = f"Movie {movie['movieId']}"

        # Display year if available
        year_str = (
            f" ({movie['startYear']})"
            if "startYear" in movie and pd.notna(movie["startYear"])
            else ""
        )

        print(f"{i}. {movie_title}{year_str} - Score: {movie['score']:.4f}")

        # Display genres if available
        if "genres" in movie and pd.notna(movie["genres"]):
            print(f"   Genres: {movie['genres']}")

    # Analyze genre preferences (vectorized)
    if "genres" in movies_df.columns:
        print("\n===== GENRE ANALYSIS =====")

        # Function to extract and count genres
        def extract_genres(df):
            # Combine all genres into a single list
            all_genres = []

            # Filter out NaN values and use vectorized string operations
            valid_genres = df["genres"].dropna()

            if not valid_genres.empty:
                # Split each genre string and flatten the result
                genre_lists = valid_genres.str.split("|")
                all_genres = [
                    genre
                    for sublist in genre_lists
                    for genre in sublist
                    if isinstance(sublist, list)
                ]

            # Count frequencies
            return pd.Series(all_genres).value_counts()

        # Get genre counts
        user_genre_counts = extract_genres(user_top_movies)
        rec_genre_counts = extract_genres(recommendations)

        print("Top genres in user's highly-rated movies:")
        print(user_genre_counts.head(5))

        print("\nTop genres in recommendations:")
        print(rec_genre_counts.head(5))

        # Calculate genre overlap (Jaccard similarity)
        if not user_genre_counts.empty and not rec_genre_counts.empty:
            user_genres_set = set(user_genre_counts.index)
            rec_genres_set = set(rec_genre_counts.index)

            overlap = user_genres_set.intersection(rec_genres_set)
            union = user_genres_set.union(rec_genres_set)

            jaccard_similarity = len(overlap) / len(union) if union else 0

            print(f"\nGenre overlap (Jaccard similarity): {jaccard_similarity:.2f}")
            print(f"Common genres: {', '.join(overlap)}")

    # Return the data for further analysis if needed
    return {"user_top_movies": user_top_movies, "recommendations": recommendations}
