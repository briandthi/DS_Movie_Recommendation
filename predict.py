import pandas as pd
from utils import create_affinity_features_optimized
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
