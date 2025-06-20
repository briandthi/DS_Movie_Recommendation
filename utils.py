import numpy as np
import pandas as pd
import tensorflow as tf


def create_affinity_features_optimized(df):
    total_rows = len(df)

    director_count = df.groupby(["userId", "director"]).size()
    actor1_count = df.groupby(["userId", "actor1"]).size()
    actor2_count = df.groupby(["userId", "actor2"]).size()
    writer_count = df.groupby(["userId", "writer"]).size()

    df["director_count"] = df.apply(
        lambda x: director_count.get((x["userId"], x["director"]), 0), axis=1
    )
    df["actor1_count"] = df.apply(
        lambda x: actor1_count.get((x["userId"], x["actor1"]), 0), axis=1
    )
    df["actor2_count"] = df.apply(
        lambda x: actor2_count.get((x["userId"], x["actor2"]), 0), axis=1
    )
    df["writer_count"] = df.apply(
        lambda x: writer_count.get((x["userId"], x["writer"]), 0), axis=1
    )

    cols = {
        "director": "directorAvgRating",
        "actor1": "actor1AvgRating",
        "actor2": "actor2AvgRating",
        "writer": "writerAvgRating",
    }

    for person_type, avg_col in cols.items():
        df[f"{person_type}_affinity"] = 0

        for idx, row in df.iterrows():
            if row[f"{person_type}_count"] > 1:
                other_movies = df[
                    (df["userId"] == row["userId"])
                    & (df[person_type] == row[person_type])
                    & (df.index != idx)
                ]

                if len(other_movies) > 0:
                    user_person_mean = other_movies["rating"].mean()
                    df.at[idx, f"{person_type}_affinity"] = (
                        user_person_mean - row[avg_col]
                    )

    return df


def create_recommendation_triplets_robust_extended(
    df_per_user, df_per_movie, mv_ratings
):
    """
    A more robust version that handles potential column name conflicts.
    Creates aligned triplets of (user features, movie features, rating) for recommendation system training
    using vectorized operations for improved performance.

    Parameters:
    -----------
    df_per_user : DataFrame
        DataFrame with userId as index, containing user features
    df_per_movie : DataFrame
        DataFrame with movieId column and movie features
    mv_ratings : DataFrame
        DataFrame with userId, movieId, and rating columns

    Returns:
    --------
    user_features : ndarray
        Array of user features for each rating
    movie_features : ndarray
        Array of movie features for each rating
    ratings : ndarray
        Array of ratings
    user_features_df : DataFrame
        DataFrame with userId and user features
    item_features_df : DataFrame
        DataFrame with movieId and movie features
    movie_ids_list : ndarray
        Array of movie IDs in the same order as in item_features_df
    """
    # Verify input dataframes
    if not isinstance(df_per_user.index, pd.Index):
        raise ValueError("df_per_user must have userId as index")
    if "movieId" not in df_per_movie.columns:
        raise ValueError("df_per_movie must contain a movieId column")
    if not all(col in mv_ratings.columns for col in ["userId", "movieId", "rating"]):
        raise ValueError("mv_ratings must contain userId, movieId, and rating columns")

    # Print column information for debugging
    print("Columns in df_per_user:")
    print(df_per_user.columns.tolist())
    print("\nColumns in df_per_movie:")
    print(df_per_movie.columns.tolist())

    # Select movie features: 'averageRating' and columns starting with capital letter
    movie_feature_cols = []

    if "averageRating" in df_per_movie.columns:
        movie_feature_cols.append("averageRating")

    # Add columns that start with capital letter
    capital_cols = [
        col for col in df_per_movie.columns if col[0].isupper() and col != "movieId"
    ]

    if capital_cols:
        movie_feature_cols.extend(capital_cols)

    # If no movie features were found, raise an error
    if not movie_feature_cols:
        print("Warning: No valid movie features found. Using only movieId.")
        movie_feature_cols = []  # Empty list, we'll just use movieId for joining

    print(f"Using movie features: {movie_feature_cols}")

    # Get valid users and movies
    valid_users = set(df_per_user.index)
    valid_movies = set(df_per_movie["movieId"])

    # Filter ratings to include only valid users and movies
    print("Filtering ratings...")
    valid_ratings = mv_ratings[
        mv_ratings["userId"].isin(valid_users)
        & mv_ratings["movieId"].isin(valid_movies)
    ].copy()

    print(f"Processing {len(valid_ratings)} valid ratings")

    # Rename user columns to avoid conflicts
    user_cols_original = list(df_per_user.columns)
    user_cols_renamed = [f"user_{col}" for col in user_cols_original]

    # Create a copy with renamed columns
    df_per_user_renamed = df_per_user.copy()
    df_per_user_renamed.columns = user_cols_renamed

    # Reset index to make userId a column for merging
    user_features_df = df_per_user_renamed.reset_index()

    # Merge user features with ratings
    print("Merging user features...")
    merged_df = pd.merge(valid_ratings, user_features_df, on="userId", how="left")

    # If we have movie features, merge them too
    if movie_feature_cols:
        # Rename movie columns to avoid conflicts
        movie_cols_renamed = [f"movie_{col}" for col in movie_feature_cols]

        # Create a copy of movie features with renamed columns
        movie_features_df = df_per_movie[["movieId"] + movie_feature_cols].copy()
        movie_features_df.columns = ["movieId"] + movie_cols_renamed

        # Merge movie features
        print("Merging movie features...")
        final_df = pd.merge(merged_df, movie_features_df, on="movieId", how="left")

        # Extract movie features with renamed columns
        movie_features = final_df[movie_cols_renamed].values
    else:
        final_df = merged_df
        # Create a dummy movie feature (just movieId)
        movie_features = final_df[["movieId"]].values

    # Extract user features with renamed columns
    user_features = final_df[user_cols_renamed].values

    # Extract ratings
    ratings = final_df["rating"].values

    # Convert to float32 to save memory
    user_features = user_features.astype(np.float32)
    movie_features = movie_features.astype(np.float32)
    ratings = ratings.astype(np.float32)

    print(
        f"Final shapes - User features: {user_features.shape}, "
        f"Movie features: {movie_features.shape}, Ratings: {ratings.shape}"
    )

    # Create DataFrames for recommendations
    # 1. User features DataFrame (with original column names)
    user_features_df_for_recs = df_per_user.reset_index().copy()

    # 2. Item features DataFrame
    item_features_df_for_recs = df_per_movie[["movieId"] + movie_feature_cols].copy()

    # 3. Movie IDs list
    movie_ids_list = item_features_df_for_recs["movieId"].values

    return (
        user_features,
        movie_features,
        ratings,
        user_features_df_for_recs,
        item_features_df_for_recs,
        movie_ids_list,
    )


def calculate_bayesian_genre_ratings_vectorized(ratings_df, movies_df, c_factor=1.0):
    # 1. Merge and explode genres
    merged_df = pd.merge(ratings_df, movies_df, on="movieId")
    merged_df["genres"] = merged_df["genres"].str.split(",")
    exploded_df = merged_df.explode("genres")

    # 2. Calculate global mean
    global_mean = ratings_df["rating"].mean()

    # 3. Calculate genre means and counts
    genre_stats = (
        exploded_df.groupby("genres")["rating"].agg(["mean", "count"]).reset_index()
    )
    genre_means = dict(zip(genre_stats["genres"], genre_stats["mean"]))

    # 4. Create user-genre count and sum matrices
    user_genre_counts = (
        exploded_df.groupby(["userId", "genres"]).size().unstack(fill_value=0)
    )
    user_genre_sums = (
        exploded_df.groupby(["userId", "genres"])["rating"].sum().unstack(fill_value=0)
    )

    # 5. Calculate raw averages (handling division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_avgs = user_genre_sums / user_genre_counts
        raw_avgs = raw_avgs.fillna(global_mean)

    # 6. Calculate C dynamically for each genre
    genre_c_values = {}
    for genre in genre_stats["genres"]:
        # Use the average number of ratings per user for this genre as C
        genre_user_counts = user_genre_counts[genre][user_genre_counts[genre] > 0]
        if len(genre_user_counts) > 0:
            # Multiply by c_factor to allow tuning
            genre_c_values[genre] = genre_user_counts.mean() * c_factor
        else:
            genre_c_values[genre] = 5 * c_factor  # Default if no data

    # 7. Apply Bayesian formula vectorized with dynamic C
    bayesian_avgs = pd.DataFrame(index=raw_avgs.index, columns=raw_avgs.columns)

    for genre in bayesian_avgs.columns:
        genre_mean = genre_means.get(genre, global_mean)
        counts = user_genre_counts[genre]
        means = raw_avgs[genre]

        # Get the C value for this genre
        c_value = genre_c_values.get(genre, 5 * c_factor)

        # Vectorized Bayesian formula with dynamic C
        bayesian_avgs[genre] = (c_value * genre_mean + counts * means) / (
            c_value + counts
        )

    # 8. Create results dataframe for reference
    results = []
    for user_id in bayesian_avgs.index:
        for genre in bayesian_avgs.columns:
            c_value = genre_c_values.get(genre, 5 * c_factor)
            results.append(
                {
                    "userId": user_id,
                    "genres": genre,
                    "raw_avg": raw_avgs.loc[user_id, genre],
                    "rating_count": user_genre_counts.loc[user_id, genre],
                    "c_value": c_value,
                    "bayesian_avg": bayesian_avgs.loc[user_id, genre],
                }
            )

    results_df = pd.DataFrame(results)

    return results_df, bayesian_avgs, genre_c_values


def create_model_2NN_reco_B(
    num_outputs=64,
    num_item_features=None,
    num_user_features=None,
    dropout_rate=0.2,
    l2_reg=0.0,
):
    """
    Creates a dual-network model with dropout regularization and dot product output.

    Parameters:
    - num_outputs: Dimension of the embedding space
    - num_user_features: Number of user features
    - num_item_features: Number of item features
    - dropout_rate: Rate for dropout layers (0 to 1)
    - l2_reg: L2 regularization factor (optional)

    Returns:
    - A Keras Model
    """
    tf.random.set_seed(1)

    # Optional L2 regularizer
    regularizer = None
    if l2_reg > 0:
        regularizer = tf.keras.regularizers.l2(l2_reg)

    # Create the user network as a Sequential model
    user_model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(num_user_features,)),
            tf.keras.layers.Dense(
                256, activation="relu", kernel_regularizer=regularizer
            ),
            tf.keras.layers.Dropout(
                dropout_rate
            ),  # Add dropout after first dense layer
            tf.keras.layers.Dense(
                128, activation="relu", kernel_regularizer=regularizer
            ),
            tf.keras.layers.Dropout(
                dropout_rate
            ),  # Add dropout after second dense layer
            tf.keras.layers.Dense(
                num_outputs, activation="linear", kernel_regularizer=regularizer
            ),
        ]
    )

    # Create the item network as a Sequential model
    item_model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(num_item_features,)),
            tf.keras.layers.Dense(
                256, activation="relu", kernel_regularizer=regularizer
            ),
            tf.keras.layers.Dropout(
                dropout_rate
            ),  # Add dropout after first dense layer
            tf.keras.layers.Dense(
                128, activation="relu", kernel_regularizer=regularizer
            ),
            tf.keras.layers.Dropout(
                dropout_rate
            ),  # Add dropout after second dense layer
            tf.keras.layers.Dense(
                num_outputs, activation="linear", kernel_regularizer=regularizer
            ),
        ]
    )

    # Create input layers for the functional API
    input_user = tf.keras.layers.Input(shape=(num_user_features,))
    input_item = tf.keras.layers.Input(shape=(num_item_features,))

    # Get embeddings from each network
    user_embedding = user_model(input_user)
    item_embedding = item_model(input_item)

    # Normalize embeddings
    user_embedding_norm = tf.keras.layers.Lambda(
        lambda x: tf.linalg.l2_normalize(x, axis=1), output_shape=(num_outputs,)
    )(user_embedding)

    item_embedding_norm = tf.keras.layers.Lambda(
        lambda x: tf.linalg.l2_normalize(x, axis=1), output_shape=(num_outputs,)
    )(item_embedding)

    # Compute dot product
    dot_product = tf.keras.layers.Dot(axes=1)(
        [user_embedding_norm, item_embedding_norm]
    )

    # Create the full model
    model = tf.keras.Model(inputs=[input_user, input_item], outputs=dot_product)

    return model


def find_similar_movies_from_model(
    movie_id, model, item_features_df, movies_df, item_scaler=None, top_n=10
):
    """
    Find movies similar to a given movie based on item embeddings from your model.

    This function is specifically designed for models created with create_model_2NN_reco.

    Parameters:
    -----------
    movie_id : int
        ID of the reference movie
    model : tf.keras.Model
        Trained recommendation model from create_model_2NN_reco
    item_features_df : DataFrame
        DataFrame with movieId and movie features
    movies_df : DataFrame
        DataFrame with movie metadata
    item_scaler : sklearn.preprocessing.StandardScaler, optional
        Scaler used for item features during training
    top_n : int, default=10
        Number of similar movies to return

    Returns:
    --------
    similar_movies : DataFrame
        DataFrame with similar movies and their similarity scores
    """

    # Check if movie exists
    if movie_id not in item_features_df["movieId"].values:
        print(f"Movie ID {movie_id} not found in the dataset.")
        return pd.DataFrame()

    # Create a model that outputs the normalized item embeddings
    item_input = model.inputs[1]  # The second input is for items

    # Find the Lambda layer that normalizes item embeddings
    # In your model, there are two Lambda layers - the second one is for items
    lambda_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Lambda)
    ]

    if len(lambda_layers) >= 2:
        # The second Lambda layer should be the item embedding normalization
        item_embedding_norm = lambda_layers[1]

        # Create a model that outputs the normalized item embeddings
        embedding_model = tf.keras.Model(
            inputs=item_input, outputs=item_embedding_norm.output
        )

        print(
            f"Created embedding model with output shape: {embedding_model.output_shape}"
        )
    else:
        print("Could not find Lambda layers. Creating a custom model.")

        # Extract the item model (the second Sequential model)
        sequential_models = [
            layer for layer in model.layers if isinstance(layer, tf.keras.Sequential)
        ]

        if len(sequential_models) >= 2:
            item_model = sequential_models[1]

            # Create a model that outputs the item embeddings (before normalization)
            embedding_model = item_model

            print(f"Using item model with output shape: {embedding_model.output_shape}")
        else:
            print("Could not find Sequential models. Using a fallback approach.")

            # Create a new model that takes the item input and outputs the last layer before the dot product
            dense_layers = [
                layer
                for layer in model.layers
                if isinstance(layer, tf.keras.layers.Dense)
            ]

            if dense_layers:
                last_dense = dense_layers[-1]
                embedding_model = tf.keras.Model(
                    inputs=item_input, outputs=last_dense.output
                )
                print(
                    f"Using last dense layer with output shape: {embedding_model.output_shape}"
                )
            else:
                print("Could not create an embedding model. Cannot proceed.")
                return pd.DataFrame()

    # Get all movie features (excluding movieId)
    all_movie_features = item_features_df.drop("movieId", axis=1).values
    all_movie_ids = item_features_df["movieId"].values

    # Scale features if needed
    if item_scaler is not None:
        all_movie_features = item_scaler.transform(all_movie_features)

    # Get embeddings for all movies
    all_embeddings = embedding_model.predict(all_movie_features, verbose=0)

    # Normalize embeddings if we're using the raw item model output
    if not any(
        isinstance(layer, tf.keras.layers.Lambda) for layer in embedding_model.layers
    ):
        # Normalize manually
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        all_embeddings = all_embeddings / norms

    # Get the index of the reference movie
    ref_idx = item_features_df[item_features_df["movieId"] == movie_id].index[0]
    ref_embedding = all_embeddings[ref_idx]

    # Calculate similarities (dot product of normalized vectors = cosine similarity)
    similarities = []
    for i, embedding in enumerate(all_embeddings):
        if all_movie_ids[i] != movie_id:  # Skip the reference movie
            similarity = np.dot(ref_embedding, embedding)
            similarities.append((all_movie_ids[i], similarity))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Get top N similar movies
    top_similar = similarities[:top_n]

    # Create DataFrame with results
    similar_movies = pd.DataFrame(top_similar, columns=["movieId", "similarity"])

    # Add movie metadata
    movie_cols = ["movieId"]
    for col in ["title", "primaryTitle", "genres", "startYear", "averageRating"]:
        if col in movies_df.columns:
            movie_cols.append(col)

    similar_movies = similar_movies.merge(
        movies_df[movie_cols], on="movieId", how="left"
    )

    return similar_movies


def create_model_2NN_reco(
    num_outputs=64, num_item_features=None, num_user_features=None
):
    """
    Creates a dual-network model with dot product output.

    Parameters:
    - num_outputs: Dimension of the embedding space
    - num_user_features: Number of user features
    - num_item_features: Number of item features

    Returns:
    - A Keras Model
    """
    tf.random.set_seed(1)

    # Create the user network as a Sequential model
    user_model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(num_user_features,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_outputs, activation="linear"),
        ]
    )

    # Create the item network as a Sequential model
    item_model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(num_item_features,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_outputs, activation="linear"),
        ]
    )

    # Create input layers for the functional API
    input_user = tf.keras.layers.Input(shape=(num_user_features,))
    input_item = tf.keras.layers.Input(shape=(num_item_features,))

    # Get embeddings from each network
    user_embedding = user_model(input_user)
    item_embedding = item_model(input_item)

    # Normalize embeddings
    user_embedding_norm = tf.keras.layers.Lambda(
        lambda x: tf.linalg.l2_normalize(x, axis=1), output_shape=(num_outputs,)
    )(user_embedding)

    item_embedding_norm = tf.keras.layers.Lambda(
        lambda x: tf.linalg.l2_normalize(x, axis=1), output_shape=(num_outputs,)
    )(item_embedding)

    # Compute dot product
    dot_product = tf.keras.layers.Dot(axes=1)(
        [user_embedding_norm, item_embedding_norm]
    )

    # Create the full model
    model = tf.keras.Model(inputs=[input_user, input_item], outputs=dot_product)

    return model
