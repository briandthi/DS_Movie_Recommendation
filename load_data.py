import time
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

from utils import (
    calculate_bayesian_genre_ratings_vectorized,
    create_model_2NN_reco,
    create_model_2NN_reco_B,
    create_recommendation_triplets_robust_extended,
)

base2_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",  # Changed from 'val_loss''val_loss',
        patience=5,
        restore_best_weights=True,
    )
]


def load_data():
    print("Chargement des données...", time.time())
    print("Chargement de user_features...", time.time())
    user_features = pd.read_csv("./data/user_features.csv")
    print("Chargement de ratings...", time.time())
    ratings = pd.read_csv("./data/ml-20m/ratings.csv")
    print("Chargement de model...", time.time())
    model = joblib.load("./data/model/final_randomforest_model_dispersion.pkl")
    print("Chargement du modèle B5 (DL)...", time.time())

    print("Chargement des scalers...")
    user_scaler = joblib.load("./data/model/user_scaler.pkl")
    print("user_scaler chargé.")
    movie_scaler = joblib.load("./data/model/movie_scaler.pkl")
    print("movie_scaler chargé.")
    scalerTarget = joblib.load("./data/model/scalerTarget.pkl")
    print("scalerTarget chargé.")

    print("Chargement des features utilisateurs et films...")
    user_features_df = pd.read_pickle("./data/model/user_features_df.pkl")
    print("user_features_df chargé. Shape:", user_features_df.shape)
    item_features_df = pd.read_pickle("./data/model/item_features_df.pkl")
    print("item_features_df chargé. Shape:", item_features_df.shape)

    print("Chargement du DataFrame des films...")
    movie_df = pd.read_csv("./data/movie_df.csv")
    print("movie_df chargé. Colonnes:", movie_df.columns.tolist())

    print("Chargement des ratings...")
    ratings = pd.read_csv("./data/ml-20m/ratings.csv")
    print("ratings chargé. Shape:", ratings.shape)

    mv_ratings = pd.read_csv("./data/mv_ratings.csv")
    results_df_per_user, bayesian_matrix_per_user, genre_c_values = (
        calculate_bayesian_genre_ratings_vectorized(mv_ratings, movie_df)
    )
    user_X, movie_X, y_rating, user_features_df, item_features_df, movie_ids_list = (
        create_recommendation_triplets_robust_extended(
            bayesian_matrix_per_user, movie_df, mv_ratings
        )
    )
    item_train, item_test = train_test_split(
        movie_X, train_size=0.80, shuffle=True, random_state=1
    )
    user_train, user_test = train_test_split(
        user_X, train_size=0.80, shuffle=True, random_state=1
    )
    y_train, y_test = train_test_split(
        y_rating, train_size=0.80, shuffle=True, random_state=1
    )
    user_train_scaled = user_scaler.fit_transform(user_train)
    item_train_scaled = movie_scaler.fit_transform(item_train)
    num_user_features = user_train_scaled.shape[1]
    num_item_features = item_train_scaled.shape[1]
    num_outputs = 64
    model_basis = create_model_2NN_reco(
        num_outputs=num_outputs,
        num_item_features=num_item_features,
        num_user_features=num_user_features,
    )
    model_B5 = create_model_2NN_reco_B(
        num_outputs=64,
        num_item_features=num_item_features,
        num_user_features=num_user_features,
        dropout_rate=0.5,
        l2_reg=0.0,
    )
    model_DNN = model_basis
    print(model_DNN.summary())
    y_train_scaled = scalerTarget.fit_transform(y_train.reshape(-1, 1)).flatten()
    tf.random.set_seed(1)
    cost_fn = tf.keras.losses.MeanSquaredError()
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model_DNN.compile(optimizer=opt, loss=cost_fn)
    history = model_DNN.fit(
        [user_train_scaled, item_train_scaled],
        y_train_scaled,
        epochs=100,
        callbacks=base2_callbacks,
        validation_split=0.2,
    )

    return {
        "movie_df": movie_df,
        "user_features_df": user_features_df,
        "item_features_df": item_features_df,
        "user_features": user_features,
        "user_scaler": user_scaler,
        "movie_scaler": movie_scaler,
        "scalerTarget": scalerTarget,
        "ratings_df": mv_ratings,
        "ratings": ratings,
        "model": model,
        "model_DNN": model_DNN,
    }


data = load_data()
