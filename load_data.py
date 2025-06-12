import time
import pandas as pd
import joblib

def load_data():
    print("Chargement des données...", time.time())
    print("Chargement de movie_df...", time.time())
    movie_df = pd.read_csv("./data/movie_df.csv")
    print("Chargement de user_features...", time.time())
    user_features = pd.read_csv("./data/user_features.csv")
    print("Chargement de ratings...", time.time())
    ratings = pd.read_csv("./data/ml-20m/ratings.csv")
    print("Chargement de model...", time.time())
    model = joblib.load("./data/model/final_randomforest_model_dispersion.pkl")
    print("Chargement terminé...", time.time())

    return {
        "movie_df": movie_df,
        "user_features": user_features,
        "ratings": ratings,
        "model": model
    }

data = load_data()
