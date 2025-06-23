import streamlit as st
import requests
import pandas as pd

st.title("Prédiction de Notes de Films")


@st.cache_data
def load_data():
    try:
        movies = pd.read_csv("data/movie_df.csv")
        users = pd.read_csv("data/user_features.csv")
        # Extraction des genres uniques
        genres = set()
        for genre_str in movies["genres"].dropna().unique():
            for g in str(genre_str).split(","):
                if g.strip() and g.strip().lower() != "(no genres listed)":
                    genres.add(g.strip())
        unique_genres = sorted(genres)
        # Extraction des décennies disponibles
        years = movies["startYear"].dropna().astype(int)
        decades = sorted(set((y // 10) * 10 for y in years if y > 1900))
        return movies, users, unique_genres, decades
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None, None, [], []


# Paramétrage du nombre de recommandations à afficher
n_recommendations = st.sidebar.slider(
    "Nombre de recommandations à afficher", min_value=1, max_value=50, value=20, step=1
)

# Sélection filtres genre et décennie
movies_df, users_df, unique_genres, unique_decades = load_data()

selected_genre = st.sidebar.selectbox(
    "Filtrer par genre", options=["Tous"] + unique_genres, index=0
)
selected_decade = st.sidebar.selectbox(
    "Filtrer par décennie", options=["Toutes"] + [str(d) for d in unique_decades], index=0
)


st.sidebar.header("Entrez les informations")

if users_df is not None:
    selected_user = st.sidebar.selectbox(
        "Sélectionnez un utilisateur", options=users_df["userId"].unique(), index=0
    )
    user_id = selected_user

    if st.sidebar.button("Obtenir des recommandations"):
        try:
            # Appel à l'API pour obtenir les recommandations et les meilleurs films notés
            reco_response = requests.get(
                "http://localhost:8000/compare_ratings_with_recommendations",
                params={
                    "user_id": int(user_id),
                    "n_recommendations": 1000,
                },
            )

            if reco_response.status_code == 200:
                reco_data = reco_response.json()
                user_top_movies = reco_data.get("user_top_movies", [])
                recommendations = reco_data.get("recommendations", [])

                # Ne pas afficher la liste des films déjà notés par l'utilisateur

                st.subheader("Recommandations du modèle (mise à jour en temps réel)")
                if recommendations:
                    reco_df = pd.DataFrame(recommendations)
                    # Filtrage par genre et décennie
                    if not reco_df.empty:
                        if selected_genre and selected_genre != "Tous":
                            reco_df = reco_df[reco_df["genres"].str.contains(rf"\b{selected_genre}\b", case=False, na=False)]
                        if selected_decade and selected_decade != "Toutes":
                            decade_int = int(selected_decade)
                            reco_df = reco_df[
                                reco_df["startYear"].apply(lambda y: str(y).isdigit() and (int(y) // 10) * 10 == decade_int)
                            ]
                    # Affichage progressif des recommandations avec note prédite
                    displayed_recos = []
                    placeholder = st.empty()
                    count = 0
                    for idx, row in reco_df.iterrows():
                        if count >= n_recommendations:
                            break
                        try:
                            pred_resp = requests.post(
                                "http://localhost:8000/predict",
                                json={
                                    "user_id": int(user_id),
                                    "movie_identifier": int(row["movieId"]),
                                },
                                timeout=5,
                            )
                            if pred_resp.status_code == 200:
                                predicted_rating = pred_resp.json()
                            else:
                                predicted_rating = None
                        except Exception:
                            predicted_rating = None
                        reco = {
                            "Titre": row["primaryTitle"],
                            "Genres": row["genres"],
                            "Année": row["startYear"],
                            "Note prédite": predicted_rating,
                        }
                        displayed_recos.append(reco)
                        # Tri dynamique à chaque ajout
                        displayed_recos_sorted = sorted(
                            displayed_recos,
                            key=lambda x: (
                                x["Note prédite"] is not None,
                                x["Note prédite"],
                            ),
                            reverse=True,
                        )
                        # Affichage progressif
                        placeholder.dataframe(pd.DataFrame(displayed_recos_sorted))
                        count += 1
                    if not displayed_recos:
                        st.info("Aucune recommandation trouvée pour cet utilisateur avec ces filtres.")
                else:
                    st.info("Aucune recommandation trouvée pour cet utilisateur.")

            else:
                st.error(
                    f"Erreur: {reco_response.json().get('detail', reco_response.text)}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Impossible de se connecter à l'API. Vérifiez que le serveur est en cours d'exécution."
            )
        except Exception as e:
            st.error(f"Une erreur s'est produite: {str(e)}")
