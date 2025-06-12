import streamlit as st
import requests
import pandas as pd

st.title("Prédiction de Notes de Films")


@st.cache_data
def load_data():
    try:
        movies = pd.read_csv("data/movie_df.csv")
        users = pd.read_csv("data/user_features.csv")
        return movies, users
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None, None


movies_df, users_df = load_data()

st.sidebar.header("Entrez les informations")

if users_df is not None:
    selected_user = st.sidebar.selectbox(
        "Sélectionnez un utilisateur",
        options=users_df["userId"].unique(),
        index=0
    )
    user_id = selected_user

if movies_df is not None:
    selected_movie = st.sidebar.selectbox(
        "Sélectionnez un film", options=movies_df["primaryTitle"].unique(), index=0
    )

    if st.sidebar.button("Prédire la note"):
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json={"user_id": int(user_id), "movie_identifier": selected_movie},
            )

            if response.status_code == 200:
                prediction = response.json()

                st.success(f"Note prédite : {prediction:.2f}/5")

                movie_details = movies_df[
                    movies_df["primaryTitle"] == selected_movie
                ].iloc[0]

                st.subheader("Détails du film")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Titre:**", movie_details["primaryTitle"])
                    st.write("**Année:**", movie_details["startYear"])
                    st.write("**Durée:**", f"{movie_details['runtimeMinutes']} minutes")

                with col2:
                    st.write("**Genres:**", movie_details["genres"])
                    st.write(
                        "**Note moyenne:**", f"{movie_details['averageRating']:.2f}/5"
                    )
                    st.write("**Nombre de votes:**", movie_details["numVotes"])

                if pd.notna(movie_details["overview"]):
                    st.write("**Synopsis:**", movie_details["overview"])

            else:
                st.error(f"Erreur: {response.json()['detail']}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Impossible de se connecter à l'API. Vérifiez que le serveur est en cours d'exécution."
            )
        except Exception as e:
            st.error(f"Une erreur s'est produite: {str(e)}")
