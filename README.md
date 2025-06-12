# Système de Recommandation de Films

Ce projet est un système de recommandation de films qui combine une API FastAPI avec une interface utilisateur Streamlit pour prédire les notes qu'un utilisateur donnerait à des films.

## Architecture du Projet

Le projet est organisé en plusieurs composants principaux :

### 1. Interface Utilisateur (Streamlit)

- Fichier : `app.py`
- Fonctionnalités :
  - Affichage d'une interface web interactive
  - Sélection d'un utilisateur et d'un film
  - Affichage des détails du film (titre, année, durée, genres, note moyenne)
  - Visualisation des prédictions de notes

### 2. API Backend (FastAPI)

- Fichier : `api.py`
- Fonctionnalités :
  - Endpoint `/predict` : Prédit la note qu'un utilisateur donnerait à un film
  - Endpoint `/health` : Vérifie que l'API est en ligne
  - Gestion des erreurs et validation des données

### 3. Moteur de Prédiction

- Fichier : `predict.py`
- Fonctionnalités :
  - Chargement du modèle de prédiction
  - Calcul des affinités utilisateur-film
  - Génération des prédictions de notes
  - Prise en compte des caractéristiques du film et de l'historique de l'utilisateur

## Données Utilisées

Le système utilise plusieurs fichiers de données :

- `data/movie_df.csv` : Informations sur les films (titre, année, genre, etc.)
- `data/user_features.csv` : Caractéristiques des utilisateurs
- `data/ratings.csv` : Historique des notes attribuées
- Modèles entraînés : `final_randomforest_model.pkl`

## Préparation de l'environnement

1. Créer un environnement virtuel

```bash
py -m venv venv
```

2. Activer l'environnement

```bash
venv\Scripts\activate
```

3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## Comment Exécuter le Projet

1. Lancer l'API backend :

```bash
python api.py
```

L'API sera accessible sur http://localhost:8000

2. Lancer l'interface Streamlit :

```bash
streamlit run app.py
```

L'interface sera accessible sur http://localhost:8501

## Dépendances Principales

- FastAPI : Framework API moderne et performant
- Streamlit : Création d'applications web pour la data science
- Pandas : Manipulation des données
- Scikit-learn : Modèles de machine learning
- Uvicorn : Serveur ASGI pour FastAPI
