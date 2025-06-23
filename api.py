from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union

from predict import predict_movie_rating, compare_ratings_with_recommendations_vectorized

app = FastAPI(
    title="Film Rating Prediction API",
    description="API pour prédire la note qu'un utilisateur donnerait à un film",
    version="1.0.0",
)


class MovieRequest(BaseModel):
    user_id: int
    movie_identifier: Union[int, str]

@app.post("/predict", response_model=float)
async def predict_rating(request: MovieRequest):
    """
    Prédit la note qu'un utilisateur donnerait à un film.

    Args:
        request: MovieRequest contenant:
            - user_id: ID de l'utilisateur
            - movie_identifier: Peut être movieId (int), tconst (str) ou primaryTitle (str)

    Returns:
        float: Note prédite entre 0 et 5
    """
    try:
        prediction = predict_movie_rating(request.user_id, request.movie_identifier)
        return float(prediction)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Endpoint pour vérifier que l'API est en ligne"""
    return {"status": "healthy"}

@app.get("/compare_ratings_with_recommendations")
async def compare_ratings_with_recommendations(user_id: int, n_recommendations: int = 10):
    """
    Compare les notes réelles d'un utilisateur avec les recommandations du modèle.

    Args:
        user_id (int): ID de l'utilisateur
        n_recommendations (int): Nombre de recommandations à retourner

    Returns:
        dict: Contient les films les mieux notés par l'utilisateur et les recommandations du modèle
    """
    try:
        result = compare_ratings_with_recommendations_vectorized(user_id, top_n=n_recommendations)
        if result is None:
            raise HTTPException(status_code=404, detail="Aucune donnée pour cet utilisateur.")
        # Sérialisation des DataFrames
        user_top_movies = result.get("user_top_movies")
        recommendations = result.get("recommendations")
        return {
            "user_top_movies": user_top_movies.to_dict(orient="records") if user_top_movies is not None else [],
            "recommendations": recommendations.to_dict(orient="records") if recommendations is not None else [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_delay=5,
        workers=1,
        log_level="info",
    )