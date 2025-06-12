from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union

from predict import predict_movie_rating

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