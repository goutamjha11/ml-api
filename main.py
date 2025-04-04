import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi import status
from model import model
from schemas import PredictionRequest, PredictionResponse
from utilities.utils import preprocess


app = FastAPI()
@app.on_event("startup")
async def startup_event():
    """Load the model when the app starts"""
    try:
        model.load()
        print("Model successfully loaded!")
    except Exception as e:
        print(f"Failed to load the model: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to load model")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict endpoint"""
    try:
        # Preprocess input features
        features = preprocess(request.features)
        if len(features.shape) == 1:  # If it's a single sample, make it 2D
            features = features.reshape(1, -1)
        # Get prediction from model
        prediction = model.predict(features)
        print(f"Prediction completed: {prediction}")
        return PredictionResponse(prediction=prediction.tolist())
        # return PredictionResponse(prediction=prediction[0])
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid model state")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {"message": "Inference API is running."}


if __name__ =="__main__":
    uvicorn.run('main:app', host='0.0.0.0',port=8000)