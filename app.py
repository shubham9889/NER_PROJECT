import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import pickle
import jwt

# Load trained model
model_path = "ner_model.pkl"
with open(model_path, "rb") as model_file:
    trained_nlp = pickle.load(model_file)

# Initialize FastAPI app
app = FastAPI()
SECRET_KEY = "your_secret_key"

# Authentication function
def authenticate(token: str):
    """Basic token authentication."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Input model
class TextInput(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict_entities(input_data: TextInput, token: str = Depends(authenticate)):
    """NER Prediction Endpoint."""
    doc = trained_nlp(input_data.text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return {"entities": entities}

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
