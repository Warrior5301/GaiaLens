import base64
import binascii
import io
import os
 
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
 
from infer import GSDPredictor
 
app = FastAPI(title="GSD estimator")
predictor = GSDPredictor(os.environ.get("GSD_MODEL", "gsd_model.pt"))
 
 
class Base64Request(BaseModel):
    image_base64: str  
 
 
@app.get("/health")
def health():
    return {"status": "ok"}
 
 
@app.post("/run")
def predict_base64(req: Base64Request):
    s = req.image_base64.strip()
    if s.startswith("data:") and "," in s:   
        s = s.split(",", 1)[1]
    try:
        data = base64.b64decode(s)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid base64 string")
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Not a valid image")
    r = predictor.predict(io.BytesIO(data))
    return {"gsd_m_per_pixel": round(r["gsd_m"], 3),
            "spread": round(r["spread"], 3),
            "n_tiles": r["n_tiles"]}