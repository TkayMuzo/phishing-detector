from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib
import numpy as np
import re
import os
from urllib.parse import urlparse

# -------------------------------
# INITIALIZE APP
# -------------------------------
app = FastAPI()

# Enable CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# LOAD MODELS
# -------------------------------
url_model = joblib.load("model/url_model_calibrated.pkl")
sms_model = joblib.load("model/sms_model_calibrated.pkl")
fusion_model = joblib.load("model/fusion_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

THRESHOLD = 0.2

# -------------------------------
# TRUSTED DOMAIN CHECK
# -------------------------------
trusted_domains = [
    "google.com", "youtube.com", "facebook.com",
    "amazon.com", "microsoft.com", "apple.com"
]

def is_trusted(url):
    return any(domain in url.lower() for domain in trusted_domains)

# -------------------------------
# INPUT SCHEMA
# -------------------------------
class InputData(BaseModel):
    url: str
    sms: str

# -------------------------------
# URL FEATURE ENGINEERING
# -------------------------------
def extract_url_features(url):
    parsed = urlparse(url)

    return [
        len(url),
        url.count('.'),
        url.count('-'),
        url.count('@'),
        url.count('?'),
        url.count('='),
        url.count('/'),
        url.count('www'),
        1 if parsed.scheme == 'https' else 0,
        1 if re.match(r"^\d+\.\d+\.\d+\.\d+", parsed.netloc) else 0,
        1 if any(k in url.lower() for k in ["login","verify","bank","secure","account"]) else 0,
        len(parsed.netloc)
    ]

# -------------------------------
# UI ROUTES
# -------------------------------
@app.get("/")
def home():
    return FileResponse(os.path.join(os.getcwd(), "index.html"))

@app.get("/ui")
def serve_ui():
    return FileResponse(os.path.join(os.getcwd(), "index.html"))

# -------------------------------
# PREDICTION ENDPOINT
# -------------------------------
@app.post("/predict")
def predict(data: InputData):
    try:
        explanations = []

        # ---------------------------
        # URL PROCESSING
        # ---------------------------
        if is_trusted(data.url):
            url_prob = 0.0
            explanations.append("Trusted domain detected")
        else:
            url_features = np.array([extract_url_features(data.url)])
            url_prob = url_model.predict_proba(url_features)[0][1]

            if "http://" in data.url:
                explanations.append("Uses insecure HTTP")
            if any(k in data.url.lower() for k in ["login","bank","verify"]):
                explanations.append("Sensitive keywords in URL")
            if len(data.url) > 75:
                explanations.append("Unusually long URL")

        # ---------------------------
        # SMS PROCESSING
        # ---------------------------
        sms_features = vectorizer.transform([data.sms])
        sms_prob = sms_model.predict_proba(sms_features)[0][1]

        if any(k in data.sms.lower() for k in ["urgent", "click", "account", "verify"]):
            explanations.append("Suspicious SMS language")

        # ---------------------------
        # FUSION
        # ---------------------------
        fusion_input = np.array([[url_prob, sms_prob]])
        final_score = fusion_model.predict_proba(fusion_input)[0][1]

        decision = "phishing" if final_score >= THRESHOLD else "legitimate"

        return {
            "url_score": float(url_prob),
            "sms_score": float(sms_prob),
            "final_score": float(final_score),
            "threshold": THRESHOLD,
            "decision": decision,
            "explanations": explanations
        }

    except Exception as e:
        return {"error": str(e)}