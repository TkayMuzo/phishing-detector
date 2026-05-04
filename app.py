from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import re
from urllib.parse import urlparse

# -------------------------------
# INITIALIZE APP
# -------------------------------
app = FastAPI()

# Enable CORS (important for frontend later)
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
# URL FEATURE ENGINEERING (12 FEATURES)
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
# EXPLAINABILITY FUNCTION
# -------------------------------
def generate_explanation(url, sms, url_prob, sms_prob, final_score):
    explanation = []

    # URL reasoning
    suspicious_keywords = ["login", "verify", "bank", "secure", "account"]
    if any(k in url.lower() for k in suspicious_keywords):
        explanation.append("URL contains phishing-related keywords.")

    if "http://" in url:
        explanation.append("URL is not secure (HTTP instead of HTTPS).")

    if len(url) > 50:
        explanation.append("URL is unusually long and may be obfuscated.")

    # SMS reasoning
    sms_keywords = ["urgent", "click", "verify", "account", "password", "now", "locked"]
    if any(k in sms.lower() for k in sms_keywords):
        explanation.append("SMS contains high-risk phishing language.")

    # Contribution reasoning
    if url_prob > sms_prob:
        explanation.append("URL contributed more to the final decision.")
    else:
        explanation.append("SMS contributed more to the final decision.")

    # Confidence reasoning
    if final_score > 0.8:
        explanation.append("Overall risk is very high.")
    elif final_score > 0.5:
        explanation.append("Moderate risk detected.")
    else:
        explanation.append("Low risk detected.")

    # fallback
    if not explanation:
        explanation.append("No significant phishing indicators detected.")

    return explanation

# -------------------------------
# ROOT ENDPOINT
# -------------------------------
@app.get("/")
def home():
    return {"message": "Phishing Detection API Running"}

# -------------------------------
# PREDICTION ENDPOINT
# -------------------------------
@app.post("/predict")
def predict(data: InputData):
    try:
        # ---------------------------
        # URL PROCESSING
        # ---------------------------
        if is_trusted(data.url):
            url_prob = 0.0
        else:
            url_features = np.array([extract_url_features(data.url)])
            url_prob = url_model.predict_proba(url_features)[0][1]

        # ---------------------------
        # SMS PROCESSING
        # ---------------------------
        sms_features = vectorizer.transform([data.sms])
        sms_prob = sms_model.predict_proba(sms_features)[0][1]

        # ---------------------------
        # FUSION
        # ---------------------------
        fusion_input = np.array([[url_prob, sms_prob]])
        final_score = fusion_model.predict_proba(fusion_input)[0][1]

        decision = "phishing" if final_score >= THRESHOLD else "legitimate"

        # ---------------------------
        # EXPLANATION
        # ---------------------------
        explanation = generate_explanation(
            data.url, data.sms, url_prob, sms_prob, final_score
        )

        return {
            "url_score": float(url_prob),
            "sms_score": float(sms_prob),
            "final_score": float(final_score),
            "threshold": THRESHOLD,
            "decision": decision,
            "explanation": explanation
        }

    except Exception as e:
        return {"error": str(e)}