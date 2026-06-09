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

# -------------------------------
# ENABLE CORS
# -------------------------------
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
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

THRESHOLD = 0.2

# -------------------------------
# TRUSTED DOMAINS
# -------------------------------
trusted_domains = [
    "google.com",
    "youtube.com",
    "youtu.be",
    "facebook.com",
    "amazon.com",
    "microsoft.com",
    "apple.com",
    "github.com"
]

# -------------------------------
# TRUST CHECK
# -------------------------------
def is_trusted(url):

    url = url.lower().strip()

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)

    domain = parsed.netloc.lower()

    return any(
        domain == td or domain.endswith("." + td)
        for td in trusted_domains
    )

# -------------------------------
# INPUT SCHEMA
# -------------------------------
class InputData(BaseModel):
    url: str
    sms: str

# -------------------------------
# URL FEATURES
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
        1 if any(
            k in url.lower()
            for k in ["login", "verify", "bank", "secure", "account"]
        ) else 0,
        len(parsed.netloc)
    ]

# -------------------------------
# UI ROUTES
# -------------------------------
@app.get("/")
def home():
    return FileResponse(
        os.path.join(os.getcwd(), "index.html")
    )

@app.get("/ui")
def serve_ui():
    return FileResponse(
        os.path.join(os.getcwd(), "index.html")
    )

# -------------------------------
# API HEALTH CHECK
# -------------------------------
@app.get("/health")
def health():
    return {"status": "running"}

# -------------------------------
# PREDICTION ENDPOINT
# -------------------------------
@app.post("/predict")
def predict(data: InputData):

    try:

        explanations = []

        # ---------------------------
        # NORMALIZE URL
        # ---------------------------
        url = data.url.strip()

        if url and not url.startswith(
            ("http://", "https://")
        ):
            url = "https://" + url

        # ---------------------------
        # URL ANALYSIS
        # ---------------------------
        if not url:

            url_prob = 0.0
            explanations.append(
                "No URL was provided."
            )

        elif is_trusted(url):

            url_prob = 0.0

            explanations.append(
                "Trusted domain detected."
            )

        else:

            url_features = np.array(
                [extract_url_features(url)]
            )

            url_prob = (
                url_model
                .predict_proba(url_features)[0][1]
            )

            if "http://" in url:
                explanations.append(
                    "Uses insecure HTTP protocol."
                )

            if any(
                k in url.lower()
                for k in [
                    "login",
                    "verify",
                    "bank",
                    "secure",
                    "account"
                ]
            ):
                explanations.append(
                    "URL contains phishing-related keywords."
                )

            if len(url) > 75:
                explanations.append(
                    "URL is unusually long and may be obfuscated."
                )

        # ---------------------------
        # SMS ANALYSIS
        # ---------------------------
        sms_features = vectorizer.transform(
            [data.sms]
        )

        sms_prob = (
            sms_model
            .predict_proba(sms_features)[0][1]
        )

        if any(
            k in data.sms.lower()
            for k in [
                "urgent",
                "click",
                "account",
                "verify",
                "password",
                "locked",
                "suspended",
                "winner",
                "claim",
                "bank"
            ]
        ):
            explanations.append(
                "SMS contains suspicious phishing language."
            )

        # ---------------------------
        # FUSION
        # ---------------------------
        final_score = max(
            url_prob,
            sms_prob
        )

        decision = (
            "phishing"
            if final_score >= THRESHOLD
            else "legitimate"
        )

        # ---------------------------
        # EXPLAIN MAIN REASON
        # ---------------------------
        if url_prob > sms_prob:

            explanations.append(
                "URL contributed most to the final decision."
            )

        elif sms_prob > url_prob:

            explanations.append(
                "SMS contributed most to the final decision."
            )

        else:

            explanations.append(
                "URL and SMS contributed equally."
            )

        # ---------------------------
        # RESPONSE
        # ---------------------------
        return {
            "url_score": float(url_prob),
            "sms_score": float(sms_prob),
            "final_score": float(final_score),
            "threshold": THRESHOLD,
            "decision": decision,
            "explanations": explanations
        }

    except Exception as e:

        return {
            "error": str(e)
        }