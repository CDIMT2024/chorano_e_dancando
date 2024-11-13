import requests
import joblib
from io import BytesIO
from functools import cache

@cache
def get_rf():
    url = "https://github.com/CDIMT2024/chorano_e_dancando/raw/refs/heads/main/spotfy-AI/my_random_forest.joblib"
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    return joblib.load(BytesIO(response.content))