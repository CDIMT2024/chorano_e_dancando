from functools import cache
import joblib


@cache
def get_rf():
    return joblib.load("my_random_forest.joblib")