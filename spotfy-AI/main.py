import os
import base64
from requests import post
import json

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = str(base64.b64encode(auth_bytes), 'utf-8')

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }

    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    return json_result['access_token']

def get_auth_header(token):
    return {
        "Authorization": "Bearer " + token,
    }



token = get_token()

