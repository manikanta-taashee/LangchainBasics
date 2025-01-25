import requests
from dotenv import load_dotenv
import os

load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = query({
    "inputs": {
        "source_sentence": "That is a happy person",
        "sentences": [
            "That is a happy dog",
            "That is a very happy person",
            "Today is a sunny day"
        ]},
})
print(output)
