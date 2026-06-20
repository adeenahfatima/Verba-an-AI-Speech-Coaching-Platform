import requests

model = "microsoft/DialoGPT-medium"
api_key = "your_huggingface_token_here"

prompt = "Give me one tip to improve my public speaking skills."

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "inputs": prompt,
    "parameters": {"max_new_tokens": 60}
}

response = requests.post(
    f"https://api-inference.huggingface.co/models/{model}",
    headers=headers,
    json=payload,
    timeout=20
)

print("Status:", response.status_code)
try:
    print("Response JSON:", response.json())
except Exception:
    print("Raw response:", response.text)
