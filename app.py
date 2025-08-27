from flask import Flask, request, jsonify
import requests
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

# Initialize models
tongue_model = YOLO("tmodel.pt") 
ROBOFLOW_CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="zUa1qfkAPDtqRQ6rRlco"
)
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    tongue_image_path = data.get("tongue_image")
    eye_image_path = data.get("eye_image")

    # --- Tongue analysis ---
    try:
        results = tongue_model(tongue_image_path)
        if results and results[0].boxes:
            tongue_condition = tongue_model.names[int(results[0].boxes.cls[0])]
            tongue_confidence = float(results[0].boxes.conf[0])
        else:
            tongue_condition, tongue_confidence = "No specific condition detected", 0.5
    except Exception as e:
        tongue_condition, tongue_confidence = f"Error: {e}", 0

    # --- Anemia detection ---
    try:
        result = ROBOFLOW_CLIENT.infer(eye_image_path, model_id="anemia-detection-v3-fyv07/1")
        anemia_result = result['predicted_classes'][0].capitalize()
        anemia_confidence = max(
            result['predictions']['anemic']['confidence'],
            result['predictions']['healthy']['confidence']
        )
    except Exception as e:
        anemia_result, anemia_confidence = f"Error: {e}", 0

    # --- AI-generated health summary ---
    prompt = f"""
    As a medical AI assistant, provide a comprehensive health analysis based on these findings:
    TONGUE: {tongue_condition} ({tongue_confidence:.2f})
    EYE (Anemia): {anemia_result} ({anemia_confidence:.2f})
    """
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1200
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        ai_response = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        ai_response = f"Error querying AI API: {e}"

    return jsonify({
        "tongue_condition": tongue_condition,
        "tongue_confidence": tongue_confidence,
        "anemia_result": anemia_result,
        "anemia_confidence": anemia_confidence,
        "ai_health_analysis": ai_response
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
