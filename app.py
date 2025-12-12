# app.py
import os
import torch
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
from torchvision import transforms
from classification_model import DeepAnn
import joblib
import pandas as pd
# near other imports in app.py

import json
import joblib
import numpy as np



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_DIR'] = 'models'

# class names for image emotions (keep as you had them)
class_names = [
    "anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]

# IMAGE transform (same as before)
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# -------------------
# Load models
# -------------------
def load_image_model():
    model_path = os.path.join(app.config['MODEL_DIR'], "deep_ann_model.pth")
    input_size = 128 * 128 * 3
    num_classes = len(class_names)
    model = DeepAnn(input_size=input_size, num_classes=num_classes)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    else:
        raise FileNotFoundError(f"Image model not found at {model_path}")
    return model

def load_text_model():
    model_path = os.path.join(app.config['MODEL_DIR'], "text_model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        # If model missing, return None and handle gracefully in routes
        return None


# Load on startup (or lazy load inside routes)
try:
    IMAGE_MODEL = load_image_model()
except Exception as e:
    IMAGE_MODEL = None
    print("Warning: image model not loaded:", e)

TEXT_MODEL = load_text_model()
if TEXT_MODEL is None:
    print("Warning: text model not found. Run train_text_model.py to create models/text_model.joblib")


def load_screening_model():
    model_path = os.path.join(app.config['MODEL_DIR'], "screen_model.joblib")
    classes_path = os.path.join(app.config['MODEL_DIR'], "screen_classes.json")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        encoders = json.load(open(classes_path, "r"))
        return model, encoders
    return None, None
SCREEN_MODEL_PATH = "models/screen_model.joblib"
SCREEN_CLASSES_PATH = "models/screen_classes.json"

if os.path.exists(SCREEN_MODEL_PATH):
    SCREEN_MODEL = joblib.load(SCREEN_MODEL_PATH)
    with open(SCREEN_CLASSES_PATH, "r") as f:
        SCREEN_CLASSES = json.load(f)
else:
    SCREEN_MODEL = None
    SCREEN_CLASSES = None
    print("Warning: Screening model not loaded.")


SCREEN_MODEL, SCREEN_ENCODERS = load_screening_model()

# -------------------
# ROUTES
# -------------------
@app.route("/")
def overview():
    return render_template("Overview.html")

@app.route("/text-voice", methods=['GET', 'POST'])
def text_voice():
    if request.method == 'POST':
        # Accept text from either name (safe approach)
        text = request.form.get('text') or request.form.get('description')

        if not text or text.strip() == "":
            return render_template("Text_voice.html", error="Please enter text to analyze.")

        # Check text model
        if TEXT_MODEL is None:
            return render_template("Text_voice.html", error="Text model not loaded.")

        # Run prediction
        prediction = TEXT_MODEL.predict([text])[0]

        return render_template("Text_voice.html", result=prediction, text=text)

    return render_template("Text_voice.html")

@app.route("/facial", methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        global IMAGE_MODEL
        if IMAGE_MODEL is None:
            try:
                IMAGE_MODEL = load_image_model()
            except Exception as e:
                return render_template('Facial.html', error=f"Model loading error: {e}")

        file = request.files.get('file')
        if file is None or file.filename == '':
            return render_template('Facial.html', error="No file selected")

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        image = Image.open(filepath)
        image_t = transform(image).unsqueeze(0)  # shape (1, 3, 128, 128)

        with torch.no_grad():
            output = IMAGE_MODEL(image_t)
            _, prediction = torch.max(output, 1)

        result = class_names[prediction.item()]
        return render_template('Facial.html', image_path=filepath, result=result)

    return render_template('Facial.html')

from flask import request, jsonify

# add this route (or replace your existing /chatbot)
@app.route("/chatbot", methods=["POST"])
def chatbot():
    """
    Lightweight context-aware chatbot.
    Expects JSON: { "message": "<text>", "results": { ... } }
    Returns JSON: { "reply": "<text>" }
    """
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    results = data.get("results") or {}

    # Safety checks
    if not user_msg:
        return jsonify({"reply": "Please type a question or ask for suggestions about your screening results."})

    # Normalize message for keyword checks
    msg = user_msg.lower()

    # Helper functions
    def get_label(k):
        """Safe getter for predicted label string"""
        try:
            v = results.get(k)
            if isinstance(v, dict) and "prediction" in v:
                return str(v["prediction"]).lower()
            # sometimes results might be mapping of label->string
            return str(v).lower() if v is not None else None
        except Exception:
            return None

    def is_high_risk_suicidal():
        lab = get_label("suicidal")
        if lab is None:
            return False
        return lab in ["yes", "likely", "high", "severe", "1", "true"]

    def severity_label(key):
        return get_label(key)  # returns string like 'moderate' or 'severe' or None

    # Build context-aware suggestions
    suggestions = []

    # Suicidal risk check (highest priority)
    if is_high_risk_suicidal():
        suggestions.append(
            "Your screening result indicates possible suicidal thoughts. "
            "If you are in immediate danger or have thoughts of harming yourself, call your local emergency number RIGHT NOW. "
            "If you are able, contact a crisis line or a trusted person. "
            "Please seek urgent professional help — I can't provide emergency services."
        )
        # Append a few immediate self-safety steps (safe, general)
        suggestions.append(
            "If safe to do so: remove access to anything you might use to hurt yourself, stay with someone you trust, "
            "and reach out to a mental-health professional or emergency services."
        )
    else:
        # depression suggestions
        dep = severity_label("depression_severity")
        if dep:
            if dep in ["severe","moderately severe", "moderate-severe", "high"]:
                suggestions.append(
                    f"Your depression severity is shown as '{dep}'. For moderate-to-severe depression, consider contacting a mental-health professional for assessment and discussing treatment options (therapy, medication). If you're struggling right now, reach out to a trusted person or a local crisis service."
                )
            elif dep in ["moderate","mild"]:
                suggestions.append(
                    f"Your depression severity is '{dep}'. Self-care (regular sleep, small daily routines, light exercise, connecting with friends) often helps. Consider talking to a counselor if symptoms persist."
                )
            else:
                suggestions.append(
                    f"Your depression severity is '{dep}'. Keep monitoring and maintain healthy routines. Seek support if things change."
                )

        # anxiety suggestions
        anx = severity_label("anxiety_severity")
        if anx:
            if anx in ["severe","high"]:
                suggestions.append(
                    f"Anxiety severity: '{anx}'. For high anxiety, consider professional help (therapy, CBT techniques). Short-term strategies include deep breathing, grounding exercises, limiting caffeine, and structured therapy."
                )
            elif anx in ["moderate","mild"]:
                suggestions.append(
                    f"Anxiety severity: '{anx}'. Try breathing exercises (4-4-4), progressive muscle relaxation, and short walks. If it interferes with daily life, consult a clinician."
                )

        # sleepiness
        sl = severity_label("sleepiness")
        if sl and sl.lower() not in ["normal","none","no","0"]:
            suggestions.append(
                f"Sleepiness level: '{sl}'. Improving sleep hygiene (consistent bedtime, reducing screen use before bed, limiting caffeine) may help. If daytime sleepiness persists, consider a medical review."
            )

    # Provide targeted treatment suggestions if user asked for 'treatment'
    if "treatment" in msg or "what should i do" in msg or "what to do" in msg:
        suggestions.append(
            "General treatment suggestions: psychotherapy (CBT), lifestyle changes (exercise, sleep, routine), "
            "peer support, and medication when recommended by a psychiatrist. Always consult a clinician before starting medications."
        )

    # If the user asked specifically about anxiety or depression
    if "anxiety" in msg and not any("anxiety" in s.lower() for s in suggestions):
        suggestions.append(
            "For anxiety: practice grounding (5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste), "
            "slow diaphragmatic breathing, and gradual exposure for specific fears. Therapy (CBT) is evidence-based for anxiety."
        )
    if "depress" in msg and not any("depress" in s.lower() for s in suggestions):
        suggestions.append(
            "For depressive symptoms: small achievable activities, regular sleep-wake times, social contact, and considering therapy. "
            "If symptoms are moderate/severe, speak with a mental-health professional about therapy and/or medications."
        )

    # Default fallback
    if not suggestions:
        suggestions.append(
            "I can explain your screening labels and offer general self-help steps (sleep, routine, talk to someone). "
            "Tell me what you'd like to know about your results (for example: 'Explain depression_severity', 'How to reduce anxiety')."
        )

    # Combine suggestions into a reply, include a short disclaimer
    reply = "\n\n".join(suggestions)
    reply += (
        "\n\nDisclaimer: I am not a clinician. This is general information only. "
        "For personalised medical advice, please consult a licensed mental-health professional."
    )

    return jsonify({"reply": reply})


@app.route("/screen", methods=["GET", "POST"])
def screen_user():
    if request.method == "POST":
        if SCREEN_MODEL is None:
            return render_template("Screen.html", error="Screening model not loaded.")

        try:
            form_data = {
                "age": int(request.form["age"]),
                "school_year": request.form["school_year"],
                "gender": request.form["gender"],
                "bmi": float(request.form["bmi"]),
                "who_bmi": request.form["who_bmi"],
                "phq_score": int(request.form["phq_score"]),
                "gad_score": int(request.form["gad_score"]),
                "epworth_score": int(request.form["epworth_score"]),
            }

            X = pd.DataFrame([form_data])

            preds = SCREEN_MODEL.predict(X)[0]
            probs = SCREEN_MODEL.predict_proba(X)

            results = {}

            for i, col in enumerate(SCREEN_CLASSES.keys()):
                class_list = SCREEN_CLASSES[col]
                predicted_label = class_list[preds[i]]

                confidence = (
                    float(np.max(probs[i][0])) if hasattr(probs[i], "__len__") else None
                )

                results[col] = {
                    "prediction": predicted_label,
                    "confidence": confidence
                }

            return render_template("Screen.html", results=results)

        except Exception as e:
            return render_template("Screen.html", error=str(e))

    return render_template("Screen.html")


@app.route("/compliance")
def compliance():
    return render_template("Compliance.html")


if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_DIR'], exist_ok=True)
    app.run(debug=True)
