# app.py
from flask import Flask, render_template, request, jsonify, session
from model import EmbeddingChatModel
import os, re
from uuid import uuid4

app = Flask(__name__, static_folder="static", template_folder="templates")
# secret key for flask session (for demo; in production use env var)
app.secret_key = os.environ.get("FLASK_SECRET", "replace_this_with_a_random_secret")

model = EmbeddingChatModel(embeddings_path="embeddings.pkl")

# In-memory conversation memory: {session_id: {"history": [(role,text,tag)], "last_tag": tag}}
# This is ephemeral and resets when app restarts. For production, use DB.
conversation_memory = {}

def ensure_session():
    if "sid" not in session:
        session["sid"] = str(uuid4())
    sid = session["sid"]
    if sid not in conversation_memory:
        conversation_memory[sid] = {"history": [], "last_tag": None}
    return sid

# Example simple follow-up rule:
# If last_tag == "order_status" and current message contains an order id (>=5 digits),
# return a templated tracking response instead of the intent response.
ORDER_ID_REGEX = re.compile(r"\b([A-Za-z0-9\-]{5,})\b")

@app.route("/")
def index():
    ensure_session()
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    sid = ensure_session()
    data = request.json or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"success": False, "error": "Empty message"}), 400

    mem = conversation_memory[sid]
    last_tag = mem.get("last_tag")

    # If last intent was order_status, check if user provided an order id now
    if last_tag == "order_status":
        match = ORDER_ID_REGEX.search(message)
        if match:
            order_id = match.group(1)
            # Example generated status — in real system call orders DB / external API
            reply = f"Thanks — I found order **{order_id}**. Current status: *In transit*. Estimated delivery: 2–4 business days."
            mem["history"].append(("user", message, None))
            mem["history"].append(("bot", reply, "order_status_followup"))
            mem["last_tag"] = "order_status_followup"
            return jsonify({"success": True, "tag": "order_status_followup", "response": reply, "confidence": 0.95})

    # Otherwise use model
    result = model.predict(message)
    mem["history"].append(("user", message, None))
    mem["history"].append(("bot", result["response"], result["tag"]))
    mem["last_tag"] = result["tag"]

    # Confidence thresholding: if low confidence, send unknown fallback
    if result["confidence"] < 0.45:
        # try to return fallback from intents.json unknown tag if present
        fallback = "Sorry, I didn't understand. Could you rephrase or provide more details?"
        return jsonify({"success": True, "tag": "unknown", "response": fallback, "confidence": result["confidence"]})

    return jsonify({"success": True, "tag": result["tag"], "response": result["response"], "confidence": result["confidence"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
