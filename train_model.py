# train_model.py
import json
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"  # fast, small, good defaults

def load_intents(path="intents.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_intent_embeddings(intents, model):
    tags = []
    intent_embeddings = []
    intent_responses = {}
    intent_patterns = {}

    for intent in intents["intents"]:
        tag = intent["tag"]
        patterns = intent.get("patterns", [])
        # ensure at least one pattern exists
        if not patterns:
            patterns = [""]  # avoid empty
        embeddings = model.encode(patterns, convert_to_numpy=True, show_progress_bar=False)
        # mean of pattern embeddings -> single intent vector
        mean_emb = embeddings.mean(axis=0)
        tags.append(tag)
        intent_embeddings.append(mean_emb)
        intent_responses[tag] = intent.get("responses", [])
        intent_patterns[tag] = patterns

    intent_embeddings = np.stack(intent_embeddings)
    return {"model_name": MODEL_NAME, "tags": tags, "embeddings": intent_embeddings, "responses": intent_responses, "patterns": intent_patterns}

def main():
    intents = load_intents("intents.json")
    print("Loading sentence-transformers model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    mapping = build_intent_embeddings(intents, model)
    Path("embeddings.pkl").write_bytes(pickle.dumps(mapping))
    print("Saved intent embeddings to embeddings.pkl")

if __name__ == "__main__":
    main()
