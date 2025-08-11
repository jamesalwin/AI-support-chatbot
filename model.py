# model.py
import pickle
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

class EmbeddingChatModel:
    def __init__(self, embeddings_path="embeddings.pkl"):
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)
        self.model_name = data.get("model_name", DEFAULT_MODEL_NAME)
        self.tags = data["tags"]
        self.intent_embeddings = data["embeddings"]  # numpy array (n_intents, dim)
        self.responses = data["responses"]
        # lazy-load embedding model (small enough to load on init)
        self.embedder = SentenceTransformer(self.model_name)

    def predict(self, text):
        # basic clean
        query_emb = self.embedder.encode([text], convert_to_numpy=True)[0].reshape(1, -1)
        sims = cosine_similarity(query_emb, self.intent_embeddings)[0]  # (n_intents,)
        best_idx = int(np.argmax(sims))
        best_tag = self.tags[best_idx]
        confidence = float(sims[best_idx])  # between -1 and 1
        # map similarity to 0-1 roughly
        conf01 = max(0.0, min(1.0, (confidence + 1) / 2))
        resp_list = self.responses.get(best_tag, [])
        response = random.choice(resp_list) if resp_list else "Sorry, I don't know how to answer that yet."
        return {"tag": best_tag, "response": response, "confidence": conf01}
