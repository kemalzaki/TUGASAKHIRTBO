import os
import pickle
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
import json

languages = ["ind", "eng", "sun"]
FEEDBACK_LOG = os.path.join(os.path.dirname(__file__), "user_feedback.json")

# Define a picklable tokenizer function at module level
def _pass_through_tokenizer(text):
    """Pass-through tokenizer that returns the text as-is (for char analyzer)"""
    return [text]


def load_dataset():
    texts = []
    labels = []
    for lang in languages:
        path = os.path.join(os.path.dirname(__file__), "..", "dataset", f"{lang}.txt")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    texts.append(t)
                    labels.append(lang)
    
    # Load additional feedback data from user corrections
    feedback_texts, feedback_labels = load_feedback_data()
    texts.extend(feedback_texts)
    labels.extend(feedback_labels)
    
    return texts, labels


def load_feedback_data():
    """Load user-corrected predictions for active learning."""
    if not os.path.exists(FEEDBACK_LOG):
        return [], []
    
    try:
        with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
            feedback_list = json.load(f)
        
        texts = [item["text"] for item in feedback_list if item.get("corrected")]
        labels = [item["correct_label"] for item in feedback_list if item.get("corrected")]
        return texts, labels
    except:
        return [], []


def save_feedback(text, predicted_label, correct_label):
    """Save user feedback for active learning."""
    try:
        feedback_list = []
        if os.path.exists(FEEDBACK_LOG):
            with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
                feedback_list = json.load(f)
        
        feedback_list.append({
            "text": text,
            "predicted": predicted_label,
            "correct_label": correct_label,
            "corrected": True
        })
        
        with open(FEEDBACK_LOG, "w", encoding="utf-8") as f:
            json.dump(feedback_list, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving feedback: {e}")


class NPLM(nn.Module):
    """Neural Probabilistic Language Model with improved architecture.
    
    Improvements:
    1. Character n-grams for better language distinction
    2. Larger embedding dimension for richer representations
    3. Dropout for regularization
    4. Multiple hidden layers for better feature extraction
    """
    def __init__(self, input_dim, emb_dim=128, hidden_size=256, output_size=3, dropout=0.3):
        super().__init__()
        # Embedding layer - produces continuous representation
        self.fc_embed = nn.Linear(input_dim, emb_dim, bias=False)
        self.dropout_embed = nn.Dropout(dropout)
        
        # Multi-layer hidden network
        self.fc1 = nn.Linear(emb_dim, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        # Layer normalization for stable training
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size // 2)

    def forward(self, x):
        # x: (batch_size, vocab_size) float tensor (BoW or TF-IDF)
        rep = self.fc_embed(x)
        rep = self.dropout_embed(rep)
        
        h = torch.relu(self.fc1(rep))
        h = self.ln1(h)
        h = self.dropout1(h)
        
        h = torch.relu(self.fc2(h))
        h = self.ln2(h)
        h = self.dropout2(h)
        
        return self.fc3(h)


def train_model(epochs=30, save_model_path=None, save_vectorizer_path=None):
    """Train model with improved preprocessing (character n-grams + TF-IDF)."""
    texts, labels = load_dataset()
    
    # Use TF-IDF with character n-grams for better language distinction
    # character n-grams capture linguistic patterns like suffixes, prefixes
    vectorizer = TfidfVectorizer(
        tokenizer=_pass_through_tokenizer,  # Use module-level function (picklable)
        analyzer='char',  # Character-level n-grams
        ngram_range=(2, 3),  # Use bigrams and trigrams
        max_features=1000,  # Limit features
        lowercase=True,
        encoding='utf-8'
    )
    X = vectorizer.fit_transform(texts).toarray().astype('float32')

    label_to_idx = {"ind": 0, "eng": 1, "sun": 2}
    y = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)

    model = NPLM(input_dim=X.shape[1], emb_dim=128, hidden_size=256, dropout=0.3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print("Epoch:", epoch, "Loss:", loss.item())

    if save_model_path:
        torch.save(model.state_dict(), save_model_path)
    if save_vectorizer_path:
        with open(save_vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)

    model.eval()
    return model, vectorizer


def load_or_create_model(model_path=None, vectorizer_path=None):
    """Load model and vectorizer from disk if available; otherwise train and save them."""
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "nplm-model.pth")
    if vectorizer_path is None:
        vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

    # Try to load existing model, but handle any errors
    try:
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            with open(vectorizer_path, "rb") as f:
                vectorizer = pickle.load(f)
            input_dim = len(vectorizer.vocabulary_)
            model = NPLM(input_dim=input_dim)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            print("✅ Model loaded successfully!")
            return model, vectorizer
    except Exception as e:
        print(f"⚠️  Could not load existing model: {e}")
        print("🔄 Retraining from scratch...")
        # Delete corrupt files
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(vectorizer_path):
            os.remove(vectorizer_path)

    # fallback: train and save
    print("📚 Training new model with improved architecture...")
    model, vectorizer = train_model(epochs=30, save_model_path=model_path, save_vectorizer_path=vectorizer_path)
    print("✅ Model training complete!")
    return model, vectorizer


def predict_text(text, model, vectorizer):
    """Return (label, confidence, debug_info) for a single text input."""
    X = vectorizer.transform([text]).toarray().astype('float32')
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        conf, pred_idx = torch.max(probs, dim=0)

    idx_to_label = {0: "ind", 1: "eng", 2: "sun"}
    label = idx_to_label.get(int(pred_idx.item()), "unknown")
    confidence = float(conf.item())
    
    # Get probability distribution for debugging
    prob_dist = {
        "ind": round(float(probs[0].item()), 4),
        "eng": round(float(probs[1].item()), 4),
        "sun": round(float(probs[2].item()), 4)
    }
    
    return label, confidence, prob_dist
