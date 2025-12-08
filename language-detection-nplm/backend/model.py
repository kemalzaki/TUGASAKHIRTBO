import os
import pickle
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize

languages = ["ind", "eng", "sun"]


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
    return texts, labels


class NPLM(nn.Module):
    """Neural Probabilistic Language Model style classifier.

    Implementation detail: instead of naive embedding lookup per token,
    we aggregate word embeddings by multiplying the BoW/TF vector with an
    embedding matrix (implemented as a linear layer without bias). This
    produces a continuous representation similar to the aggregated
    context representation used in NPLM papers, then a small feed-forward
    network maps it to class logits.
    """
    def __init__(self, input_dim, emb_dim=64, hidden_size=64, output_size=3):
        super().__init__()
        # fc_embed performs X (1 x V) @ W (V x emb_dim) -> (1 x emb_dim)
        self.fc_embed = nn.Linear(input_dim, emb_dim, bias=False)
        self.fc1 = nn.Linear(emb_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, vocab_size) float tensor (BoW or TF-IDF)
        rep = self.fc_embed(x)
        h = torch.relu(self.fc1(rep))
        return self.fc2(h)


def train_model(epochs=30, save_model_path=None, save_vectorizer_path=None):
    texts, labels = load_dataset()
    vectorizer = CountVectorizer(tokenizer=word_tokenize)
    X = vectorizer.fit_transform(texts).toarray().astype('float32')

    label_to_idx = {"ind": 0, "eng": 1, "sun": 2}
    y = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)

    model = NPLM(input_dim=X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
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

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        input_dim = len(vectorizer.vocabulary_)
        model = NPLM(input_dim=input_dim)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model, vectorizer

    # fallback: train and save
    model, vectorizer = train_model(epochs=30, save_model_path=model_path, save_vectorizer_path=vectorizer_path)
    return model, vectorizer


def predict_text(text, model, vectorizer):
    """Return (label, confidence) for a single text input."""
    X = vectorizer.transform([text]).toarray().astype('float32')
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        conf, pred_idx = torch.max(probs, dim=0)

    idx_to_label = {0: "ind", 1: "eng", 2: "sun"}
    return idx_to_label.get(int(pred_idx.item()), "unknown"), float(conf.item())
