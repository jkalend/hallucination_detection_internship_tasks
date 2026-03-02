import numpy as np
import re
from collections import Counter

def preprocess(text):
    # Very basic preprocessing: lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return words

class Word2VecSGNS:
    def __init__(self, vocab_size, embedding_dim=50, learning_rate=0.01, n_negs=5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        self.n_negs = n_negs
        
        # Initialize weights
        # W1: Input-to-hidden (Target word embeddings)
        # W2: Hidden-to-output (Context word embeddings)
        self.W1 = np.random.uniform(-0.5 / embedding_dim, 0.5 / embedding_dim, (vocab_size, embedding_dim))
        self.W2 = np.random.uniform(-0.5 / embedding_dim, 0.5 / embedding_dim, (vocab_size, embedding_dim))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

    def train_step(self, target_idx, context_idx, negative_indices):
        # Forward pass
        # Target embedding
        v_w = self.W1[target_idx]
        
        # Positive sample
        u_pos = self.W2[context_idx]
        z_pos = np.dot(v_w, u_pos)
        y_pos = self.sigmoid(z_pos)
        
        # Negative samples
        u_negs = self.W2[negative_indices]
        z_negs = np.dot(u_negs, v_w)
        y_negs = self.sigmoid(z_negs)
        
        # Loss (Negative Sampling Objective)
        # J = -log(sigmoid(u_pos . v_w)) - sum(log(sigmoid(-u_neg . v_w)))
        loss = -np.log(y_pos + 1e-10) - np.sum(np.log(1 - y_negs + 1e-10))
        
        # Gradients
        grad_z_pos = y_pos - 1
        grad_z_negs = y_negs
        
        grad_u_pos = grad_z_pos * v_w
        grad_u_negs = grad_z_negs.reshape(-1, 1) * v_w.reshape(1, -1)
        
        grad_v_w = grad_z_pos * u_pos + np.dot(grad_z_negs, u_negs)
        
        # Update weights
        np.add.at(self.W1, target_idx, -self.lr * grad_v_w)
        np.add.at(self.W2, context_idx, -self.lr * grad_u_pos)
        np.add.at(self.W2, negative_indices, -self.lr * grad_u_negs)
        
        return loss

def get_batches(words, word_to_id, window_size=2):
    for i, word in enumerate(words):
        target_idx = word_to_id[word]
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        
        for j in range(start, end):
            if i == j: 
                continue
            context_idx = word_to_id[words[j]]
            yield target_idx, context_idx

def train():
    # Small example dataset
    text = """
    the quick brown fox jumps over the lazy dog
    machine learning is a field of artificial intelligence
    natural language processing is a subfield of linguistics
    word embeddings are a type of word representation
    deep learning models are used for many tasks
    """
    words = preprocess(text)
    vocab = sorted(list(set(words)))
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}
    vocab_size = len(vocab)
    
    # Negative sampling distribution
    counts = Counter(words)
    probs = np.array([counts[id_to_word[i]] for i in range(vocab_size)])
    probs = np.power(probs, 0.75)
    probs /= np.sum(probs)
    
    model = Word2VecSGNS(vocab_size, embedding_dim=10, learning_rate=0.05, n_negs=5)
    
    all_ids = np.arange(vocab_size)
    epochs = 100
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for target, context in get_batches(words, word_to_id):
            # Sample negative indices
            candidate_ids = all_ids[all_ids != context]
            candidate_probs = probs[candidate_ids]
            candidate_probs /= candidate_probs.sum()
            replace = model.n_negs > candidate_ids.size
            negs = np.random.choice(
                candidate_ids, size=model.n_negs, replace=replace, p=candidate_probs
            )
            loss = model.train_step(target, context, negs)
            total_loss += loss
            count += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/count:.4f}")

    # Simple similarity test
    def get_sim(w1, w2):
        v1 = model.W1[word_to_id[w1]]
        v2 = model.W1[word_to_id[w2]]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    print("\nSimilarity examples:")
    try:
        print(f"learning - machine: {get_sim('learning', 'machine'):.4f}")
        print(f"learning - dog: {get_sim('learning', 'dog'):.4f}")
    except KeyError:
        pass

if __name__ == "__main__":
    train()
