import math
import random

# ================================================
# 1. Minimal Autograd Engine
# ================================================
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    __radd__ = __add__

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (1/other)

    def __pow__(self, power):
        out = Value(self.data ** power, (self,), f'**{power}')
        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


# ================================================
# 2. Helper Functions
# ================================================
def softmax(vals):
    # vals: list of Value objects
    max_val = max(v.data for v in vals)
    shifted = [(v - Value(max_val)).exp() for v in vals]
    sum_exp = sum(v.data for v in shifted)
    return [v.data / (sum_exp + 1e-9) for v in shifted]

# ================================================
# 3. Basic Neural Modules (Linear, ReLU)
# ================================================
class Linear:
    def __init__(self, nin, nout):
        # weights: shape (nout, nin) and bias: shape (nout,)
        self.weights = [[Value(random.uniform(-1, 1)) for _ in range(nin)] for _ in range(nout)]
        self.bias = [Value(0.0) for _ in range(nout)]
    def __call__(self, x):
        # x: list of Value (length = nin)
        out = []
        for w, b in zip(self.weights, self.bias):
            dot = sum(wi * xi for wi, xi in zip(w, x)) + b
            out.append(dot)
        return out
    def parameters(self):
        params = []
        for row in self.weights:
            params.extend(row)
        params.extend(self.bias)
        return params

def relu(x):
    # x: list of Value
    return [xi if xi.data > 0 else Value(0.0) for xi in x]

# ================================================
# 4. Self-Attention (Single-Head) Module
# ================================================
class SelfAttention:
    def __init__(self, d_model):
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)
    def __call__(self, X):
        # X: list of token embeddings (each is list of Value of length d_model)
        Q = [self.Wq(x) for x in X]
        K = [self.Wk(x) for x in X]
        V = [self.Wv(x) for x in X]
        outputs = []
        for i in range(len(X)):
            qi = Q[i]
            scores = []
            for j in range(len(X)):
                kj = K[j]
                score = sum(qij * kjj for qij, kjj in zip(qi, kj))
                scores.append(score)
            probs = softmax(scores)
            attended = [Value(0.0) for _ in range(len(V[0]))]
            for j, p in enumerate(probs):
                vj = V[j]
                for k in range(len(vj)):
                    attended[k] = attended[k] + Value(p) * vj[k]
            outputs.append(attended)
        out = [self.Wo(x) for x in outputs]
        return out
    def parameters(self):
        params = []
        for layer in [self.Wq, self.Wk, self.Wv, self.Wo]:
            params.extend(layer.parameters())
        return params

# ================================================
# 5. Feed-Forward Module
# ================================================
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
    def __call__(self, x):
        out = self.linear1(x)
        out = relu(out)
        out = self.linear2(out)
        return out
    def parameters(self):
        return self.linear1.parameters() + self.linear2.parameters()

# ================================================
# 6. Transformer Block (with a very basic normalization)
# ================================================
class TransformerBlock:
    def __init__(self, d_model, d_ff):
        self.attn = SelfAttention(d_model)
        self.ff = FeedForward(d_model, d_ff)
        # Very simple "norm" parameters (not full layer norm)
        self.norm1_scale = [Value(1.0) for _ in range(d_model)]
        self.norm1_bias = [Value(0.0) for _ in range(d_model)]
        self.norm2_scale = [Value(1.0) for _ in range(d_model)]
        self.norm2_bias = [Value(0.0) for _ in range(d_model)]
    def simple_layer_norm(self, x, scale, bias, eps=1e-6):
        mean = sum(x, Value(0.0)) / len(x)
        var = sum((xi - mean) ** 2 for xi in x) / len(x)
        std = Value(math.sqrt(var.data + eps))
        return [scale[i] * ((x[i] - mean) / std) + bias[i] for i in range(len(x))]
    def __call__(self, X):
        # Self-attention sublayer with residual connection
        attn_out = self.attn(X)
        X1 = []
        for i in range(len(X)):
            res = [X[i][j] + attn_out[i][j] for j in range(len(X[i]))]
            normed = self.simple_layer_norm(res, self.norm1_scale, self.norm1_bias)
            X1.append(normed)
        # Feed-forward sublayer with residual connection
        X2 = []
        for i in range(len(X1)):
            ff_out = self.ff(X1[i])
            res = [X1[i][j] + ff_out[j] for j in range(len(X1[i]))]
            normed = self.simple_layer_norm(res, self.norm2_scale, self.norm2_bias)
            X2.append(normed)
        return X2
    def parameters(self):
        params = []
        params.extend(self.attn.parameters())
        params.extend(self.ff.parameters())
        params.extend(self.norm1_scale + self.norm1_bias + self.norm2_scale + self.norm2_bias)
        return params

# ================================================
# 7. Transformer Language Model
# ================================================
class TransformerLM:
    def __init__(self, vocab, d_model, d_ff, num_layers, max_len):
        self.vocab = vocab  # list of tokens (here, characters)
        self.vocab_size = len(vocab)
        self.d_model = d_model
        # Embedding table: map each token to a vector (list of Value)
        self.embeddings = { token: [Value(random.uniform(-1,1)) for _ in range(d_model)] for token in vocab }
        self.blocks = [TransformerBlock(d_model, d_ff) for _ in range(num_layers)]
        self.final_linear = Linear(d_model, self.vocab_size)
    def forward(self, tokens):
        # tokens: list of tokens (strings)
        X = [self.embeddings[token] for token in tokens]
        # (For simplicity, we omit positional encoding here)
        for block in self.blocks:
            X = block(X)
        logits = [self.final_linear(x) for x in X]
        return logits
    def parameters(self):
        params = []
        for emb in self.embeddings.values():
            params.extend(emb)  # note: embeddings are Value objects
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_linear.parameters())
        return params

# ================================================
# 8. Tokenization and Loss
# ================================================
def tokenize(text):
    # Simple character-level tokenizer
    return list(text)

def cross_entropy_loss(logits, target_index, eps=1e-9):
    probs = softmax(logits)
    loss = -math.log(probs[target_index] + eps)
    return Value(loss)

# ================================================
# 9. Training Loop (Naive SGD)
# ================================================
def train(model, data, epochs=5, lr=0.01):
    tokens = tokenize(data)
    seq_len = 10  # use very short sequences for demonstration
    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(len(tokens) - seq_len - 1):
            inp = tokens[i : i + seq_len]
            target = tokens[i + 1 : i + seq_len + 1]
            logits_seq = model.forward(inp)
            loss = Value(0.0)
            for j, logits in enumerate(logits_seq):
                try:
                    target_index = model.vocab.index(target[j])
                except ValueError:
                    continue
                loss = loss + cross_entropy_loss(logits, target_index)
            loss.backward()
            total_loss += loss.data
            for p in model.parameters():
                p.data -= lr * p.grad
                p.grad = 0.0
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")

# ================================================
# 10. Chat Function (Sampling)
# ================================================
def chat(model, prompt, max_length=50):
    tokens = tokenize(prompt)
    generated = tokens[:]
    for _ in range(max_length):
        logits_seq = model.forward(generated)
        last_logits = logits_seq[-1]
        probs = softmax(last_logits)
        r = random.random()
        cum = 0.0
        chosen = 0
        for i, p in enumerate(probs):
            cum += p
            if cum >= r:
                chosen = i
                break
        next_token = model.vocab[chosen]
        generated.append(next_token)
        if next_token == "\n":
            break
    return "".join(generated)

# ================================================
# 11. Chat with Basic Reasoning
# ================================================
def chat_with_reasoning(model, prompt, reasoning_max=30, answer_max=50):
    # Step 1: Ask the model to think step by step.
    reasoning_prompt = "Let's think step by step: " + prompt + "\nChain-of-thought:"
    reasoning_response = chat(model, reasoning_prompt, max_length=reasoning_max)
    # Extract the chain-of-thought part (remove the prompt)
    chain_of_thought = reasoning_response[len(reasoning_prompt):].strip()
    print("Chain-of-Thought:", chain_of_thought)
    
    # Step 2: Now ask for an answer using the reasoning.
    answer_prompt = reasoning_response + "\nAnswer:"
    answer_response = chat(model, answer_prompt, max_length=answer_max)
    # Extract the final answer part.
    answer = answer_response[len(answer_prompt):].strip()
    print("Answer:", answer)
    return answer_response

# ================================================
# 12. Main: Train and Chat with Reasoning
# ================================================
if __name__ == "__main__":
    # Use a tiny sample text for training (toy dataset)
    sample_text = "Hello, how are you?\nI'm fine, thank you.\nTell me a joke.\nWhy did the chicken cross the road? To get to the other side!\nThat's funny.\n"
    # Build vocabulary (character-level)
    vocab = sorted(list(set(sample_text)))
    # Initialize model with small dimensions (for demonstration)
    model = TransformerLM(vocab, d_model=16, d_ff=32, num_layers=1, max_len=50)
    # Train on repeated sample text
    train_data = sample_text * 10
    train(model, train_data, epochs=5, lr=0.01)
    
    # Use the new chat function with basic reasoning
    prompt = "Tell me a joke."
    print("\n--- Chat with Basic Reasoning ---")
    chat_with_reasoning(model, prompt, reasoning_max=30, answer_max=50)
