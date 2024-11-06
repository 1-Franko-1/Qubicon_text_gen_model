"""
=======================================================================================================================================
==================================================== QUBICON Text Generation PTTM-1 ===================================================
=======================================================================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from colorama import Fore, Style, init

# Header with color
print(Fore.CYAN + "=" * 128)
print(Fore.CYAN + "=" * 48 + " QUBICON Text Generation PTTM-1 " + "=" * 48)
print(Fore.CYAN + "=" * 128)
print(Fore.WHITE)

# Parameters!
d_model = 512  # Dimensionality of the model
num_heads = 8  # Number of attention heads in each MultiHeadAttention layer
num_layers = 6  # Number of encoder and decoder layers
d_ff = 2048  # Dimensionality of the feedforward layer
max_seq_length = 150  # Maximum sequence length for inputs
dropout = 0.1  # Dropout rate to prevent overfitting
learning_rate = 1e-4  # Learning rate for the optimizer
batch_size = 64  # Batch size for training
num_epochs = 15  # Number of training epochs
beam_width = 3  # Beam width for beam search during text generation
temperature = 0.7  # Temperature for controlling randomness in generation
max_len = 50  # Maximum length for generated sequences
weight_decay = 1e-5  # Weight decay for L2 regularization
grad_clip = 1.0  # Gradient clipping to prevent exploding gradients

# Define the MultiHeadAttention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            attn_scores += (mask * -1e9)
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x):
        batch_size = x.size(0)
        seq_len = x.size(2)
        x = x.permute(0, 2, 1, 3)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

# Define the PositionWiseFeedForward class
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# Define the PositionalEncoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.positional_encoding = self.create_positional_encoding()

    def create_positional_encoding(self):
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pos_encodings = np.zeros((self.max_len, self.d_model))
        pos_encodings[:, 0::2] = np.sin(position * div_term)
        pos_encodings[:, 1::2] = np.cos(position * div_term)
        return torch.tensor(pos_encodings, dtype=torch.float32)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError("Sequence length exceeds maximum length.")
        pos_encodings = self.positional_encoding[:seq_len, :]
        return x + pos_encodings.unsqueeze(0)

# Define the EncoderLayer class
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.attn(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ffn_output))
        return x

# Define the DecoderLayer class
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttention(d_model, num_heads)
        self.attn2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output1 = self.attn1(x, x, x, tgt_mask)
        x = self.layer_norm1(x + self.dropout(attn_output1))
        attn_output2 = self.attn2(x, enc_output, enc_output, src_mask)
        x = self.layer_norm2(x + self.dropout(attn_output2))
        ffn_output = self.ffn(x)
        x = self.layer_norm3(x + self.dropout(ffn_output))
        return x

# Define the Transformer model class
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_length=300, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        emb_src = self.dropout(self.positional_encoding(self.embedding(src)))
        for layer in self.encoder_layers:
            emb_src = layer(emb_src, src_mask)
        
        emb_tgt = self.dropout(self.positional_encoding(self.embedding(tgt)))
        for layer in self.decoder_layers:
            emb_tgt = layer(emb_tgt, emb_src, src_mask, tgt_mask)
        
        return self.fc(emb_tgt)

# Tokenization and detokenization functions
def tokenize(text, word_to_token):
    return [word_to_token.get(word, 0) for word in text.split()]

def detokenize(tokens, token_to_word):
    return ' '.join([token_to_word.get(token, '') for token in tokens if token != 0])

# Define a Dataset class for the text data
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, knowledge_base, word_to_token, max_length=150):  # Ensure max_length matches model's max_seq_length
        self.max_length = max_length
        self.data = [self.process_sequence(entry['source'], word_to_token) for entry in knowledge_base]

    def process_sequence(self, text, word_to_token):
        tokenized = self.tokenize(text, word_to_token)
        if len(tokenized) > self.max_length:
            tokenized = tokenized[:self.max_length]  # Truncate if the sequence is too long
        elif len(tokenized) < self.max_length:
            tokenized = tokenized + [word_to_token['<pad>']] * (self.max_length - len(tokenized))  # Pad if the sequence is too short
        return tokenized

    def tokenize(self, text, word_to_token):
        return [word_to_token.get(word, word_to_token['<pad>']) for word in text.split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        return torch.tensor(sequence, dtype=torch.long)

# Load the knowledge base from a JSON file
with open('data.json', 'r') as f:
    knowledge_base = json.load(f)

# Create word-to-token and token-to-word mappings
word_to_token = {}
token_to_word = {}
current_token = 2

for entry in knowledge_base:
    for word in entry['source'].split():
        if word not in word_to_token:
            word_to_token[word] = current_token
            token_to_word[current_token] = word
            current_token += 1

word_to_token['<pad>'] = 0
word_to_token['<start>'] = 1
token_to_word[0] = '<pad>'
token_to_word[1] = '<start>'

# Train-test split
train_data, val_data = train_test_split(knowledge_base, test_size=0.1)


# Create datasets
train_dataset = TextDataset(train_data, word_to_token)
val_dataset = TextDataset(val_data, word_to_token)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
vocab_size = len(word_to_token)
model = TransformerModel(vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_length=150, dropout=0.1)

# Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)

def train_epoch():
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        src = batch[:, :-1].to(device)
        tgt = batch[:, 1:].to(device)
        output = model(src, tgt)
        loss = criterion(output.reshape(-1, vocab_size), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Initialize the model
vocab_size = len(word_to_token)
model = TransformerModel(vocab_size, d_model=d_model, num_heads=num_heads, num_layers=num_layers, d_ff=d_ff, max_seq_length=max_seq_length, dropout=dropout)

# Define the model save path
model_save_path = "PTTM_1.pth"

# Check if pretrained model exists
if os.path.exists(model_save_path):
    # Load the pretrained model
    model.load_state_dict(torch.load(model_save_path))
    model.eval()  # Set the model to evaluation mode after loading
    print(f"Pretrained model loaded from {model_save_path}")
else:
    # Pretrained model not found, start training
    print("No pretrained model found, starting from scratch.")
    
    # Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # Training Loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    def train_epoch():
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            src = batch[:, :-1].to(device)
            tgt = batch[:, 1:].to(device)
            output = model(src, tgt)
            loss = criterion(output.reshape(-1, vocab_size), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    # Training Loop
    for epoch in range(num_epochs):
        train_loss = train_epoch()
        print(f"Epoch {epoch + 1}, Loss: {train_loss}")
        scheduler.step()

    # Save the model after training
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Beam Search for Text Generation
def beam_search(model, start_token, word_to_token, token_to_word, beam_width=beam_width, max_len=max_len):
    model.eval()
    tokens = [start_token]
    current_beam = [(tokens, 0)]  # (sequence, score)
    
    for _ in range(max_len):
        all_candidates = []
        for seq, score in current_beam:
            input_seq = torch.tensor(seq).unsqueeze(0).to(device)
            output = model(input_seq, input_seq)  # Self-attention decoding
            probs = torch.nn.functional.softmax(output[0, -1] / temperature, dim=-1)
            top_probs, top_tokens = probs.topk(beam_width)
            
            for prob, token in zip(top_probs, top_tokens):
                candidate = (seq + [token.item()], score - torch.log(prob).item())  # Minimize negative log-likelihood
                all_candidates.append(candidate)
        
        current_beam = sorted(all_candidates, key=lambda x: x[1])[:beam_width]
        tokens = current_beam[0][0]  # Best sequence

    return detokenize(tokens[1:], token_to_word)

# Chat Interface
def chat_with_ai():
    while True:
        user_input = input("You: ")
        
        # Tokenize input and add batch dimension, ensure dtype is long
        tokenized_input = torch.tensor(tokenize(user_input, word_to_token), dtype=torch.long).unsqueeze(0).to(device)
        
        # Run model inference without gradients
        with torch.no_grad():
            # Pass the tokenized input to the model
            output = model(tokenized_input, tokenized_input)
            
            # Get the predicted token indices (after applying argmax)
            predicted_tokens = output.argmax(dim=-1).squeeze(0).tolist()
        
        # Detokenize the predicted tokens (skip <start> token, index 1)
        response = detokenize(predicted_tokens[1:], token_to_word)
        print(f"AI: {response}")

# Start the chat
chat_with_ai()