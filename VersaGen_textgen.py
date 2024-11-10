"""
=======================================================================================================================================
=                                                       VersaGen TEXT GENERATION                                                      =
=======================================================================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import math
import os
from collections import defaultdict
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from colorama import Fore, Style, init

# Header with color
print(Fore.CYAN + "=" * 128)
print(Fore.CYAN + "=" + " " * 53 + " VersaGen Text Gen " + " " * 54 + "=")
print(Fore.CYAN + "=" * 128)
print(Fore.WHITE)

# Define the model save path
model_save_path = "PTTM_1.pth"

def select_parameters():
    print("\nPlease select the model configuration based on your system's capabilities:")
    print("1. High-end (2.5 billion parameters)")
    print("2. Mid-range (352.82 million parameters)")
    print("3. Low-end (44.19 million parameters)")
    
    choice = input("Enter the number corresponding to your choice: ")
    
    if choice == '1':
        # High-end configuration
        d_model = 2048
        num_heads = 32
        num_layers = 24
        d_ff = 8192
        dropout = 0.1
        learning_rate = 1e-4
        batch_size = 64
        beam_width = 3
        temperature = 0.7
        weight_decay = 1e-5
        grad_clip = 1.0
    elif choice == '2':
        # Mid-range configuration
        d_model = 1024
        num_heads = 16
        num_layers = 12
        d_ff = 4096
        dropout = 0.1
        learning_rate = 1e-4
        batch_size = 64
        beam_width = 3
        temperature = 0.7
        weight_decay = 1e-5
        grad_clip = 1.0
    elif choice == '3':
        # Low-end configuration
        d_model = 512
        num_heads = 8
        num_layers = 6
        d_ff = 2048
        dropout = 0.1
        learning_rate = 1e-4
        batch_size = 64
        beam_width = 3
        temperature = 0.7
        weight_decay = 1e-5
        grad_clip = 1.0
    else:
        print("Invalid choice")
        exit()

    # Return the selected configuration
    return d_model, num_heads, num_layers, d_ff, dropout, learning_rate, batch_size, beam_width, temperature, weight_decay, grad_clip

# Call the function to select parameters
d_model, num_heads, num_layers, d_ff, dropout, learning_rate, batch_size, beam_width, temperature, weight_decay, grad_clip = select_parameters()

num_epochs = 10
max_len = 100
max_seq_length = 300
datasetfile = 'data.json'  # Dataset file

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
            mask = mask.unsqueeze(0)  # Add a batch dimension if missing
            attn_scores += mask * -1e9
            
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
        
        # Only keep embedding, positional encoding, and decoder layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_length)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None):
        emb_tgt = self.dropout(self.positional_encoding(self.embedding(tgt)))
        
        for layer in self.decoder_layers:
            emb_tgt = layer(emb_tgt, emb_tgt, tgt_mask, tgt_mask)  # Only use the target (no encoder input)
        
        return self.fc(emb_tgt)

    def sample_next_token(self, logits, temperature=1.0, top_k=0, top_p=0.0):
        # Sample token from the logits (same function as before)
        if temperature != 1.0:
            logits = logits / temperature
        
        pad_token_id = word_to_token['<pad>']
        logits[pad_token_id] = -float('Inf')  # Ensure padding token is not selected
        
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Ensure top_k is not greater than vocab size
            values, indices = torch.topk(logits, k=top_k)
            logits = torch.zeros_like(logits).scatter_(-1, indices, values)
        
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_logits[sorted_indices_to_remove] = -float('Inf')
            logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
        
        logits = torch.clamp(logits, min=-100.0, max=100.0)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        return next_token

    def generate_text(self, prompt, max_len=50, temperature=1.0, top_k=50, top_p=0.9):
        self.eval()  # Set the model to evaluation mode
        
        # Tokenize the input prompt
        tokens = torch.tensor(tokenize(prompt, word_to_token), dtype=torch.long).unsqueeze(0).to(device)
        
        generated = tokens  # Start with the prompt (initial tokens)

        # Create a progress bar for token generation
        for _ in tqdm(range(max_len), desc="Generating text", ncols=100):
            seq_len = generated.size(1)
            tgt_mask = self.generate_nopeak_mask(seq_len)  # Generate the target mask
            
            # Get the logits from the model for the current generated sequence
            output = self(generated, tgt_mask=tgt_mask)  # Only use the target (no encoder input)
            logits = output[0, -1]  # Get logits for the last token
            
            # Sample the next token based on temperature, top_k, and top_p
            next_token = self.sample_next_token(logits, temperature, top_k, top_p)
            
            # Append the next token to the generated sequence
            generated = torch.cat([generated, torch.tensor([[next_token]]).to(device)], dim=1)
        
        # Convert generated token IDs back to words
        generated_text = detokenize(generated.squeeze(0).tolist(), token_to_word)
        return generated_text

    def generate_nopeak_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(device)

def tokenize(text, subword_vocab):
    tokens = []
    for word in text.split():
        if word in subword_vocab:
            tokens.append(subword_vocab[word])
        else:
            # Split OOV words into subwords by character
            tokens.extend(subword_vocab.get(char, 0) for char in word)
    return tokens

def detokenize(tokens, idx_to_word):
    words = []
    
    for token in tokens:
        word = idx_to_word.get(token, '')

        # Skip <start> and <pad> tokens
        if word == '<start>' or word == '<pad>':
            continue

        # If the token represents a recognized word/subword, add it as-is.
        if word:
            if word.startswith("▁"):
                words.append(word.replace("▁", " "))  # Handle subword token correctly
            else:
                words.append(" " + word)  # Add space before word if no subword marker
        else:
            continue

    # Join words and handle excessive whitespace
    text = ''.join(words).strip()
    return re.sub(r'\s+', ' ', text)  # Clean up any extra spaces

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

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
with open(datasetfile, 'r') as f:
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

# Training Loop
def train_epoch():
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        
        # Source (input sequence) and Target (next-word prediction)
        src = batch[:, :-1].to(device)
        tgt = batch[:, 1:].to(device)
        
        # Forward pass: predict next word
        output = model(src, tgt)
        
        # Calculate loss based on next-word prediction
        loss = criterion(output.reshape(-1, vocab_size), tgt.reshape(-1))
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        # Optimization step
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

# Initialize the model
vocab_size = len(word_to_token)
model = TransformerModel(vocab_size, d_model=d_model, num_heads=num_heads, num_layers=num_layers, d_ff=d_ff, max_seq_length=max_seq_length, dropout=dropout)

# Count parameters
total_params = count_parameters(model)
print(f"Total number of parameters in the model: {total_params / 1e6:.2f} million")

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
        print(f"Epoch {epoch + 1} / {num_epochs}, Loss: {train_loss}")
        scheduler.step()

    # Save the model after training
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Chat Interface
def chat_with_ai():
    while True:
        user_input = input("You: ")
        
        # Generate text based on the user input
        generated_text = model.generate_text(user_input, max_len=50, temperature=0.7, top_k=50, top_p=0.9)
        
        print(f"AI: {generated_text}")
        
# Start the chat
chat_with_ai()