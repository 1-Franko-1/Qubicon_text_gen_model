"""
=======================================================================================================================================
=                                                       VersaGen TEXT GENERATION                                                      =
=======================================================================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from colorama import Fore, Style, init
import threading
import signal
import time

# Initialize colorama to enable color support on Windows
init()

scaler = GradScaler()  # Initialize GradScaler for mixed precision training

# Header with color
print(Style.BRIGHT + Fore.CYAN + "=" * 128)
print(Fore.CYAN + "=" + " " * 126 + "=")
print(Fore.CYAN + "=" + " " * 53 + Fore.WHITE + " VersaGen Text Gen " + Fore.BLUE + " " * 54 + "=")
print(Fore.CYAN + "=" + " " * 126 + "=")
print(Fore.CYAN + "=" * 128 + Fore.WHITE)

# Define the model save path
model_save_path = "VersaGen_textmodel.pth"

def select_parameters():
    print(Fore.CYAN + "=" + " " * 126 + "=")
    print(Fore.CYAN + "=" + " " * 26 + Fore.WHITE + "Please select the model configuration based on your system's capabilities:" + Fore.CYAN + " " * 26 + "=")
    print(Fore.CYAN + "=" + " " * 45 + Fore.RED + "1. Impossible (~2 trilion parameters)" + Fore.CYAN + " " * 44 + "=")
    print(Fore.CYAN + "=" + " " * 45 + Fore.RED + "2. Ultra (~50 billion parameters)" + Fore.CYAN + " " * 48 + "=")
    print(Fore.CYAN + "=" + " " * 45 + Fore.YELLOW + "3. High (~7.5 billion parameters)" + Fore.CYAN + " " * 48 + "=")
    print(Fore.CYAN + "=" + " " * 45 + Fore.YELLOW + "4. Mid (~1.6 billion parameters)" + Fore.CYAN + " " * 49 + "=")
    print(Fore.CYAN + "=" + " " * 45 + Fore.GREEN + "5. Low (~600 million parameters)" + Fore.CYAN + " " * 49 + "=")
    print(Fore.CYAN + "=" + " " * 45 + Fore.GREEN + "6. Ultra Low (~25 million parameters)" + Fore.CYAN + " " * 44 + "=")
    print(Fore.CYAN + "=" + " " * 126 + "=")
    print(Fore.CYAN + "=" * 128)
    print(Fore.CYAN + "=" + " " * 126 + "=")

    choice = input(Fore.CYAN + "= " + Fore.WHITE + "Enter the number corresponding to your choice: ")

    print(Fore.CYAN + "=" + " " * 126 + "=")
    print(Fore.CYAN + "=" * 128 + Fore.WHITE)
    
    if choice == '1':
        # Impossible configuration (~2 trillion parameters)
        d_model = 20000
        num_heads = 256
        num_layers = 160
        d_ff = 80000
    elif choice == '2':
        # Ultra configuration (~50 billion parameters)
        d_model = 8192   
        num_heads = 128
        num_layers = 80
        d_ff = 32768
    elif choice == '3':
        # High configuration (~7.5 billion parameters)
        d_model = 4096
        num_heads = 64
        num_layers = 48
        d_ff = 16384
        grad_clip = 1.0
    elif choice == '4':
        # Mid configuration (~1.5 billion parameters)
        d_model = 2048
        num_heads = 32
        num_layers = 24
        d_ff = 8192
    elif choice == '5':
        # Low configuration (~600 million parameters)
        d_model = 1536
        num_heads = 24
        num_layers = 16
        d_ff = 6144
    elif choice == '6':
        # Ultra Low configuration (~25 million parameters)
        d_model = 512
        num_heads = 8
        num_layers = 6
        d_ff = 2048
    elif choice == '7':
        # Scaled down config (~250 parameters)
        d_model = 2    # Kept from previous 2
        num_heads = 1  # Kept from previous 1
        num_layers = 1 # Kept from previous 1
        d_ff = 2       # Further reduced from 4
    else:
        print("Invalid choice")
        exit()

    # Return the selected configuration
    return d_model, num_heads, num_layers, d_ff, choice

# Call the function to select parameters
d_model, num_heads, num_layers, d_ff, choice = select_parameters()

dropout = 0.1
learning_rate = 1e-4
batch_size = 256
temperature = 0.7
weight_decay = 1e-5
grad_clip = 1.0
num_epochs = 1
max_len = 65000
max_seq_length = 25000
datasetfile = '14kdata.txt'

def stop_thread(signum, frame):
    global stop_generation
    stop_generation = True

# Register signal handler for interrupt (Ctrl+C)
signal.signal(signal.SIGINT, stop_thread)

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
            # Make sure the mask has the correct shape: (batch_size, num_heads, seq_len, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)  # Add dimensions for num_heads and seq_len
            attn_scores = attn_scores + mask  # Assuming mask is of shape (batch_size, 1, seq_len, seq_len)
            
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
                if stop_generation:
                    print(Fore.WHITE + "\nGeneration stopped!")
                    break

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
    tokens = [
        subword_vocab[word] if word in subword_vocab 
        else [subword_vocab.get(char, 0) for char in word]
        for word in text.split()
    ]
    # Flatten the list in case of out-of-vocabulary words
    return [item for sublist in tokens for item in (sublist if isinstance(sublist, list) else [sublist])]

def detokenize(tokens, idx_to_word):
    words = []
    for token in tokens:
        word = idx_to_word.get(token, '')

        # Skip special tokens
        if word in {'<start>', '<pad>'}:
            continue

        if word:
            if word.startswith("▁"):  # Subword marker
                words.append(word[1:])  # Remove the subword marker and concatenate
            else:
                words.append(word)  # Regular subword or character

    # Join and clean excessive spaces efficiently
    return ' '.join(words).strip()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

# Define a Dataset class for the text data
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, knowledge_base, word_to_token, max_length=150):
        self.max_length = max_length
        self.data = [self.process_sequence(entry['source'], word_to_token) for entry in knowledge_base]

    def process_sequence(self, text, word_to_token):
        tokenized = self.tokenize(text, word_to_token)
        if len(tokenized) > self.max_length:
            tokenized = tokenized[:self.max_length]  # Truncate if too long
        elif len(tokenized) < self.max_length:
            tokenized = tokenized + [word_to_token['<pad>']] * (self.max_length - len(tokenized))  # Pad if too short
        return tokenized

    def tokenize(self, text, word_to_token):
        return [word_to_token.get(word, word_to_token['<pad>']) for word in text.split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        return torch.tensor(sequence, dtype=torch.long)

# Load the knowledge base from a .txt file
knowledge_base = []
with open(datasetfile, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        if line.strip():  # Skip empty lines
            knowledge_base.append({'source': line.strip()})

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

# Add special tokens
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

# Training function
def train_epoch(model, data_loader, optimizer, criterion, scaler, grad_clip):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training", ncols=100):
        inputs = batch.to(device)
        targets = batch.to(device)
        
        optimizer.zero_grad()

        with autocast():  # Mixed precision training
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(data_loader)

# Validation function
def validate_epoch(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating", ncols=100):
            inputs = batch.to(device)
            targets = batch.to(device)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()
    return total_loss / len(data_loader)

# Initialize the model
vocab_size = len(word_to_token)
model = TransformerModel(vocab_size, d_model=d_model, num_heads=num_heads, num_layers=num_layers, d_ff=d_ff, max_seq_length=max_seq_length, dropout=dropout)

# Count parameters
total_params = count_parameters(model)
print(Fore.CYAN + "=" + " " * 126 + "=")
print(Fore.CYAN + "=" + " " * 26 + Fore.WHITE + f"Total number of parameters in the model: {total_params / 1e6:.2f} million")

# Check if pretrained model exists
if os.path.exists(model_save_path):
    # Load the pretrained model
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model.eval()  # Set the model to evaluation mode after loading
    print(Fore.CYAN + "=" + " " * 26 + Fore.WHITE + f"Pretrained model loaded from {model_save_path}" + Fore.CYAN + " " * 49 + "=")
    print(Fore.CYAN + "=" + " " * 126 + "=")
    print(Fore.CYAN + "=" * 128 + Fore.WHITE)
else:
    # Pretrained model not found, start training
    print(Fore.CYAN + "=" + " " * 26 + Fore.WHITE + "No pretrained model found, starting from scratch." + Fore.CYAN + " " * 51 + "=")
    print(Fore.CYAN + "=" + " " * 126 + "=")
    print(Fore.CYAN + "=" * 128 + Fore.WHITE)

    # Training Loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training Loop
    best_val_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, grad_clip)
        val_loss = validate_epoch(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved with validation loss: {val_loss:.4f}")

        # Step the scheduler
        scheduler.step(val_loss)

    # Save the model after training
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

generation_thread = None 
stop_event = threading.Event()

def user_feedback():
    feedback = input(Fore.WHITE + "Was the response correct? (yes/no): ").strip().lower()
    return feedback

def adjust_learning_rate(optimizer, feedback, reward_factor=1.1, punishment_factor=0.9):
    # Adjust the learning rate based on feedback
    for param_group in optimizer.param_groups:
        if feedback == "yes":
            param_group['lr'] *= reward_factor  # Increase learning rate for reward
            print("Learning rate increased.")
        else:
            param_group['lr'] *= punishment_factor  # Decrease learning rate for punishment
            print("Learning rate decreased.")

def generate_in_thread():
    global stop_generation
    while not stop_generation:
        user_input = input(Fore.WHITE + "You: ")
        if user_input.lower() == "exit":  # Allow user to type 'exit' to stop the chat
            print(Fore.GREEN + "Exiting chat...")
            stop_event.set()
            generation_thread.set()
            quit()
        try:
            # Generate text based on the user input
            generated_text = model.generate_text(user_input, max_len=50, temperature=0.7, top_k=50, top_p=0.9)
            if not stop_generation:  # Only print if generation was not stopped
                print(Fore.GREEN + f"AI: {generated_text}")
                feedback = user_feedback()
                adjust_learning_rate(optimizer, feedback)
        except Exception as e:
            print(Fore.RED + f"Error: {str(e)}")
            break

def start_text_generation():
    global stop_generation, generation_thread
    stop_generation = False  # Reset the stop flag
    generation_thread = threading.Thread(target=generate_in_thread)
    generation_thread.start()

def main():
    try:
        while True:
            if not generation_thread or not generation_thread.is_alive():
                start_text_generation()  # Start a new generation thread if not already running

            while generation_thread.is_alive():
                time.sleep(1)  # Main thread can do other work here

    except KeyboardInterrupt:
        print(Fore.WHITE + "Program manually stopped by user.")
        stop_generation = True
        if generation_thread:
            generation_thread.join()  # Wait for the generation thread to finish
        sys.exit(0)  # Exit the program gracefully

if __name__ == "__main__":
    main()