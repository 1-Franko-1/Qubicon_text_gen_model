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
import json
import sys
import math
import os
from collections import defaultdict
import re
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
    """
    Prompts the user to select a model configuration based on system capabilities.
    
    Returns:
        tuple: Model parameters (d_model, num_heads, num_layers, d_ff).
    """

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
    return d_model, num_heads, num_layers, d_ff

# Call the function to select parameters
d_model, num_heads, num_layers, d_ff = select_parameters()

dropout = 0.1
learning_rate = 1e-4
batch_size = 256
temperature = 0.7
weight_decay = 1e-5
grad_clip = 1.0
num_epochs = 50
max_len = 65000
max_seq_length = 25000
datasetfile = 'data.json'  # Dataset file

def stop_thread(signum, frame):
    """
    Sets a flag to stop the generation thread when receiving a signal (e.g., interrupt).
    
    Args:
        signum (int): Signal number.
        frame (frame): Current stack frame.
    """

    global stop_generation
    stop_generation = True

# Register signal handler for interrupt (Ctrl+C)
signal.signal(signal.SIGINT, stop_thread)

# Define the MultiHeadAttention class
class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention mechanism used in transformer models.

    Attributes:
        d_model (int): The dimension of the model (input/output size).
        num_heads (int): The number of attention heads.
        d_k (int): The dimension of each attention head.
        W_q (nn.Linear): Linear layer for the query.
        W_k (nn.Linear): Linear layer for the key.
        W_v (nn.Linear): Linear layer for the value.
        W_o (nn.Linear): Linear layer for the output.
    """

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
        """
        Performs scaled dot-product attention.
        
        Args:
            Q (Tensor): Query tensor.
            K (Tensor): Key tensor.
            V (Tensor): Value tensor.
            mask (Tensor, optional): Mask to apply to attention scores (default is None).
        
        Returns:
            Tensor: The result of attention mechanism.
        """

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            # Make sure the mask has the correct shape: (batch_size, num_heads, seq_len, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)  # Add dimensions for num_heads and seq_len
            attn_scores = attn_scores + mask  # Assuming mask is of shape (batch_size, 1, seq_len, seq_len)
            
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        """
        Splits the input tensor into multiple attention heads.

        Args:
            x (Tensor): The input tensor to split.

        Returns:
            Tensor: Tensor with shape adjusted for multi-head attention.
        """

        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x):
        """
        Combines the attention heads into a single tensor.

        Args:
            x (Tensor): The tensor from multiple heads to combine.

        Returns:
            Tensor: Tensor after combining all heads.
        """

        batch_size = x.size(0)
        seq_len = x.size(2)
        x = x.permute(0, 2, 1, 3)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass for the multi-head attention mechanism.

        Args:
            Q (Tensor): Query tensor.
            K (Tensor): Key tensor.
            V (Tensor): Value tensor.
            mask (Tensor, optional): Mask to apply to attention scores.

        Returns:
            Tensor: Output after applying multi-head attention.
        """

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

# Define the PositionWiseFeedForward class
class PositionWiseFeedForward(nn.Module):
    """
    Implements a position-wise feed-forward network, used in transformer models to process each position independently.

    Attributes:
        fc1 (nn.Linear): First linear transformation (input dimension to hidden dimension).
        fc2 (nn.Linear): Second linear transformation (hidden dimension to output dimension).
    """
    def __init__(self, d_model, d_ff):
        """
        Initializes the feed-forward network with two linear layers.

        Args:
            d_model (int): The dimension of the input.
            d_ff (int): The dimension of the hidden layer.
        """
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # First linear layer
        self.fc2 = nn.Linear(d_ff, d_model)  # Second linear layer

    def forward(self, x):
        """
        Applies the position-wise feed-forward transformation.

        Args:
            x (Tensor): The input tensor, typically the output from the previous transformer layer.

        Returns:
            Tensor: The transformed output after applying ReLU and the two linear layers.
        """
        return self.fc2(torch.relu(self.fc1(x)))  # Apply ReLU activation in between the linear transformations


# Define the PositionalEncoding class
class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding used in transformer models to inject information about the relative position of tokens.

    Attributes:
        d_model (int): The dimension of the model (input/output size).
        max_len (int): The maximum sequence length for which positional encoding will be generated.
        positional_encoding (Tensor): A precomputed tensor representing the positional encodings.
    """
    def __init__(self, d_model, max_len=300):
        """
        Initializes the positional encoding with the given model dimension and maximum sequence length.

        Args:
            d_model (int): The dimension of the model (input/output size).
            max_len (int, optional): The maximum length of the input sequence. Default is 300.
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.positional_encoding = self.create_positional_encoding()

    def create_positional_encoding(self):
        """
        Generates the positional encoding matrix using sine and cosine functions.
        
        Returns:
            Tensor: A tensor containing the positional encodings for all positions up to max_len.
        """
        position = np.arange(self.max_len)[:, np.newaxis]  # Generate positions
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))  # Calculate divisors for sine and cosine
        pos_encodings = np.zeros((self.max_len, self.d_model))  # Initialize tensor
        pos_encodings[:, 0::2] = np.sin(position * div_term)  # Apply sine to even indices
        pos_encodings[:, 1::2] = np.cos(position * div_term)  # Apply cosine to odd indices
        return torch.tensor(pos_encodings, dtype=torch.float32)  # Convert to tensor

    def forward(self, x):
        """
        Adds positional encodings to the input tensor.

        Args:
            x (Tensor): The input tensor (e.g., from embedding layer).

        Returns:
            Tensor: The input tensor with added positional encodings.
        
        Raises:
            ValueError: If the input sequence length exceeds the maximum length defined for positional encodings.
        """
        seq_len = x.size(1)  # Get the sequence length
        if seq_len > self.max_len:
            raise ValueError("Sequence length exceeds maximum length.")  # Raise error if input sequence is too long
        pos_encodings = self.positional_encoding[:seq_len, :]  # Get positional encodings for current sequence length
        return x + pos_encodings.unsqueeze(0)  # Add positional encodings to input


# Define the DecoderLayer class
class DecoderLayer(nn.Module):
    """
    Implements a single decoder layer of the transformer model, which consists of multi-head attention,
    feed-forward network, and layer normalization.

    Attributes:
        attn1 (MultiHeadAttention): The self-attention mechanism used for the decoder.
        attn2 (MultiHeadAttention): The encoder-decoder attention mechanism.
        ffn (PositionWiseFeedForward): The position-wise feed-forward network applied to the output.
        layer_norm1 (nn.LayerNorm): Layer normalization for the first attention block.
        layer_norm2 (nn.LayerNorm): Layer normalization for the second attention block.
        layer_norm3 (nn.LayerNorm): Layer normalization for the feed-forward network.
        dropout (nn.Dropout): Dropout layer for regularization.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        Initializes the components of the decoder layer, including attention mechanisms, feed-forward network,
        and layer normalization.

        Args:
            d_model (int): The dimension of the model (input/output size).
            num_heads (int): The number of attention heads.
            d_ff (int): The dimension of the feed-forward network.
            dropout (float): The dropout rate for regularization.
        """
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttention(d_model, num_heads)  # Self-attention mechanism
        self.attn2 = MultiHeadAttention(d_model, num_heads)  # Encoder-decoder attention mechanism
        self.ffn = PositionWiseFeedForward(d_model, d_ff)  # Feed-forward network
        self.layer_norm1 = nn.LayerNorm(d_model)  # Layer normalization for self-attention
        self.layer_norm2 = nn.LayerNorm(d_model)  # Layer normalization for encoder-decoder attention
        self.layer_norm3 = nn.LayerNorm(d_model)  # Layer normalization for feed-forward network
        self.dropout = nn.Dropout(dropout)  # Dropout layer

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        The forward pass for the decoder layer, which includes self-attention, encoder-decoder attention, 
        and a feed-forward network.

        Args:
            x (Tensor): The input tensor to the decoder layer (usually from the previous layer).
            enc_output (Tensor): The output from the encoder, used in the encoder-decoder attention.
            src_mask (Tensor): The source mask to apply during attention.
            tgt_mask (Tensor): The target mask to apply during attention.

        Returns:
            Tensor: The processed output after applying attention and feed-forward layers.
        """
        attn_output1 = self.attn1(x, x, x, tgt_mask)  # Apply self-attention
        x = self.layer_norm1(x + self.dropout(attn_output1))  # Add and normalize self-attention output
        attn_output2 = self.attn2(x, enc_output, enc_output, src_mask)  # Apply encoder-decoder attention
        x = self.layer_norm2(x + self.dropout(attn_output2))  # Add and normalize encoder-decoder attention output
        ffn_output = self.ffn(x)  # Apply feed-forward network
        x = self.layer_norm3(x + self.dropout(ffn_output))  # Add and normalize feed-forward output
        return x  # Return the final output

# Define the Transformer model class
class TransformerModel(nn.Module):
    """
    Implements a transformer model consisting of an embedding layer, positional encoding, multiple decoder layers,
    and a final linear layer to predict the next token in a sequence.

    Attributes:
        vocab_size (int): The size of the vocabulary (number of unique tokens).
        d_model (int): The dimension of the model (embedding and output size).
        num_heads (int): The number of attention heads in the multi-head attention mechanism.
        num_layers (int): The number of transformer decoder layers.
        d_ff (int): The dimension of the feed-forward layer in each decoder.
        max_seq_length (int): The maximum sequence length for input.
        dropout_rate (float): The dropout rate used in the model to prevent overfitting.
        embedding (nn.Embedding): The embedding layer to transform token indices to embeddings.
        positional_encoding (PositionalEncoding): The positional encoding added to the input embeddings.
        decoder_layers (nn.ModuleList): A list of transformer decoder layers.
        fc (nn.Linear): The final linear layer to map the decoder output to the vocabulary size.
        dropout (nn.Dropout): Dropout layer for regularization.
    """
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_length=300, dropout=0.1):
        """
        Initializes the transformer model components, including embedding, positional encoding, 
        decoder layers, and output projection layer.

        Args:
            vocab_size (int): The size of the vocabulary (number of unique tokens).
            d_model (int, optional): The dimension of the model (embedding and output size). Default is 512.
            num_heads (int, optional): The number of attention heads in the multi-head attention mechanism. Default is 8.
            num_layers (int, optional): The number of transformer decoder layers. Default is 6.
            d_ff (int, optional): The dimension of the feed-forward layer in each decoder. Default is 2048.
            max_seq_length (int, optional): The maximum sequence length for input. Default is 300.
            dropout (float, optional): The dropout rate used in the model. Default is 0.1.
        """
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout
        
        # Initialize the components of the model
        self.embedding = nn.Embedding(vocab_size, d_model)  # Embedding layer for input tokens
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_length)  # Positional encoding
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])  # Decoder layers
        self.fc = nn.Linear(d_model, vocab_size)  # Output projection layer to map to vocab size
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

    def forward(self, tgt, tgt_mask=None):
        """
        Defines the forward pass of the transformer model, which includes embedding, adding positional encoding,
        passing through decoder layers, and outputting the predictions for the next token.

        Args:
            tgt (Tensor): The target sequence of token indices.
            tgt_mask (Tensor, optional): A mask to prevent attending to certain tokens (e.g., future tokens for autoregression).

        Returns:
            Tensor: The predicted logits for the next token in the sequence.
        """
        emb_tgt = self.dropout(self.positional_encoding(self.embedding(tgt)))  # Apply embedding and positional encoding
        
        # Pass through each decoder layer
        for layer in self.decoder_layers:
            emb_tgt = layer(emb_tgt, emb_tgt, tgt_mask, tgt_mask)  # Only use the target (no encoder input)
        
        return self.fc(emb_tgt)  # Project the output to the vocabulary space

    def sample_next_token(self, logits, temperature=1.0, top_k=0, top_p=0.0):
        """
        Samples the next token from the logits output by the model using temperature scaling, top-k, and top-p sampling.

        Args:
            logits (Tensor): The output logits from the model.
            temperature (float, optional): Scaling factor for the logits (default is 1.0, no scaling).
            top_k (int, optional): The number of top-k logits to consider for sampling (default is 0, no restriction).
            top_p (float, optional): The cumulative probability threshold for top-p sampling (default is 0.0, no filtering).

        Returns:
            int: The index of the sampled token.
        """
        if temperature != 1.0:
            logits = logits / temperature  # Apply temperature scaling
        
        pad_token_id = word_to_token['<pad>']
        logits[pad_token_id] = -float('Inf')  # Ensure padding token is not selected
        
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Ensure top_k is not greater than vocab size
            values, indices = torch.topk(logits, k=top_k)
            logits = torch.zeros_like(logits).scatter_(-1, indices, values)  # Apply top-k filtering
        
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_logits[sorted_indices_to_remove] = -float('Inf')  # Apply top-p filtering
            logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
        
        logits = torch.clamp(logits, min=-100.0, max=100.0)  # Clamp logits to avoid extreme values
        probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
        next_token = torch.multinomial(probs, 1).item()  # Sample from the distribution

        return next_token

    def generate_text(self, prompt, max_len=50, temperature=1.0, top_k=50, top_p=0.9):
        """
        Generates text by autoregressively predicting the next token, given a prompt.

        Args:
            prompt (str): The initial text prompt to begin generation.
            max_len (int, optional): The maximum length of the generated sequence. Default is 50.
            temperature (float, optional): The temperature for scaling logits (default is 1.0).
            top_k (int, optional): The number of top-k logits to consider for sampling (default is 50).
            top_p (float, optional): The cumulative probability threshold for top-p sampling (default is 0.9).

        Returns:
            str: The generated text as a string.
        """
        self.eval()  # Set the model to evaluation mode
        
        # Tokenize the input prompt
        tokens = torch.tensor(tokenize(prompt, word_to_token), dtype=torch.long).unsqueeze(0).to(device)
        
        generated = tokens  # Start with the prompt (initial tokens)

        # Generate the next tokens autoregressively
        for _ in tqdm(range(max_len), desc="Generating text", ncols=100):
            if stop_generation:
                print(Fore.WHITE + "\nGeneration stopped!")
                break

            seq_len = generated.size(1)
            tgt_mask = self.generate_nopeak_mask(seq_len)  # Generate the target mask
            
            # Get the logits for the current generated sequence
            output = self(generated, tgt_mask=tgt_mask)
            logits = output[0, -1]  # Get logits for the last token
            
            # Sample the next token using the specified sampling method
            next_token = self.sample_next_token(logits, temperature, top_k, top_p)
            
            # Append the next token to the generated sequence
            generated = torch.cat([generated, torch.tensor([[next_token]]).to(device)], dim=1)
        
        # Convert generated token IDs back to words
        generated_text = detokenize(generated.squeeze(0).tolist(), token_to_word)
        return generated_text

    def generate_nopeak_mask(self, size):
        """
        Generates a mask to prevent attention to future tokens during autoregressive generation.

        Args:
            size (int): The size of the sequence (number of tokens).

        Returns:
            Tensor: A triangular mask (upper half filled with ones, lower half with zeros).
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()  # Upper triangular matrix
        return mask.to(device)

def tokenize(text, subword_vocab):
    """
    Tokenizes a given text into a list of subword tokens or character-level tokens if the word is out-of-vocabulary.
    
    Args:
        text (str): The input text string to tokenize.
        subword_vocab (dict): A dictionary mapping words to subword tokens (e.g., BPE or subword tokenization).

    Returns:
        list: A list of tokenized subwords or characters.
    """
    tokens = [
        subword_vocab[word] if word in subword_vocab 
        else [subword_vocab.get(char, 0) for char in word]  # Use character-level tokenization for OOV words
        for word in text.split()
    ]
    
    # Flatten the list in case of out-of-vocabulary words (i.e., when a word is split into characters)
    return [item for sublist in tokens for item in (sublist if isinstance(sublist, list) else [sublist])]

def detokenize(tokens, idx_to_word):
    """
    Converts a list of token indices back to a human-readable string. Skips special tokens like '<start>' and '<pad>'.
    
    Args:
        tokens (list): A list of token indices to detokenize.
        idx_to_word (dict): A dictionary mapping token indices to words or subwords.

    Returns:
        str: The detokenized text.
    """
    words = []
    for token in tokens:
        word = idx_to_word.get(token, '')  # Get the word corresponding to the token
        
        # Skip special tokens like <start> or <pad>
        if word in {'<start>', '<pad>'}:
            continue

        if word:
            if word.startswith("▁"):  # Subword marker (e.g., in sentencepiece tokenization)
                words.append(word[1:])  # Remove the subword marker and concatenate
            else:
                words.append(word)  # Regular word or subword without marker

    # Join and clean excessive spaces efficiently
    return ' '.join(words).strip()

def count_parameters(model):
    """
    Counts the total number of parameters in a given PyTorch model.
    
    Args:
        model (torch.nn.Module): The PyTorch model whose parameters are to be counted.
    
    Returns:
        int: The total number of parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

# Define a Dataset class for the text data
class TextDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset class for processing text data, including tokenization, padding, and truncation.

    Args:
        knowledge_base (list of dict): A list of dictionary entries containing the text data.
        word_to_token (dict): A dictionary mapping words to token IDs.
        max_length (int, optional): The maximum length of the sequences. Default is 150.
    """
    def __init__(self, knowledge_base, word_to_token, max_length=150):
        """
        Initializes the dataset by processing each sequence in the knowledge base.

        Args:
            knowledge_base (list of dict): A list containing data entries, each having a 'source' field with text.
            word_to_token (dict): A dictionary mapping words to token IDs.
            max_length (int, optional): The maximum sequence length. Default is 150.
        """
        self.max_length = max_length
        self.data = [self.process_sequence(entry['source'], word_to_token) for entry in knowledge_base]

    def process_sequence(self, text, word_to_token):
        """
        Tokenizes the input text and ensures it fits within the maximum length by padding or truncating.
        
        Args:
            text (str): The text to process (tokenize, pad, or truncate).
            word_to_token (dict): A dictionary mapping words to token IDs.

        Returns:
            list: A list of tokenized IDs, padded or truncated to fit the max length.
        """
        tokenized = self.tokenize(text, word_to_token)
        if len(tokenized) > self.max_length:
            tokenized = tokenized[:self.max_length]  # Truncate if the sequence is too long
        elif len(tokenized) < self.max_length:
            tokenized = tokenized + [word_to_token['<pad>']] * (self.max_length - len(tokenized))  # Pad if the sequence is too short
        return tokenized

    def tokenize(self, text, word_to_token):
        """
        Tokenizes a text sequence into token IDs using the provided word-to-token mapping.

        Args:
            text (str): The input text to tokenize.
            word_to_token (dict): A dictionary mapping words to token IDs.

        Returns:
            list: A list of token IDs corresponding to the words in the text.
        """
        return [word_to_token.get(word, word_to_token['<pad>']) for word in text.split()]

    def __len__(self):
        """
        Returns the total number of sequences in the dataset.
        
        Returns:
            int: The number of sequences in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a specific sequence from the dataset by index.

        Args:
            idx (int): The index of the sequence to retrieve.

        Returns:
            torch.Tensor: The tokenized sequence as a PyTorch tensor.
        """
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

# Train function
def train_epoch():
    """
    Trains the model for one epoch. This function computes the loss, performs backpropagation,
    and updates the model parameters using gradient scaling for mixed precision training.

    Returns:
        float: The average loss over the epoch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0  # Initialize total loss to accumulate during training

    # Iterate through the training data
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()  # Clear previous gradients
        src = batch[:, :-1].to(device)  # Source sequence (all but last token)
        tgt = batch[:, 1:].to(device)  # Target sequence (all but first token)

        device_type = 'cuda' if next(model.parameters()).is_cuda else 'cpu'  # Determine device type

        # Forward pass with automatic mixed precision
        with torch.amp.autocast(device_type=device_type):
            output = model(src, tgt)
            loss = criterion(output.reshape(-1, vocab_size), tgt.reshape(-1))  # Compute loss
        
        # Backpropagation with gradient scaling for mixed precision
        scaler.scale(loss).backward()  # Scale the loss for backward pass
        scaler.unscale_(optimizer)  # Unscale gradients before optimization
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)  # Gradient clipping to prevent exploding gradients
        scaler.step(optimizer)  # Step the optimizer
        scaler.update()  # Update the scaler

        total_loss += loss.item()  # Accumulate loss for the epoch

    # Return the average loss for the epoch
    return total_loss / len(train_loader)

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
    
    # Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # Training Loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training Loop
    for epoch in range(num_epochs):
        train_loss = train_epoch()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")
        scheduler.step()

    # Save the model after training
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

generation_thread = None 
stop_event = threading.Event()

def generate_in_thread():
    """
    Continuously generates text based on user input in a separate thread. The function listens for user input,
    generates text, and prints the AI's response. It will stop if the user types 'exit' or an error occurs.
    This function runs in a loop until the `stop_generation` flag is set.

    The function is designed to run in a separate thread to allow asynchronous generation without blocking the main program.

    Returns:
        None
    """
    global stop_generation
    while not stop_generation:
        user_input = input(Fore.WHITE + "You: ")  # Get input from the user
        if user_input.lower() == "exit":  # If the user types 'exit', stop generation
            print(Fore.GREEN + "Exiting chat...")
            stop_event.set()  # Set stop event to signal termination
            sys.exit(0)  # Exit gracefully
        try:
            # Generate text based on the user input
            generated_text = model.generate_text(user_input, max_len=50, temperature=0.7, top_k=50, top_p=0.9)
            if not stop_generation:  # Only print if generation was not stopped
                print(Fore.GREEN + f"AI: {generated_text}")
        except Exception as e:
            print(Fore.RED + f"Error: {str(e)}")  # Handle any exceptions during generation
            break

def start_text_generation():
    """
    Starts the text generation process in a new thread. The function sets the `stop_generation` flag to False,
    then starts a new thread that runs the `generate_in_thread` function.

    Returns:
        None
    """
    global stop_generation, generation_thread
    stop_generation = False  # Reset the stop flag before starting
    generation_thread = threading.Thread(target=generate_in_thread)  # Create a new thread for generation
    generation_thread.start()  # Start the text generation thread

def main():
    """
    Main entry point for running the text generation system. This function handles starting the text generation
    in a separate thread and allows for interactive user input. It continuously checks if the generation thread
    is running, and if not, it restarts it. The program can be stopped gracefully with a KeyboardInterrupt.

    Returns:
        None
    """
    try:
        while True:
            if not generation_thread or not generation_thread.is_alive():
                start_text_generation()  # Start a new generation thread if not already running

            while generation_thread.is_alive():
                time.sleep(1)  # Main thread can do other work here

    except KeyboardInterrupt:
        # Handle the case where the user stops the program manually
        print(Fore.WHITE + "Program manually stopped by user.")
        stop_generation = True  # Set the flag to stop generation
        if generation_thread:
            generation_thread.join()  # Wait for the generation thread to finish
        sys.exit(0)  # Exit the program gracefully

if __name__ == "__main__":
    main()