"""
=======================================================================================================================================
==================================================== QUBICON Text Generation PTTM-1 ===================================================
=======================================================================================================================================
"""

import tensorflow as tf
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
import concurrent.futures
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Header with color
print(Fore.CYAN + "=" * 128)
print(Fore.CYAN + "=" * 48 + " QUBICON Text Generation PTTM-1 " + "=" * 48)
print(Fore.CYAN + "=" * 128)

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define the MultiHeadAttention class
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = tf.keras.layers.Dense(d_model)
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_v = tf.keras.layers.Dense(d_model)
        self.W_o = tf.keras.layers.Dense(d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        if mask is not None:
            attn_scores += (mask * -1e9)
        attn_probs = tf.nn.softmax(attn_scores, axis=-1)
        output = tf.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_length, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[2]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, seq_length, self.d_model))

    def call(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

# Define the PositionWiseFeedForward class
class PositionWiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.fc2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        return self.fc2(self.fc1(x))

# Define the PositionalEncoding class
class PositionalEncoding(tf.keras.layers.Layer):
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
        return tf.constant(pos_encodings, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        if seq_len > self.max_len:
            raise ValueError("Sequence length exceeds maximum length.")
        pos_encodings = self.positional_encoding[:seq_len, :]
        x = x + pos_encodings[tf.newaxis, :, :]
        return x

# Define the EncoderLayer class
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, src_mask):
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Define the DecoderLayer class
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm3 = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_output, src_mask, tgt_mask):
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

# Define the Transformer model class
class TransformerModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model=12288, num_heads=96, num_layers=96, d_ff=49152, max_seq_length=300, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_length)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        self.decoder_layers = [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super(TransformerModel, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'd_ff': self.d_ff,
            'max_seq_length': self.max_seq_length,
            'dropout': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Check which keys are present in the config and provide defaults if necessary
        required_keys = ['vocab_size', 'd_model', 'num_heads', 'num_layers', 'd_ff', 'max_seq_length', 'dropout']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing key in config: {key}")
        
        # Extract only the relevant parameters for TransformerModel
        relevant_config = {key: config[key] for key in required_keys}
        return cls(**relevant_config)

    def generate_nopeak_mask(self, size):
        size = int(size)
        mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones((size, size))).to_dense()
        return mask

    def call(self, src, tgt, src_mask=None, tgt_mask=None):
        emb_src = self.dropout(self.positional_encoding(self.embedding(src)))
        for layer in self.encoder_layers:
            emb_src = layer(emb_src, src_mask)
        
        emb_tgt = self.dropout(self.positional_encoding(self.embedding(tgt)))
        for layer in self.decoder_layers:
            emb_tgt = layer(emb_tgt, emb_src, src_mask, tgt_mask)
        
        return self.fc(emb_tgt)


    def sample_next_token(self, logit, temperature=1.0, top_k=0, top_p=0.0):
        # Apply temperature scaling
        if temperature != 1.0:
            logit /= temperature
        
        # Top-k sampling
        if top_k > 0:
            values, indices = tf.nn.top_k(logit, k=top_k)
            logit = tf.scatter_nd(tf.expand_dims(indices, 1), values, tf.shape(logit))
        
        # Top-p (nucleus) sampling``
        if top_p > 0.0:
            sorted_logits, sorted_indices = tf.nn.top_k(logit, k=tf.shape(logit)[-1])
            cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits), axis=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            logit = tf.scatter_nd(
                tf.expand_dims(sorted_indices[~sorted_indices_to_remove], 1),
                sorted_logits[~sorted_indices_to_remove],
                tf.shape(logit)
            )
        
        # Sample from the processed logit distribution
        next_token = tf.random.categorical([logit], num_samples=1)
        
        # Clip token to vocab range, if needed
        vocab_size = tf.shape(logit)[-1]
        next_token = tf.clip_by_value(tf.cast(next_token, tf.int64), tf.constant(0, dtype=tf.int64), tf.cast(vocab_size, tf.int64) - 1)
            
        return next_token

    def beam_search(self, src, start_sequence, beam_width=5, max_len=50, start_token=1, temperature=1.0, top_k=50, top_p=0.9, length_penalty=1.0):
        src = tf.convert_to_tensor(src, dtype=tf.int32)
        start_sequence = tf.convert_to_tensor(start_sequence, dtype=tf.int32)
        beams = [(start_sequence, 0.0)]
        
        for step in range(max_len):
            all_candidates = []
            
            for seq, score in beams:
                seq = tf.convert_to_tensor(seq, dtype=tf.int32)
                if len(seq.shape) == 1:
                    seq = tf.expand_dims(seq, 0)
                
                seq_len = tf.shape(seq)[1]
                tgt_mask = self.generate_nopeak_mask(seq_len)
                predictions = self(src, seq, src_mask=None, tgt_mask=tgt_mask)
                logits = predictions[:, -1, :]  # Get logits for the last token

                # Set probability of <start> token to zero (prevent it from reappearing)
                logits = tf.where(tf.range(self.vocab_size) == start_token, -float('Inf'), logits)
                
                # Sample next token with temperature, top_k, and top_p constraints
                next_tokens = []
                for logit in logits:
                    next_token = self.sample_next_token(logit, temperature, top_k, top_p)
                    next_tokens.append(next_token)

                # Process each candidate sequence for the next step
                for next_token in next_tokens:
                    next_token = tf.cast(tf.reshape(next_token, [-1]), dtype=tf.int32)
                    candidate_seq = tf.concat([seq, tf.expand_dims(next_token, 1)], axis=1)
                    
                    next_token_prob = tf.nn.softmax(logits, axis=-1)[0, next_token[0]]
                    candidate_score = score + tf.math.log(next_token_prob).numpy()
                    candidate_score /= (len(candidate_seq[0]) ** length_penalty)
                    all_candidates.append((candidate_seq, candidate_score))

            # Sort candidates by score, select top beam_width sequences
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]

        best_sequence = beams[0][0].numpy().tolist()
        return best_sequence
        
# Tokenization and detokenization functions
def tokenize(text, word_to_token):
    return [word_to_token.get(word, 0) for word in text.split()]

def detokenize(tokens, token_to_word):
    # Flatten the list if necessary
    if isinstance(tokens[0], list):
        tokens = [token for sublist in tokens for token in sublist]
    return ' '.join([token_to_word.get(token, '') for token in tokens if token != 0])

# Define a Dataset class for the text data
class TextDataset:
    def __init__(self, knowledge_base, word_to_token, max_length=300):
        self.max_length = max_length
        self.data = [self.truncate_sequence(self.tokenize(entry['source'], word_to_token)) for entry in knowledge_base]

    def tokenize(self, text, word_to_token):
        return [word_to_token.get(word, 0) for word in text.split()]

    def truncate_sequence(self, sequence):
        return sequence[:self.max_length]

    # In your TextDataset class
    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            lambda: self.data,
            output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
        dataset = dataset.padded_batch(32, padded_shapes=[self.max_length])  # Increase batch size
        dataset = dataset.map(self.collate_fn)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch for efficiency
        return dataset

    def collate_fn(self, batch):
        max_len = tf.shape(batch)[1]
        attention_mask = tf.cast(batch != 0, tf.float32)
        return batch, attention_mask

# Load the knowledge base from a JSON file
with open('50kdataset.json', 'r') as f:
    knowledge_base = json.load(f)

# Check and handle knowledge base
if not knowledge_base:
    raise ValueError("The knowledge base is empty.")
    
print(f"Number of samples in knowledge_base: {len(knowledge_base)}")
print(f"Sample entry: {knowledge_base[0]}")

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

vocab_size = len(word_to_token)

# Split the knowledge base into training and validation sets
if len(knowledge_base) > 1:
    train_data, val_data = train_test_split(knowledge_base, test_size=0.1, random_state=42)
else:
    print("Not enough data to split. Using the entire dataset for training.")
    train_data = knowledge_base
    val_data = []

# Create datasets for training and validation
train_dataset = TextDataset(train_data, word_to_token)
val_dataset = TextDataset(val_data, word_to_token)

train_tf_dataset = train_dataset.get_dataset().repeat()
val_tf_dataset = val_dataset.get_dataset().repeat()

# Paths for saving and loading the model
full_model_path = 'model.weights.h5'
checkpoint_path = 'model_checkpoint.weights.h5'

def calculate_parameters(vocab_size, d_model, num_heads, num_layers, d_ff):
    embedding_params = vocab_size * d_model
    attention_per_layer = num_heads * 4 * (d_model ** 2)
    attention_total = num_layers * attention_per_layer
    ff_per_layer = d_model * d_ff + d_ff * d_model
    ff_total = num_layers * ff_per_layer
    norm_total = num_layers * (2 * d_model)
    
    total_parameters = (embedding_params + attention_total + ff_total + norm_total)
    return total_parameters

#parameters
num_epochs = 50
d_model = 768
num_heads = 12
num_layers = 24
d_ff = 2048
max_seq_length=150
dropout=0.1
beam_width=5
max_len=50
temperature=0.7
top_k=50
top_p=0.9
length_penalty=1.0

total_params = calculate_parameters(vocab_size, d_model, num_heads, num_layers, d_ff)

# Initialize the model
model = TransformerModel(vocab_size, d_model=d_model, num_heads=num_heads, num_layers=num_layers, d_ff=d_ff, max_seq_length=max_seq_length, dropout=dropout)

# Initialize optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

# Function to process each batch with color-coded messages for loss
def process_batch(batch, model, optimizer, criterion, vocab_size):
    with tf.GradientTape() as tape:
        tgt = batch[:, 1:]
        output = model(batch[:, :-1], tgt)
        output = tf.reshape(output, (-1, vocab_size))
        tgt_batch = tf.reshape(tgt, (-1,))
        loss = criterion(tgt_batch, output)
        loss = tf.reduce_mean(loss)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Print loss with color
    print(Fore.GREEN + f"Batch Loss: {loss.numpy():.4f}")
    return loss

def train_model():
    # Check for GPU availability and set the device accordingly
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available and will be used for training.")
        device_name = '/GPU:0'  # You can also use '/GPU:0' or '/GPU:1', etc., based on your setup
    else:
        print("GPU is not available. Training will be performed on the CPU.")
        device_name = '/CPU:0'
        
    tf.config.optimizer.set_jit(True)  # Enable XLA

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        # Use the selected device context
        with tf.device(device_name):
            # Training phase
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for batch, attention_mask in train_tf_dataset:
                    futures.append(executor.submit(process_batch, batch, model, optimizer, criterion, vocab_size))

                for future in concurrent.futures.as_completed(futures):
                    total_loss += future.result().numpy()  # Get the loss from each future
                    num_batches += 1

            # Validation phase
            val_loss = 0
            num_val_batches = 0
            for batch, attention_mask in val_tf_dataset:
                tgt = batch[:, 1:]
                output = model(batch[:, :-1], tgt)
                output = tf.reshape(output, (-1, vocab_size))
                tgt_batch = tf.reshape(tgt, (-1,))
                loss = criterion(tgt_batch, output)
                val_loss += tf.reduce_mean(loss).numpy()
                num_val_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else np.nan
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else np.nan

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        # Save the model checkpoints after training
        model.save_weights(checkpoint_path)

    # Save the model weights after training
    model.save_weights(full_model_path)
    print(f"Model training completed. Weights saved to {full_model_path}")

# Check if the full model exists
if os.path.exists(full_model_path):
    print(f"Loading weights from {full_model_path}")
    model.build((None, None))  # Ensure the model is built before loading weights
    model.load_weights(full_model_path)
# Check if a checkpoint exists
elif os.path.exists(checkpoint_path):
    print(f"Loading checkpoint weights from {checkpoint_path}")
    model.load_weights(checkpoint_path)  # Ensure to load checkpoint weights
    train_model()
else:
    print("Training model from scratch")
    train_model()

# During inference (modify chat_with_ai)
def chat_with_ai():
    print(Fore.GREEN + f'Total Parameters: {total_params}')
    print(Fore.GREEN + "Welcome to the AI chat! Type 'exit' to end the conversation.")
    while True:
        user_input = input(Fore.CYAN + "You: ")
        
        if user_input.lower() == 'exit':
            print(Fore.RED + "Ending the chat. Goodbye!")
            break
        
        # Tokenize the user input
        tokenized_input = [word_to_token['<start>']] + tokenize(user_input, word_to_token)
        tokenized_input = tf.constant([tokenized_input], dtype=tf.int32)  # Ensure it's 2D

        # Batch processing - you can modify this if necessary
        generated_response = model.beam_search(
            tf.constant([[word_to_token['<start>']]]), 
            tokenized_input, 
            beam_width=beam_width, 
            max_len=max_len, 
            start_token=word_to_token['<start>'], 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p, 
            length_penalty=length_penalty
        )

        # Detokenize the generated response
        response_text = detokenize(generated_response, token_to_word)
        print(Fore.GREEN + f"Model:{response_text.replace('<start>', '')}")

# Call the chat function to start the conversation
chat_with_ai()
