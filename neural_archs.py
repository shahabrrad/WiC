import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: In addition to __init__() and forward(), feel free to add
# other functions or attributes you might need.
class DAN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embeddings=None):
        # TODO: Declare DAN architecture
        super(DAN, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if not(embeddings is None):
            self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float())
        else:
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # Fully connected layers
        # The input dimension is calculated as follows:
        # embedding_dim * 2 (for s1 and s2) + embedding_dim (for W) + 1 (for x)
        self.fc1 = nn.Linear(embedding_dim * 3 + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, s1, s2, W, x):
        # Embed sequences and target word
        s1_embedded = self.embedding(s1).mean(1) # Average embeddings across the sequence length for s1
        s2_embedded = self.embedding(s2).mean(1) # Average embeddings across the sequence length for s2
        W_embedded = self.embedding(W).mean(1) # Embedding for W
        
        # Concatenate the averaged embeddings, W's embedding, and x
        combined = torch.cat((s1_embedded, s2_embedded, W_embedded, x.unsqueeze(1)), 1)
        
        # Pass through fully connected layers
        hidden = F.relu(self.fc1(combined))
        output = self.fc2(hidden)
        
        # Use sigmoid activation for binary classification
        return torch.sigmoid(output)


class RNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate=0.5, embeddings=None, bidirectional=False):
        # TODO: Declare RNN model architecture
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if not(embeddings is None):
            self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float())
        else:
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layers for processing sequences s1 and s2 with dropout
        self.rnn_s1 = nn.RNN(embedding_dim, hidden_dim, bidirectional=bidirectional, num_layers=2, batch_first=True, dropout=dropout_rate)
        self.rnn_s2 = nn.RNN(embedding_dim, hidden_dim, bidirectional=bidirectional, num_layers=2, batch_first=True, dropout=dropout_rate)
        
        # RNN for the target word W with dropout
        self.rnn_w = nn.RNN(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout_rate)
        
        # Linear layer to process the numerical input x
        self.linear_x = nn.Linear(1, hidden_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.linear_out = nn.Linear(hidden_dim * 4, output_dim)


    def forward(self, s1, s2, w, x):
        # Embedding lookups
        embedded_s1 = self.embedding(s1)
        embedded_s2 = self.embedding(s2)
        embedded_w = self.embedding(w)
        
        # Apply dropout to embeddings
        embedded_s1 = self.dropout(embedded_s1)
        embedded_s2 = self.dropout(embedded_s2)
        embedded_w = self.dropout(embedded_w)
        
        # LSTM outputs
        _, hidden_s1 = self.rnn_s1(embedded_s1)
        _, hidden_s2 = self.rnn_s2(embedded_s2)
        _, hidden_w = self.rnn_w(embedded_w)
        
        # Process numerical input x through a linear layer with ReLU activation
        x = x.view(-1, 1) # Ensure x is the right shape
        x = F.relu(self.linear_x(x))
        
        # Concatenate all features
        combined_features = torch.cat((hidden_s1[-1], hidden_s2[-1], hidden_w[-1], x), dim=1)
        
        # Apply dropout before the final layer
        combined_features = self.dropout(combined_features)
        
        # Final classification layer
        output = self.linear_out(combined_features)
        output = torch.sigmoid(output) # Assuming binary classification
        
        return output





class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate=0.5, embeddings=None, bidirectional=False):
        # TODO: Declare LSTM model architecture
        super(LSTM, self).__init__()
        
        # Embedding layer for sequences s1, s2, and word W
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if not(embeddings is None):
            self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float())
        else:
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers for processing sequences s1 and s2 with dropout
        self.lstm_s1 = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, num_layers=2, batch_first=True, dropout=dropout_rate)
        self.lstm_s2 = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, num_layers=2, batch_first=True, dropout=dropout_rate)
        
        # LSTM for the target word W with dropout
        self.lstm_w = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout_rate)
        
        # Linear layer to process the numerical input x
        self.linear_x = nn.Linear(1, hidden_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.linear_out = nn.Linear(hidden_dim * 4, output_dim)


    def forward(self, s1, s2, w, x):
        # Embedding lookups
        embedded_s1 = self.embedding(s1)
        embedded_s2 = self.embedding(s2)
        embedded_w = self.embedding(w)
        
        # Apply dropout to embeddings
        embedded_s1 = self.dropout(embedded_s1)
        embedded_s2 = self.dropout(embedded_s2)
        embedded_w = self.dropout(embedded_w)
        
        # LSTM outputs
        _, (hidden_s1, _) = self.lstm_s1(embedded_s1)
        _, (hidden_s2, _) = self.lstm_s2(embedded_s2)
        _, (hidden_w, _) = self.lstm_w(embedded_w)
        
        # Process numerical input x through a linear layer with ReLU activation
        x = x.view(-1, 1) # Ensure x is the right shape
        x = F.relu(self.linear_x(x))
        
        # Concatenate all features
        combined_features = torch.cat((hidden_s1[-1], hidden_s2[-1], hidden_w[-1], x), dim=1)
        
        # Apply dropout before the final layer
        combined_features = self.dropout(combined_features)
        
        # Final classification layer
        output = self.linear_out(combined_features)
        output = torch.sigmoid(output) # Assuming binary classification
        
        return output

