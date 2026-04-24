import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicLSTM(nn.Module):
    """
    An Attention-based Bidirectional LSTM (Att-BiLSTM).
    Highly cited in speech and audio processing literature because the attention 
    mechanism allows the model to weigh the importance of different time frames,
    providing both higher accuracy and model interpretability.
    """
    def __init__(self, one_batch=None, num_classes=10, hidden_size=128, num_layers=2, dropout=0.3):
        super(DynamicLSTM, self).__init__()
        
        # -------------------------
        # Dynamic Input Handling
        # -------------------------
        if one_batch is not None:
            _, C, H, W = one_batch.shape
            self.input_channels = C
            self.feature_dim = H
            self.seq_len = W
        else:
            self.input_channels = 1
            self.feature_dim = 64
            self.seq_len = 32

        # The input to the LSTM will be C * H features per time step
        self.lstm_input_size = self.input_channels * self.feature_dim
        
        # -------------------------
        # Bidirectional LSTM
        # -------------------------
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # -------------------------
        # Temporal Attention Mechanism
        # -------------------------
        # This projects the hidden state at each time step to a single importance score
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # -------------------------
        # Classifier Head
        # -------------------------
        # We classify based on the attention-weighted context vector
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape arrives as: (Batch, Channels, Height, Width)
        B, C, H, W = x.shape
        
        # 1. Permute to (Batch, Width, Channels, Height)
        x = x.permute(0, 3, 1, 2)
        
        # 2. Reshape to (Batch, SequenceLength, Features) -> (B, W, C * H)
        x = x.reshape(B, W, -1)
        
        # 3. Pass through LSTM
        # lstm_out shape: (Batch, SequenceLength, hidden_size * 2)
        lstm_out, _ = self.lstm(x)
        
        # 4. Calculate Attention Weights
        # attn_scores shape: (Batch, SequenceLength, 1)
        attn_scores = self.attention(lstm_out)
        
        # Normalize scores to probabilities across the time dimension
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 5. Apply Attention Weights to LSTM outputs (Context Vector)
        # Multiply each time step's hidden state by its attention weight, then sum
        # context shape: (Batch, hidden_size * 2)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # 6. Classify
        return self.fc(context)