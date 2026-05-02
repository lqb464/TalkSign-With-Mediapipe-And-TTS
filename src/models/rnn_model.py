import torch
import torch.nn as nn
from src.models.base_model import SignLanguageModel

class SequenceRNNClassifier(SignLanguageModel):
    def __init__(self, input_dim, num_classes, model_type="gru", hidden_dim=64, num_layers=2, dropout=0.1, bidirectional=True):
        super().__init__()
        
        rnn_cls = nn.GRU if model_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        fc_input_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, _ = self.rnn(x)
        # Lấy output của frame cuối cùng
        last_out = out[:, -1, :]
        return self.fc(last_out)