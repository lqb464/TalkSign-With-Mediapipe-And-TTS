import torch
import torch.nn as nn
from src.models.base_model import SignLanguageModel

class TransformerClassifier(SignLanguageModel):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        # Đảm bảo d_model chia hết cho nhead
        assert d_model % nhead == 0, "d_model phải chia hết cho nhead"
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model)) 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq, feat]
        x = self.input_projection(x) 
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.dropout(x)
        out = self.transformer_encoder(x)
        return self.fc(out.mean(dim=1))