import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=18, d_model=128, nhead=8, num_layers=3, 
                 dim_feedforward=256, dropout=0.1, num_market_indices=11):
        super(TimeSeriesTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_market_indices = num_market_indices
        
        # Separate embedding for each market index (optional)
        self.index_embeddings = nn.Parameter(torch.randn(num_market_indices, d_model))
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Cross-attention between market indices
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # Output layers
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Prediction head
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, input_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_market_indices, sequence_length, input_dim)
        Returns:
            output: (batch_size, num_market_indices, input_dim)
        """
        batch_size, num_indices, seq_len, input_dim = x.shape
        
        # Reshape for transformer
        x = x.view(batch_size * num_indices, seq_len, input_dim)
        
        # Project input
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        x = self.transformer(x)  # (batch*num_indices, seq_len, d_model)
        
        # Take last timestep
        x = x[:, -1, :]  # (batch*num_indices, d_model)
        
        # Reshape back
        x = x.view(batch_size, num_indices, self.d_model)
        
        # Add index embeddings
        x = x + self.index_embeddings.unsqueeze(0)
        
        # Cross-attention between market indices
        x = x.transpose(0, 1)  # (num_indices, batch, d_model)
        x, _ = self.cross_attention(x, x, x)
        x = x.transpose(0, 1)  # (batch, num_indices, d_model)
        
        # Apply layer norm and residual
        x = self.layer_norm1(x)
        x = self.dropout(x)
        
        # Output projection
        output = self.output_projection(x)
        
        return output

class MarketAggregator(nn.Module):
    """
    Advanced market encoder:
    (B, N_MARKET, DIM) → (B, DIM)

    Features:
    - Multi-head self-attention (cross-market dependencies)
    - Learnable global query (CLS-style pooling)
    - Gated feature fusion (FiLM-like)
    - Residual + LayerNorm
    """

    def __init__(
        self,
        dim: int = 18,
        num_heads: int = 6,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

        self.norm2 = nn.LayerNorm(dim)

        self.global_query = nn.Parameter(torch.randn(1, 1, dim))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N_MARKET, DIM)
        """

        B = x.size(0)

        attn_out, _ = self.self_attn(x, x, x)  # (B, N, D)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        query = self.global_query.expand(B, -1, -1)  # (B, 1, D)
        pooled, _ = self.cross_attn(query, x, x)     # (B, 1, D)
        pooled = pooled.squeeze(1)                  # (B, D)

        gate = self.gate(pooled)                    # (B, D)
        pooled = pooled * gate

        z = self.out_proj(pooled)

        return z

class FusionV2(nn.Module):
    """
    Advanced fusion:
    Inputs:
        s: (B, D) → stock embedding
        m: (B, D) → market embedding

    Output:
        (B, D) → market-aware stock representation
    """

    def __init__(
        self,
        dim: int = 18,
        hidden_dim: int = 64,
        num_heads: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.film = nn.Sequential(
            nn.Linear(dim, dim * 2)  # gamma, beta
        )

        self.bilinear = nn.Bilinear(dim, dim, dim)


        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        s, m: (B, D)
        """

        gamma, beta = self.film(m).chunk(2, dim=-1)
        s_film = gamma * s + beta   # (B, D)

        bilinear_feat = self.bilinear(s, m)  # (B, D)

        s_q = s.unsqueeze(1)  # (B, 1, D)
        m_kv = m.unsqueeze(1)

        attn_out, _ = self.cross_attn(s_q, m_kv, m_kv)
        attn_out = attn_out.squeeze(1)  # (B, D)

        interaction = s * m

        fusion_stack = torch.cat(
            [s_film, bilinear_feat, attn_out, interaction],
            dim=-1
        ) 

        fused = self.mlp(fusion_stack)

        gate_input = torch.cat([s, m, fused], dim=-1)
        gate = self.gate(gate_input)

        out = gate * fused + (1 - gate) * s

        out = self.norm(out + s)

        return out# (B, DIM)
    
class SurrogateHead(nn.Module):
    """
    Stronger surrogate head for representation learning.
    Outputs raw logits (no sigmoid).
    """

    def __init__(
        self,
        dim: int = 18,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),

            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # (B, 1) → RAW LOGITS