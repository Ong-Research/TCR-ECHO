import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class BalancedAtchleyAttention(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads, 
        dropout=0.1, 
        enable_monitoring=False
    ):
        super().__init__()
        # Separate projections for q, k, v
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Bilinear form for Atchley factors
        self.U = nn.Parameter(torch.randn(num_heads, 5, 5) * 0.02)
        
        # Mix parameter
        self.mix_param = nn.Parameter(torch.tensor(0.0))
        
        # Optional monitoring
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            self.register_buffer("mix_param_history", torch.zeros(100))
            self.history_idx = 0
        
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, seq1, seq2, atc1, atc2):
        B, L1, _ = seq1.shape
        _, L2, _ = seq2.shape
        
        # (dropping the padding tokens)
        if atc1.size(1) > L1:
            atc1 = atc1[:, :L1, :]
            
        if atc2.size(1) > L2:
            atc2 = atc2[:, :L2, :]
        
        seq1 = seq1.to(self.q_proj.weight.dtype)
        seq2 = seq2.to(self.q_proj.weight.dtype)
        q = self.q_proj(seq1)
        k = self.k_proj(seq2)
        v = self.v_proj(seq2)
        
        # Reshape for multi-head attention
        q = q.view(B, L1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L2, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L2, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Standard attention
        std_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Compute biophysical bias for each head separately
        bio_scores = torch.zeros_like(std_scores)
        
        for h in range(self.num_heads):
            # Extract bilinear form for this head
            U_h = self.U[h]  # [5, 5]
            
            # Compute bias for this head
            head_bias = torch.einsum("bip,pq,bjq->bij", atc1, U_h, atc2)
            
            # Add to the appropriate slice of bio_scores
            bio_scores[:, h] = head_bias
        
        # Mix with bounded parameter
        mix = torch.tanh(self.mix_param)
        
        # Update history if monitoring is enabled
        if self.enable_monitoring:
            with torch.no_grad():
                self.mix_param_history[self.history_idx] = mix.item()
                self.history_idx = (self.history_idx + 1) % 100
        
        # Normalize each attention mechanism before mixing
        std_scores_norm = F.softmax(std_scores, dim=-1)
        bio_scores_norm = F.softmax(bio_scores, dim=-1)
        
        # Interpolate between attention mechanisms
        mix_ratio = (mix + 1) / 2  # Convert from [-1,1] to [0,1]
        attn = (1 - mix_ratio) * std_scores_norm + mix_ratio * bio_scores_norm
        
        # Apply dropout and compute weighted values
        attn = self.dropout(attn)
        out = attn @ v
        
        # Reshape and project to output dimension
        out = out.transpose(1, 2).reshape(B, L1, self.dim)
        return self.out_proj(out)
        
    def get_bias_strength(self):
        """Return current bias strength for monitoring"""
        return torch.tanh(self.mix_param).item()
    
    
class StandardCrossAttention(nn.Module):
    """
    Standard cross-attention mechanism without Atchley factor integration.
    Can serve as a baseline or be used in combination with other attention types.
    """
    def __init__(self, dim, num_heads, dropout=0.1, enable_monitoring=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Separate projections for queries, keys, and values
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, seq1, seq2, *args):
        """
        Args:
            seq1: First sequence tensor [B, L1, D]
            seq2: Second sequence tensor [B, L2, D]
            *args: Additional arguments (ignored in this class)
        
        Returns:
            output: Attention output [B, L1, D]
        """
        B, L1, _ = seq1.shape
        _, L2, _ = seq2.shape
        
        # Project to queries, keys, values
        q = self.q_proj(seq1)
        k = self.k_proj(seq2)
        v = self.v_proj(seq2)
        
        # Reshape for multi-head attention
        q = q.view(B, L1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L2, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L2, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v
        
        # Reshape and project output
        out = out.transpose(1, 2).reshape(B, L1, self.dim)
        return self.out_proj(out)