class MultiHeadAttention(nn.Module):
    """Multi-head Attention（教育目的の実装）"""
    
    def __init__(self, d_model, num_heads, use_rope=False, use_cosine=False):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        self.use_cosine = use_cosine
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        if use_rope:
            freqs_cos, freqs_sin = precompute_rope_freqs(self.d_k, 2048)
            self.register_buffer('freqs_cos', freqs_cos)
            self.register_buffer('freqs_sin', freqs_sin)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 線形射影
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # RoPEの適用（オプション）
        if self.use_rope:
            Q = apply_rope(Q, self.freqs_cos, self.freqs_sin)
            K = apply_rope(K, self.freqs_cos, self.freqs_sin)
        
        # Attention計算
        if self.use_cosine:
            output, _ = cosine_attention(Q, K, V, mask=mask)
        else:
            output, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
        
        # ヘッドを結合
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(output)
