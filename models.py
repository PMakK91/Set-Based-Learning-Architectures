# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 21:12:27 2026

@author: user
"""


# -*- coding: utf-8 -*-
"""
Updated Models using Custom MultiHeadAttention and PositionWiseFeedForward
"""

import torch
import torch.nn as nn
import math

# =====================================================================
# 1. Custom Components 
# =====================================================================

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
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, -1)
        output = torch.matmul(attn_probs, V)
        return output
    
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):  
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# =====================================================================
# 2. Βασικά Μοντέλα (MLP, Deep Sets, Janossy, PointNet++)
# =====================================================================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class DeepSets(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, phi_layers=[64, 64], rho_layers=[64]):
        super().__init__()
        self.phi = MLP(input_dim=input_dim, hidden_dims=phi_layers, output_dim=hidden_dim)
        self.rho = MLP(input_dim=hidden_dim, hidden_dims=rho_layers, output_dim=output_dim)
        
    def forward(self, x, lengths=None):
        phi_x = self.phi(x)
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device)[None, :] < lengths[:, None]
            phi_x = phi_x * mask.unsqueeze(-1).float()
        sum_x = torch.sum(phi_x, dim=1)
        out = self.rho(sum_x)
        return out


# =====================================================================
# 3. Set Transformer (Με χρήση των custom MultiHeadAttention & FFN)
# =====================================================================

class SAB(nn.Module):
    """Set Attention Block (Χρησιμοποιεί το custom MultiHeadAttention)"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=dim, num_heads=num_heads)
        self.ff = PositionWiseFeedForward(d_model=dim, d_ff=dim * 2)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # Στο custom MultiHeadAttention, το mask περιμένει διάσταση κατάλληλη ή None.
        # Αν έχουμε mask, το μετατρέπουμε σε κατάλληλο σχήμα για το attention score.
        attn_mask = None
        if mask is not None:
            # mask shape: [Batch, Seq_Len] -> [Batch, 1, Seq_Len, Seq_Len] ή παρόμοιο
            # Εδώ το mask περνάει στη scaled_dot_product_attention όπου mask == 0 γεμίζει με -1e9
            attn_mask = mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, N]
            
        attn_out = self.mha(x, x, x, mask=attn_mask)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x


class PMA(nn.Module):
    """Pooling by Multihead Attention (Χρησιμοποιεί το custom MultiHeadAttention)"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.seed = nn.Parameter(torch.Tensor(1, 1, dim))
        nn.init.xavier_uniform_(self.seed)
        self.mha = MultiHeadAttention(d_model=dim, num_heads=num_heads)
        self.ff = PositionWiseFeedForward(d_model=dim, d_ff=dim * 2)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        query = self.seed.expand(batch_size, -1, -1)
        
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, N]
            
        attn_out = self.mha(query, x, x, mask=attn_mask)
        out = self.ln1(query + attn_out)
        ff_out = self.ff(out)
        out = self.ln2(out + ff_out)
        return out


class SetTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super().__init__()
        self.enc = nn.Linear(input_dim, hidden_dim)
        self.sab = SAB(dim=hidden_dim, num_heads=num_heads)
        self.pma = PMA(dim=hidden_dim, num_heads=num_heads)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths=None):
        x = self.enc(x)
        mask = None
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device)[None, :] < lengths[:, None]
            
        x = self.sab(x, mask=mask)
        x = self.pma(x, mask=mask)
        x = x.squeeze(1)
        out = self.classifier(x)
        return out


# =====================================================================
# 4. Janossy Pooling & PointNet++
# =====================================================================

class JanossyPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_permutations=1):
        super().__init__()
        self.num_permutations = num_permutations
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, lengths=None):
        batch_size, max_len, _ = x.size()
        if lengths is None:
            lengths = torch.full((batch_size,), max_len, dtype=torch.long, device=x.device)
            
        final_outputs = []
        for _ in range(self.num_permutations):
            noise = torch.rand(batch_size, max_len, device=x.device)
            mask = torch.arange(max_len, device=x.device)[None, :] < lengths[:, None]
            noise[~mask] = -float('inf')
            
            perm_indices = torch.argsort(noise, dim=1, descending=True)
            perm_indices_expanded = perm_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
            x_permuted = torch.gather(x, 1, perm_indices_expanded)
            
            lstm_out, _ = self.lstm(x_permuted)
            last_valid_indices = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, lstm_out.size(-1))
            last_out = torch.gather(lstm_out, 1, last_valid_indices).squeeze(1)
            final_outputs.append(last_out)
            
        avg_out = torch.stack(final_outputs, dim=0).mean(dim=0)
        out = self.classifier(avg_out)
        return out


class PointNetPlusPlus(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k_neighbors=3):
        super().__init__()
        self.k = k_neighbors
        self.local_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x, lengths=None):
        B, N, D = x.size()
        dist = torch.cdist(x, x)
        
        mask = None
        if lengths is not None:
            mask = torch.arange(N, device=x.device)[None, :] < lengths[:, None]
            dist = dist.masked_fill(~mask.unsqueeze(1), float('inf'))
            
        current_k = min(self.k, N)
        _, knn_idx = torch.topk(dist, k=current_k, largest=False, dim=-1)
        
        knn_idx_expanded = knn_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        x_local = torch.gather(x.unsqueeze(1).expand(-1, N, -1, -1), 2, knn_idx_expanded)
        
        local_features = self.local_mlp(x_local)
        if mask is not None:
            valid_neighbors = torch.gather(mask.unsqueeze(1).expand(-1, N, -1), 2, knn_idx)
            local_features = local_features.masked_fill(~valid_neighbors.unsqueeze(-1), float('-inf'))
            
        local_pooled, _ = torch.max(local_features, dim=2)
        global_features = self.global_mlp(local_pooled)
        
        if mask is not None:
            global_features = global_features.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            
        global_pooled, _ = torch.max(global_features, dim=1)
        out = self.classifier(global_pooled)
        return out