import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaController(nn.Module):
    def __init__(self, task_feat_dim, layer_feat_dim, hidden_dim, 
                 ranks, bitwidths, sparsities, num_layers):
        super().__init__()
        self.task_encoder = nn.Sequential(
            nn.Linear(task_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layer_encoder = nn.Sequential(
            nn.Linear(layer_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # three heads: rank, bitwidth, sparsity (output per layer)
        self.rank_head = nn.Linear(hidden_dim, len(ranks))
        self.bit_head  = nn.Linear(hidden_dim, len(bitwidths))
        self.sparse_head = nn.Linear(hidden_dim, len(sparsities))
        # store options
        self.ranks      = ranks
        self.bitwidths  = bitwidths
        self.sparsities = sparsities
        self.num_layers = num_layers

    def forward(self, task_feats, layer_feats):
        # task_feats: [batch, task_feat_dim]
        # layer_feats: [batch, num_layers, layer_feat_dim]
        batch_size = task_feats.size(0)
        
        t = self.task_encoder(task_feats)                   # [batch,h]
        l = self.layer_encoder(layer_feats.view(-1, layer_feats.size(-1)))  # [batch*num_layers,h]
        l = l.view(batch_size, self.num_layers, -1)         # [batch,num_layers,h]
        
        # aggregate: sum task + layer per layer
        h = t.unsqueeze(1) + l                              # [batch,num_layers,h]
        
        # Process each layer separately instead of flattening all
        rank_logits_list = []
        bit_logits_list = []
        sp_logits_list = []
        
        for layer_idx in range(self.num_layers):
            layer_h = h[:, layer_idx, :]  # [batch, h]
            
            # Each head predicts for one layer
            rank_logits_list.append(self.rank_head(layer_h).unsqueeze(1))  # [batch, 1, R]
            bit_logits_list.append(self.bit_head(layer_h).unsqueeze(1))    # [batch, 1, B] 
            sp_logits_list.append(self.sparse_head(layer_h).unsqueeze(1))  # [batch, 1, S]
        
        # Concatenate across layers
        rank_logits = torch.cat(rank_logits_list, dim=1)  # [batch, num_layers, R]
        bit_logits = torch.cat(bit_logits_list, dim=1)    # [batch, num_layers, B]
        sp_logits = torch.cat(sp_logits_list, dim=1)      # [batch, num_layers, S]
        
        rank_probs = F.softmax(rank_logits, dim=-1)
        bit_probs  = F.softmax(bit_logits,  dim=-1)
        sp_probs   = F.softmax(sp_logits,   dim=-1)
        return rank_probs, bit_probs, sp_probs
