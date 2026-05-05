import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch.nn.functional import normalize


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.1,
                 maxlen: int = 750):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class HierarchicalMemoryUnit(torch.nn.Module):
    """
    Hierarchical Frame Memory replacing the flat HistoryUnit.

    Maintains K hierarchical memory banks at coarsening temporal scales:
        Level 0  -> recent / fine-grained slots
        Level 1  -> mid-term / partially aggregated slots
        Level k  -> long-term / event-level abstraction

    Each level has a fixed-capacity bank populated by a content-aware
    update rule:
      * For every incoming (per-level) feature x_t, cosine similarity is
        computed against existing slots.
      * If the best slot exceeds the merge threshold, x_t is *merged*
        into that slot via a similarity-gated update (redundancy
        suppression).
      * Otherwise x_t is *inserted* into the lowest-importance slot
        (uninitialised first, else min(usage - lambda * age)).
      * Evicted / overwritten slots are detached from the autograd graph
        to keep memory and compute bounded.

    The output is the multi-scale memory bank (concatenation of all
    levels with learnable scale embeddings) followed by transformer-based
    self-refinement (cross-scale interaction) and short-window grounding.
    Shape returned: (M_total, B, D) - drop-in compatible with the existing
    `history_anchor_decoder_block1` cross-attention in MYNET.
    """

    def __init__(self, opt):
        super(HierarchicalMemoryUnit, self).__init__()
        self.n_feature = opt["feat_dim"]
        n_class = opt["num_of_class"]
        d = opt["hidden_dim"]
        dropout = 0.3

        # ---- hierarchy configuration (toggleable / configurable) ----
        self.num_levels = int(opt.get("mem_num_levels", 3))
        default_sizes = [16, 12, 8][: self.num_levels]
        default_strides = [1, 2, 4][: self.num_levels]
        self.mem_sizes = list(opt.get("mem_sizes", default_sizes))
        self.level_strides = list(opt.get("mem_level_strides", default_strides))
        assert len(self.mem_sizes) == self.num_levels
        assert len(self.level_strides) == self.num_levels
        self.merge_thresh = float(opt.get("mem_merge_thresh", 0.75))
        self.age_decay = float(opt.get("mem_age_decay", 0.01))
        self.short_window_size = 16

        self.total_slots = int(sum(self.mem_sizes))

        # learnable per-level priors (used when long_x is empty / for unused slots)
        self.mem_init = nn.ParameterList([
            nn.Parameter(torch.zeros(m, 1, d)) for m in self.mem_sizes
        ])
        # per-level abstraction projections (level k -> level k+1)
        self.level_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
            for _ in range(self.num_levels)
        ])
        # per-level scale embeddings (so anchors / decoders know which scale)
        self.scale_embed = nn.Parameter(torch.zeros(self.num_levels, 1, 1, d))
        # per-level merge-gate net: maps similarity scalar to merge strength
        self.merge_gate = nn.ModuleList([
            nn.Sequential(nn.Linear(1, d // 4), nn.GELU(), nn.Linear(d // 4, 1), nn.Sigmoid())
            for _ in range(self.num_levels)
        ])

        # multi-scale self-refinement (lets levels talk to each other)
        self.history_positional_encoding = PositionalEncoding(d, dropout, maxlen=400)
        self.mem_self_refine = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d, nhead=4, dropout=dropout, activation='gelu'),
            num_layers=2,
            norm=nn.LayerNorm(d),
        )
        # ground memory in current short-window observation
        self.mem_cross_refine = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d, nhead=4, dropout=dropout, activation='gelu'),
            num_layers=2,
            norm=nn.LayerNorm(d),
        )

        # snippet head -- multi-scale snippet supervision (input scales with M_total)
        self.snip_head = nn.Sequential(nn.Linear(d, d // 4), nn.ReLU())
        self.snip_classifier = nn.Sequential(
            nn.Linear(self.total_slots * (d // 4), (self.total_slots * (d // 4)) // 4),
            nn.ReLU(),
            nn.Linear((self.total_slots * (d // 4)) // 4, n_class),
        )

        self.norm2 = nn.LayerNorm(d)
        self.dropout2 = nn.Dropout(0.1)

        self.best_loss = 1000000
        self.best_map = 0

    def _build_level_sequences(self, long_x):
        """Build per-level feature sequences by progressive temporal aggregation.

        long_x: (L, B, D)
        returns: list of length num_levels, each (T_k, B, D)
        """
        seqs = [long_x]
        cur = long_x
        for k in range(1, self.num_levels):
            # window = stride[k] / stride[k-1]
            win = max(1, self.level_strides[k] // max(1, self.level_strides[k - 1]))
            L = cur.size(0)
            L_trunc = (L // win) * win
            if L_trunc == 0:
                # fall back to a single mean-pooled token; keeps memory non-empty
                pooled = cur.mean(dim=0, keepdim=True)
                seqs.append(self.level_proj[k](pooled))
                cur = seqs[-1]
                continue
            B, D = cur.size(1), cur.size(2)
            # reshape (not view) handles non-contiguous slices from the
            # permuted base tensor without an explicit .contiguous() copy.
            windowed = cur[:L_trunc].reshape(L_trunc // win, win, B, D)
            agg = self.level_proj[k](windowed.mean(dim=1))  # (T_trunc, B, D)
            seqs.append(agg)
            cur = agg
        return seqs

    def _stream_update(self, seq, level_idx):
        """Content-aware streaming update for one level.

        seq:        (T, B, D)
        returns:    (M, B, D)
        """
        M = self.mem_sizes[level_idx]
        T, B, D = seq.size(0), seq.size(1), seq.size(2)
        device = seq.device

        # initialise from learnable prior (broadcast across batch)
        mem = self.mem_init[level_idx].to(device).expand(-1, B, -1).contiguous()  # (M, B, D)
        age = torch.zeros(M, B, device=device)
        usage = torch.zeros(M, B, device=device)
        initialized = torch.zeros(M, B, device=device, dtype=torch.bool)

        merge_gate = self.merge_gate[level_idx]
        tau = self.merge_thresh

        eye = torch.eye(M, device=device)  # (M, M) -- for one-hot scatters

        for t in range(T):
            x_t = seq[t]  # (B, D)
            slots_n = F.normalize(mem, dim=-1)
            x_n = F.normalize(x_t, dim=-1).unsqueeze(0)  # (1, B, D)
            sim = (slots_n * x_n).sum(-1)                # (M, B)

            # ignore uninitialised slots when picking a merge target
            sim_for_merge = sim.masked_fill(~initialized, float('-inf'))
            max_sim, max_idx = sim_for_merge.max(dim=0)  # (B,), (B,)
            valid_max = torch.isfinite(max_sim)
            merge_mask = valid_max & (max_sim > tau)     # (B,)

            # eviction target: prefer uninitialised, else min(importance)
            any_uninit = (~initialized).any(dim=0)
            first_uninit_idx = (~initialized).int().argmax(dim=0)  # arbitrary if none
            importance = usage - self.age_decay * age
            evict_idx = importance.argmin(dim=0)  # if all initialised, pick min importance
            evict_idx = torch.where(any_uninit, first_uninit_idx, evict_idx)  # (B,)

            final_idx = torch.where(merge_mask, max_idx, evict_idx)  # (B,)

            # gather current slot value at final_idx -> (B, D)
            cur_slot = mem.gather(0, final_idx.view(1, B, 1).expand(1, B, D)).squeeze(0)

            # gated merge alpha; for inserts we fully overwrite (alpha = 1, slot detached)
            alpha_merge = merge_gate(max_sim.clamp(-1, 1).unsqueeze(-1))  # (B, 1)
            alpha_insert = torch.ones_like(alpha_merge)
            alpha = torch.where(merge_mask.unsqueeze(-1), alpha_merge, alpha_insert)

            # detach the previous slot when we are evicting/inserting -- this prevents
            # the autograd graph from spanning the entire history on overwritten slots
            cur_slot_used = torch.where(
                merge_mask.unsqueeze(-1), cur_slot, cur_slot.detach()
            )

            new_slot = (1.0 - alpha) * cur_slot_used + alpha * x_t  # (B, D)

            # scatter new_slot into mem at final_idx
            idx_expand = final_idx.view(1, B, 1).expand(1, B, D)
            mem = mem.scatter(0, idx_expand, new_slot.unsqueeze(0))

            # bookkeeping (one-hot mask of touched slots)
            touched = eye[final_idx].t()  # (M, B), float
            usage = usage + touched
            age = age + 1.0
            age = age * (1.0 - touched)
            initialized = initialized | touched.bool()

        return mem

    def forward(self, long_x, encoded_x):
        # long_x:    (L, B, D)
        # encoded_x: (S, B, D)  -- short-window encoded features
        B = encoded_x.size(1)
        device = encoded_x.device

        # 1. build per-level (stride-aggregated) sequences
        if long_x is None or long_x.size(0) == 0:
            mems = [
                self.mem_init[k].to(device).expand(-1, B, -1).contiguous()
                for k in range(self.num_levels)
            ]
        else:
            seqs = self._build_level_sequences(long_x)
            mems = [self._stream_update(seqs[k], k) for k in range(self.num_levels)]

        # 2. add scale embeddings and concatenate
        scaled = [mems[k] + self.scale_embed[k] for k in range(self.num_levels)]
        full_mem = torch.cat(scaled, dim=0)  # (M_total, B, D)

        # 3. multi-scale self-refinement (cross-scale interactions)
        full_mem_pe = self.history_positional_encoding(full_mem)
        refined = self.mem_self_refine(full_mem_pe)

        # 4. ground memory in the current short-window observation
        refined = self.mem_cross_refine(refined, encoded_x)
        refined = refined + self.dropout2(full_mem_pe)
        refined = self.norm2(refined)

        # 5. multi-scale snippet supervision
        snippet_feat = self.snip_head(refined)
        snippet_feat = torch.flatten(snippet_feat.permute(1, 0, 2), start_dim=1)
        snip_cls = self.snip_classifier(snippet_feat)

        return refined, snip_cls


class HistoryUnit(torch.nn.Module):
    def __init__(self, opt):
        super(HistoryUnit, self).__init__()
        self.n_feature=opt["feat_dim"] 
        n_class=opt["num_of_class"]
        n_embedding_dim=opt["hidden_dim"]
        n_hist_dec_head = 4
        n_hist_dec_layer = 5
        n_hist_dec_head_2 = 4
        n_hist_dec_layer_2 = 2
        self.anchors=opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        self.anchors_stride=[]
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        

        self.history_positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)   

        self.history_encoder_block1 = nn.TransformerDecoder(
                                            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_hist_dec_head, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_hist_dec_layer, 
                                            nn.LayerNorm(n_embedding_dim))  
        
        
        self.history_encoder_block2 = nn.TransformerDecoder(
                                            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_hist_dec_head_2, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_hist_dec_layer_2, 
                                            nn.LayerNorm(n_embedding_dim))  
        
        

        self.snip_head = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim//4), nn.ReLU())     
        self.snip_classifier = nn.Sequential(nn.Linear(self.history_tokens*n_embedding_dim//4, (self.history_tokens*n_embedding_dim//4)//4), nn.ReLU(), nn.Linear((self.history_tokens*n_embedding_dim//4)//4,n_class))                      
        

        self.history_token = nn.Parameter(torch.zeros(self.history_tokens, 1, n_embedding_dim))
        # self.history_token_extra = nn.Parameter(torch.zeros(self.history_tokens*2, 1, n_embedding_dim))

        self.norm2 = nn.LayerNorm(n_embedding_dim)
        self.dropout2 = nn.Dropout(0.1)


    def forward(self, long_x, encoded_x):
        

        ## History Encoder
        hist_pe_x = self.history_positional_encoding(long_x)
        history_token = self.history_token.expand(-1, hist_pe_x.shape[1], -1)  
        hist_encoded_x_1 = self.history_encoder_block1(history_token, hist_pe_x)
        hist_encoded_x_2 = self.history_encoder_block2(hist_encoded_x_1, encoded_x)
        hist_encoded_x_2 = hist_encoded_x_2 + self.dropout2(hist_encoded_x_1)
        hist_encoded_x = self.norm2(hist_encoded_x_2)
   
        ## Snippet Classfication Head
        snippet_feat = self.snip_head(hist_encoded_x_1)
        snippet_feat = torch.flatten(snippet_feat.permute(1, 0, 2), start_dim=1)
        
        snip_cls = self.snip_classifier(snippet_feat)
        
        return hist_encoded_x, snip_cls



class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature=opt["feat_dim"] 
        n_class=opt["num_of_class"]
        n_embedding_dim=opt["hidden_dim"]
        n_enc_layer=opt["enc_layer"]
        n_enc_head=opt["enc_head"]
        n_dec_layer=opt["dec_layer"]
        n_dec_head=opt["dec_head"]
        n_comb_dec_head = 4
        n_comb_dec_layer = 5
        n_seglen=opt["segment_size"]
        self.anchors=opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        self.anchors_stride=[]
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0

        self.feature_reduction_rgb = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        self.feature_reduction_flow = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        
        self.positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)      
        
        self.encoder = nn.TransformerEncoder(
                                            nn.TransformerEncoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_enc_head, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_enc_layer, 
                                            nn.LayerNorm(n_embedding_dim))
                                            
        self.decoder = nn.TransformerDecoder(
                                            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_dec_head, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_dec_layer, 
                                            nn.LayerNorm(n_embedding_dim))  

        # Toggleable hierarchical memory. Defaults to True so the new
        # mechanism is used out-of-the-box; set opt["use_hier_memory"]=False
        # to fall back to the original flat HistoryUnit for ablation.
        if bool(opt.get("use_hier_memory", True)):
            self.history_unit = HierarchicalMemoryUnit(opt)
        else:
            self.history_unit = HistoryUnit(opt)


        self.history_anchor_decoder_block1 = nn.TransformerDecoder(
                                            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_comb_dec_head, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_comb_dec_layer, 
                                            nn.LayerNorm(n_embedding_dim))  
            

        self.classifier = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim,n_class))
        self.regressor = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim,2))    
                           
        
        self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))


        self.norm1 = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(0.1)

        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2])
        # base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:])
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2].float())
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:].float())
        base_x = torch.cat([base_x_rgb,base_x_flow],dim=-1)
        
        base_x = base_x.permute([1,0,2])# seq_len x batch x featsize x 

        short_x = base_x[-self.short_window_size:]

        long_x = base_x[:-self.short_window_size]
        
        ## Anchor Feature Generator
        pe_x = self.positional_encoding(short_x)
        encoded_x = self.encoder(pe_x)   
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)  
        decoded_x = self.decoder(decoder_token, encoded_x) 
        decoded_x = decoded_x

        ## Future-Supervised History Module
        hist_encoded_x, snip_cls = self.history_unit(long_x, encoded_x)


        ## History Driven Anchor Refinement
        decoded_anchor_feat = self.history_anchor_decoder_block1(decoded_x, hist_encoded_x)
        decoded_anchor_feat = decoded_anchor_feat + self.dropout1(decoded_x)
        decoded_anchor_feat = self.norm1(decoded_anchor_feat)
        decoded_anchor_feat = decoded_anchor_feat.permute([1, 0, 2])
        
        # Predition Module
        anc_cls = self.classifier(decoded_anchor_feat)
        anc_reg = self.regressor(decoded_anchor_feat)
        
        return anc_cls, anc_reg, snip_cls

 
class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class=opt["num_of_class"]-1
        n_seglen=opt["segment_size"]
        n_embedding_dim=2*n_seglen
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        # FC layers for the 2 streams
        
        self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2 = nn.Linear(n_embedding_dim, 1)
        self.norm = nn.InstanceNorm1d(n_class)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        #inputs - batch x seq_len x class
        
        base_x = inputs.permute([0,2,1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        
        return x
        
