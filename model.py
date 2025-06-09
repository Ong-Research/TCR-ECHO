import torch
import torch.nn as nn
from torch.nn import functional as F
from peft import get_peft_model, LoraConfig, TaskType
from transformers import EsmModel, EsmConfig
import esm
from esm.models.esmc import ESMC
from attentions import BalancedAtchleyAttention, StandardCrossAttention

class Model(nn.Module):
    def __init__(
        self,
        esm1_name: str,
        esm2_name: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: list,
        contrastive_temp: float,
        lambda_enc: float,
        lambda_int: float,
        classifier_hidden: int,
        dropout: float,
        focal_gamma: float,
        class_balance: float,
        use_lora: bool = False,
        num_heads: int = 8,
        enable_monitoring: bool = True,
        cross_attn_dropout: float = 0.1,
        second_contrastive: bool = True,
        random_init: bool = False
    ):
        super().__init__()
        # Load pretrained ESM backbones 
        self.esm1_alphabet = None
        self.esm2_alphabet = None
        self.esm1_name = esm1_name
        self.esm2_name = esm2_name
        
        if random_init:
            cfg = EsmConfig.from_pretrained(f'facebook/{esm1_name}')
            self.esm1 = EsmModel(cfg)  
            self.esm2 = EsmModel(cfg)
            hidden_dim = cfg.hidden_size 
        else:
            if esm1_name.startswith("esmc"):
                self.esm1 = ESMC.from_pretrained( esm1_name)
            else:
                self.esm1 = EsmModel.from_pretrained(f'facebook/{esm1_name}')
            if esm2_name.startswith("esmc"):
                self.esm2 = ESMC.from_pretrained( esm1_name)
            else:
                self.esm2 = EsmModel.from_pretrained(f'facebook/{esm2_name}')

        if use_lora:
            # PEFT LoRA config
            peft_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                bias='none',
                lora_dropout=lora_dropout,
                target_modules=['attn.layernorm_qkv.1'] if 'esmc' in esm1_name else ['attention.self.key', 'attention.self.value'],
                layers_to_transform= lora_target_modules
            )
            
            self.esm1 = get_peft_model(self.esm1, peft_cfg)
            self.esm2 = get_peft_model(self.esm2, peft_cfg)

            # Freeze base parameters
            for name, param in self.esm1.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False
            for name, param in self.esm2.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False

        # hidden dimension
        hidden_dim = getattr(self.esm1.config, 'hidden_size', getattr(self.esm1, 'embed_dim', None))

        
        self.cross_att_tpep = BalancedAtchleyAttention(hidden_dim, 
                                                  num_heads=num_heads,
                                                  dropout=cross_attn_dropout,
                                                  enable_monitoring=enable_monitoring)
        self.cross_att_pep_t = BalancedAtchleyAttention(hidden_dim, 
                                                  num_heads=num_heads,
                                                  dropout=cross_attn_dropout,
                                                  enable_monitoring=enable_monitoring)
        self.second_contrastive = second_contrastive
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1)
        )
        self.no_cross_att = False
        # Loss params
        self.focal_gamma = focal_gamma
        self.class_balance = class_balance
        self.contrastive_temp = contrastive_temp
        self.lambda_enc = lambda_enc
        self.lambda_int = lambda_int

    def forward(
        self,
        inp1, mask1,
        inp2, mask2,
        atchley1, atchley2,
        labels
    ):
        # Encode and remove CLS token
        if self.esm1_name.startswith("esmc"):
            
            out1 = self.esm1(sequence_tokens=inp1).embeddings
            out2 = self.esm2(sequence_tokens=inp2).embeddings
        
        else:
            out1 = self.esm1(input_ids=inp1, attention_mask=mask1).last_hidden_state
            out2 = self.esm2(input_ids=inp2, attention_mask=mask2).last_hidden_state

        tcr_enc = out1[:, 1:, :]            
        pep_enc = out2[:, 1:, :]

        # Encoder-level contrastive
        
        loss_enc = flexible_peptide_contrastive(pep_enc, tcr_enc, labels, temp=self.contrastive_temp)

        # Cross-attention + pooling
        tcr_att = self.cross_att_tpep(tcr_enc, pep_enc, atchley1, atchley2)
        pep_att = self.cross_att_pep_t(pep_enc,tcr_enc, atchley2, atchley1)
        tcr_pool = tcr_att.mean(dim=1)
        pep_pool = pep_att.mean(dim=1)
        
        # Interaction-level contrastive (use labels)
        if self.second_contrastive:
            loss_int = flexible_peptide_contrastive(pep_pool, tcr_pool, labels, temp=self.contrastive_temp)
        
        # Fusion & classification
        seq = torch.cat([tcr_pool, pep_pool], dim=-1)
        logits = self.classifier(self.dropout(seq)).squeeze(-1)

        # Compute loss if labels provided
        if labels is not None:
            loss_focal = focal_loss(logits, labels, gamma=self.focal_gamma, alpha=self.class_balance)
            if self.second_contrastive:
                total_loss = loss_focal + self.lambda_enc * loss_enc + self.lambda_int * loss_int
            else:
                total_loss = loss_focal + self.lambda_enc * loss_enc
            return logits, total_loss
        return logits, total_loss
    
    

def flexible_peptide_contrastive(pmhc, tcr, labels, temp=0.2, mode='pooled'):
    """
    Peptide-anchored contrastive loss supporting both pooled and residue-level contrasts.
    Each peptide is an anchor, with its paired TCR as positive and other TCRs as negatives.
    
    Args:
      pmhc: Tensor of shape [B, L_p, H] or [B, H] — pMHC embeddings.
      tcr: Tensor of shape [B, L_t, H] or [B, H] — TCR embeddings.
      labels: Tensor of shape [B], binary (1 = positive binding, 0 = negative).
      temp: Float, temperature scaling factor.
      mode: String, one of ['pooled', 'residue_wise']
      
    Returns:
      Scalar loss (mean over all peptide anchors).
    """
    batch_size = pmhc.shape[0]
    device = pmhc.device
    
    # Verify input dimensions
    pmhc_is_sequence = pmhc.dim() == 3
    tcr_is_sequence = tcr.dim() == 3
    
    # Handle different modes
    if mode == 'pooled':
        # Pool sequences
        if pmhc_is_sequence:
            pmhc_pooled = pmhc.mean(dim=1)   # [B, H]
        else:
            pmhc_pooled = pmhc
            
        if tcr_is_sequence:
            tcr_pooled = tcr.mean(dim=1)   # [B, H]
        else:
            tcr_pooled = tcr
        
        # Normalize embeddings
        pmhc_norm = F.normalize(pmhc_pooled, p=2, dim=1)  # [B, H]
        tcr_norm = F.normalize(tcr_pooled, p=2, dim=1)    # [B, H]
        
        # Compute similarity matrix
        sim = torch.matmul(pmhc_norm, tcr_norm.T) / temp  # [B, B]
        
    elif mode == 'residue_wise':
        if not (pmhc_is_sequence and tcr_is_sequence):
            raise ValueError("Residue-wise mode requires sequence dimensions for both pMHC and TCR")
        
        # Sequence lengths
        L_p = pmhc.shape[1]
        L_t = tcr.shape[1]
        
        # Reshape to combine batch and sequence dims for normalization
        pmhc_flat = pmhc.reshape(-1, pmhc.shape[-1])  # [B*L_p, H]
        tcr_flat = tcr.reshape(-1, tcr.shape[-1])     # [B*L_t, H]
        
        # Normalize embeddings
        pmhc_norm = F.normalize(pmhc_flat, p=2, dim=1).view(batch_size, L_p, -1)  # [B, L_p, H]
        tcr_norm = F.normalize(tcr_flat, p=2, dim=1).view(batch_size, L_t, -1)    # [B, L_t, H]
        
        # Compute residue-to-residue similarity for each pair
        # For each (pmhc, tcr) pair, compute similarity matrix of shape [L_p, L_t]
        # Then average over residues to get a single similarity score
        sim = torch.zeros(batch_size, batch_size, device=device)
        
        for i in range(batch_size):
            for j in range(batch_size):
                # [L_p, H] @ [H, L_t] -> [L_p, L_t]
                residue_sim = torch.matmul(pmhc_norm[i], tcr_norm[j].T) / temp
                # Average over all residue pairs
                sim[i, j] = residue_sim.mean()
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Numerical stability: subtract max per row
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max.detach()
    exp_sim = torch.exp(sim)  # [B, B]
    
    # Create a positive mask where each peptide's positive is only its matched TCR
    # This is a diagonal matrix where (i,i) indicates the positive pair
    positive_mask = torch.eye(batch_size, device=device).bool()
    
    # Compute numerator: exp_sim for positive pairs (diagonal)
    numer = exp_sim.diagonal()  # [B]
    
    # Compute denominator: sum over all TCRs for each peptide
    denom = exp_sim.sum(dim=1)  # [B]
    
    # InfoNCE loss per peptide anchor
    eps = 1e-8
    losses = -torch.log((numer + eps) / (denom + eps))  # [B]
    
    positive_anchor_mask = labels.bool()
    if positive_anchor_mask.sum() > 0:
        positive_loss = losses[positive_anchor_mask].mean()
    else:
        positive_loss = torch.tensor(0.0, device=device)
    
    return losses.mean()

def focal_loss(logits, labels, gamma, alpha):
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
    p_t = labels * probs + (1 - labels) * (1 - probs)
    alpha_t = labels * alpha + (1 - labels) * (1 - alpha)
    loss = alpha_t * (1 - p_t) ** gamma * bce
    return loss.mean()