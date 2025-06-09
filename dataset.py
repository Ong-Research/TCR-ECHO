import torch
from torch.utils.data import Dataset

class TCRPeptideDataset(Dataset):
    def __init__(self, df, tokenizer, atchley_map, cols,
                 mask_prob: float = 0.0,
                 tcr_max_len: int = 25,
                 pep_max_len: int = 15):
        """
        df: DataFrame with TCR, peptide, and label columns
        tokenizer: HuggingFace tokenizer for ESM models
        atchley_map: dict mapping amino acid -> Atchley vector list
        cols: dict with 'tcr', 'peptide', 'label' keys
        mask_prob: probability of randomly masking each token
        tcr_max_len: fixed length for TCR sequences
        pep_max_len: fixed length for peptide sequences
        """
        print('mask_prob:', mask_prob)
        self.seqs1 = df[cols['tcr']].astype(str).tolist()
        self.seqs2 = df[cols['peptide']].astype(str).tolist()
        self.labels = df[cols['label']].tolist()
        self.tokenizer = tokenizer
        self.atchley_map = atchley_map
        self.mask_prob = mask_prob
        self.tcr_max_len = tcr_max_len
        self.pep_max_len = pep_max_len
        self.mask_token_id = getattr(tokenizer, 'mask_token_id', None)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq1 = self.seqs1[idx]
        seq2 = self.seqs2[idx]
        label = self.labels[idx]

        # Tokenize with fixed lengths
        enc1 = self.tokenizer(
            seq1,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.tcr_max_len
        )
        enc2 = self.tokenizer(
            seq2,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.pep_max_len
        )
        input_ids1 = enc1['input_ids'].squeeze(0)
        mask1      = enc1['attention_mask'].squeeze(0)
        input_ids2 = enc2['input_ids'].squeeze(0)
        mask2      = enc2['attention_mask'].squeeze(0)

        # Random masking
        if self.mask_prob > 0.0 and self.mask_token_id is not None:
            rand1 = torch.rand(input_ids1.shape)
            mask_idx1 = (rand1 < self.mask_prob) & mask1.bool()
            input_ids1[mask_idx1] = self.mask_token_id
            rand2 = torch.rand(input_ids2.shape)
            mask_idx2 = (rand2 < self.mask_prob) & mask2.bool()
            input_ids2[mask_idx2] = self.mask_token_id

        # Build fixed-size Atchley factor tensors
        # TCR
        tcr_factors = [
            self.atchley_map.get(aa, [0.0]*len(next(iter(self.atchley_map.values()))))
            for aa in seq1[:self.tcr_max_len]
        ]
        while len(tcr_factors) < self.tcr_max_len:
            tcr_factors.append([0.0]*len(next(iter(self.atchley_map.values()))))
        at1 = torch.tensor(tcr_factors, dtype=torch.float)

        # Peptide
        pep_factors = [
            self.atchley_map.get(aa, [0.0]*len(next(iter(self.atchley_map.values()))))
            for aa in seq2[:self.pep_max_len]
        ]
        while len(pep_factors) < self.pep_max_len:
            pep_factors.append([0.0]*len(next(iter(self.atchley_map.values()))))
        at2 = torch.tensor(pep_factors, dtype=torch.float)

        return (
            input_ids1, mask1,
            input_ids2, mask2,
            at1, at2,
            torch.tensor(label, dtype=torch.long)
        )
