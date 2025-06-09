import os
import yaml
import torch
import wandb
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import Model
from dataset import TCRPeptideDataset
from utils import save_checkpoint, compute_metrics, load_atchley, load_checkpoint


def train_one(cfg, epochs, run_name_suffix=""):
    """
    Single training run for given config and epoch count. Returns best validation AUC.
    """
    run = wandb.init(
        project=cfg['wandb']['project'],
        config=cfg,
        name=f"{cfg.get('run_name', 'exp')}{run_name_suffix}_{cfg['wandb']['run']}"
    )

    # Data preparation
    df_train = pd.read_csv(cfg['dataset']['train_csv'])
    if cfg['dataset'].get('val_csv'):
        df_val = pd.read_csv(cfg['dataset']['val_csv'])
        df_test = pd.read_csv(cfg['dataset']['test_csv'])
    else:
        df_test = pd.read_csv(cfg['dataset']['test_csv'])
        train_peps = set(df_train[cfg['dataset']['columns']['peptide']].unique())
        mask = df_test[cfg['dataset']['columns']['peptide']].isin(train_peps)
        df_val = df_test[~mask]
        df_test = df_test[mask]
    pos = (df_train['label'] == 1).sum()
    neg = (df_train['label'] == 0).sum()
    class_imbalance = neg / (pos + neg)
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/esm2_t6_8M_UR50D")
    atchley_map = load_atchley(cfg.get('atchley_path'))

    train_ds = TCRPeptideDataset(df_train, tokenizer, atchley_map, cfg['dataset']['columns'], mask_prob=cfg['dataset'].get('mask_prob', 0.0))
    val_ds   = TCRPeptideDataset(df_val, tokenizer, atchley_map, cfg['dataset']['columns'])

    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg['training']['batch_size'])

    # Model
    lora_preset = cfg['lora']['presets'][cfg['esm']['encoder1']]

    model_params = {
        'esm1_name':            cfg['esm']['encoder1'],
        'esm2_name':            cfg['esm']['encoder2'],
        'use_lora':             cfg['use_lora'],
        'lora_r':               lora_preset['r'],
        'lora_alpha':           lora_preset['alpha'],
        'lora_dropout':         lora_preset['dropout'],
        'lora_target_modules':  lora_preset['layers_to_transform'],
        'contrastive_temp':     cfg['contrastive']['temperature'],
        'lambda_enc':           cfg['contrastive']['lambda_enc'],
        'lambda_int':           cfg['contrastive']['lambda_int'],
        'classifier_hidden':    cfg['classifier_hidden'],
        'dropout':              cfg['training']['dropout'],
        'focal_gamma':          cfg['training']['focal_gamma'],
        'class_balance':        class_imbalance,
        'second_contrastive':      cfg['training']['second_contrastive'],
    }
    
    model = Model(**model_params).to(cfg.get('device','cpu'))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay']
    )

    best_auc = 0
    no_improve = 0
    for epoch in range(epochs):
        # train step
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            *inputs, labels = [b.to(cfg.get('device','cpu')) for b in batch]
            logits, loss = model(*inputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)

        # validation
        val_metrics,_,_ = compute_metrics(model, val_loader, device=cfg.get('device','cpu'))
        run.log({'train_loss': avg_loss, **{f'val_{k}': v for k,v in val_metrics.items()}, 'epoch': epoch})

        # track best
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            save_checkpoint(model, optimizer, cfg, best_auc, cfg['training']['output_dir'])
            no_improve = 0
        else:
            no_improve += 1
        # if no_improve >= cfg['training']['early_stopping']['patience']:
        #     break
    run.finish()
    return best_auc


def main(config):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    best_auc = train_one(cfg, epochs=cfg['training']['epochs'])

    # test evaluation
    model,_,run_cfg = load_checkpoint(Model, cfg['training']['output_dir'], device=cfg.get('device','cpu'))
    df_test = pd.read_csv(run_cfg['dataset']['test_csv'])
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{run_cfg['esm']['encoder1']}")
    atchley_map = load_atchley(run_cfg.get('atchley_path'))
    test_ds = TCRPeptideDataset(df_test, tokenizer, atchley_map, run_cfg['dataset']['columns'])
    test_loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'])
    test_metrics = compute_metrics(model, test_loader, device=cfg.get('device','cpu'))
    # save predictions
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            *inputs, lbls = [b.to(cfg.get('device','cpu')) for b in batch]
            logits = model(*inputs, labels=lbls)
            preds.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(lbls.cpu().tolist())
    out_df = pd.DataFrame({'pred':preds,'label':labels})
    out_df.to_csv(os.path.join(cfg['training']['output_dir'],'test_predictions.csv'),index=False)
    print("Final test metrics:", test_metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ESM-ViT model")
    parser.add_argument("--config", type=str, default=r"params/config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
