import os
import json
import pandas as pd
import numpy as np
import torch

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)

def save_final_json(log_dir, per_fold, summary):
    result = {"per_fold": per_fold, **summary}
    save_json(result, os.path.join(log_dir, "result.json"))


def append_aggregate_csv(csv_path, run_id, summary, hyper):
    row = {
        'id': run_id,
        'hyper': hyper,
        'acc_avg': summary['average']['accuracy'], 'acc_std': summary['std']['accuracy'],
        'sen_avg': summary['average']['sensitivity'], 'sen_std': summary['std']['sensitivity'],
        'spe_avg': summary['average']['specificity'], 'spe_std': summary['std']['specificity'],
        'f1_avg': summary['average']['f1-score'], 'f1_std': summary['std']['f1-score'],
        'roc_avg': summary['average']['roc-auc'], 'roc_std': summary['std']['roc-auc'],
    }
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)

def save_split_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)

def save_arrays(log_dir, k, adjs, pccs, d_e, d_h):
    np.save(os.path.join(log_dir, f"adj_{k}.npy"), torch.cat(adjs).cpu().numpy())
    np.save(os.path.join(log_dir, f"pcc_{k}.npy"), torch.cat(pccs).cpu().numpy())
    np.save(os.path.join(log_dir, f"dist_e_{k}.npy"), np.concatenate([t.cpu().numpy() for sub in d_e for t in sub]))
    np.save(os.path.join(log_dir, f"dist_h_{k}.npy"), np.concatenate([t.cpu().numpy() for sub in d_h for t in sub]))