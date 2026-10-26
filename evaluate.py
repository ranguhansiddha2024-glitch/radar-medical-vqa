"""Evaluate the trained projection heads on held-out cached image/report pairs."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from train_contrastive import ProjectionModel

CACHE = Path("data/processed/features.npz")
CKPT = Path("outputs/projection_head.pt")

def main():
    if not CACHE.exists() or not CKPT.exists():
        raise FileNotFoundError("Run `python train_contrastive.py` first.")
    d = np.load(CACHE, allow_pickle=False)
    c = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = ProjectionModel(c["dim"], c["proj_dim"])
    model.load_state_dict(c["state_dict"])
    model.eval()
    image = torch.tensor(d["image"], dtype=torch.float32)
    report = torch.tensor(d["report"], dtype=torch.float32)
    pos = torch.tensor(d["positive"], dtype=torch.float32)
    neg = torch.tensor(d["negative"], dtype=torch.float32)
    has_pair = torch.tensor(d["has_pair"], dtype=torch.bool)
    test_idx = torch.tensor(c["test_indices"], dtype=torch.long)
    with torch.inference_mode():
        iz, tz, scale = model(image[test_idx], report[test_idx])
        sims = iz @ tz.T
        targets = torch.arange(len(test_idx))
        # For a square held-out batch, this is retrieval among the held-out reports.
        pred = sims.argmax(dim=1)
        recall1 = float((pred == targets).float().mean())
        valid = test_idx[has_pair[test_idx]]
        contra_acc = None
        if len(valid):
            iv, pv, _ = model(image[valid], pos[valid])
            _, nv, _ = model(image[valid], neg[valid])
            ps = (iv * pv).sum(-1)
            ns = (iv * nv).sum(-1)
            contra_acc = float((ps > ns).float().mean())
    result = {
        "test_examples": int(len(test_idx)),
        "image_to_report_recall_at_1_within_test_batch": recall1,
        "contradiction_pair_preference_accuracy": contra_acc,
        "warning": "Small subset, weak labels, and small test batch; metrics are exploratory, not clinical validation."
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/evaluation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
