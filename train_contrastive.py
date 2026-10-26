"""CPU-friendly contrastive projection-head training on a small X-Raydar subset.

The large pretrained BiomedCLIP encoders are frozen. We precompute embeddings, then train
small projection heads with symmetric InfoNCE and an explicit positive-vs-negated finding
margin loss. This is project-specific fine-tuning of projection heads, NOT full BiomedCLIP
pretraining. Only use this dataset under its stated non-commercial research/evaluation terms.
"""
import json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from src.model import load_biomedclip

SEED = 41
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DATA_ROOT = Path("data/xraydar-multimodal")
CACHE_DIR = Path("data/processed")
OUT_DIR = Path("outputs")
MAX_SAMPLES = 120  # increase after a successful first run; full dataset has 979 images
EPOCHS = 12
BATCH_SIZE = 8
DEVICE = "cpu"

class ProjectionModel(nn.Module):
    def __init__(self, dim=512, proj_dim=256):
        super().__init__()
        self.image_proj = nn.Sequential(nn.Linear(dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.text_proj = nn.Sequential(nn.Linear(dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, image_features, text_features):
        image_z = F.normalize(self.image_proj(image_features), dim=-1)
        text_z = F.normalize(self.text_proj(text_features), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100)
        return image_z, text_z, scale

def load_rows():
    ann = DATA_ROOT / "annotations.jsonl"
    if not ann.exists():
        raise FileNotFoundError(
            f"{ann} not found. First run: python download_dataset.py. "
            "Read the X-Raydar non-commercial research/evaluation terms."
        )
    rows = [json.loads(line) for line in ann.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("report_text") and r.get("image_file")]
    random.shuffle(rows)
    return rows[:MAX_SAMPLES]

@torch.inference_mode()
def build_cache(rows):
    model, preprocess, tokenizer = load_biomedclip()
    images, reports, pos_texts, neg_texts, valid_rows = [], [], [], [], []
    for i, row in enumerate(rows):
        path = DATA_ROOT / row["image_file"]
        if not path.exists():
            continue
        try:
            img = Image.open(path).convert("RGB")
            x = preprocess(img).unsqueeze(0)
            imf = model.encode_image(x).float().cpu().squeeze(0)
            report = row["report_text"][:1000]
            txt = tokenizer([report])
            tf = model.encode_text(txt).float().cpu().squeeze(0)
            labels = row.get("consensus_labels") or []
            # Only construct a contradiction pair from an explicit positive consensus label.
            # Label presence is used as a weak annotation, not a definitive clinical diagnosis.
            label = next((lab for lab in labels if isinstance(lab, str) and lab.strip()), None)
            if label:
                ptext = f"A chest X-ray showing {label.replace('_', ' ')}."
                ntext = f"A chest X-ray showing no {label.replace('_', ' ')}."
                pfeat = model.encode_text(tokenizer([ptext])).float().cpu().squeeze(0)
                nfeat = model.encode_text(tokenizer([ntext])).float().cpu().squeeze(0)
            else:
                pfeat = torch.zeros(512)
                nfeat = torch.zeros(512)
            images.append(imf); reports.append(tf); pos_texts.append(pfeat); neg_texts.append(nfeat)
            valid_rows.append({"xray_id": row.get("xray_id"), "has_contradiction_pair": bool(label)})
        except Exception as exc:
            print(f"Skipping row {i}: {exc}")
        if (i + 1) % 10 == 0:
            print(f"Encoded {i + 1}/{len(rows)} records")
    if len(images) < 12:
        raise RuntimeError(f"Only encoded {len(images)} rows; need at least 12 usable images/reports.")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_DIR / "features.npz",
        image=np.stack([x.numpy() for x in images]),
        report=np.stack([x.numpy() for x in reports]),
        positive=np.stack([x.numpy() for x in pos_texts]),
        negative=np.stack([x.numpy() for x in neg_texts]),
        has_pair=np.array([r["has_contradiction_pair"] for r in valid_rows], dtype=bool),
        ids=np.array([str(r["xray_id"]) for r in valid_rows]),
    )
    print(f"Cached {len(images)} image/report pairs in {CACHE_DIR / 'features.npz'}")

def main():
    OUT_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / "features.npz"
    if not cache_path.exists():
        rows = load_rows()
        print(f"Preparing frozen embeddings for up to {len(rows)} examples. First run may take a while on CPU.")
        build_cache(rows)
    data = np.load(cache_path, allow_pickle=False)
    image = torch.tensor(data["image"], dtype=torch.float32)
    report = torch.tensor(data["report"], dtype=torch.float32)
    positive = torch.tensor(data["positive"], dtype=torch.float32)
    negative = torch.tensor(data["negative"], dtype=torch.float32)
    has_pair = torch.tensor(data["has_pair"], dtype=torch.bool)
    n = len(image)
    order = torch.randperm(n)
    n_train = max(8, int(0.70 * n))
    n_val = max(1, int(0.15 * n))
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train+n_val]
    test_idx = order[n_train+n_val:]
    if len(test_idx) == 0:
        test_idx = val_idx
    model = ProjectionModel().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(EPOCHS):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        losses = []
        for start in range(0, len(perm), BATCH_SIZE):
            idx = perm[start:start+BATCH_SIZE]
            if len(idx) < 2:
                continue
            iz, tz, scale = model(image[idx], report[idx])
            logits = scale * (iz @ tz.T)
            targets = torch.arange(len(idx))
            loss_clip = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
            valid = has_pair[idx]
            loss_contra = torch.tensor(0.0)
            if valid.any():
                izc, pz, _ = model(image[idx][valid], positive[idx][valid])
                _, nz, _ = model(image[idx][valid], negative[idx][valid])
                pos_sim = (izc * pz).sum(dim=-1)
                neg_sim = (izc * nz).sum(dim=-1)
                loss_contra = F.relu(0.20 - pos_sim + neg_sim).mean()
            loss = loss_clip + 0.5 * loss_contra
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch + 1, "train_loss": avg_loss})
        print(f"Epoch {epoch+1:02d}/{EPOCHS} loss={avg_loss:.4f}")
    torch.save({
        "state_dict": model.state_dict(),
        "dim": 512, "proj_dim": 256,
        "seed": SEED, "epochs": EPOCHS,
        "train_indices": train_idx.numpy(),
        "val_indices": val_idx.numpy(),
        "test_indices": test_idx.numpy(),
    }, OUT_DIR / "projection_head.pt")
    (OUT_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
    print("Saved outputs/projection_head.pt and outputs/training_history.json")
    print("Now run: python evaluate.py")

if __name__ == "__main__":
    main()
