"""BiomedCLIP inference helpers. Model weights are downloaded from Hugging Face on first use."""
import torch
import open_clip
from .lexicon import LEXICON

MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

def load_biomedclip():
    # The official Hugging Face model card documents this OpenCLIP loading pattern.
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    model = model.to("cpu").eval()
    return model, preprocess, tokenizer

@torch.inference_mode()
def embed_image(model, preprocess, image):
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to("cpu")
    features = model.encode_image(tensor)
    return torch.nn.functional.normalize(features.float(), dim=-1)

@torch.inference_mode()
def embed_texts(model, tokenizer, texts):
    tokens = tokenizer(texts).to("cpu")
    features = model.encode_text(tokens)
    return torch.nn.functional.normalize(features.float(), dim=-1)

@torch.inference_mode()
def rank_findings(model, tokenizer, image_embedding, finding_names=None, projection=None):
    """Compare positive and negative candidate prompts; scores are similarities, not probabilities."""
    names = finding_names or list(LEXICON.keys())
    prompts = []
    for name in names:
        prompts.extend([LEXICON[name]["positive"], LEXICON[name]["negative"]])
    text_features = embed_texts(model, tokenizer, prompts)
    if projection is not None:
        # Optional project-trained projection heads, learned by train_contrastive.py.
        image_features = torch.nn.functional.normalize(projection.image_proj(image_embedding), dim=-1)
        text_features = torch.nn.functional.normalize(projection.text_proj(text_features), dim=-1)
    else:
        image_features = image_embedding
    sims = (image_features @ text_features.T).squeeze(0).cpu().tolist()
    rows = []
    for idx, name in enumerate(names):
        pos, neg = sims[2 * idx], sims[2 * idx + 1]
        rows.append({
            "finding": name,
            "positive_similarity": float(pos),
            "negative_similarity": float(neg),
            "preferred_status": "positive" if pos >= neg else "negative",
            "margin": float(pos - neg),
        })
    return sorted(rows, key=lambda x: abs(x["margin"]), reverse=True)
