# Multimodal Medical VQA with Radiological Report Cross-Checking

A CPU-oriented academic research prototype using Microsoft's pretrained **BiomedCLIP** image–text encoder, a radiological terminology lexicon, a conservative report consistency checker, and an optional contrastive projection-head training experiment.

## Important scope statement

This is a research demonstration, **not a medical device** and not suitable for clinical decisions. Image–text similarities are not calibrated probabilities. The current application ranks a fixed set of finding prompts; it is not a free-form generative VQA model. Once `outputs/projection_head.pt` exists, the app loads the trained projection head automatically. The rule-based report checker is a heuristic and may miss or misinterpret negation, uncertainty, laterality, and context.

## Model and data

- Model: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, loaded with OpenCLIP using the official model-card pattern.
- Model files are roughly 789 MB, so first launch requires internet and disk space.
- Optional training dataset: `dnamodel/xraydar-multimodal` from Hugging Face. The dataset is approximately 2.81 GB and is restricted to non-commercial research/evaluation under its dataset terms. Read its dataset card and terms before downloading or using it.
- Training freezes the pretrained encoders and learns small projection heads with symmetric InfoNCE plus a margin-based positive-vs-negated-finding penalty. This is **not** full BiomedCLIP pretraining.

## 1. Requirements

Use Python 3.11 on Windows. Install Python from https://www.python.org/downloads/ and select **Add Python to PATH** during installation. Ensure at least 8 GB RAM and several GB free disk space. First-time model and dataset downloads require internet.

Open PowerShell in the project folder.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, use this command for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install CPU PyTorch first, then the project dependencies:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Check imports:

```powershell
python -c "import torch, open_clip, streamlit; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## 2. Run the application

```powershell
streamlit run app.py
```

The first run downloads BiomedCLIP model files and may take several minutes. Streamlit prints a local URL, normally `http://localhost:8501`. Upload an image, enter a question about a finding in the lexicon, optionally paste a de-identified report, and select **Analyze image**.

For a basic app smoke test, use a legitimate de-identified research image from an authorized dataset. Do not upload identifiable patient data.

## 3. Run automated tests

```powershell
pytest -q
```

The rule-based tests do not require downloading the model or dataset.

## 4. Optional dataset download for training

Read the dataset card first:
https://huggingface.co/datasets/dnamodel/xraydar-multimodal

The data is for non-commercial research/evaluation only. Its download is about 2.81 GB.

```powershell
python download_dataset.py
```

## 5. Run contrastive training

The first training run precomputes BiomedCLIP embeddings for up to 120 examples (configurable in `train_contrastive.py`) and can be slow on CPU. Begin with the default small subset. This training learns project-specific projection heads, not the whole foundation model.

```powershell
python train_contrastive.py
python evaluate.py
```

Expected output files:
- `data/processed/features.npz` — cached frozen embeddings
- `outputs/projection_head.pt` — trained projection head and split indices
- `outputs/training_history.json` — per-epoch loss
- `outputs/evaluation.json` — held-out exploratory metrics

The evaluation's Recall@1 is calculated within a small held-out test batch and should not be presented as a benchmark result. For stronger research claims, use more data, patient-level splits where IDs are available, a larger independent test set, and report confidence intervals. The current starter training script uses random row splits and therefore is a learning prototype, not a publication-grade validation pipeline.

## 6. How the components work

1. BiomedCLIP embeds an uploaded image and candidate radiological descriptions.
2. For each finding, the app compares the similarity of positive and negative prompts.
3. The user question is mapped to a finding in the lexicon.
4. If a reference report is supplied, the consistency checker looks for explicit positive/negative mentions and uncertainty phrases.
5. The app returns a model-preferred polarity, similarity margin, consistency status, and ranked findings.

## 7. Scientific limitations to state in the report

- The initial app uses a pretrained general biomedical image–text model; it is not a clinically validated chest-X-ray diagnostic system.
- Prompt similarity is not a probability or proof of image evidence.
- The initial VQA component is candidate-based, not a generative answer model.
- Lexicon matching is deliberately simple and can mishandle sentence scope, negation, laterality, and uncertainty.
- The optional training subset is small; exploratory metrics must not be advertised as clinical performance.
- Report and image labels may legitimately differ.
- Do not claim improvement unless the baseline and trained variant are evaluated on the same independent test set.

## 8. Suggested 5-day schedule

**Day 1:** Install environment, run tests, launch app, confirm first model download.  
**Day 2:** Test with permitted research images; record screenshots and document limitations.  
**Day 3:** Download dataset if feasible; run the embedding cache and training experiment.  
**Day 4:** Run evaluation, analyze failure cases, prepare baseline-vs-trained comparison.  
**Day 5:** Finalize report, citations, architecture diagram, setup instructions, test results, and demo script.

## References

- BiomedCLIP model card: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- BiomedCLIP paper: https://arxiv.org/abs/2303.00915
- X-Raydar multimodal dataset: https://huggingface.co/datasets/dnamodel/xraydar-multimodal
- Streamlit installation docs: https://docs.streamlit.io/get-started/installation
- OpenCLIP: https://github.com/mlfoundations/open_clip
