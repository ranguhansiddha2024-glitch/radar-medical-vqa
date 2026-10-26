import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from src.model import load_biomedclip, embed_image, rank_findings
from src.lexicon import LEXICON, check_consistency
from train_contrastive import ProjectionModel

st.set_page_config(page_title="Medical VQA Research Prototype", page_icon="🩻", layout="wide")
st.title("Multimodal Medical VQA with Radiological Report Cross-Checking")
st.caption("Research prototype — not a diagnostic tool. Similarity scores are not probabilities.")

with st.sidebar:
    st.header("About this prototype")
    st.write("Uses Microsoft's pretrained BiomedCLIP image–text encoder and a small radiology lexicon.")
    st.warning("Do not use this application for clinical decisions. Model outputs can be wrong.")
    st.write("The report checker only checks explicit terms and simple negation patterns.")

@st.cache_resource(show_spinner="Loading BiomedCLIP model (first load downloads model files)…")
def get_model():
    return load_biomedclip()

@st.cache_resource
def get_projection():
    path = Path("outputs/projection_head.pt")
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    projection = ProjectionModel(checkpoint.get("dim", 512), checkpoint.get("proj_dim", 256))
    projection.load_state_dict(checkpoint["state_dict"])
    projection.eval()
    return projection

uploaded = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])
question = st.text_input(
    "Clinical question",
    value="Is pleural effusion present?",
    help="This prototype ranks a fixed set of radiological finding prompts; it is not a free-form generative VQA model."
)
report = st.text_area(
    "Optional reference radiology report",
    placeholder="Paste a de-identified report excerpt here for a rule-based consistency check.",
    height=120,
)

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns([1, 1])
        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with right:
            st.subheader("Model output")
            if st.button("Analyze image", type="primary"):
                with st.spinner("Computing image–text similarities on CPU…"):
                    model, preprocess, tokenizer = get_model()
                    image_embedding = embed_image(model, preprocess, image)
                    projection = get_projection()
                    ranked = rank_findings(model, tokenizer, image_embedding, projection=projection)
                    if projection is not None:
                        st.success("Using the project-trained contrastive projection head.")
                    else:
                        st.info("Using pretrained BiomedCLIP embeddings. Run the optional training experiment to add the project-trained projection head.")
                    question_lower = question.lower()
                    selected = next(
                        (name for name, info in LEXICON.items()
                         if name in question_lower or any(s in question_lower for s in info["synonyms"])),
                        None
                    )
                    if selected is None:
                        st.info("The question does not match a known finding in the lexicon. Showing the highest-margin finding instead.")
                        selected_row = ranked[0]
                    else:
                        selected_row = next(row for row in ranked if row["finding"] == selected)
                    st.markdown(f"**Finding queried:** {selected_row['finding']}")
                    st.markdown(f"**Model-preferred polarity:** {selected_row['preferred_status']}")
                    st.metric("Positive vs negative similarity margin", f"{selected_row['margin']:.4f}")
                    st.caption("A positive margin means the positive prompt matched more strongly than its negative counterpart. It is not a calibrated probability or diagnosis.")
                    consistency = check_consistency(
                        selected_row["finding"], selected_row["preferred_status"], report
                    )
                    st.subheader("Reference-report consistency")
                    st.write(f"**Status:** `{consistency['status']}`")
                    st.write(consistency["message"])
                    if consistency["matched_claims"]:
                        st.dataframe(pd.DataFrame(consistency["matched_claims"]), use_container_width=True)
                    st.subheader("Ranked findings")
                    st.dataframe(pd.DataFrame(ranked), use_container_width=True)
                    st.download_button(
                        "Download result table (CSV)",
                        pd.DataFrame(ranked).to_csv(index=False).encode("utf-8"),
                        file_name="finding_similarity_results.csv",
                        mime="text/csv",
                    )
    except Exception as exc:
        st.error(f"Could not process the image: {exc}")
        st.exception(exc)
else:
    st.info("Upload an image to enable analysis.")

st.divider()
st.caption("Model: BiomedCLIP (Microsoft). Dataset training is a separate experiment; this app initially uses the pretrained checkpoint without claiming project-specific fine-tuning.")
