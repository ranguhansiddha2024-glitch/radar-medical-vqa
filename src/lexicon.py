"""Small, explicit radiology terminology lexicon and conservative consistency checks."""
import re

LEXICON = {
    "pleural effusion": {
        "synonyms": ["pleural effusion", "pleural fluid", "fluid in the pleural space"],
        "positive": "A chest X-ray showing pleural effusion.",
        "negative": "A chest X-ray showing no pleural effusion.",
    },
    "pneumothorax": {
        "synonyms": ["pneumothorax", "pleural air", "air in the pleural space"],
        "positive": "A chest X-ray showing pneumothorax.",
        "negative": "A chest X-ray showing no pneumothorax.",
    },
    "cardiomegaly": {
        "synonyms": ["cardiomegaly", "enlarged cardiac silhouette", "enlarged heart"],
        "positive": "A chest X-ray showing cardiomegaly.",
        "negative": "A chest X-ray showing no cardiomegaly.",
    },
    "atelectasis": {
        "synonyms": ["atelectasis", "atelectatic change", "lung volume loss"],
        "positive": "A chest X-ray showing atelectasis.",
        "negative": "A chest X-ray showing no atelectasis.",
    },
    "consolidation": {
        "synonyms": ["consolidation", "air-space consolidation", "airspace opacity"],
        "positive": "A chest X-ray showing consolidation.",
        "negative": "A chest X-ray showing no consolidation.",
    },
    "pulmonary edema": {
        "synonyms": ["pulmonary edema", "pulmonary oedema", "interstitial edema", "interstitial oedema"],
        "positive": "A chest X-ray showing pulmonary edema.",
        "negative": "A chest X-ray showing no pulmonary edema.",
    },
    "fracture": {
        "synonyms": ["fracture", "rib fracture", "bony fracture"],
        "positive": "A chest X-ray showing a fracture.",
        "negative": "A chest X-ray showing no fracture.",
    },
    "opacity": {
        "synonyms": ["opacity", "opacities", "lung opacity", "airspace opacity"],
        "positive": "A chest X-ray showing an opacity.",
        "negative": "A chest X-ray showing no focal opacity.",
    },
}

NEGATION_PATTERNS = [
    r"\bno\b", r"\bwithout\b", r"\babsent\b", r"\bnegative for\b",
    r"\bnot seen\b", r"\bnot identified\b", r"\bno evidence of\b",
    r"\bfree of\b", r"\babsence of\b",
]
UNCERTAINTY_PATTERNS = [
    r"\bpossible\b", r"\bpossibly\b", r"\bmay represent\b", r"\bcannot exclude\b",
    r"\bquestionable\b", r"\bsuspected\b", r"\bprobably\b", r"\buncertain\b",
]

def _term_pattern(term):
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)


def detect_report_claims(report_text):
    """Extract finding mentions using sentence-local negation and uncertainty."""
    text = (report_text or "").lower()
    claims = []

    # Process each sentence separately so negation does not leak across sentences.
    sentences = re.split(r'(?<=[.!?;])\s+|\n+', text)

    for sentence in sentences:
        for canonical, info in LEXICON.items():
            for synonym in sorted(info["synonyms"], key=len, reverse=True):
                match = _term_pattern(synonym).search(sentence)
                if not match:
                    continue

                # Only inspect words immediately before the finding.
                prefix = sentence[max(0, match.start() - 60):match.start()]
                uncertain = any(
                    re.search(p, prefix) for p in UNCERTAINTY_PATTERNS
                )
                negated = any(
                    re.search(p, prefix) for p in NEGATION_PATTERNS
                )

                status = (
                    "uncertain" if uncertain
                    else "negative" if negated
                    else "positive"
                )

                claims.append({
                    "finding": canonical,
                    "status": status,
                    "matched_phrase": synonym,
                    "context": sentence.strip(),
                })
                break

    return claims

def check_consistency(predicted_finding, predicted_status, report_text):
    """Check a model-preferred claim against an optional reference report."""
    if not report_text or not report_text.strip():
        return {
            "status": "not_checked",
            "message": "No reference report supplied; report consistency cannot be assessed.",
            "matched_claims": [],
        }
    claims = detect_report_claims(report_text)
    relevant = [c for c in claims if c["finding"] == predicted_finding]
    if not relevant:
        return {
            "status": "insufficient_evidence",
            "message": f"The supplied report has no explicit lexicon match for {predicted_finding}.",
            "matched_claims": claims,
        }
    statuses = {c["status"] for c in relevant}
    if "uncertain" in statuses:
        return {
            "status": "uncertain",
            "message": f"The report uses uncertain language about {predicted_finding}; do not treat it as a definite finding.",
            "matched_claims": relevant,
        }
    if predicted_status in ("positive", "negative") and (
        ("positive" in statuses and predicted_status == "negative") or
        ("negative" in statuses and predicted_status == "positive")
    ):
        return {
            "status": "potential_contradiction",
            "message": f"The model prefers '{predicted_status}' for {predicted_finding}, but the supplied report explicitly indicates the opposite. Review the image and report.",
            "matched_claims": relevant,
        }
    return {
        "status": "consistent_with_report",
        "message": f"No direct polarity contradiction was detected for {predicted_finding}. This is not proof of clinical correctness.",
        "matched_claims": relevant,
    }
