from src.lexicon import detect_report_claims, check_consistency

def test_detect_positive_and_negative_claims():
    claims = detect_report_claims("Small pleural effusion. No pneumothorax.")
    by_name = {c["finding"]: c["status"] for c in claims}
    assert by_name["pleural effusion"] == "positive"
    assert by_name["pneumothorax"] == "negative"

def test_contradiction_detected():
    result = check_consistency("pneumothorax", "positive", "No pneumothorax is seen.")
    assert result["status"] == "potential_contradiction"

def test_missing_report_not_checked():
    result = check_consistency("atelectasis", "positive", "")
    assert result["status"] == "not_checked"

def test_uncertain_language():
    claims = detect_report_claims("Possible pleural effusion.")
    assert any(c["finding"] == "pleural effusion" and c["status"] == "uncertain" for c in claims)
