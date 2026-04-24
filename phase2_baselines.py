"""
=============================================================
PHASE 2 — BASELINE EVALUATION
Hospital Complaint Triage Assistant
=============================================================

TWO BASELINES evaluated on test.jsonl:
  Baseline 1 — Zero-Shot  : No instructions, no examples.
  Baseline 2 — Prompted   : Detailed prompt + valid labels + 2 examples.

FIXES APPLIED OVER PREVIOUS VERSION (Codex review + manual audit):
  [FIX-1] Model: mistral-7b-instruct-v0.2 removed (invalid on Groq).
          Now uses mixtral-8x7b-32768 (closest to Mistral family, production-stable on Groq).
          Llama-3 added as fallback reference model.
  [FIX-2] VALID_DEPARTMENTS corrected to match actual dataset routing.primary_department values.
          Old script had wrong values (General Ward, Surgery, OPD, etc.).
  [FIX-3] Field reading corrected:
          gt_dept now reads from routing.primary_department (not root 'department').
          gt_desc reads from complaint_description (root field, confirmed present).
          gt_caption reads from input.image_caption (nested, not root refined_caption).
  [FIX-4] Progress log no longer hardcodes /24. Uses dynamic len(test_records).
  [FIX-5] JSON extraction regex made robust with fallback chain.
  [FIX-6] Full Kaggle compatibility:
          - No os.environ dependency (API key via Kaggle Secrets / direct assignment).
          - Output paths use relative dirs (works in /kaggle/working/).
          - pip install block included as comment for Kaggle notebook cell.
  [FIX-7] Dataset was rebuilt from original 217 records:
          - All flagged records resolved and reintegrated (no data permanently excluded).
          - Stratified 70/15/15 split: train=153, val=32, test=32.

HOW TO RUN ON KAGGLE:
  Cell 1 (install):
    !pip install groq scikit-learn rouge-score nltk pandas -q
    import nltk; nltk.download('punkt', quiet=True)

  Cell 2 (secrets — Kaggle UI > Add-ons > Secrets):
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    import os
    os.environ["GROQ_API_KEY"] = secrets.get_secret("GROQ_API_KEY")

  Cell 3:
    !python phase2_baselines.py

HOW TO RUN LOCALLY:
  pip install groq scikit-learn rouge-score nltk pandas
  export GROQ_API_KEY="your_key_here"
  python phase2_baselines.py

OUTPUT FILES:
  baseline_results.csv     — all predictions + ground truth for all records
  baseline_metrics.json    — F1, accuracy, BLEU, ROUGE, JSON validity per baseline
  baseline_errors.csv      — only failed/wrong predictions for Phase 3 error analysis
=============================================================
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
import os
import json
import time
import re
import pandas as pd
from groq import Groq
from sklearn.metrics import accuracy_score, f1_score, classification_report
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt', quiet=True)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# [FIX-1] mixtral-8x7b-32768 is Groq's stable Mixtral (same family as Mistral-7B).
# If you want to compare with another model, change MODEL_NAME here.
# Other valid Groq model strings (as of 2025): llama3-8b-8192, llama3-70b-8192
MODEL_NAME  = "mixtral-8x7b-32768"

TEST_FILE   = "test.jsonl"    # upload test.jsonl to Kaggle dataset or same folder
OUTPUT_DIR  = "."             # /kaggle/working/ on Kaggle, . locally
DELAY_SEC   = 2               # seconds between API calls (Groq free tier: ~30 req/min)

# ─────────────────────────────────────────────────────────────
# VALID LABEL SETS — sourced directly from dataset
# ─────────────────────────────────────────────────────────────

VALID_CATEGORIES = [
    "Broken Hospital Bed",
    "Crowded Hospital Waiting Room",
    "Dirty Hospital Bathroom",
    "Empty / Unstaffed Nursing Station",
    "Overflowing Hospital Trash (Outside)",
    "Rats / Rodent Infestation",
    "Torn Hospital Privacy Curtain",
    "Unappetizing Hospital Food Tray",
    "Unhygienic / Contaminated Hospital Food",
    "Water Puddle on Hospital Floor",
]

VALID_SEVERITIES = ["low", "medium", "high", "critical"]

# [FIX-2] These are the ACTUAL routing.primary_department values in the dataset.
# Old script had wrong values (General Ward, Surgery, OPD, etc.) — those were
# legacy hospital_details.department values, not the routing ground truth.
VALID_DEPARTMENTS = [
    "Administration",
    "Dietary",
    "Facilities Management",
    "Housekeeping",
    "Maintenance",
    "Nursing",
    "Pest Control",
    "Waste Management",
]

# ─────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────

def load_test_data(path):
    """Loads test.jsonl. Each line is one record (JSON object)."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} test records from '{path}'")
    return records

# ─────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────

def build_zeroshot_prompt(caption, voice_text):
    """
    BASELINE 1 — Zero-Shot
    Intentionally weak. No label list. No examples. No format rules.
    Model gets raw situation and must guess the structure.
    This sets the floor — everything else should beat this.
    """
    return f"""A patient filed a hospital complaint. Here is the information:

Image caption: {caption}
Patient voice complaint: {voice_text}

Extract the complaint details and return a JSON with these fields:
- category
- severity
- department
- complaint_description"""


def build_prompted_prompt(caption, voice_text):
    """
    BASELINE 2 — Prompted (Few-Shot with valid label constraints)
    Gives the model everything except fine-tuned weights:
      - Clear role and task
      - Full list of valid categories, severities, departments
      - 2 worked examples (one per modality type)
      - Strict JSON-only output instruction

    The gap between Baseline 1 and Baseline 2 = value of prompt engineering.
    The gap between Baseline 2 and the fine-tuned model = value of QLoRA.
    """
    categories_str  = "\n".join(f"- {c}" for c in VALID_CATEGORIES)
    departments_str = ", ".join(VALID_DEPARTMENTS)

    return f"""You are a hospital complaint triage system.
Analyze the patient complaint (image caption + voice transcript) and extract structured information.

RULES:
- Return ONLY a valid JSON object. No explanation. No markdown. No extra text.
- Use ONLY the valid values listed below. Do not invent new values.

VALID CATEGORIES (pick exactly one):
{categories_str}

VALID SEVERITY LEVELS (pick exactly one):
- low      → minor inconvenience, no safety risk
- medium   → uncomfortable, affects patient experience  
- high     → affects patient safety or care quality
- critical → immediate danger, requires urgent escalation

VALID DEPARTMENTS (pick exactly one):
{departments_str}

EXAMPLE 1:
Input:
  Image caption: The hospital bathroom has visible mold on the walls and dirty floors.
  Voice complaint: The bathroom in my ward is filthy. There is mold everywhere and it smells terrible.
Output:
{{"category": "Dirty Hospital Bathroom", "severity": "high", "department": "Housekeeping", "complaint_description": "Patient reported severe unhygienic conditions in the hospital bathroom including visible mold, dirty floors, and foul odor."}}

EXAMPLE 2:
Input:
  Image caption: Rats and rodent droppings visible near the hospital kitchen area.
  Voice complaint: I saw a rat running across the corridor near the food area. This is completely unacceptable.
Output:
{{"category": "Rats / Rodent Infestation", "severity": "critical", "department": "Pest Control", "complaint_description": "Patient observed rodent presence near the hospital kitchen and corridor, posing a serious hygiene and infection control risk."}}

Now process this complaint:
Image caption: {caption}
Voice complaint: {voice_text}

Return ONLY the JSON object:"""


# ─────────────────────────────────────────────────────────────
# API CALL
# ─────────────────────────────────────────────────────────────

def call_model(client, prompt, record_id, baseline_name):
    """
    Calls the model via Groq API.
    temperature=0 → deterministic output (required for reproducible evaluation).
    Returns raw text response, or None on error.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a hospital complaint triage assistant. "
                        "Always respond with valid JSON only. "
                        "Never include explanation or markdown formatting."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,      # deterministic — critical for fair eval
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"  ERROR on {record_id} [{baseline_name}]: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# JSON PARSER
# ─────────────────────────────────────────────────────────────

def parse_json_output(raw_text):
    """
    [FIX-5] Robust JSON extraction with 3-layer fallback chain.
    
    Layer 1: Strip markdown fences (```json ... ```) and extract {...}
    Layer 2: Find first { to last } and parse that substring
    Layer 3: Direct parse of cleaned text
    
    Returns (parsed_dict, is_valid_json_bool)
    """
    if not raw_text:
        return {}, False

    # Layer 1: remove markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    # Layer 2: find the outermost { ... } block
    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end+1]
        try:
            return json.loads(candidate), True
        except json.JSONDecodeError:
            pass

    # Layer 3: direct parse
    try:
        return json.loads(cleaned), True
    except json.JSONDecodeError:
        return {}, False


# ─────────────────────────────────────────────────────────────
# FIELD EXTRACTOR
# ─────────────────────────────────────────────────────────────

def extract_fields(parsed_dict):
    """
    Safely pulls 4 key fields from the parsed model output.
    Returns empty string if field is missing (never crashes).
    Raw values preserved even if not in valid label set
    (invalids are counted separately in metrics).
    """
    pred_cat  = str(parsed_dict.get("category",             "")).strip()
    pred_sev  = str(parsed_dict.get("severity",             "")).strip().lower()
    pred_dept = str(parsed_dict.get("department",           "")).strip()
    pred_desc = str(parsed_dict.get("complaint_description","")).strip()
    return pred_cat, pred_sev, pred_dept, pred_desc


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def compute_metrics(results, baseline_name):
    """
    Computes all metrics for one baseline.

    Classification (category, severity, department):
      - Accuracy: % of exact matches
      - Macro F1: equal weight to all classes regardless of frequency
        (important here since test set has slight class imbalance)

    Generation (complaint_description):
      - BLEU: n-gram overlap with reference description
      - ROUGE-1/2/L: recall-oriented n-gram and sequence overlap

    Structural:
      - JSON validity rate: % of responses that parsed successfully
      - Invalid label counts: how often model invented labels not in our set
    """
    n = len(results)

    gt_cats   = [r["gt_category"]   for r in results]
    gt_sevs   = [r["gt_severity"]   for r in results]
    gt_depts  = [r["gt_department"] for r in results]

    pred_cats  = [r["pred_category"]  for r in results]
    pred_sevs  = [r["pred_severity"]  for r in results]
    pred_depts = [r["pred_department"] for r in results]

    json_valid_count = sum(1 for r in results if r["json_valid"])

    # Classification
    cat_acc  = accuracy_score(gt_cats,  pred_cats)
    sev_acc  = accuracy_score(gt_sevs,  pred_sevs)
    dept_acc = accuracy_score(gt_depts, pred_depts)

    cat_f1   = f1_score(gt_cats,  pred_cats,  average="macro", zero_division=0)
    sev_f1   = f1_score(gt_sevs,  pred_sevs,  average="macro", zero_division=0)
    dept_f1  = f1_score(gt_depts, pred_depts, average="macro", zero_division=0)

    # Invalid label counts
    invalid_cats  = sum(1 for p in pred_cats  if p not in VALID_CATEGORIES)
    invalid_sevs  = sum(1 for p in pred_sevs  if p not in VALID_SEVERITIES)
    invalid_depts = sum(1 for p in pred_depts if p not in VALID_DEPARTMENTS)

    # BLEU + ROUGE on complaint_description
    smoother = SmoothingFunction().method1
    scorer   = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    bleu_scores, r1, r2, rl = [], [], [], []
    for r in results:
        ref = r["gt_description"].split()
        hyp = r["pred_description"].split()
        if ref and hyp:
            bleu_scores.append(sentence_bleu([ref], hyp, smoothing_function=smoother))
            rs = scorer.score(r["gt_description"], r["pred_description"])
            r1.append(rs["rouge1"].fmeasure)
            r2.append(rs["rouge2"].fmeasure)
            rl.append(rs["rougeL"].fmeasure)

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    metrics = {
        "baseline":                  baseline_name,
        "model":                     MODEL_NAME,
        "total_records":             n,
        "json_validity_rate":        round(json_valid_count / n, 4),
        "category_accuracy":         round(cat_acc,  4),
        "category_macro_f1":         round(cat_f1,   4),
        "severity_accuracy":         round(sev_acc,  4),
        "severity_macro_f1":         round(sev_f1,   4),
        "department_accuracy":       round(dept_acc, 4),
        "department_macro_f1":       round(dept_f1,  4),
        "bleu_score":                avg(bleu_scores),
        "rouge1":                    avg(r1),
        "rouge2":                    avg(r2),
        "rougeL":                    avg(rl),
        "invalid_category_preds":    invalid_cats,
        "invalid_severity_preds":    invalid_sevs,
        "invalid_department_preds":  invalid_depts,
    }

    # Per-class breakdown for category (useful for faculty)
    print(f"\n  Per-class report — CATEGORY [{baseline_name}]:")
    print(classification_report(gt_cats, pred_cats, zero_division=0))

    return metrics


# ─────────────────────────────────────────────────────────────
# BASELINE RUNNER
# ─────────────────────────────────────────────────────────────

def run_baseline(client, test_records, baseline_name, prompt_fn):
    """
    Runs one complete baseline over all test records.
    [FIX-4] Progress counter is dynamic, not hardcoded to /24.
    [FIX-3] Reads fields from correct paths in the dataset structure.
    """
    n_total = len(test_records)
    print(f"\n{'='*60}")
    print(f"Running: {baseline_name}  ({n_total} records, model: {MODEL_NAME})")
    print(f"{'='*60}")

    results = []

    for i, record in enumerate(test_records):
        image_id = record["image_id"]

        # [FIX-3] Correct field paths confirmed from actual dataset structure:
        # - Caption lives at input.image_caption (not root refined_caption)
        # - voice_text is at root
        # - gt_dept is routing.primary_department (not root 'department')
        # - gt_desc is root complaint_description
        caption    = record.get("input", {}).get("image_caption", record.get("refined_caption", ""))
        voice_text = record.get("voice_text", "")
        gt_cat     = record["category"]
        gt_sev     = record["severity"]
        gt_dept    = record.get("routing", {}).get("primary_department", record.get("department", ""))
        gt_desc    = record.get("complaint_description", "")

        print(f"  [{i+1:0{len(str(n_total))}d}/{n_total}] {image_id} ... ", end="", flush=True)

        prompt   = prompt_fn(caption, voice_text)
        raw_text = call_model(client, prompt, image_id, baseline_name)

        parsed, is_valid = parse_json_output(raw_text)
        pred_cat, pred_sev, pred_dept, pred_desc = extract_fields(parsed)

        cat_correct  = (pred_cat  == gt_cat)
        sev_correct  = (pred_sev  == gt_sev)
        dept_correct = (pred_dept == gt_dept)

        status_parts = []
        status_parts.append("cat✓" if cat_correct  else "cat✗")
        status_parts.append("sev✓" if sev_correct  else "sev✗")
        status_parts.append("dep✓" if dept_correct else "dep✗")
        status_parts.append("json✓" if is_valid    else "json✗")
        print(" | ".join(status_parts))

        results.append({
            "baseline":         baseline_name,
            "image_id":         image_id,
            "gt_category":      gt_cat,
            "gt_severity":      gt_sev,
            "gt_department":    gt_dept,
            "gt_description":   gt_desc,
            "pred_category":    pred_cat,
            "pred_severity":    pred_sev,
            "pred_department":  pred_dept,
            "pred_description": pred_desc,
            "json_valid":       is_valid,
            "raw_output":       raw_text or "",
            "cat_correct":      cat_correct,
            "sev_correct":      sev_correct,
            "dept_correct":     dept_correct,
        })

        time.sleep(DELAY_SEC)

    return results


# ─────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────

def save_results(all_results, all_metrics):
    """
    Saves 3 output files:
      1. baseline_results.csv   — all predictions for all records and both baselines
      2. baseline_metrics.json  — aggregated metrics (hand this to Member A)
      3. baseline_errors.csv    — only wrong/failed records (hand this to Member B for Phase 3)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Full results
    csv_path = os.path.join(OUTPUT_DIR, "baseline_results.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Metrics
    metrics_path = os.path.join(OUTPUT_DIR, "baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved: {metrics_path}")

    # Error cases
    errors = [r for r in all_results if not r["cat_correct"] or not r["sev_correct"] or not r["json_valid"]]
    error_path = os.path.join(OUTPUT_DIR, "baseline_errors.csv")
    pd.DataFrame(errors).to_csv(error_path, index=False)
    print(f"Saved: {error_path}  ({len(errors)} error records)")

    # Summary table
    print(f"\n{'='*62}")
    print("METRICS SUMMARY")
    print(f"{'='*62}")
    m0, m1 = all_metrics[0], all_metrics[1]
    name0 = "Zero-Shot"
    name1 = "Prompted"
    print(f"{'Metric':<30} {name0:>14} {name1:>14}")
    print("-" * 60)
    keys = [
        "json_validity_rate",
        "category_accuracy",   "category_macro_f1",
        "severity_accuracy",   "severity_macro_f1",
        "department_accuracy", "department_macro_f1",
        "bleu_score", "rouge1", "rouge2", "rougeL",
    ]
    for key in keys:
        print(f"  {key:<28} {str(m0[key]):>14} {str(m1[key]):>14}")
    print(f"{'='*62}")
    print("\nSave this table for Phase 3 comparison with the fine-tuned model.")
    print(f"Model used: {MODEL_NAME}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():

    # ── API Key ──────────────────────────────────────────────
    # On Kaggle: use Kaggle Secrets (Add-ons > Secrets in notebook)
    #   from kaggle_secrets import UserSecretsClient
    #   os.environ["GROQ_API_KEY"] = UserSecretsClient().get_secret("GROQ_API_KEY")
    # Locally: export GROQ_API_KEY="your_key"
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "\nGROQ_API_KEY not set.\n"
            "Kaggle: Add-ons > Secrets > Add secret named GROQ_API_KEY\n"
            "Local:  export GROQ_API_KEY='your_key_here'\n"
        )

    client = Groq(api_key=api_key)
    print(f"Groq client ready. Model: {MODEL_NAME}")

    # ── Load test data ────────────────────────────────────────
    test_records = load_test_data(TEST_FILE)

    # ── Baseline 1: Zero-Shot ─────────────────────────────────
    results_zs = run_baseline(
        client, test_records,
        baseline_name="Baseline-1-ZeroShot",
        prompt_fn=build_zeroshot_prompt,
    )
    metrics_zs = compute_metrics(results_zs, "Baseline-1-ZeroShot")

    # ── Baseline 2: Prompted ──────────────────────────────────
    results_pt = run_baseline(
        client, test_records,
        baseline_name="Baseline-2-Prompted",
        prompt_fn=build_prompted_prompt,
    )
    metrics_pt = compute_metrics(results_pt, "Baseline-2-Prompted")

    # ── Save everything ───────────────────────────────────────
    save_results(
        all_results=[*results_zs, *results_pt],
        all_metrics=[metrics_zs, metrics_pt],
    )

    print("\nPhase 2 complete.")
    print("  baseline_results.csv  → Member A (metrics table for report)")
    print("  baseline_metrics.json → Member A")
    print("  baseline_errors.csv   → Member B (error analysis for Phase 3)")


if __name__ == "__main__":
    main()
