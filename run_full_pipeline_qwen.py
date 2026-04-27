import csv
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import hospital_multillm_rag as hm


def main() -> None:
    root = Path("/Users/Genai Project")
    load_dotenv(root / ".env")

    out_dir = root / "full_pipeline_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_path = root / "test.jsonl"
    master_path = root / "master_with_splits.json"

    orch = hm.build_orchestrator(str(master_path))

    rows = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    results = []
    errors = []

    for i, rec in enumerate(rows, 1):
        image_id = rec.get("image_id", f"row_{i}")
        caption = (rec.get("input", {}) or {}).get("image_caption", rec.get("refined_caption", ""))
        voice = rec.get("voice_text", "")
        complaint = voice.strip() or caption.strip() or "No complaint text provided."

        payload = hm.ComplaintInput(
            name=rec.get("name", f"AutoUser_{i}"),
            complaint=complaint,
            hospital_name=rec.get("hospital_name", "General Hospital"),
            ward=rec.get("ward", "General Ward"),
            image_caption=caption,
            voice_text=voice,
            metadata={"image_id": image_id},
        )

        print(f"[{i:02d}/{len(rows)}] {image_id} ... ", end="", flush=True)
        try:
            out = orch.process(payload)
            out["image_id"] = image_id
            out["gt_category"] = rec.get("category", "")
            out["gt_severity"] = str(rec.get("severity", "")).lower().strip()
            out["gt_department"] = ((rec.get("routing", {}) or {}).get("primary_department") or rec.get("department", "")).strip()
            out["cat_correct"] = out.get("category", "") == out["gt_category"]
            out["sev_correct"] = str(out.get("severity", "")).lower().strip() == out["gt_severity"]
            out["dept_correct"] = out.get("department", "") == out["gt_department"]
            results.append(out)
            print(
                f"cat:{'✓' if out['cat_correct'] else '✗'} "
                f"sev:{'✓' if out['sev_correct'] else '✗'} "
                f"dep:{'✓' if out['dept_correct'] else '✗'}"
            )
        except Exception as e:
            errors.append(
                {
                    "image_id": image_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            print("ERROR")

    results_json = out_dir / "pipeline_results.json"
    errors_json = out_dir / "pipeline_errors.json"
    summary_json = out_dir / "pipeline_summary.json"
    flat_csv = out_dir / "pipeline_results_flat.csv"

    results_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    errors_json.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "image_id",
        "category",
        "severity",
        "department",
        "complaint_description",
        "needs_human_review",
        "structured_json_valid",
        "second_validation_score",
        "cat_correct",
        "sev_correct",
        "dept_correct",
        "gt_category",
        "gt_severity",
        "gt_department",
    ]
    with open(flat_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    n = len(rows)
    ok = len(results)
    cat_acc = (sum(1 for r in results if r.get("cat_correct")) / ok) if ok else 0.0
    sev_acc = (sum(1 for r in results if r.get("sev_correct")) / ok) if ok else 0.0
    dep_acc = (sum(1 for r in results if r.get("dept_correct")) / ok) if ok else 0.0
    json_rate = (sum(1 for r in results if r.get("structured_json_valid")) / ok) if ok else 0.0
    human_review_rate = (sum(1 for r in results if r.get("needs_human_review")) / ok) if ok else 0.0

    summary = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "total_test_records": n,
        "processed_successfully": ok,
        "failed_records": len(errors),
        "config": {
            "qwen_backend": os.getenv("QWEN_BACKEND"),
            "qwen_model_id": os.getenv("QWEN_MODEL_ID"),
            "mistral_backend": os.getenv("MISTRAL_BACKEND"),
            "mistral_model_id": os.getenv("MISTRAL_HF_MODEL"),
            "llama_enabled": os.getenv("LLAMA_BASELINE_ENABLED"),
        },
        "metrics_on_successful_records": {
            "category_accuracy": round(cat_acc, 4),
            "severity_accuracy": round(sev_acc, 4),
            "department_accuracy": round(dep_acc, 4),
            "structured_json_validity_rate": round(json_rate, 4),
            "needs_human_review_rate": round(human_review_rate, 4),
        },
        "output_files": {
            "results_json": str(results_json),
            "errors_json": str(errors_json),
            "results_flat_csv": str(flat_csv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDONE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
