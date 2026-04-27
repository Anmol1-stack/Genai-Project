import argparse
import json
from pathlib import Path
from typing import Dict, List


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_case_text(r: Dict) -> str:
    image_id = r.get("image_id", "")
    caption = (r.get("input", {}) or {}).get("image_caption") or r.get("refined_caption", "")
    voice = r.get("voice_text", "")
    category = r.get("category", "")
    severity = str(r.get("severity", "")).lower()
    department = (r.get("routing", {}) or {}).get("primary_department") or r.get("department", "")
    desc = r.get("complaint_description", "")

    return (
        f"Case ID: {image_id}\n"
        f"Observed caption: {caption}\n"
        f"Voice complaint: {voice}\n"
        f"Resolved category: {category}\n"
        f"Resolved severity: {severity}\n"
        f"Resolved department: {department}\n"
        f"Clinical complaint summary: {desc}\n"
        "---\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fallback RAG docs from train+val JSONL")
    parser.add_argument("--train", required=True, help="Path to train.jsonl")
    parser.add_argument("--val", required=True, help="Path to val.jsonl")
    parser.add_argument("--out-dir", default="data/raw", help="Output doc directory")
    args = parser.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    rows = train_rows + val_rows
    if not rows:
        raise RuntimeError("No records loaded from train/val.")

    casebook = out_dir / "casebook_from_train_val.txt"
    with casebook.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(to_case_text(r))

    categories = sorted({str(r.get("category", "")).strip() for r in rows if r.get("category")})
    severities = sorted({str(r.get("severity", "")).lower().strip() for r in rows if r.get("severity")})
    departments = sorted(
        {
            (
                (r.get("routing", {}) or {}).get("primary_department")
                or r.get("department", "")
            ).strip()
            for r in rows
            if ((r.get("routing", {}) or {}).get("primary_department") or r.get("department"))
        }
    )

    taxonomy = out_dir / "triage_taxonomy_from_train_val.txt"
    with taxonomy.open("w", encoding="utf-8") as f:
        f.write("Triage category labels:\n")
        for c in categories:
            f.write(f"- {c}\n")
        f.write("\nSeverity labels:\n")
        for s in severities:
            f.write(f"- {s}\n")
        f.write("\nDepartment labels:\n")
        for d in departments:
            f.write(f"- {d}\n")

    print(f"Saved: {casebook}")
    print(f"Saved: {taxonomy}")
    print(f"Total records used: {len(rows)} (train+val)")


if __name__ == "__main__":
    main()

