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

    # Force local fine-tuned adapters for both Qwen and Mistral.
    os.environ["QWEN_BACKEND"] = "local_peft"
    os.environ["QWEN_BASE_MODEL_ID"] = "Qwen/Qwen2.5-1.5B-Instruct"
    os.environ["MISTRAL_BACKEND"] = "local_peft"
    os.environ["MISTRAL_BASE_MODEL_ID"] = "mistralai/Mistral-7B-Instruct-v0.2"
    os.environ["MISTRAL_ADAPTER_DIR"] = "/Users/Genai Project/finetuning models/mistral/mistral_v02_adapter_fixed_for_runtime"
    os.environ["LLAMA_BASELINE_ENABLED"] = "false"
    os.environ["MAX_NEW_TOKENS_STRUCTURED"] = "100"
    os.environ["MAX_NEW_TOKENS_DESCRIPTION"] = "100"

    out_dir = root / "full_pipeline_outputs_local"
    out_dir.mkdir(parents=True, exist_ok=True)

    tests = [
        {
            "name": "Local Test 1",
            "complaint": "Hospital bed in ward is broken and unsafe.",
            "hospital_name": "General Hospital",
            "ward": "Ward-3",
            "image_caption": "Broken hospital bed with damaged side rail in patient room",
            "voice_text": "My bed is broken and unsafe please replace it",
        },
        {
            "name": "Local Test 2",
            "complaint": "There is a water puddle near ICU entrance and patients may slip.",
            "hospital_name": "General Hospital",
            "ward": "ICU Entrance",
            "image_caption": "Water puddle on hospital floor near ICU corridor",
            "voice_text": "Please clean the water spill before someone falls",
        },
        {
            "name": "Local Test 3",
            "complaint": "Rat seen near food preparation area in cafeteria.",
            "hospital_name": "General Hospital",
            "ward": "Cafeteria",
            "image_caption": "Rat near kitchen food trays in hospital canteen",
            "voice_text": "I saw a rat near the food area this is unsafe",
        },
    ]

    started = datetime.utcnow().isoformat()
    master_json = str(root / "master_with_splits.json")

    results = []
    errors = []

    try:
        orch = hm.build_orchestrator(master_json)
    except Exception as e:
        fatal = {
            "stage": "build_orchestrator",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "started_utc": started,
        }
        (out_dir / "fatal_error.json").write_text(json.dumps(fatal, indent=2), encoding="utf-8")
        raise

    for i, case in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] Running {case['name']} ...", flush=True)
        payload = hm.ComplaintInput(
            name=case["name"],
            complaint=case["complaint"],
            hospital_name=case["hospital_name"],
            ward=case["ward"],
            image_caption=case["image_caption"],
            voice_text=case["voice_text"],
            metadata={"test_id": i},
        )
        try:
            out = orch.process(payload)
            out["test_id"] = i
            out["test_name"] = case["name"]
            results.append(out)
            print(
                f"  -> ok | category={out.get('category')} | severity={out.get('severity')} | dept={out.get('department')}",
                flush=True,
            )
        except Exception as e:
            err = {
                "test_id": i,
                "test_name": case["name"],
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            errors.append(err)
            print(f"  -> error | {e}", flush=True)

    summary = {
        "started_utc": started,
        "ended_utc": datetime.utcnow().isoformat(),
        "total_tests": len(tests),
        "success_count": len(results),
        "error_count": len(errors),
        "config": {
            "qwen_backend": os.getenv("QWEN_BACKEND"),
            "mistral_backend": os.getenv("MISTRAL_BACKEND"),
            "llama_enabled": os.getenv("LLAMA_BASELINE_ENABLED"),
            "qwen_adapter_dir": os.getenv("QWEN_ADAPTER_DIR"),
            "mistral_adapter_dir": os.getenv("MISTRAL_ADAPTER_DIR"),
            "sqlite_path": os.getenv("SQLITE_PATH", "hospital_triage.db"),
            "chroma_path": os.getenv("CHROMA_PATH"),
        },
    }

    (out_dir / "inputs.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")
    (out_dir / "outputs.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDone.")
    print(json.dumps(summary, indent=2))
    print(f"Saved outputs at: {out_dir}")


if __name__ == "__main__":
    main()
