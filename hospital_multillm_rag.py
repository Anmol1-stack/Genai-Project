from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


class LightweightHashEmbedder:
    """Offline fallback when SentenceTransformer model download/load fails."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9_]+", (text or "").lower())
        if not tokens:
            return vec
        for tok in tokens:
            idx = abs(hash(tok)) % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec) + 1e-12
        return vec / norm

    def encode(self, texts: Any, normalize_embeddings: bool = True) -> np.ndarray:
        single = isinstance(texts, str)
        items = [texts] if single else list(texts or [])

        if not items:
            out = np.zeros((0, self.dim), dtype=np.float32)
            return out[0] if single else out

        mat = np.stack([self._embed_one(t) for t in items]).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
            mat = mat / norms
        return mat[0] if single else mat


def _build_embedder(model_name: str) -> Any:
    if _env_bool("EMBEDDING_FALLBACK_ONLY", False):
        print("INFO: EMBEDDING_FALLBACK_ONLY=true -> using LightweightHashEmbedder.")
        return LightweightHashEmbedder()
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        print(
            f"WARNING: Could not load embedding model '{model_name}'. "
            f"Falling back to LightweightHashEmbedder. reason={exc}"
        )
        return LightweightHashEmbedder()


def _resolve_default_chroma_path() -> str:
    env_path = os.getenv("CHROMA_PATH", "").strip()
    candidates = [
        env_path,
        "/Users/Genai Project/vectordb/chromadb",
        "/content/Genai Project/vectordb/chromadb",
        "vectordb/chromadb",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return p
    # Fallback when path does not exist yet.
    return next((p for p in candidates if p), "vectordb/chromadb")


SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
RANK_TO_SEVERITY = {v: k for k, v in SEVERITY_RANK.items()}

DEFAULT_CATEGORIES = [
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

DEFAULT_DEPARTMENTS = [
    "Administration",
    "Dietary",
    "Facilities Management",
    "Housekeeping",
    "Maintenance",
    "Nursing",
    "Pest Control",
    "Waste Management",
]

DEFAULT_SEVERITIES = ["low", "medium", "high", "critical"]

# Auto-detected project fine-tuned artifacts (from your provided folder).
DEFAULT_QWEN_ADAPTER_DIR = "/Users/Genai Project/finetuning models/qwen/content/qwen2_5_1_5b_instruct_adapter"
DEFAULT_MISTRAL_ADAPTER_DIR = "/Users/Genai Project/finetuning models/mistral/mistral_v02_adapter"
DEFAULT_MISTRAL_ADAPTER_WEIGHT = "/Users/Genai Project/finetuning models/mistral/adapter_model.safetensors"


# Category-driven minimum severity floor.
CATEGORY_MIN_SEVERITY = {
    "Broken Hospital Bed": "high",
    "Crowded Hospital Waiting Room": "medium",
    "Dirty Hospital Bathroom": "high",
    "Empty / Unstaffed Nursing Station": "critical",
    "Overflowing Hospital Trash (Outside)": "high",
    "Rats / Rodent Infestation": "critical",
    "Torn Hospital Privacy Curtain": "low",
    "Unappetizing Hospital Food Tray": "low",
    "Unhygienic / Contaminated Hospital Food": "critical",
    "Water Puddle on Hospital Floor": "medium",
}


CATEGORY_ALIASES = {
    "unappetizing hospital food": "Unappetizing Hospital Food Tray",
    "unappetizing food tray": "Unappetizing Hospital Food Tray",
    "rodent infestation": "Rats / Rodent Infestation",
    "rats infestation": "Rats / Rodent Infestation",
    "empty nursing station": "Empty / Unstaffed Nursing Station",
    "unstaffed nursing station": "Empty / Unstaffed Nursing Station",
    "water puddle": "Water Puddle on Hospital Floor",
}


class ComplaintInput(BaseModel):
    name: str = Field(..., description="Patient or complainant name.")
    complaint: str = Field(..., description="Primary complaint message from user.")
    hospital_name: str = Field(..., description="Hospital name.")
    ward: str = Field(..., description="Ward/unit name.")
    image_caption: str = Field(..., description="Caption generated from image.")
    voice_text: str = Field(..., description="Transcribed voice text.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TriageStructuredOutput(BaseModel):
    category: str
    severity: str
    department: str
    complaint_description: str


@dataclass
class PipelineConfig:
    # Retrieval
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_path: str = _resolve_default_chroma_path()
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", os.getenv("CHROMA_SOP_COLLECTION", "hospital_sops"))
    chroma_trainval_collection: str = os.getenv("CHROMA_TRAINVAL_COLLECTION", "hospital_trainval")
    chroma_predictions_collection: str = os.getenv("CHROMA_PREDICTIONS_COLLECTION", "hospital_predictions")
    top_k: int = int(os.getenv("TOP_K", "4"))
    use_trainval_fallback: bool = _env_bool("USE_TRAINVAL_FALLBACK", True)
    trainval_top_k: int = int(os.getenv("TRAINVAL_TOP_K", "2"))
    store_predictions_in_chroma: bool = _env_bool("STORE_PREDICTIONS_IN_CHROMA", True)

    # Qwen structured extraction — fine-tuned model running locally via MLX (Apple Silicon).
    qwen_backend: str = os.getenv("QWEN_BACKEND", "mlx")  # mlx | groq | hf | local_peft
    qwen_mlx_model: str = os.getenv("QWEN_MLX_MODEL", "Vani0235/hospital-qwen-finetuned")
    qwen_model_id: str = os.getenv("QWEN_MODEL_ID", "llama-3.1-8b-instant")
    qwen_base_model_id: str = os.getenv("QWEN_BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    qwen_adapter_dir: str = os.getenv("QWEN_ADAPTER_DIR", DEFAULT_QWEN_ADAPTER_DIR)

    # Mistral description generation — fine-tuned model running locally via MLX (Apple Silicon).
    mistral_backend: str = os.getenv("MISTRAL_BACKEND", "mlx")  # mlx | groq | hf | local_peft
    mistral_mlx_model: str = os.getenv("MISTRAL_MLX_MODEL", "./models/mistral-mlx-4bit")
    mistral_groq_model: str = os.getenv("MISTRAL_GROQ_MODEL", "llama-3.1-8b-instant")
    mistral_hf_model: str = os.getenv("MISTRAL_HF_MODEL", "Vani0235/hospital-mistral-finetuned")
    mistral_base_model_id: str = os.getenv("MISTRAL_BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
    mistral_adapter_dir: str = os.getenv("MISTRAL_ADAPTER_DIR", DEFAULT_MISTRAL_ADAPTER_DIR)
    hf_token_env: str = os.getenv("HF_TOKEN_ENV", "HF_TOKEN")
    hf_provider: Optional[str] = os.getenv("HF_PROVIDER", "").strip() or None

    # Llama baseline prompt model — disabled; kept for reference only
    llama_enabled: bool = _env_bool("LLAMA_BASELINE_ENABLED", False)
    llama_model: str = os.getenv("LLAMA_MODEL", "llama-3.1-8b-instant")
    groq_api_key_env: str = os.getenv("GROQ_API_KEY_ENV", "GROQ_API_KEY")

    # Generation
    structured_temperature: float = float(os.getenv("STRUCTURED_TEMP", "0.0"))
    description_temperature: float = float(os.getenv("DESCRIPTION_TEMP", "0.2"))
    max_new_tokens_structured: int = int(os.getenv("MAX_NEW_TOKENS_STRUCTURED", "220"))
    max_new_tokens_description: int = int(os.getenv("MAX_NEW_TOKENS_DESCRIPTION", "200"))

    # Validation and safety
    enable_second_validation: bool = _env_bool("ENABLE_SECOND_VALIDATION", True)
    second_validation_threshold: float = float(os.getenv("SECOND_VALIDATION_THRESHOLD", "0.35"))

    # Storage
    sqlite_path: str = os.getenv("SQLITE_PATH", "hospital_triage.db")


def load_taxonomy_from_master(master_json_path: Optional[str]) -> Tuple[List[str], List[str], List[str]]:
    if not master_json_path:
        return DEFAULT_CATEGORIES, DEFAULT_SEVERITIES, DEFAULT_DEPARTMENTS

    path = Path(master_json_path)
    if not path.exists():
        return DEFAULT_CATEGORIES, DEFAULT_SEVERITIES, DEFAULT_DEPARTMENTS

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_CATEGORIES, DEFAULT_SEVERITIES, DEFAULT_DEPARTMENTS

    if not isinstance(payload, list):
        return DEFAULT_CATEGORIES, DEFAULT_SEVERITIES, DEFAULT_DEPARTMENTS

    cats = sorted({str(x.get("category", "")).strip() for x in payload if x.get("category")})
    sevs = sorted({str(x.get("severity", "")).lower().strip() for x in payload if x.get("severity")})
    depts = sorted(
        {
            (
                (x.get("routing", {}) or {}).get("primary_department")
                or x.get("department", "")
            ).strip()
            for x in payload
            if ((x.get("routing", {}) or {}).get("primary_department") or x.get("department"))
        }
    )
    return (
        cats or DEFAULT_CATEGORIES,
        sevs or DEFAULT_SEVERITIES,
        depts or DEFAULT_DEPARTMENTS,
    )


def canonicalize_category(pred: str, valid_categories: List[str]) -> str:
    raw = _normalize_space(pred)
    if raw in valid_categories:
        return raw

    low = raw.lower()
    if low in CATEGORY_ALIASES:
        alias = CATEGORY_ALIASES[low]
        if alias in valid_categories:
            return alias

    # Case-insensitive match fallback.
    for c in valid_categories:
        if c.lower() == low:
            return c
    return raw


def canonicalize_department(pred: str, valid_departments: List[str]) -> str:
    raw = _normalize_space(pred)
    if raw in valid_departments:
        return raw
    low = raw.lower()
    for d in valid_departments:
        if d.lower() == low:
            return d
    return raw


def canonicalize_severity(pred: str) -> str:
    raw = _normalize_space(pred).lower()
    if raw in SEVERITY_RANK:
        return raw
    if raw in {"urgent", "severe"}:
        return "high"
    if raw in {"very high", "immediate"}:
        return "critical"
    return raw


def robust_parse_json(raw_text: str) -> Tuple[Dict[str, Any], bool]:
    if not raw_text:
        return {}, False
    cleaned = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate), True
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(cleaned), True
    except json.JSONDecodeError:
        return {}, False


class GroqChatClient:
    def __init__(self, api_key: str):
        from groq import Groq

        self.client = Groq(api_key=api_key)

    def complete(self, model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


class HFInferenceClient:
    """Calls HuggingFace Inference API via direct HTTP requests.

    The huggingface_hub SDK's InferenceClient requires provider routing that
    is only available for pre-registered models. Custom fine-tuned models
    (e.g. Vani0235/hospital-*) are served directly through the standard
    inference endpoint — direct HTTP bypasses the SDK routing layer.
    """

    _HF_API = "https://api-inference.huggingface.co/models"

    def __init__(self, token: str, provider: Optional[str] = None):  # noqa: ARG002
        import requests as _requests
        self._session = _requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {token}"})
        self._token = token

    @staticmethod
    def _format_chat_prompt(model_id: str, system_prompt: str, user_prompt: str) -> str:
        """Build a single text prompt using the model family's chat template."""
        m = model_id.lower()
        if "qwen" in m:
            return (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        if "mistral" in m or "mixtral" in m:
            return f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

    def _generate(self, model_id: str, prompt: str, max_new_tokens: int, temperature: float) -> str:
        url = f"{self._HF_API}/{model_id}"
        params: Dict[str, Any] = {"max_new_tokens": max_new_tokens, "return_full_text": False}
        if temperature > 0:
            params["temperature"] = temperature
            params["do_sample"] = True
        else:
            params["do_sample"] = False
        resp = self._session.post(url, json={"inputs": prompt, "parameters": params}, timeout=120)
        if resp.status_code == 503:
            # Model is loading on HF servers — wait and retry once.
            import time; time.sleep(20)
            resp = self._session.post(url, json={"inputs": prompt, "parameters": params}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            return (data[0].get("generated_text") or "").strip()
        return ""

    def chat(self, model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        model_id = (model or "").strip().split(":")[0]
        prompt = self._format_chat_prompt(model_id, system_prompt, user_prompt)
        return self._generate(model_id, prompt, max_tokens, temperature)

    def text(self, model: str, prompt: str, temperature: float, max_new_tokens: int) -> str:
        model_id = (model or "").strip().split(":")[0]
        return self._generate(model_id, prompt, max_new_tokens, temperature)


class LocalPeftRunner:
    def __init__(self, base_model_id: str, adapter_dir: str, runner_name: str):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not adapter_dir:
            raise ValueError(f"{runner_name} adapter_dir is required for local_peft backend.")
        fixed_adapter_dir = self._resolve_or_repair_adapter_dir(base_model_id, adapter_dir, runner_name)

        self.torch = torch
        # Prefer adapter tokenizer if valid; fallback to base tokenizer for repaired/corrupted adapter layouts.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(fixed_adapter_dir, trust_remote_code=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        if torch.cuda.is_available():
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config=quant_cfg,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)

        self.model = PeftModel.from_pretrained(base, fixed_adapter_dir)
        self.model.eval()

    @staticmethod
    def _default_lora_config(base_model_id: str) -> Dict[str, Any]:
        # Matches your fine-tuning notebook defaults.
        return {
            "alpha_pattern": {},
            "auto_mapping": None,
            "base_model_name_or_path": base_model_id,
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layer_replication": None,
            "layers_pattern": None,
            "layers_to_transform": None,
            "loftq_config": {},
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "megatron_config": None,
            "megatron_core": "megatron.core",
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": 16,
            "rank_pattern": {},
            "revision": None,
            "target_modules": [
                "q_proj",
                "gate_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
            ],
            "task_type": "CAUSAL_LM",
            "use_dora": False,
            "use_rslora": False,
        }

    @staticmethod
    def _is_valid_json(path: Path) -> bool:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return isinstance(payload, dict)
        except Exception:
            return False

    @classmethod
    def _resolve_or_repair_adapter_dir(cls, base_model_id: str, adapter_dir: str, runner_name: str) -> str:
        adir = Path(adapter_dir)
        if not adir.exists():
            raise FileNotFoundError(f"{runner_name} adapter dir not found: {adapter_dir}")

        cfg_path = adir / "adapter_config.json"
        weight_paths = [
            adir / "adapter_model.safetensors",
            adir / "adapter_model.bin",
            adir.parent / "adapter_model.safetensors",
            adir.parent / "adapter_model.bin",
            Path(DEFAULT_MISTRAL_ADAPTER_WEIGHT),
        ]
        weight_file = next((p for p in weight_paths if p.exists()), None)

        has_valid_cfg = cfg_path.exists() and cls._is_valid_json(cfg_path)
        if has_valid_cfg and weight_file and (adir / weight_file.name).exists():
            return str(adir)

        # Repair into a deterministic sibling directory.
        repaired = adir.parent / f"{adir.name}_fixed_for_runtime"
        repaired.mkdir(parents=True, exist_ok=True)

        # Copy tokenizer artifacts if present.
        for fname in [
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "README.md",
        ]:
            src = adir / fname
            if src.exists():
                shutil.copy2(src, repaired / fname)

        # Write/repair adapter_config.json.
        if has_valid_cfg:
            shutil.copy2(cfg_path, repaired / "adapter_config.json")
        else:
            (repaired / "adapter_config.json").write_text(
                json.dumps(cls._default_lora_config(base_model_id), indent=2),
                encoding="utf-8",
            )

        if not weight_file:
            raise FileNotFoundError(
                f"{runner_name} adapter weights not found in {adir} or parent directory."
            )
        shutil.copy2(weight_file, repaired / "adapter_model.safetensors")
        return str(repaired)

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        do_sample = temperature > 0
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class MLXRunner:
    """Runs fine-tuned models locally on Apple Silicon using mlx-lm."""

    def __init__(self, model_path: str, runner_name: str):
        from mlx_lm import load, generate as _gen
        print(f"[MLX] Loading {runner_name} from {model_path} ...")
        self._model, self._tokenizer = load(model_path)
        self._gen = _gen
        self._model_path = model_path
        print(f"[MLX] {runner_name} ready.")

    @staticmethod
    def _format_prompt(model_path: str, system_prompt: str, user_prompt: str) -> str:
        m = model_path.lower()
        if "qwen" in m:
            return (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        return f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int, temperature: float) -> str:
        from mlx_lm.sample_utils import make_sampler
        prompt = self._format_prompt(self._model_path, system_prompt, user_prompt)
        result = self._gen(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=make_sampler(temp=temperature),
            verbose=False,
        )
        return (result or "").strip()


class HospitalMultiLLMOrchestrator:
    def __init__(self, cfg: PipelineConfig, categories: List[str], severities: List[str], departments: List[str]):
        self.cfg = cfg
        self.valid_categories = categories
        self.valid_severities = severities
        self.valid_departments = departments

        self.embedder = _build_embedder(cfg.embedding_model)

        self.chroma_client = chromadb.PersistentClient(path=cfg.chroma_path)
        try:
            self.collection = self.chroma_client.get_collection(cfg.chroma_collection)
        except Exception as exc:
            raise RuntimeError(
                f"Chroma collection '{cfg.chroma_collection}' not found at {cfg.chroma_path}. "
                "Create it first with prepare_collection_* helpers."
            ) from exc

        self.trainval_collection = None
        if (
            cfg.use_trainval_fallback
            and cfg.chroma_trainval_collection
            and cfg.chroma_trainval_collection != cfg.chroma_collection
        ):
            try:
                self.trainval_collection = self.chroma_client.get_collection(cfg.chroma_trainval_collection)
            except Exception:
                self.trainval_collection = None

        self.predictions_collection = None
        if cfg.store_predictions_in_chroma:
            self.predictions_collection = self.chroma_client.get_or_create_collection(
                name=cfg.chroma_predictions_collection,
                metadata={"hnsw:space": "cosine"},
            )

        self._groq = None
        groq_key = os.getenv(cfg.groq_api_key_env, "")
        if groq_key:
            self._groq = GroqChatClient(groq_key)

        hf_token = os.getenv(cfg.hf_token_env, "")
        self._hf = HFInferenceClient(hf_token, cfg.hf_provider) if hf_token else None

        self._local_qwen = None
        if cfg.qwen_backend == "local_peft":
            self._local_qwen = LocalPeftRunner(cfg.qwen_base_model_id, cfg.qwen_adapter_dir, "qwen")

        self._local_mistral = None
        if cfg.mistral_backend == "local_peft":
            self._local_mistral = LocalPeftRunner(cfg.mistral_base_model_id, cfg.mistral_adapter_dir, "mistral")

        self._mlx_qwen = None
        if cfg.qwen_backend == "mlx":
            self._mlx_qwen = MLXRunner(cfg.qwen_mlx_model, "Qwen")

        self._mlx_mistral = None
        if cfg.mistral_backend == "mlx":
            self._mlx_mistral = MLXRunner(cfg.mistral_mlx_model, "Mistral")

        self._init_storage()

    def _init_storage(self) -> None:
        db = Path(self.cfg.sqlite_path)
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS complaints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    name TEXT NOT NULL,
                    complaint TEXT NOT NULL,
                    hospital_name TEXT NOT NULL,
                    ward TEXT NOT NULL,
                    image_caption TEXT NOT NULL,
                    voice_text TEXT NOT NULL,
                    similarity_score REAL NOT NULL,
                    fusion_mode TEXT NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    department TEXT NOT NULL,
                    complaint_description TEXT NOT NULL,
                    second_validation_score REAL,
                    needs_human_review INTEGER NOT NULL,
                    validation_errors_json TEXT NOT NULL,
                    retrieved_context_json TEXT NOT NULL,
                    raw_structured_output TEXT NOT NULL,
                    raw_description_output TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _compute_similarity_and_fusion(self, image_caption: str, voice_text: str, complaint_text: str) -> Dict[str, Any]:
        img = _normalize_space(image_caption)
        voice = _normalize_space(voice_text or complaint_text)
        complaint = _normalize_space(complaint_text)

        emb_img = self.embedder.encode(img, normalize_embeddings=True).astype(np.float32)
        emb_voice = self.embedder.encode(voice, normalize_embeddings=True).astype(np.float32)
        score = _cosine(emb_img, emb_voice)

        if score > 0.7:
            mode = "high_similarity"
            image_weight = 0.5
            voice_weight = 0.5
            policy = "use_both_equally"
        elif score >= 0.4:
            mode = "medium_similarity"
            image_weight = 0.45
            voice_weight = 0.55
            policy = "consider_both"
        else:
            mode = "low_similarity"
            image_weight = 0.2
            voice_weight = 0.8
            policy = "prioritize_voice"

        fused = (
            f"Fusion policy: {policy}\n"
            f"Similarity score: {score:.4f}\n"
            f"Image weight: {image_weight:.2f}, Voice weight: {voice_weight:.2f}\n"
            f"Image caption: {img}\n"
            f"Voice text: {voice}\n"
            f"Direct complaint: {complaint}"
        )
        return {
            "similarity_score": float(score),
            "fusion_mode": mode,
            "fusion_policy": policy,
            "fused_text": fused,
            "image_text": img,
            "voice_text": voice,
        }

    def _query_collection(
        self,
        collection: Any,
        collection_name: str,
        query_embedding: List[float],
        n_results: int,
    ) -> List[Dict[str, Any]]:
        if not collection or n_results <= 0:
            return []
        out = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        rows: List[Dict[str, Any]] = []
        docs = (out.get("documents") or [[]])[0]
        metas = (out.get("metadatas") or [[]])[0]
        dists = (out.get("distances") or [[]])[0]
        ids = (out.get("ids") or [[]])[0]
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            rows.append(
                {
                    "rank": 0,
                    "collection": collection_name,
                    "id": ids[i] if i < len(ids) else f"{collection_name}_{i}",
                    "source": (meta or {}).get("source", "unknown"),
                    "score": float(round(1 - float(dist), 6)),
                    "text": doc,
                }
            )
        return rows

    def _retrieve_context(self, query_text: str) -> List[Dict[str, Any]]:
        q_emb = self.embedder.encode(query_text, normalize_embeddings=True).astype(np.float32).tolist()
        rows: List[Dict[str, Any]] = []
        rows.extend(
            self._query_collection(
                collection=self.collection,
                collection_name=self.cfg.chroma_collection,
                query_embedding=q_emb,
                n_results=self.cfg.top_k,
            )
        )
        if self.trainval_collection is not None:
            rows.extend(
                self._query_collection(
                    collection=self.trainval_collection,
                    collection_name=self.cfg.chroma_trainval_collection,
                    query_embedding=q_emb,
                    n_results=self.cfg.trainval_top_k,
                )
            )

        # Deduplicate by (collection, id), then rank by score.
        dedup: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r in rows:
            key = (str(r.get("collection", "")), str(r.get("id", "")))
            if key not in dedup or float(r.get("score", 0.0)) > float(dedup[key].get("score", 0.0)):
                dedup[key] = r
        merged = sorted(dedup.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)
        for i, r in enumerate(merged, start=1):
            r["rank"] = i
        return merged

    def _context_block(self, contexts: List[Dict[str, Any]], max_docs: Optional[int] = None) -> str:
        """
        Build a readable context block for LLM prompts.

        max_docs:
          - None  → include all retrieved docs   (used by Mistral for rich description)
          - 1     → include only top-ranked doc   (used by Qwen for classification grounding)
        """
        if not contexts:
            return "No retrieval context found."
        docs = contexts[:max_docs] if max_docs is not None else contexts
        parts = []
        for c in docs:
            parts.append(
                f"[Context {c['rank']} | score={c['score']:.4f} | "
                f"collection={c.get('collection', 'unknown')} | source={c['source']}]\n{c['text']}"
            )
        return "\n\n".join(parts)

    def _qwen_structured(self, complaint: ComplaintInput, fusion: Dict[str, Any], contexts: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str, bool]:
        """
        Structured extraction LLM (simulates fine-tuned Qwen2.5-1.5B-Instruct via Groq/HF API).

        INPUT ROUTING POLICY — Qwen only receives complaint-signal fields:
          ✅ Sent:     ward (for department routing), image_caption, voice_text, complaint text
          ✅ Sent:     top-1 RAG doc only (grounding, not full context)
          ✅ Sent:     valid taxonomy lists (categories, severities, departments)
          ❌ Excluded: patient name  — PII; irrelevant for classification
          ❌ Excluded: hospital_name — irrelevant for category/severity/department
          ❌ Excluded: full RAG context — not needed; Mistral handles description enrichment

        OUTPUT: {"category": ..., "severity": ..., "department": ...}
        """
        categories = "\n".join(f"- {c}" for c in self.valid_categories)
        departments = "\n".join(f"- {d}" for d in self.valid_departments)
        severities = "\n".join(f"- {s}" for s in self.valid_severities)

        # Qwen receives only the single best-matching SOP doc for lightweight grounding.
        # Full RAG context is routed exclusively to Mistral (description generator).
        top1_context = self._context_block(contexts, max_docs=1)

        # Enriched system prompt simulating fine-tuned Qwen2.5-1.5B-Instruct adapter
        # behaviour via structured few-shot instructions — fully API-based, no local weights.
        system_prompt = (
            "You are a hospital complaint triage AI fine-tuned on clinical hospital-complaint data.\n"
            "Your ONLY job is to classify a patient complaint into exactly one category, severity, and department.\n"
            "Rules:\n"
            "  1. category MUST be chosen verbatim from the provided category list.\n"
            "  2. severity MUST be exactly one of: low, medium, high, critical.\n"
            "     Apply these mandatory minimums based on category:\n"
            "     - 'Rats / Rodent Infestation' or 'Unhygienic / Contaminated Hospital Food' -> critical\n"
            "     - 'Empty / Unstaffed Nursing Station' -> critical\n"
            "     - 'Broken Hospital Bed' or 'Dirty Hospital Bathroom' or 'Overflowing Hospital Trash (Outside)' -> at least high\n"
            "     - 'Crowded Hospital Waiting Room' or 'Water Puddle on Hospital Floor' -> at least medium\n"
            "  3. department MUST be chosen verbatim from the provided department list.\n"
            "     Use ward location to resolve ambiguous department routing.\n"
            "  4. Return ONLY a valid JSON object. No markdown, no explanation, no extra keys.\n"
            "Few-shot examples:\n"
            '{"category":"Water Puddle on Hospital Floor","severity":"medium","department":"Housekeeping"}\n'
            '{"category":"Rats / Rodent Infestation","severity":"critical","department":"Pest Control"}\n'
            '{"category":"Empty / Unstaffed Nursing Station","severity":"critical","department":"Nursing"}'
        )

        user_prompt = (
            # Only ward is passed — it aids department routing without leaking PII or
            # hospital-level metadata that is irrelevant to classification.
            f"Ward/Unit: {complaint.ward}\n\n"
            # Core complaint signal: fused image+voice+complaint text
            f"{fusion['fused_text']}\n\n"
            # Top-1 SOP doc only — for category grounding
            "Relevant SOP reference (top match):\n"
            f"{top1_context}\n\n"
            "Valid categories (choose exactly one verbatim):\n"
            f"{categories}\n\n"
            "Valid severities (choose exactly one verbatim):\n"
            f"{severities}\n\n"
            "Valid departments (choose exactly one verbatim):\n"
            f"{departments}\n\n"
            "Return ONLY JSON with keys category, severity, department:\n"
            '{"category":"...","severity":"...","department":"..."}'
        )

        if self.cfg.qwen_backend == "mlx":
            if not self._mlx_qwen:
                raise RuntimeError("QWEN_BACKEND=mlx but MLX Qwen runner not initialized.")
            raw = self._mlx_qwen.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=self.cfg.max_new_tokens_structured,
                temperature=self.cfg.structured_temperature,
            )
        elif self.cfg.qwen_backend == "local_peft":
            raw = self._local_qwen.generate(
                prompt=f"{system_prompt}\n\n{user_prompt}",
                max_new_tokens=self.cfg.max_new_tokens_structured,
                temperature=self.cfg.structured_temperature,
            )
        elif self.cfg.qwen_backend == "groq":
            if not self._groq:
                raise RuntimeError("QWEN_BACKEND=groq but Groq client is not configured.")
            raw = self._groq.complete(
                model=self.cfg.qwen_model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.cfg.structured_temperature,
                max_tokens=self.cfg.max_new_tokens_structured,
            )
        elif self.cfg.qwen_backend == "hf":
            if not self._hf:
                raise RuntimeError("QWEN_BACKEND=hf but HF client is not configured.")
            raw = self._hf.chat(
                model=self.cfg.qwen_model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.cfg.structured_temperature,
                max_tokens=self.cfg.max_new_tokens_structured,
            )
        else:
            raise ValueError(f"Unsupported qwen_backend: {self.cfg.qwen_backend}")

        parsed, ok = robust_parse_json(raw)
        return parsed, raw, ok

    def _mistral_description(self, complaint: ComplaintInput, fusion: Dict[str, Any], contexts: List[Dict[str, Any]], structured: Dict[str, Any]) -> str:
        """
        Description generation LLM — fine-tuned Mistral-7B-Instruct-v0.2 via HF Inference API.

        INPUT ROUTING POLICY — Mistral receives the full enriched context:
          ✅ Sent:     ward, hospital_name (narrative grounding for description)
          ✅ Sent:     image_caption + voice_text (core complaint signal)
          ✅ Sent:     all retrieved RAG docs (full context — enriches description quality)
          ✅ Sent:     already-extracted category/severity/department (anchors the description)
          ✅ Sent:     contextual metadata signals (physically_harmed, internally_reported, etc.)
          ❌ Excluded: patient name — PII; description should not be personalized
          ❌ Excluded: taxonomy lists — structured fields already resolved by Qwen

        OUTPUT: plain-text complaint_description (1-2 sentences, formal clinical language)
        """
        full_context = self._context_block(contexts, max_docs=None)

        # Extract contextual enrichment signals from complaint metadata.
        # These are non-primary fields (not predicted outputs) used only to make
        # the description richer and more clinically accurate.
        meta = complaint.metadata or {}
        contextual_parts: List[str] = []
        if meta.get("physically_harmed") not in (None, "", False, "false", "no", "No"):
            contextual_parts.append(f"Physical harm reported: {meta['physically_harmed']}")
        if meta.get("internally_reported") not in (None, "", False, "false", "no", "No"):
            contextual_parts.append(f"Already internally reported: {meta['internally_reported']}")
        for key in ("risk_level", "affected_area", "recurrence", "staff_notified", "additional_context"):
            if meta.get(key):
                label = key.replace("_", " ").title()
                contextual_parts.append(f"{label}: {meta[key]}")
        contextual_block = (
            "Contextual signals (use to enrich description, do not repeat verbatim):\n"
            + "\n".join(f"  - {p}" for p in contextual_parts)
            + "\n\n"
        ) if contextual_parts else ""

        system_prompt = (
            "You are a hospital triage clinical documentation AI fine-tuned on real hospital complaint records.\n"
            "Your task is to write a single concise complaint_description (1-2 sentences) that:\n"
            "  - States the specific issue observed (what is wrong, where in the hospital, and the risk it poses).\n"
            "  - Incorporates any contextual signals (harm reported, internal escalation, recurrence) where relevant.\n"
            "  - Is written in formal clinical language.\n"
            "  - Does NOT include recommendations or suggested actions.\n"
            "  - Does NOT start with 'The patient' or any generic filler phrase.\n"
            "  - Must NOT include the patient's name.\n"
            "Return ONLY the description text. No JSON, no labels, no markdown.\n"
            "Example: 'A water puddle on the floor of Ward 3 at City Hospital creates a slip-and-fall hazard for patients and staff, with no housekeeping response reported despite prior internal notification.'"
        )
        user_prompt = (
            f"Hospital: {complaint.hospital_name}\n"
            f"Ward/Unit: {complaint.ward}\n\n"
            f"{fusion['fused_text']}\n\n"
            "Structured fields already extracted:\n"
            f"  Category:   {structured.get('category', '')}\n"
            f"  Severity:   {structured.get('severity', '')}\n"
            f"  Department: {structured.get('department', '')}\n\n"
            f"{contextual_block}"
            "Retrieved SOP / training context (use to enrich description):\n"
            f"{full_context}\n\n"
            "Write exactly one concise complaint_description (1-2 sentences, formal clinical language):"
        )

        if self.cfg.mistral_backend == "mlx":
            if not self._mlx_mistral:
                raise RuntimeError("MISTRAL_BACKEND=mlx but MLX Mistral runner not initialized.")
            return self._mlx_mistral.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=self.cfg.max_new_tokens_description,
                temperature=self.cfg.description_temperature,
            )

        if self.cfg.mistral_backend == "groq":
            if not self._groq:
                raise RuntimeError("MISTRAL_BACKEND=groq but Groq client is not configured.")
            return self._groq.complete(
                model=self.cfg.mistral_groq_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.cfg.description_temperature,
                max_tokens=self.cfg.max_new_tokens_description,
            )

        if self.cfg.mistral_backend == "local_peft":
            if not self._local_mistral:
                raise RuntimeError("MISTRAL_BACKEND=local_peft but local model is not initialized.")
            return self._local_mistral.generate(
                prompt=f"{system_prompt}\n\n{user_prompt}\n\nDescription:",
                max_new_tokens=self.cfg.max_new_tokens_description,
                temperature=self.cfg.description_temperature,
            )

        if self.cfg.mistral_backend == "hf":
            if not self._hf:
                raise RuntimeError("HF token is required for Mistral HF description generation.")
            try:
                return self._hf.chat(
                    model=self.cfg.mistral_hf_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt + "\n\nReturn only the description text.",
                    temperature=self.cfg.description_temperature,
                    max_tokens=self.cfg.max_new_tokens_description,
                )
            except Exception:
                full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nDescription:"
                return self._hf.text(
                    model=self.cfg.mistral_hf_model,
                    prompt=full_prompt,
                    temperature=self.cfg.description_temperature,
                    max_new_tokens=self.cfg.max_new_tokens_description,
                )

        raise ValueError(f"Unsupported mistral_backend: {self.cfg.mistral_backend}")

    def _llama_baseline(self, complaint: ComplaintInput, fusion: Dict[str, Any], contexts: List[Dict[str, Any]]) -> Optional[str]:
        if not self.cfg.llama_enabled or not self._groq:
            return None
        prompt = (
            "You are baseline triage assistant. Return JSON with category, severity, department, complaint_description.\n\n"
            f"{fusion['fused_text']}\n\n"
            f"Context:\n{self._context_block(contexts)}"
        )
        return self._groq.complete(
            model=self.cfg.llama_model,
            system_prompt="Output valid JSON only.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=260,
        )

    def _apply_hybrid_severity(self, category: str, llm_severity: str, fusion_text: str, contexts: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        llm_norm = canonicalize_severity(llm_severity)
        llm_rank = SEVERITY_RANK.get(llm_norm, 1)

        category_floor = CATEGORY_MIN_SEVERITY.get(category, "low")
        floor_rank = SEVERITY_RANK.get(category_floor, 1)

        combined_text = f"{fusion_text}\n" + "\n".join(c["text"] for c in contexts)
        text_low = combined_text.lower()
        if any(k in text_low for k in ["rodent", "rat", "contaminated food", "septic", "infection outbreak"]):
            floor_rank = max(floor_rank, SEVERITY_RANK["critical"])
        if any(k in text_low for k in ["slip", "fall risk", "broken equipment", "electrical", "biohazard"]):
            floor_rank = max(floor_rank, SEVERITY_RANK["high"])

        final_rank = max(llm_rank, floor_rank)
        final_sev = RANK_TO_SEVERITY[final_rank]
        return final_sev, {
            "llm_severity": llm_norm,
            "llm_rank": llm_rank,
            "rule_floor": RANK_TO_SEVERITY[floor_rank],
            "final": final_sev,
        }

    def _validate_output(self, category: str, severity: str, department: str, complaint_description: str, fusion: Dict[str, Any]) -> Tuple[List[str], Optional[float]]:
        errors: List[str] = []
        if category not in self.valid_categories:
            errors.append("Invalid category label.")
        if severity not in self.valid_severities:
            errors.append("Invalid severity label.")
        if department not in self.valid_departments:
            errors.append("Invalid department label.")
        if not complaint_description or len(_normalize_space(complaint_description)) < 12:
            errors.append("Description is too short or empty.")

        second_score: Optional[float] = None
        if self.cfg.enable_second_validation:
            emb_a = self.embedder.encode(fusion["fused_text"], normalize_embeddings=True).astype(np.float32)
            emb_b = self.embedder.encode(complaint_description, normalize_embeddings=True).astype(np.float32)
            second_score = _cosine(emb_a, emb_b)
            if second_score < self.cfg.second_validation_threshold:
                errors.append("Second-level semantic validation score is below threshold.")
        return errors, second_score

    def _store_record(
        self,
        complaint: ComplaintInput,
        fusion: Dict[str, Any],
        final_payload: Dict[str, Any],
        validation_errors: List[str],
        second_score: Optional[float],
        contexts: List[Dict[str, Any]],
        raw_structured: str,
        raw_description: str,
    ) -> int:
        conn = sqlite3.connect(self.cfg.sqlite_path)
        try:
            cur = conn.execute(
                """
                INSERT INTO complaints (
                    created_at_utc, name, complaint, hospital_name, ward, image_caption, voice_text,
                    similarity_score, fusion_mode, category, severity, department, complaint_description,
                    second_validation_score, needs_human_review, validation_errors_json,
                    retrieved_context_json, raw_structured_output, raw_description_output
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now_utc_iso(),
                    complaint.name,
                    complaint.complaint,
                    complaint.hospital_name,
                    complaint.ward,
                    complaint.image_caption,
                    complaint.voice_text,
                    float(fusion["similarity_score"]),
                    fusion["fusion_mode"],
                    final_payload["category"],
                    final_payload["severity"],
                    final_payload["department"],
                    final_payload["complaint_description"],
                    second_score,
                    int(final_payload["needs_human_review"]),
                    json.dumps(validation_errors, ensure_ascii=False),
                    json.dumps(contexts, ensure_ascii=False),
                    raw_structured,
                    raw_description,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def _build_prediction_document(self, complaint: ComplaintInput, final_payload: Dict[str, Any]) -> str:
        return (
            f"Name: {complaint.name}\n"
            f"Hospital: {complaint.hospital_name}\n"
            f"Ward: {complaint.ward}\n"
            f"Complaint: {complaint.complaint}\n"
            f"Image caption: {complaint.image_caption}\n"
            f"Voice text: {complaint.voice_text}\n\n"
            f"Predicted category: {final_payload.get('category', '')}\n"
            f"Predicted severity: {final_payload.get('severity', '')}\n"
            f"Predicted department: {final_payload.get('department', '')}\n"
            f"Predicted complaint_description: {final_payload.get('complaint_description', '')}\n"
        )

    def _store_prediction_vector(
        self,
        record_id: int,
        complaint: ComplaintInput,
        final_payload: Dict[str, Any],
        contexts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not self.cfg.store_predictions_in_chroma:
            return {"enabled": False, "stored": False, "reason": "disabled_by_config"}
        if self.predictions_collection is None:
            return {"enabled": True, "stored": False, "reason": "collection_not_initialized"}

        doc = self._build_prediction_document(complaint, final_payload)
        emb = self.embedder.encode(doc, normalize_embeddings=True).astype(np.float32).tolist()
        pred_id = f"pred_{record_id}"
        metadata = {
            "source": "llm_prediction",
            "record_id": str(record_id),
            "category": str(final_payload.get("category", "")),
            "severity": str(final_payload.get("severity", "")),
            "department": str(final_payload.get("department", "")),
            "needs_human_review": str(bool(final_payload.get("needs_human_review", False))).lower(),
            "retrieved_context_count": str(len(contexts)),
            "created_at_utc": _now_utc_iso(),
        }
        self.predictions_collection.upsert(
            ids=[pred_id],
            embeddings=[emb],
            documents=[doc],
            metadatas=[metadata],
        )
        return {
            "enabled": True,
            "stored": True,
            "collection": self.cfg.chroma_predictions_collection,
            "id": pred_id,
        }

    def process(self, complaint: ComplaintInput) -> Dict[str, Any]:
        fusion = self._compute_similarity_and_fusion(
            image_caption=complaint.image_caption,
            voice_text=complaint.voice_text,
            complaint_text=complaint.complaint,
        )
        contexts = self._retrieve_context(fusion["fused_text"])

        structured_raw_dict, raw_structured_text, structured_json_valid = self._qwen_structured(
            complaint=complaint,
            fusion=fusion,
            contexts=contexts,
        )

        category = canonicalize_category(str(structured_raw_dict.get("category", "")), self.valid_categories)
        severity = canonicalize_severity(str(structured_raw_dict.get("severity", "")))
        department = canonicalize_department(str(structured_raw_dict.get("department", "")), self.valid_departments)

        descr = self._mistral_description(
            complaint=complaint,
            fusion=fusion,
            contexts=contexts,
            structured={
                "category": category,
                "severity": severity,
                "department": department,
            },
        )
        complaint_description = _normalize_space(descr)

        final_severity, severity_trace = self._apply_hybrid_severity(
            category=category,
            llm_severity=severity,
            fusion_text=fusion["fused_text"],
            contexts=contexts,
        )

        validation_errors, second_score = self._validate_output(
            category=category,
            severity=final_severity,
            department=department,
            complaint_description=complaint_description,
            fusion=fusion,
        )

        # llama_baseline_raw = self._llama_baseline(complaint, fusion, contexts)
        needs_review = (len(validation_errors) > 0) or (not structured_json_valid)

        final_payload: Dict[str, Any] = {
            "category": category,
            "severity": final_severity,
            "department": department,
            "complaint_description": complaint_description,
            "needs_human_review": needs_review,
            "fusion": {
                "similarity_score": fusion["similarity_score"],
                "fusion_mode": fusion["fusion_mode"],
                "fusion_policy": fusion["fusion_policy"],
            },
            "severity_trace": severity_trace,
            "retrieved_context": contexts,
            "validation_errors": validation_errors,
            "second_validation_score": second_score,
            "structured_json_valid": structured_json_valid,
            "raw_outputs": {
                "qwen_structured_raw": raw_structured_text,
                "mistral_description_raw": descr,
                # "llama_baseline_raw": llama_baseline_raw,
            },
        }

        record_id = self._store_record(
            complaint=complaint,
            fusion=fusion,
            final_payload=final_payload,
            validation_errors=validation_errors,
            second_score=second_score,
            contexts=contexts,
            raw_structured=raw_structured_text,
            raw_description=descr,
        )
        prediction_store_status: Dict[str, Any]
        try:
            prediction_store_status = self._store_prediction_vector(
                record_id=record_id,
                complaint=complaint,
                final_payload=final_payload,
                contexts=contexts,
            )
        except Exception as exc:
            prediction_store_status = {
                "enabled": self.cfg.store_predictions_in_chroma,
                "stored": False,
                "reason": f"exception: {type(exc).__name__}: {str(exc)}",
            }
        final_payload["record_id"] = record_id
        final_payload["prediction_vector_store"] = prediction_store_status
        return final_payload

    def list_complaints(self, only_urgent: bool = False, limit: int = 100) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.cfg.sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            if only_urgent:
                rows = conn.execute(
                    """
                    SELECT * FROM complaints
                    WHERE severity IN ('critical', 'high')
                    ORDER BY
                        CASE severity WHEN 'critical' THEN 4 WHEN 'high' THEN 3 WHEN 'medium' THEN 2 ELSE 1 END DESC,
                        id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM complaints
                    ORDER BY
                        CASE severity WHEN 'critical' THEN 4 WHEN 'high' THEN 3 WHEN 'medium' THEN 2 ELSE 1 END DESC,
                        id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def build_orchestrator(master_json_path: Optional[str] = None) -> HospitalMultiLLMOrchestrator:
    categories, severities, departments = load_taxonomy_from_master(master_json_path)
    cfg = PipelineConfig()
    return HospitalMultiLLMOrchestrator(cfg, categories, severities, departments)


def _chunk_for_rag(text: str, source: str, chunk_chars: int = 1800, overlap_chars: int = 200) -> List[Dict[str, Any]]:
    cleaned = _normalize_space(text)
    if not cleaned:
        return []
    chunks: List[Dict[str, Any]] = []
    step = max(1, chunk_chars - overlap_chars)
    start = 0
    idx = 0
    n = len(cleaned)
    while start < n:
        end = min(start + chunk_chars, n)
        piece = cleaned[start:end].strip()
        if len(piece) >= 50:
            chunks.append(
                {
                    "id": f"{Path(source).stem}_chunk_{idx}",
                    "source": source,
                    "text": piece,
                }
            )
            idx += 1
        if end == n:
            break
        start += step
    return chunks


def prepare_collection_from_sop_folder(
    sop_dir: str,
    cfg: Optional[PipelineConfig] = None,
    recreate: bool = True,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = cfg or PipelineConfig()
    folder = Path(sop_dir)
    if not folder.exists():
        raise FileNotFoundError(f"SOP folder not found: {folder}")

    supported = {".txt", ".md"}
    files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in supported]
    if not files:
        raise RuntimeError(f"No .txt/.md files found in SOP folder: {folder}")

    all_chunks: List[Dict[str, Any]] = []
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        all_chunks.extend(_chunk_for_rag(text, fp.name))

    if not all_chunks:
        raise RuntimeError("SOP docs were found, but chunking produced no chunks.")

    embedder = _build_embedder(cfg.embedding_model)
    vectors = embedder.encode([c["text"] for c in all_chunks], normalize_embeddings=True).astype(np.float32).tolist()

    target_collection = collection_name or cfg.chroma_collection
    client = chromadb.PersistentClient(path=cfg.chroma_path)
    if recreate:
        try:
            client.delete_collection(target_collection)
        except Exception:
            pass
        coll = client.create_collection(name=target_collection, metadata={"hnsw:space": "cosine"})
    else:
        coll = client.get_or_create_collection(name=target_collection, metadata={"hnsw:space": "cosine"})

    ids = [c["id"] for c in all_chunks]
    docs = [c["text"] for c in all_chunks]
    metas = [{"source": c["source"]} for c in all_chunks]
    batch = 128
    for i in range(0, len(ids), batch):
        coll.add(
            ids=ids[i : i + batch],
            embeddings=vectors[i : i + batch],
            documents=docs[i : i + batch],
            metadatas=metas[i : i + batch],
        )

    return {
        "collection": target_collection,
        "chroma_path": cfg.chroma_path,
        "source_docs": len(files),
        "chunks_indexed": len(all_chunks),
    }


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prepare_collection_from_train_val(
    train_jsonl: str,
    val_jsonl: str,
    cfg: Optional[PipelineConfig] = None,
    recreate: bool = True,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fallback strategy when SOP docs are not available.
    Uses train+val only to avoid test leakage.
    """
    cfg = cfg or PipelineConfig()
    train = _read_jsonl(train_jsonl)
    val = _read_jsonl(val_jsonl)
    rows = train + val
    if not rows:
        raise RuntimeError("No rows loaded from train/val.")

    docs: List[Tuple[str, str]] = []
    for r in rows:
        rid = str(r.get("image_id", "unknown"))
        caption = (r.get("input", {}) or {}).get("image_caption") or r.get("refined_caption", "")
        voice = r.get("voice_text", "")
        category = r.get("category", "")
        severity = str(r.get("severity", "")).lower()
        department = (r.get("routing", {}) or {}).get("primary_department") or r.get("department", "")
        description = r.get("complaint_description", "")
        text = (
            f"Case ID: {rid}\n"
            f"Observed caption: {caption}\n"
            f"Voice complaint: {voice}\n"
            f"Known category: {category}\n"
            f"Known severity: {severity}\n"
            f"Known department: {department}\n"
            f"Known complaint_description: {description}\n"
        )
        docs.append((f"trainval_{rid}.txt", text))

    all_chunks: List[Dict[str, Any]] = []
    for source, text in docs:
        all_chunks.extend(_chunk_for_rag(text, source))

    embedder = _build_embedder(cfg.embedding_model)
    vectors = embedder.encode([c["text"] for c in all_chunks], normalize_embeddings=True).astype(np.float32).tolist()
    target_collection = collection_name or cfg.chroma_trainval_collection
    client = chromadb.PersistentClient(path=cfg.chroma_path)

    if recreate:
        try:
            client.delete_collection(target_collection)
        except Exception:
            pass
        coll = client.create_collection(name=target_collection, metadata={"hnsw:space": "cosine"})
    else:
        coll = client.get_or_create_collection(name=target_collection, metadata={"hnsw:space": "cosine"})

    ids = [c["id"] for c in all_chunks]
    docs_txt = [c["text"] for c in all_chunks]
    metas = [{"source": c["source"]} for c in all_chunks]
    batch = 128
    for i in range(0, len(ids), batch):
        coll.add(
            ids=ids[i : i + batch],
            embeddings=vectors[i : i + batch],
            documents=docs_txt[i : i + batch],
            metadatas=metas[i : i + batch],
        )

    return {
        "collection": target_collection,
        "chroma_path": cfg.chroma_path,
        "rows_used": len(rows),
        "chunks_indexed": len(all_chunks),
        "note": "train+val fallback corpus created; test set excluded.",
    }


def prepare_dual_collections(
    sop_dir: str,
    train_jsonl: str,
    val_jsonl: str,
    cfg: Optional[PipelineConfig] = None,
    recreate: bool = True,
) -> Dict[str, Any]:
    cfg = cfg or PipelineConfig()
    sop_out = prepare_collection_from_sop_folder(
        sop_dir=sop_dir,
        cfg=cfg,
        recreate=recreate,
        collection_name=cfg.chroma_collection,
    )
    trainval_out = prepare_collection_from_train_val(
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        cfg=cfg,
        recreate=recreate,
        collection_name=cfg.chroma_trainval_collection,
    )
    return {
        "mode": "dual_collection",
        "sop_collection": sop_out,
        "trainval_collection": trainval_out,
    }


def check_adapters(cfg: Optional[PipelineConfig] = None, repair: bool = False) -> Dict[str, Any]:
    cfg = cfg or PipelineConfig()

    def _check_one(name: str, base_model_id: str, adapter_dir: str) -> Dict[str, Any]:
        adir = Path(adapter_dir)
        result: Dict[str, Any] = {
            "name": name,
            "adapter_dir": str(adir),
            "exists": adir.exists(),
            "status": "MISSING",
            "details": [],
            "resolved_adapter_dir": None,
        }
        if not adir.exists():
            result["details"].append("Adapter directory not found.")
            return result

        cfg_path = adir / "adapter_config.json"
        in_dir_weight = (adir / "adapter_model.safetensors").exists() or (adir / "adapter_model.bin").exists()
        parent_weight = (adir.parent / "adapter_model.safetensors").exists() or (adir.parent / "adapter_model.bin").exists()
        cfg_valid = cfg_path.exists() and LocalPeftRunner._is_valid_json(cfg_path)

        if cfg_valid:
            result["details"].append("adapter_config.json valid.")
        else:
            if cfg_path.exists():
                result["details"].append("adapter_config.json exists but is invalid/corrupted.")
            else:
                result["details"].append("adapter_config.json missing.")

        if in_dir_weight:
            result["details"].append("Adapter weight found inside adapter directory.")
        elif parent_weight:
            result["details"].append("Adapter weight found in parent directory (repairable layout).")
        else:
            result["details"].append("Adapter weight missing.")

        if cfg_valid and in_dir_weight:
            result["status"] = "OK"
            result["resolved_adapter_dir"] = str(adir)
            return result

        if repair:
            try:
                fixed = LocalPeftRunner._resolve_or_repair_adapter_dir(base_model_id, str(adir), name)
                result["status"] = "REPAIRED"
                result["resolved_adapter_dir"] = fixed
                result["details"].append(f"Repaired adapter directory created: {fixed}")
                return result
            except Exception as exc:
                result["status"] = "INVALID"
                result["details"].append(f"Repair failed: {exc}")
                return result

        result["status"] = "INVALID"
        result["resolved_adapter_dir"] = str(adir)
        return result

    qwen = _check_one("qwen", cfg.qwen_base_model_id, cfg.qwen_adapter_dir)
    mistral = _check_one("mistral", cfg.mistral_base_model_id, cfg.mistral_adapter_dir)

    return {
        "repair_mode": repair,
        "qwen": qwen,
        "mistral": mistral,
        "summary": {
            "qwen_status": qwen["status"],
            "mistral_status": mistral["status"],
        },
    }


# ---------------------------------------------------------------------------
# Pre-built 3-test-case runner — API-only, no local weights required
# ---------------------------------------------------------------------------



def run_three_test_cases(
    master_json_path: Optional[str] = None,
    output_path: Optional[str] = None,
    test_cases: Optional[List[ComplaintInput]] = None,
) -> List[Dict[str, Any]]:
    """
    Run the full RAG pipeline (ChromaDB retrieval + guardrails + multi-LLM) for 3 test cases
    entirely via API calls — no local model weights required.

    Returns a list of 3 structured JSON outputs matching the trained dataset schema.
    If output_path is provided, also writes results to that file.
    """
    orch = build_orchestrator(master_json_path)
    cases = test_cases if test_cases else []

    results: List[Dict[str, Any]] = []
    for idx, inp in enumerate(cases, start=1):
        print(f"\n{'='*60}")
        print(f"Processing test case {idx}/{len(cases)}: {inp.name} @ {inp.hospital_name}")
        print(f"{'='*60}")
        try:
            out = orch.process(inp)
            # Attach test-case index for readability
            out["test_case_index"] = idx
            out["input_summary"] = {
                "name": inp.name,
                "hospital_name": inp.hospital_name,
                "ward": inp.ward,
                "complaint_snippet": inp.complaint[:120] + ("..." if len(inp.complaint) > 120 else ""),
                "image_caption_snippet": inp.image_caption[:120] + ("..." if len(inp.image_caption) > 120 else ""),
                "voice_text_snippet": inp.voice_text[:120] + ("..." if len(inp.voice_text) > 120 else ""),
            }
            results.append(out)
            print(json.dumps(out, indent=2, ensure_ascii=False))
        except Exception as exc:
            err = {
                "test_case_index": idx,
                "error": str(exc),
                "input_summary": {
                    "name": inp.name,
                    "hospital_name": inp.hospital_name,
                },
            }
            results.append(err)
            print(f"ERROR processing test case {idx}: {exc}")

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResults written to: {output_path}")

    return results


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Hospital multi-LLM RAG core pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sop = sub.add_parser("prepare-sop", help="Build Chroma collection from SOP .txt/.md docs")
    p_sop.add_argument("--sop-dir", required=True)
    p_sop.add_argument("--no-recreate", action="store_true")
    p_sop.add_argument("--collection", default=None, help="Override target collection name for SOP indexing.")

    p_tv = sub.add_parser("prepare-trainval", help="Build fallback Chroma collection from train+val jsonl")
    p_tv.add_argument("--train", required=True)
    p_tv.add_argument("--val", required=True)
    p_tv.add_argument("--no-recreate", action="store_true")
    p_tv.add_argument("--collection", default=None, help="Override target collection name for train+val indexing.")

    p_both = sub.add_parser("prepare-both", help="Build both SOP and train+val collections together.")
    p_both.add_argument("--sop-dir", required=True)
    p_both.add_argument("--train", required=True)
    p_both.add_argument("--val", required=True)
    p_both.add_argument("--no-recreate", action="store_true")

    p_run = sub.add_parser("process", help="Run full multi-LLM process for one complaint")
    p_run.add_argument("--name", required=True)
    p_run.add_argument("--complaint", required=True)
    p_run.add_argument("--hospital", required=True)
    p_run.add_argument("--ward", required=True)
    p_run.add_argument("--image-caption", required=True)
    p_run.add_argument("--voice-text", required=True)
    p_run.add_argument("--master-json", default=None)

    p_chk = sub.add_parser("check-adapters", help="Check fine-tuned adapter artifacts for Qwen and Mistral.")
    p_chk.add_argument("--repair", action="store_true", help="Auto-repair broken adapter layout if possible.")

    p_tests = sub.add_parser(
        "run-tests",
        help="Run 3 built-in test cases end-to-end and output structured JSON results.",
    )
    p_tests.add_argument(
        "--master-json",
        default=None,
        help="Path to master_with_splits.json for taxonomy loading (optional).",
    )
    p_tests.add_argument(
        "--output",
        default=None,
        help="Optional path to write the 3 JSON results to a file.",
    )

    args = parser.parse_args()

    if args.cmd == "prepare-sop":
        out = prepare_collection_from_sop_folder(
            args.sop_dir,
            recreate=not args.no_recreate,
            collection_name=args.collection,
        )
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "prepare-trainval":
        out = prepare_collection_from_train_val(
            args.train,
            args.val,
            recreate=not args.no_recreate,
            collection_name=args.collection,
        )
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "prepare-both":
        out = prepare_dual_collections(
            sop_dir=args.sop_dir,
            train_jsonl=args.train,
            val_jsonl=args.val,
            recreate=not args.no_recreate,
        )
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "process":
        orch = build_orchestrator(args.master_json)
        payload = ComplaintInput(
            name=args.name,
            complaint=args.complaint,
            hospital_name=args.hospital,
            ward=args.ward,
            image_caption=args.image_caption,
            voice_text=args.voice_text,
        )
        out = orch.process(payload)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    if args.cmd == "check-adapters":
        out = check_adapters(repair=args.repair)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    if args.cmd == "run-tests":
        run_three_test_cases(
            master_json_path=getattr(args, "master_json", None),
            output_path=getattr(args, "output", None),
        )
        return


if __name__ == "__main__":
    _cli()
