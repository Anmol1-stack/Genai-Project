# LLM Baseline Comparison — Zero-Shot vs Prompted

This folder contains the baseline evaluation notebooks comparing **Zero-Shot** and **Prompted (Few-Shot)** performance of the two LLMs used in this project.

| Notebook | Model | Baselines Covered |
|----------|-------|-------------------|
| `Mistral_7B_ZeroShot_vs_Prompted.ipynb` | Mistral-7B-Instruct-v0.2 | Baseline-1 (Zero-Shot) vs Baseline-2 (Prompted) |
| `Qwen_1_5B_ZeroShot_vs_Prompted.ipynb` | Qwen2.5-1.5B-Instruct | Baseline-1 (Zero-Shot) vs Baseline-2 (Prompted) |

## Key Results (from Table VI)

| Metric | Mistral B1 (Zero-Shot) | Mistral B2 (Prompted) | Qwen B2 (Prompted) |
|--------|----------------------|---------------------|-------------------|
| Category Accuracy | 0.0000 | 0.8750 | 0.9375 |
| Category Macro F1 | 0.0000 | 0.7214 | 0.9217 |
| Department Accuracy | 0.1250 | 0.9062 | 0.5938 |
| ROUGE-1 | 0.3550 | 0.3250 | 0.2949 |

These are standalone baselines. The final fine-tuned pipeline is in [`../running_files/`](../running_files/).
