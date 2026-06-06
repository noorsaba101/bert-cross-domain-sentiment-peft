# Robust Cross-Domain Sentiment Analysis with BERT

BERT-based binary sentiment analysis comparing full fine-tuning, LoRA, and prompt-tuning across SST-2, Yelp, IMDB, and Amazon reviews.

The project studies not only in-domain sentiment classification performance, but also cross-domain generalization, robustness to input perturbations, confidence calibration, and computational efficiency.

## Overview

Sentiment models often perform well on the domain they are trained on, but degrade when applied to text from another domain. This project investigates that behavior using `bert-base-uncased`.

Models are trained on SST-2 and evaluated on:

- SST-2 validation data
- Yelp Polarity
- IMDB
- Amazon Polarity

## Methods

The notebook compares three training strategies:

| Method | Description |
|---|---|
| Full fine-tuning | Updates all BERT parameters and the classification head |
| LoRA | Adds trainable low-rank adaptation matrices while keeping most base model parameters frozen |
| Prompt-tuning | Adds trainable virtual prompt embeddings with very few trainable parameters |

The final full run uses three random seeds:

```text
7, 42, 2026
```

## Results Highlights

Final numbers are averaged over the three seeds and computed from `outputs/results_registry_full.json`. The table shows the **best configuration of each method**:

| Method | SST-2 dev acc | Cross-domain avg acc | ECE | Trainable params |
|---|---:|---:|---:|---:|
| Full fine-tuning | 0.9289 | 0.8720 | 0.0468 | 109.5M |
| LoRA (r=8, lr=3e-5) | 0.9037 | 0.8610 | **0.0142** | 1.34M (~81.6x fewer) |
| Tuned prompt (vt=20) | 0.8486 | 0.7804 | 0.0277 | 16.9K (~6,480x fewer) |

Full fine-tuning gives the best raw accuracy, but **LoRA is the best overall trade-off**: near-baseline accuracy, the lowest ECE (best calibration), about 81.6x fewer trainable parameters, and the smallest robustness drops. Tuned prompt-tuning is the lightest option by far but trails on accuracy. Among the perturbations, character-level noise is the hardest stress test, while synonym replacement is too mild (it edits only a few sentiment words) to be conclusive.

## Main Files

```text
.
|-- README.md
|-- sentiment_cross_domain.ipynb
|-- docs/
|   |-- project_report.md
|   |-- project_rationale.md
|   |-- proposal.md
|   `-- proposal_submission.pdf
|-- outputs/
|   |-- results_registry_dev.json
|   `-- results_registry_full.json
|-- requirements.txt
|-- requirements_runpod.txt
|-- setup_runpod.sh
`-- LICENSE
```

Large model checkpoint files such as `.pt`, `.pth`, `.bin`, and `.safetensors` are not committed to the repository. They are generated locally during training and should be ignored to keep the repository lightweight.

## Documentation

- [docs/project_report.md](docs/project_report.md) is the main project report. It includes the research questions, datasets, metrics, notebook workflow, development-mode sanity checks, completed result analysis, LoRA ablations, exploratory prompt-tuning follow-up runs, and final conclusions.
- [docs/proposal.md](docs/proposal.md) contains the original class project proposal.
- [docs/project_rationale.md](docs/project_rationale.md) explains the experiment design rationale, including the planned deliverables and risk / mitigation notes.
- [docs/proposal_submission.pdf](docs/proposal_submission.pdf) is the submitted proposal document for the course project.

## Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

Depending on your GPU and CUDA version, you may need to install a PyTorch build that matches your system.

## Running the Notebook

Open and run:

```text
sentiment_cross_domain.ipynb
```

Run the cells from top to bottom.

For a quick sanity check:

```python
DEVELOPMENT_MODE = True
CLEAR_OLD_REGISTRY = True
```

For the final experiment run:

```python
DEVELOPMENT_MODE = False
CLEAR_OLD_REGISTRY = True
```

After the final run starts producing results, set:

```python
CLEAR_OLD_REGISTRY = False
```

This allows interrupted runs to resume without deleting completed results.

## Outputs

The notebook writes lightweight result registries to:

```text
outputs/results_registry_dev.json
outputs/results_registry_full.json
```

The registry files store metrics for each run, including SST-2 performance, cross-domain performance, robustness metrics, calibration results, trainable parameter count, training time, and CUDA memory usage.

## Relationship to the Seminal Paper

This project uses **BERT (Devlin et al., 2019)** as its seminal paper. We reproduce the paper's SST-2 full fine-tuning baseline and extend it with parameter-efficient methods.

**Implementation fidelity.** The full fine-tuning setup follows the BERT fine-tuning recipe (§4.1, Appendix A.3): `bert-base-uncased`, a linear head on the `[CLS]` token, AdamW, learning rate `3e-5` (with a `2e-5` vs `3e-5` ablation), batch size `16`, `3` epochs, dropout `0.1`, and a linear warmup/decay schedule. Every hyperparameter falls inside the ranges the paper specifies.

**Results reproduction.** The BERT paper reports **BERT-BASE SST-2 = 93.5%** (Table 1, GLUE test server). Our full fine-tuning baseline reaches **0.9289 SST-2 dev accuracy** (mean over seeds `7, 42, 2026`). Because SST-2 test labels are hidden, we evaluate on the dev split, so a small dev-vs-test gap is expected. The notebook's sanity-check cell compares against ~93% (a representative BERT-base figure, slightly rounded from the paper's exact 93.5%); the result is within roughly 1 percentage point either way, confirming a close reproduction.

**Methods beyond the BERT paper.** LoRA and prompt-tuning are not part of Devlin et al. (2019); they follow their own source papers (cited below).

A fuller paper-to-implementation mapping is in [docs/project_report.md](docs/project_report.md).

## References

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).  
**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.**  
NAACL-HLT 2019.  
https://arxiv.org/abs/1810.04805

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021).  
**LoRA: Low-Rank Adaptation of Large Language Models.**  
https://arxiv.org/abs/2106.09685

Lester, B., Al-Rfou, R., & Constant, N. (2021).  
**The Power of Scale for Parameter-Efficient Prompt Tuning.**  
https://arxiv.org/abs/2104.08691

## Course Context

This repository was developed for the CS5143 Natural Language Processing class project, Spring 2026.

## Group Members

- Noor us Saba - K247625
- Kanza Syed - K247604

## License

This project is released under the MIT License.
