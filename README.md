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

## Main Files

```text
.
|-- README.md
|-- sentiment_cross_domain.ipynb
|-- RESULT_ANALYSIS.md
|-- docs/
|   |-- experiment_details.md
|   |-- proposal.md
|   |-- project_rationale.md
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

- [docs/experiment_details.md](docs/experiment_details.md) describes the research questions, datasets, metrics, notebook workflow, development-mode sanity results, and guidance for interpreting final results.
- [RESULT_ANALYSIS.md](RESULT_ANALYSIS.md) contains the completed result analysis, including baseline comparison, LoRA ablations, exploratory prompt-tuning follow-up runs, and final conclusions.
- [docs/proposal.md](docs/proposal.md) contains the original class project proposal.
- [docs/project_rationale.md](docs/project_rationale.md) explains the experiment design rationale.
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

## Reference

This project is based on the standard BERT fine-tuning setup introduced in:

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).  
**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.**  
NAACL-HLT 2019.  
https://arxiv.org/abs/1810.04805

## Course Context

This repository was developed for the CS5143 Natural Language Processing class project, Spring 2026.

## Group Members

- Noor us Saba - K247625
- Kanza Syed - K247604

## License

This project is released under the MIT License.
