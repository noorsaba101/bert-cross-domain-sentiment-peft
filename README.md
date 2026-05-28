# Robust Cross-Domain Sentiment Analysis with BERT: PEFT vs Full Fine-Tuning

BERT-based sentiment analysis comparing full fine-tuning, LoRA, and prompt-tuning across SST-2, Yelp, IMDB, and Amazon reviews, with evaluations for cross-domain generalization, robustness, calibration, and efficiency.

This repository contains a robust cross-domain sentiment analysis project using BERT. The project compares full fine-tuning with parameter-efficient fine-tuning (PEFT) methods, specifically LoRA and prompt-tuning, for binary sentiment classification.

The main goal is to study not only in-domain performance on SST-2, but also cross-domain generalization, robustness to input perturbations, confidence calibration, and computational efficiency.

## Project Overview

Sentiment models often perform well on the domain they are trained on, but their performance can degrade when applied to text from a different domain. For example, a model trained on sentence-level sentiment data may behave differently on movie reviews, restaurant reviews, or product reviews.

This project investigates this problem using `bert-base-uncased` as the base model. The model is trained on SST-2 and evaluated on both in-domain and cross-domain datasets.

## Research Questions

This project is designed to answer the following questions:

1. Can PEFT methods such as LoRA and prompt-tuning approach the performance of full fine-tuning on SST-2?
2. Which method generalizes better across unseen domains such as Yelp, IMDB, and Amazon reviews?
3. How robust are the methods to text perturbations such as punctuation removal, character noise, and synonym replacement?
4. How well-calibrated are the model confidence scores, and can temperature scaling improve calibration?
5. What is the trade-off between performance and computational cost across the different methods?

## Methods Compared

The notebook compares three fine-tuning strategies:

| Method | Description |
|---|---|
| Full fine-tuning | Updates all BERT parameters and the classification head |
| LoRA | Adds trainable low-rank adaptation matrices while keeping most base model parameters frozen |
| Prompt-tuning | Adds trainable virtual prompt embeddings with very few trainable parameters |

## Datasets

### In-domain Training and Validation

- **SST-2 (GLUE)**  
  Used for training and in-domain validation.

### Cross-domain Evaluation

The following datasets are used only for evaluation and are not used for training:

- **Yelp Polarity**
- **IMDB**
- **Amazon Polarity**

This setup allows the trained models to be evaluated under domain shift.

## Evaluation Metrics

The project reports both quality and efficiency metrics.

### Classification Metrics

- Accuracy
- Macro F1-score

### Cross-domain Evaluation

Each trained model is evaluated on:

- SST-2 validation set
- Yelp test set
- IMDB test set
- Amazon Polarity test set

### Robustness Evaluation

The notebook evaluates robustness under three text perturbations:

- Punctuation removal
- Character-level noise
- Simple synonym replacement

### Calibration Evaluation

The notebook computes:

- Expected Calibration Error (ECE)
- Temperature scaling on the SST-2 validation set

### Cost and Efficiency Metrics

The notebook also logs:

- Total parameter count
- Trainable parameter count
- Training time
- Peak CUDA memory usage

## Ablation Study

The project includes targeted LoRA ablations:

| Ablation | Values |
|---|---|
| LoRA rank | `r=8`, `r=16` |
| Learning rate | `2e-5`, `3e-5` |

The final full run uses three random seeds:

```text
7, 42, 2026
```

This supports mean and standard deviation reporting across seeds.

## Repository Structure

```text
.
├── README.md
├── sentiment_cross_domain.ipynb   # Main experiment notebook
├── docs/
│   ├── proposal.md                # Original class project proposal
│   ├── project_rationale.md       # Rationale and experiment design notes
│   └── proposal_submission.pdf    # Submitted proposal document
├── outputs/
│   ├── results_registry_dev.json  # Development/sanity-run results
│   └── results_registry_full.json # Final full-run results
├── .gitignore
└── LICENSE
```

> Note: Large model checkpoint files such as `.pt`, `.pth`, `.bin`, and `.safetensors` are not committed to the repository. They are generated locally during training and should be ignored to keep the repository lightweight.

## Project Documents

Additional project documentation is available in the `docs/` folder:

- `proposal.md` — original class project proposal describing the project scope, motivation, datasets, methods, and expected outcomes.
- `project_rationale.md` — detailed rationale explaining why each experiment component is included.
- `proposal_submission.pdf` — submitted proposal document for the course project.

## Notebook Workflow

The main notebook follows this structure:

1. Environment and package validation
2. Dataset loading
3. Tokenization and dataloader preparation
4. Model construction for full fine-tuning, LoRA, and prompt-tuning
5. Training with early stopping and best-checkpoint restore
6. In-domain evaluation on SST-2
7. Cross-domain evaluation on Yelp, IMDB, and Amazon
8. Robustness testing using input perturbations
9. Calibration analysis using ECE and temperature scaling
10. Multi-seed runs and LoRA ablations
11. Performance vs cost comparison
12. Result interpretation and limitations

## Development vs Full Run

The notebook includes a workflow switch:

```python
DEVELOPMENT_MODE = True
```

When `DEVELOPMENT_MODE = True`, the notebook runs a fast sanity check with:

- 1 epoch
- 1 seed
- Subsampled cross-domain evaluation
- No ablations

For final experiments, set:

```python
DEVELOPMENT_MODE = False
```

This enables:

- 3 epochs
- 3 seeds
- Full cross-domain evaluation
- LoRA rank and learning-rate ablations

The notebook stores development and full results separately:

```text
outputs/results_registry_dev.json
outputs/results_registry_full.json
```

## Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install torch transformers datasets accelerate peft scikit-learn pandas numpy matplotlib seaborn nltk
```

Depending on your GPU and CUDA version, you may need to install a PyTorch build that matches your system.

## Running the Notebook

Open the notebook:

```text
sentiment_cross_domain.ipynb
```

Run the cells from top to bottom.

For a quick test:

```python
DEVELOPMENT_MODE = True
CLEAR_OLD_REGISTRY = True
```

For the final run:

```python
DEVELOPMENT_MODE = False
CLEAR_OLD_REGISTRY = True
```

After the final run starts producing results, set:

```python
CLEAR_OLD_REGISTRY = False
```

This allows interrupted runs to resume without deleting completed results.

## Expected Outputs

After running the notebook, the `outputs/` folder may contain lightweight result files such as:

```text
results_registry_dev.json
results_registry_full.json
```

The registry files store metrics for each run, including:

- SST-2 dev accuracy and F1
- Yelp, IMDB, and Amazon accuracy/F1
- Robustness metrics
- ECE and temperature value
- Trainable parameter count
- Training time
- CUDA memory usage

Model checkpoint files such as `.pt` are generated during training but are not tracked in Git because they can be large. They can be regenerated by running the notebook.

## Development-Mode Sanity Results

Before running the full experiment grid, a development-mode sanity run was performed with one seed, one epoch, and shuffled cross-domain subsampling. This was used to verify that the training, evaluation, robustness, calibration, and logging pipeline worked correctly before launching the full multi-seed and ablation runs.

### Effect of Shuffled Cross-Domain Sampling

Initially, cross-domain subsets were selected using the first `n` examples. This produced suspicious IMDB results, especially a large mismatch between accuracy and macro F1-score. The sampling was updated to use shuffled selection:

```python
ds.shuffle(seed=BASE_SEED).select(range(n))
```

After this fix, IMDB results became more balanced and credible.

| Method | Previous IMDB Acc/F1 | New IMDB Acc/F1 | Interpretation |
|---|---:|---:|---|
| Full fine-tuning | `0.887 / 0.470` | `0.843 / 0.843` | More credible after shuffled sampling |
| Prompt-tuning | `0.038 / 0.037` | `0.532 / 0.441` | Fixed abnormal subset behavior |
| LoRA r=8 | `0.848 / 0.459` | `0.816 / 0.816` | More credible after shuffled sampling |

### Latest Development-Mode Results

| Method | SST-2 Dev Acc | Yelp Acc | IMDB Acc | Amazon Acc | Trainable Params |
|---|---:|---:|---:|---:|---:|
| Full fine-tuning | 0.923 | 0.875 | 0.843 | 0.880 | 109.48M |
| LoRA r=8 | 0.892 | 0.877 | 0.816 | 0.867 | 1.34M |
| Prompt-tuning | 0.557 | 0.507 | 0.532 | 0.517 | 16.9K |

These results are not the final experimental results because development mode uses only one seed, one epoch, and subsampled cross-domain evaluation. However, they confirm that the pipeline behaves sensibly and that LoRA provides a strong performance-efficiency trade-off compared with full fine-tuning, while prompt-tuning is highly parameter-efficient but weaker in this setup.

## How to Analyze the Final Results

After the full experiment grid completes, the main analysis is based on the metrics saved in:

```text
outputs/results_registry_full.json
```

Each run contains results for SST-2 development performance, cross-domain evaluation, robustness checks, calibration, and computational cost. The final comparison should use the mean and standard deviation across the three seeds: `7`, `42`, and `2026`.

### 1. Cross-Domain Generalization

Cross-domain generalization is analyzed by comparing each model's in-domain SST-2 performance with its performance on the unseen review datasets:

- Yelp
- IMDB
- Amazon Polarity

The key metrics are:

- `dev.accuracy`
- `dev.f1_macro`
- `yelp.accuracy`
- `yelp.f1_macro`
- `imdb.accuracy`
- `imdb.f1_macro`
- `amazon.accuracy`
- `amazon.f1_macro`

A model generalizes better if it maintains strong performance on Yelp, IMDB, and Amazon after being trained only on SST-2.

A useful way to report this is to compute the average cross-domain score:

```text
cross_domain_avg_accuracy = mean(Yelp accuracy, IMDB accuracy, Amazon accuracy)
cross_domain_avg_f1 = mean(Yelp F1, IMDB F1, Amazon F1)
```

Then compare the drop from SST-2:

```text
generalization_gap = SST-2 dev accuracy - cross-domain average accuracy
```

Lower gap means better cross-domain generalization.

Interpretation:

- If full fine-tuning has the highest SST-2 accuracy but a larger drop on Yelp/IMDB/Amazon, it may be more domain-specific.
- If LoRA has slightly lower SST-2 accuracy but similar cross-domain performance, it may offer a better generalization-efficiency trade-off.
- If prompt-tuning performs weakly across all domains, it suggests that the current prompt setup is underfitting or needs different hyperparameters.

### 2. Robustness to Input Perturbations

Robustness is analyzed using the perturbed SST-2 development evaluations saved as:

- `robust_dev_punctuation`
- `robust_dev_char_noise`
- `robust_dev_synonym`

These correspond to:

- punctuation removal
- character-level noise
- synonym replacement

For each method, compare the clean SST-2 dev performance with the perturbed performance.

Example:

```text
punctuation_drop = clean dev accuracy - punctuation robustness accuracy
char_noise_drop = clean dev accuracy - char-noise robustness accuracy
synonym_drop = clean dev accuracy - synonym robustness accuracy
```

A more robust method has a smaller drop under perturbation.

Typical interpretation:

- Character noise is usually the hardest perturbation because it directly affects tokenization.
- Punctuation removal may cause a moderate drop.
- Synonym replacement may cause little change if the replacements are mild or limited.

The robustness section should answer:

```text
Which method loses the least performance when the input text is slightly modified?
```

### 3. Confidence Calibration

Calibration is analyzed using:

- `ece_dev`
- `temp`

ECE means Expected Calibration Error. Lower ECE is better.

The notebook applies temperature scaling on the SST-2 development set. The final analysis should compare ECE across methods.

Interpretation:

- Lower `ece_dev` means the model's confidence scores are more reliable.
- A high accuracy model can still be poorly calibrated if it is overconfident.
- If LoRA has similar accuracy but lower ECE than full fine-tuning, LoRA may be more reliable in confidence estimation.
- The learned `temp` value shows how much temperature scaling adjusted the logits. A value greater than 1 usually softens overconfident predictions.

Report something like:

```text
Calibration was evaluated using Expected Calibration Error (ECE) after temperature scaling on the SST-2 development set. Lower ECE indicates better confidence reliability.
```

### 4. Computational Efficiency

Efficiency is analyzed using:

- `params.total`
- `params.trainable`
- `train_seconds`
- `cuda_mem.max_allocated_mb`
- `cuda_mem.reserved_mb`

The most important metric for PEFT comparison is trainable parameters.

Expected pattern:

- Full fine-tuning trains all BERT parameters.
- LoRA trains far fewer parameters while keeping most of BERT frozen.
- Prompt-tuning trains the fewest parameters.

Efficiency should be discussed together with performance.

Useful comparisons:

```text
trainable parameter reduction = full trainable params / PEFT trainable params
training time difference = full train time - PEFT train time
memory difference = full max CUDA memory - PEFT max CUDA memory
```

Interpretation:

- If LoRA reaches close to full fine-tuning performance with much fewer trainable parameters, it gives a strong performance-cost trade-off.
- If prompt-tuning uses very few parameters but has much lower accuracy, it is efficient but not effective in this setup.
- If full fine-tuning gives the best accuracy but highest trainable parameter count, it remains the strongest but most expensive baseline.

### 5. Ablation Analysis

The ablation results compare LoRA settings:

- rank `r=8` vs `r=16`
- learning rate `2e-5` vs `3e-5`

The goal is to see whether increasing LoRA rank or changing learning rate improves performance enough to justify the extra cost.

Analyze:

- SST-2 dev accuracy/F1
- cross-domain average accuracy/F1
- robustness scores
- ECE
- trainable parameters
- training time
- CUDA memory

Interpretation:

- If `r=16` improves accuracy only slightly but increases trainable parameters, `r=8` may be the better efficiency choice.
- If `2e-5` improves stability or calibration, it may be preferable despite similar accuracy.
- If results vary across seeds, report mean ± standard deviation rather than relying on one run.

### 6. Final Conclusion Pattern

The final conclusion should combine all four dimensions:

```text
Full fine-tuning achieved the strongest in-domain performance, but required training all BERT parameters. LoRA achieved competitive SST-2 and cross-domain performance while using far fewer trainable parameters, making it the best performance-efficiency trade-off in this experiment. Prompt-tuning was the most parameter-efficient method but underperformed in accuracy and cross-domain transfer, suggesting that it may require further tuning, more epochs, or different prompt initialization. Robustness and calibration results further show whether each method remains reliable under input perturbations and confidence-based evaluation.
```

### Summary Mapping

| Claim in README | What to use from notebook |
|---|---|
| Cross-domain generalization | Yelp, IMDB, Amazon accuracy/F1 vs SST-2 dev |
| Robustness | `robust_dev_punctuation`, `robust_dev_char_noise`, `robust_dev_synonym` |
| Calibration | `ece_dev`, `temp` |
| Efficiency | `trainable params`, `train_seconds`, `cuda_mem` |

The final discussion should be based on the full multi-seed results, not only a single run. Single-run results are useful for sanity checking, but final conclusions should rely on mean and standard deviation across the planned seeds.

## Reference Paper

This project is based on the standard BERT fine-tuning setup introduced in:

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).  
**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.**  
NAACL-HLT 2019.  
https://arxiv.org/abs/1810.04805

## Course Context

This repository was developed for the CS5143 Natural Language Processing class project, Spring 2026.

## Group Members

- Noor us Saba — K247625
- Kanza Syed — K247604

## License

This project is released under the MIT License.