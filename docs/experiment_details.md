# Experiment Details and Results

This document contains the research questions, methods, datasets, metrics, workflow, sanity results, completed result analysis, and final conclusions.

## Report Outline

1. Project overview and research questions
2. Methods, datasets, metrics, and ablation setup
3. Notebook workflow and output files
4. Development-mode sanity checks
5. Analysis framework for interpreting results
6. Completed results, ablations, exploratory prompt-tuning, and final conclusions

## Project Overview

This repository contains a robust cross-domain sentiment analysis project using BERT. The project compares full fine-tuning with parameter-efficient fine-tuning (PEFT) methods, specifically LoRA and prompt-tuning, for binary sentiment classification.

The main goal is to study not only in-domain performance on SST-2, but also cross-domain generalization, robustness to input perturbations, confidence calibration, and computational efficiency.

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

### In-Domain Training and Validation

- **SST-2 (GLUE)**: used for training and in-domain validation.

### Cross-Domain Evaluation

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

### Cross-Domain Evaluation

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

## Project Documents

Additional project documentation is available in the `docs/` folder:

- `proposal.md`: original class project proposal describing the project scope, motivation, datasets, methods, and expected outcomes.
- `project_rationale.md`: detailed rationale explaining why each experiment component is included.
- `proposal_submission.pdf`: submitted proposal document for the course project.

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

## Installation Command

The README uses `requirements.txt` for installation. The equivalent explicit package install command from the original project notes is:

```bash
pip install torch transformers datasets accelerate peft scikit-learn pandas numpy matplotlib seaborn nltk
```

Depending on your GPU and CUDA version, you may need to install a PyTorch build that matches your system.

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

## Analysis Framework

The main analysis is based on the metrics saved in:

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
- If results vary across seeds, report mean +/- standard deviation rather than relying on one run.

### 6. Conclusion Template

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

## Completed Results Analysis

The completed experiments compare three training strategies for BERT-based binary sentiment classification:

- Full fine-tuning
- LoRA-based parameter-efficient fine-tuning
- Prompt-tuning-based parameter-efficient fine-tuning

In addition to the main baseline and ablation experiments, exploratory prompt-tuning follow-up runs were conducted with prompt-specific hyperparameters. These exploratory runs are reported separately from the main baseline comparison.

The models were trained on SST-2 and evaluated on both in-domain and cross-domain datasets. The cross-domain datasets were Yelp Polarity, IMDB, and Amazon Polarity. The analysis focuses on five main aspects:

- Cross-domain generalization
- Robustness to input perturbations
- Confidence calibration
- Computational efficiency
- Exploratory prompt-tuning sensitivity

All reported baseline values are averaged across three seeds: `7`, `42`, and `2026`.

### 1. Baseline Comparison: Full Fine-Tuning vs LoRA vs Prompt-Tuning

| Method | SST-2 Dev Accuracy | Cross-Domain Avg Accuracy | ECE | Trainable Parameters | CUDA Memory |
|---|---:|---:|---:|---:|---:|
| Full fine-tuning | **0.9289 +/- 0.0050** | **0.8720 +/- 0.0040** | 0.0468 | 109.48M | 2365 MB |
| LoRA r=8 | 0.9037 +/- 0.0020 | 0.8610 +/- 0.0012 | **0.0142** | 1.34M | 1751 MB |
| Prompt-tuning | 0.5535 +/- 0.0614 | 0.5170 +/- 0.0257 | 0.0941 | **16.9K** | **1297 MB** |

Lower ECE is better.

Full fine-tuning achieved the highest in-domain and cross-domain accuracy because it updates the entire BERT model. It reached the best SST-2 development performance and also the best average performance across Yelp, IMDB, and Amazon.

LoRA achieved slightly lower accuracy than full fine-tuning, but it required far fewer trainable parameters. Compared with full fine-tuning, LoRA was only about:

```text
2.5 percentage points lower on SST-2 dev accuracy
1.1 percentage points lower on average cross-domain accuracy
```

while training about:

```text
81.6x fewer parameters than full fine-tuning
```

This makes LoRA the strongest parameter-efficient method in this experiment.

Prompt-tuning was the most parameter-efficient method, training only about `16.9K` parameters. However, its performance remained weak and close to chance level on several datasets. Therefore, prompt-tuning did not provide a competitive accuracy-efficiency trade-off under the current configuration.

This baseline prompt-tuning result should be interpreted only under the shared training setup. Later exploratory prompt-tuning runs show that prompt-tuning improves substantially when using a prompt-specific learning rate and longer training.

### 2. Cross-Domain Generalization

Cross-domain generalization was measured by evaluating each model on Yelp, IMDB, and Amazon after training only on SST-2.

The cross-domain average accuracy was computed as:

```text
cross_domain_avg_accuracy = mean(Yelp accuracy, IMDB accuracy, Amazon accuracy)
```

The generalization gap was computed as:

```text
generalization_gap = SST-2 dev accuracy - cross-domain average accuracy
```

| Method | SST-2 Dev Accuracy | Cross-Domain Avg Accuracy | Generalization Gap |
|---|---:|---:|---:|
| Full fine-tuning | 0.9289 | **0.8720** | 0.0569 |
| LoRA r=8 | 0.9037 | 0.8610 | **0.0427** |
| Prompt-tuning | 0.5535 | 0.5170 | 0.0365 |

Lower generalization gap is better, but only when the clean in-domain performance is strong enough to make the gap meaningful.

Full fine-tuning achieved the highest absolute cross-domain accuracy. However, LoRA had a smaller generalization gap than full fine-tuning, meaning that LoRA retained more of its SST-2 performance when transferred to unseen review domains.

The prompt-tuning gap is not very meaningful because its base SST-2 performance was already low. Therefore, the main useful comparison is between full fine-tuning and LoRA.

The exploratory tuned prompt runs are discussed separately because they use a different hyperparameter regime. They improve cross-domain transfer substantially compared with baseline prompt-tuning, but they still remain below LoRA and full fine-tuning.

Across the cross-domain datasets, IMDB was the hardest dataset. Yelp and Amazon results were closer to each other, while IMDB accuracy was consistently lower for both full fine-tuning and LoRA. This is likely because IMDB reviews are longer and more narrative than SST-2 sentences, while Yelp and Amazon reviews are more directly sentiment-oriented.

### 3. Robustness to Input Perturbations

Robustness was evaluated on the SST-2 development set using three perturbation types:

- Punctuation perturbation
- Character-level noise
- Synonym replacement

The robustness drop was calculated as:

```text
robustness_drop = clean SST-2 dev accuracy - perturbed accuracy
```

| Method | Punctuation Drop | Character Noise Drop | Synonym Drop |
|---|---:|---:|---:|
| Full fine-tuning | 0.0199 | 0.0447 | 0.0004 |
| LoRA r=8 | **0.0092** | **0.0329** | 0.0000 |
| Prompt-tuning | -0.0050 | -0.0065 | 0.0000 |

Lower robustness drop is better.

For the meaningful comparison between full fine-tuning and LoRA, LoRA showed smaller drops under both punctuation perturbation and character noise. This suggests that LoRA was more robust relative to its clean performance.

Character noise caused the largest drop for both full fine-tuning and LoRA. This is expected because character-level corruption can interfere with BERT tokenization and produce less reliable subword representations.

Synonym perturbation caused almost no performance change. This suggests that the implemented synonym replacement was mild or affected only a small portion of tokens. Therefore, synonym robustness should be interpreted cautiously.

Prompt-tuning robustness is less meaningful because the clean prompt-tuning performance was already weak. In some cases, the perturbed score was slightly higher than the clean score, which does not indicate true robustness; rather, it reflects instability and weak learning.

### 4. Confidence Calibration

Confidence calibration was measured using Expected Calibration Error (ECE) on the SST-2 development set after temperature scaling.

Lower ECE indicates better calibration.

| Method | Mean ECE |
|---|---:|
| LoRA r=8 | **0.0142** |
| Full fine-tuning | 0.0468 |
| Prompt-tuning | 0.0941 |

LoRA achieved the best calibration among the three methods. This is important because it means LoRA's confidence estimates were more reliable, even though full fine-tuning achieved higher raw accuracy.

Full fine-tuning had higher accuracy but worse calibration than LoRA. Its learned temperature values were around `4.6-4.8`, suggesting that the full fine-tuned models were more overconfident and required stronger softening during temperature scaling.

Prompt-tuning had the weakest calibration overall, which is consistent with its unstable and low classification performance.

The exploratory tuned prompt runs also improved calibration compared with baseline prompt-tuning. Baseline prompt-tuning had ECE `0.0941`, while tuned prompt `vt=20` reduced ECE to `0.0277` and tuned prompt `vt=30` reduced it further to `0.0234`. However, LoRA still had the best calibration overall among the main methods with ECE `0.0142`. These tuned prompt calibration results are exploratory because they come from a prompt-specific hyperparameter regime rather than the shared baseline setup.

### 5. Computational Efficiency

Efficiency was analyzed using:

- Trainable parameters
- Training time
- CUDA memory usage

| Method | Mean Train Time | Max CUDA Memory | Trainable Parameters |
|---|---:|---:|---:|
| Full fine-tuning | ~514.8 sec | ~2365 MB | 109.48M |
| LoRA r=8 | ~636.5 sec | ~1751 MB | 1.34M |
| Prompt-tuning | ~442.0 sec | ~1297 MB | 16.9K |

Full fine-tuning required training all BERT parameters, making it the most flexible but also the most expensive in terms of trainable parameter count.

LoRA trained only about `1.34M` parameters, which is approximately `81.6x` fewer than full fine-tuning. It also used less CUDA memory than full fine-tuning while maintaining competitive accuracy and better calibration.

Prompt-tuning used the fewest trainable parameters and the lowest memory, but the performance was too weak to make it competitive in this experiment.

The exploratory tuned prompt runs remained extremely lightweight. The `vt=20` tuned prompt setting used about `16.9K` trainable parameters, while `vt=30` used about `24.6K` trainable parameters. Even the `vt=30` setting trained far fewer parameters than LoRA's `1.34M` trainable parameters. However, the tuned prompt methods still had lower accuracy than LoRA, so they are best interpreted as ultra-lightweight alternatives rather than the strongest PEFT method.

One interesting observation is that LoRA did not train faster than full fine-tuning in wall-clock time. This can happen because PEFT wrappers introduce implementation overhead, and runtime is not determined only by the number of trainable parameters. Therefore, LoRA's main efficiency advantage in this notebook is:

```text
far fewer trainable parameters
lower memory usage
competitive accuracy
better calibration
```

rather than faster training time.

### 6. LoRA Ablation Analysis

LoRA ablations were used to study the effect of:

- LoRA rank: `r=8` vs `r=16`
- Learning rate: `3e-5` vs `2e-5`

| LoRA Setting | SST-2 Dev Accuracy | Cross-Domain Avg Accuracy | ECE | Trainable Parameters |
|---|---:|---:|---:|---:|
| r=8, lr=3e-5 | 0.9037 | 0.8610 | **0.0142** | 1.34M |
| r=8, lr=2e-5 | 0.8972 | 0.8532 | 0.0188 | 1.34M |
| r=16, lr=2e-5 | 0.9006 | 0.8592 | 0.0151 | 2.68M |
| r=16, lr=3e-5 | **0.9052** | **0.8617** | 0.0204 | 2.68M |

#### Rank 8 vs Rank 16

LoRA rank controls the capacity of the adapter. A higher rank gives the adapter more expressive power, but also increases the number of trainable parameters.

Increasing the rank from `8` to `16` doubled the trainable parameters:

```text
r=8:  1.34M trainable parameters
r=16: 2.68M trainable parameters
```

However, the performance improvement was very small. Compared with the baseline LoRA setting `r=8, lr=3e-5`, the best rank-16 setting improved only about:

```text
+0.15 percentage points on SST-2 dev accuracy
+0.07 percentage points on average cross-domain accuracy
```

This improvement is too small to clearly justify doubling the number of LoRA trainable parameters. Therefore, `r=8` provides the better efficiency balance.

#### Learning Rate 2e-5 vs 3e-5

For LoRA, the learning rate `3e-5` performed better than `2e-5`.

The setting `r=8, lr=2e-5` performed worse than `r=8, lr=3e-5` on both SST-2 dev accuracy and cross-domain average accuracy. This suggests that the lower learning rate under-trained the LoRA adapters in this setup.

Overall, the original LoRA baseline setting was already a strong choice:

```text
LoRA r=8, lr=3e-5
```

### 7. Exploratory Prompt-Tuning Follow-Up

The main baseline comparison used the same general training setup for full fine-tuning, LoRA, and prompt-tuning. Under that shared setup, prompt-tuning performed weakly, with an average SST-2 development accuracy of `0.5535` and cross-domain average accuracy of `0.5170`.

Because prompt-tuning often requires a different hyperparameter regime than full fine-tuning or LoRA, additional exploratory prompt-tuning runs were conducted. These runs are reported separately from the main baseline comparison and are not treated as replacements for the baseline prompt-tuning results.

The exploratory prompt-tuning setup used:

```text
learning rate = 1e-3
epochs = 5
seeds = [7, 42, 2026]
```

Two virtual-token settings were tested:

```text
virtual_tokens = 20
virtual_tokens = 30
```

These runs were labeled as `exploratory_prompt` in the registry to keep them separate from the main baseline and LoRA ablation results.

#### Exploratory Prompt-Tuning Results

| Method | SST-2 Dev Accuracy | Cross-Domain Avg Accuracy | ECE | Trainable Parameters | CUDA Memory |
|---|---:|---:|---:|---:|---:|
| Prompt baseline | 0.5535 | 0.5170 | 0.0941 | 16.9K | 1297 MB |
| Tuned prompt, vt=20 | 0.8486 | 0.7804 | 0.0277 | 16.9K | 1293 MB |
| Tuned prompt, vt=30 | **0.8490** | **0.7974** | **0.0234** | 24.6K | 1339 MB |

The tuned prompt experiments improved substantially over the baseline prompt-tuning setup.

Compared with baseline prompt-tuning:

```text
vt=20 improved SST-2 dev accuracy by about 29.5 percentage points
vt=20 improved cross-domain average accuracy by about 26.3 percentage points

vt=30 improved SST-2 dev accuracy by about 29.6 percentage points
vt=30 improved cross-domain average accuracy by about 28.0 percentage points
```

This shows that the original prompt-tuning baseline was not necessarily weak because prompt-tuning itself is ineffective. Rather, prompt-tuning was highly sensitive to the training setup. A higher learning rate and longer training allowed the prompt parameters to learn much more effectively.

#### Virtual Tokens 20 vs 30

The `vt=30` setting performed slightly better than `vt=20`.

| Setting | SST-2 Dev Accuracy | Cross-Domain Avg Accuracy | ECE |
|---|---:|---:|---:|
| vt=20 | 0.8486 | 0.7804 | 0.0277 |
| vt=30 | **0.8490** | **0.7974** | **0.0234** |

The SST-2 development accuracy was almost the same for both settings. However, `vt=30` gave better cross-domain performance, improving the cross-domain average by about `1.7` percentage points over `vt=20`.

This suggests that increasing the number of virtual tokens gave the prompt more capacity, which helped transfer to Yelp, IMDB, and Amazon. However, it also increased the number of trainable parameters from about `16.9K` to about `24.6K`.

#### Tuned Prompt-Tuning vs LoRA

Although tuned prompt-tuning improved substantially, it still did not outperform LoRA.

| Method | SST-2 Dev Accuracy | Cross-Domain Avg Accuracy | Trainable Parameters |
|---|---:|---:|---:|
| LoRA r=8 | **0.9037** | **0.8610** | 1.34M |
| Tuned prompt, vt=30 | 0.8490 | 0.7974 | 24.6K |

LoRA remained stronger by about:

```text
5.5 percentage points on SST-2 dev accuracy
6.4 percentage points on average cross-domain accuracy
```

However, tuned prompt-tuning used far fewer trainable parameters than LoRA. The `vt=30` tuned prompt setup trained only about `24.6K` parameters, while LoRA trained about `1.34M` parameters.

Therefore, tuned prompt-tuning can be viewed as an ultra-lightweight alternative, but LoRA remains the better overall performance-efficiency trade-off when accuracy and transfer performance are prioritized.

#### Interpretation

The exploratory prompt-tuning results change the interpretation of prompt-tuning.

The baseline prompt-tuning result showed that prompt-tuning was not competitive under the shared baseline setup. However, the tuned prompt results show that prompt-tuning can improve substantially when given a more suitable prompt-specific setup.

The main conclusion is therefore:

```text
Prompt-tuning is highly sensitive to hyperparameters. Under the shared baseline setup, it performed weakly. With a higher learning rate, longer training, and more virtual tokens, it improved substantially, but still remained below LoRA and full fine-tuning in accuracy.
```

These exploratory results should remain separate from the main baseline comparison because they use a different hyperparameter regime. The main fair comparison remains:

```text
full fine-tuning vs LoRA vs baseline prompt-tuning under the shared setup
```

The tuned prompt results are best reported as a follow-up experiment showing that prompt-tuning can become more effective when configured specifically for prompt learning.

### 8. Conceptual Interpretation of the Methods

#### Full Fine-Tuning

Full fine-tuning updates all or most of BERT's parameters. This gives the model the maximum flexibility to adapt to SST-2 sentiment classification.

```text
Full fine-tuning = change the model itself
```

This explains why full fine-tuning achieved the highest overall accuracy.

#### LoRA

LoRA freezes the original BERT weights and adds small trainable low-rank matrices inside the model's linear layers. These adapters allow the model to adjust internal representations without updating all BERT parameters.

```text
LoRA = add small trainable internal updates
```

This explains why LoRA performed close to full fine-tuning while training far fewer parameters.

#### Prompt-Tuning

Prompt-tuning freezes BERT and learns a small set of virtual prompt tokens at the input level. It does not modify the internal transformer layers.

```text
Prompt-tuning = add trainable input prompts
```

This explains why prompt-tuning was weak under the shared baseline setup: it had less control over the internal model layers and required a more suitable prompt-specific training configuration. The exploratory tuned prompt runs showed that prompt-tuning can improve substantially with a higher learning rate, longer training, and more virtual tokens, although it still remained below LoRA and full fine-tuning.

### 9. Accuracy and Macro-F1

The notebook reports both accuracy and macro-F1.

Macro-F1 is computed separately from accuracy using scikit-learn's macro-F1 calculation. It is not copied from accuracy.

The values are similar for full fine-tuning and LoRA because the datasets are binary and mostly balanced, and the stronger models perform similarly on both sentiment classes. In this situation, accuracy and macro-F1 naturally become very close.

However, macro-F1 is still useful because it reveals class imbalance or class-specific weakness. For example, prompt-tuning sometimes has accuracy around `0.51`, but macro-F1 around `0.35`, showing that it is not learning both classes well.

Micro-F1 was not used because, for single-label binary classification, micro-F1 is usually almost identical to accuracy. Since accuracy is already reported, micro-F1 would add little additional information. Macro-F1 is more informative for this comparison.

### 10. Notes on Synonym Perturbation

The synonym perturbation produced negligible performance changes for all methods.

This likely means that the synonym replacement function was mild, replaced only a small number of words, or often returned the original sentence. Therefore, synonym robustness should not be over-interpreted.

A suitable limitation statement is:

```text
Synonym perturbation caused negligible change, suggesting that the implemented synonym replacement was mild or affected only a small portion of tokens. Future work could use stronger semantic perturbation methods or controlled paraphrase generation.
```

### 11. Final Conclusion

Overall, full fine-tuning produced the best raw accuracy, but LoRA provided the best performance-efficiency trade-off.

Full fine-tuning achieved the highest in-domain and cross-domain accuracy, but required updating all `109.48M` BERT parameters. LoRA achieved competitive performance with about `81.6x` fewer trainable parameters, lower GPU memory usage, better calibration, and smaller robustness drops under perturbations.

Baseline prompt-tuning was the most parameter-efficient main method, training only `16.9K` parameters, but it performed weakly under the shared baseline configuration. However, the exploratory tuned prompt runs showed that prompt-tuning can improve substantially with prompt-specific hyperparameters. The best tuned prompt setting, `vt=30`, reached `0.8490` SST-2 dev accuracy and `0.7974` cross-domain average accuracy while training only `24.6K` parameters. Despite this improvement, tuned prompt-tuning still remained below LoRA and full fine-tuning in accuracy.

The LoRA ablation results showed that increasing rank from `8` to `16` only slightly improved accuracy while doubling the number of trainable parameters. Therefore, LoRA with rank `8` offered the better efficiency balance in this experiment.

The final takeaway is:

```text
Full fine-tuning is best for maximum accuracy.
LoRA is best for the overall performance-efficiency trade-off.
Baseline prompt-tuning is weakest under the shared setup.
Tuned prompt-tuning shows that prompt methods are highly hyperparameter-sensitive and can become a useful ultra-lightweight alternative, but still remain below LoRA in this experiment.
```
