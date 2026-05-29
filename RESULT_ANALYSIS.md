## Results Analysis

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

---

## 1. Baseline Comparison: Full Fine-Tuning vs LoRA vs Prompt-Tuning

| Method | SST-2 Dev Accuracy | Cross-Domain Avg Accuracy | ECE ↓ | Trainable Parameters | CUDA Memory |
|---|---:|---:|---:|---:|---:|
| Full fine-tuning | **0.9289 ± 0.0050** | **0.8720 ± 0.0040** | 0.0468 | 109.48M | 2365 MB |
| LoRA r=8 | 0.9037 ± 0.0020 | 0.8610 ± 0.0012 | **0.0142** | 1.34M | 1751 MB |
| Prompt-tuning | 0.5535 ± 0.0614 | 0.5170 ± 0.0257 | 0.0941 | **16.9K** | **1297 MB** |

Full fine-tuning achieved the highest in-domain and cross-domain accuracy because it updates the entire BERT model. It reached the best SST-2 development performance and also the best average performance across Yelp, IMDB, and Amazon.

LoRA achieved slightly lower accuracy than full fine-tuning, but it required far fewer trainable parameters. Compared with full fine-tuning, LoRA was only about:

```text
2.5 percentage points lower on SST-2 dev accuracy
1.1 percentage points lower on average cross-domain accuracy
```

while training about:

```text
81.6× fewer parameters than full fine-tuning
```

This makes LoRA the strongest parameter-efficient method in this experiment.

Prompt-tuning was the most parameter-efficient method, training only about `16.9K` parameters. However, its performance remained weak and close to chance level on several datasets. Therefore, prompt-tuning did not provide a competitive accuracy-efficiency trade-off under the current configuration.

This baseline prompt-tuning result should be interpreted only under the shared training setup. Later exploratory prompt-tuning runs show that prompt-tuning improves substantially when using a prompt-specific learning rate and longer training.

---

## 2. Cross-Domain Generalization

Cross-domain generalization was measured by evaluating each model on Yelp, IMDB, and Amazon after training only on SST-2.

The cross-domain average accuracy was computed as:

```text
cross_domain_avg_accuracy = mean(Yelp accuracy, IMDB accuracy, Amazon accuracy)
```

The generalization gap was computed as:

```text
generalization_gap = SST-2 dev accuracy - cross-domain average accuracy
```

| Method | SST-2 Dev Accuracy | Cross-Domain Avg Accuracy | Generalization Gap ↓ |
|---|---:|---:|---:|
| Full fine-tuning | 0.9289 | **0.8720** | 0.0569 |
| LoRA r=8 | 0.9037 | 0.8610 | **0.0427** |
| Prompt-tuning | 0.5535 | 0.5170 | 0.0365 |

Full fine-tuning achieved the highest absolute cross-domain accuracy. However, LoRA had a smaller generalization gap than full fine-tuning, meaning that LoRA retained more of its SST-2 performance when transferred to unseen review domains.

The prompt-tuning gap is not very meaningful because its base SST-2 performance was already low. Therefore, the main useful comparison is between full fine-tuning and LoRA.

The exploratory tuned prompt runs are discussed separately because they use a different hyperparameter regime. They improve cross-domain transfer substantially compared with baseline prompt-tuning, but they still remain below LoRA and full fine-tuning.

Across the cross-domain datasets, IMDB was the hardest dataset. Yelp and Amazon results were closer to each other, while IMDB accuracy was consistently lower for both full fine-tuning and LoRA. This is likely because IMDB reviews are longer and more narrative than SST-2 sentences, while Yelp and Amazon reviews are more directly sentiment-oriented.

---

## 3. Robustness to Input Perturbations

Robustness was evaluated on the SST-2 development set using three perturbation types:

- Punctuation perturbation
- Character-level noise
- Synonym replacement

The robustness drop was calculated as:

```text
robustness_drop = clean SST-2 dev accuracy - perturbed accuracy
```

| Method | Punctuation Drop ↓ | Character Noise Drop ↓ | Synonym Drop ↓ |
|---|---:|---:|---:|
| Full fine-tuning | 0.0199 | 0.0447 | 0.0004 |
| LoRA r=8 | **0.0092** | **0.0329** | 0.0000 |
| Prompt-tuning | -0.0050 | -0.0065 | 0.0000 |

For the meaningful comparison between full fine-tuning and LoRA, LoRA showed smaller drops under both punctuation perturbation and character noise. This suggests that LoRA was more robust relative to its clean performance.

Character noise caused the largest drop for both full fine-tuning and LoRA. This is expected because character-level corruption can interfere with BERT tokenization and produce less reliable subword representations.

Synonym perturbation caused almost no performance change. This suggests that the implemented synonym replacement was mild or affected only a small portion of tokens. Therefore, synonym robustness should be interpreted cautiously.

Prompt-tuning robustness is less meaningful because the clean prompt-tuning performance was already weak. In some cases, the perturbed score was slightly higher than the clean score, which does not indicate true robustness; rather, it reflects instability and weak learning.

---

## 4. Confidence Calibration

Confidence calibration was measured using Expected Calibration Error (ECE) on the SST-2 development set after temperature scaling.

Lower ECE indicates better calibration.

| Method | Mean ECE ↓ |
|---|---:|
| LoRA r=8 | **0.0142** |
| Full fine-tuning | 0.0468 |
| Prompt-tuning | 0.0941 |

LoRA achieved the best calibration among the three methods. This is an important result because it means LoRA’s confidence estimates were more reliable, even though full fine-tuning achieved higher raw accuracy.

Full fine-tuning had higher accuracy but worse calibration than LoRA. Its learned temperature values were around `4.6–4.8`, suggesting that the full fine-tuned models were more overconfident and required stronger softening during temperature scaling.

Prompt-tuning had the weakest calibration overall, which is consistent with its unstable and low classification performance.

The exploratory tuned prompt runs also improved calibration compared with baseline prompt-tuning. Baseline prompt-tuning had ECE `0.0941`, while tuned prompt `vt=20` reduced ECE to `0.0277` and tuned prompt `vt=30` reduced it further to `0.0234`. However, LoRA still had the best calibration overall among the main methods with ECE `0.0142`. These tuned prompt calibration results are exploratory because they come from a prompt-specific hyperparameter regime rather than the shared baseline setup.

---

## 5. Computational Efficiency

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

LoRA trained only about `1.34M` parameters, which is approximately `81.6×` fewer than full fine-tuning. It also used less CUDA memory than full fine-tuning while maintaining competitive accuracy and better calibration.

Prompt-tuning used the fewest trainable parameters and the lowest memory, but the performance was too weak to make it competitive in this experiment.

The exploratory tuned prompt runs remained extremely lightweight. The `vt=20` tuned prompt setting used about `16.9K` trainable parameters, while `vt=30` used about `24.6K` trainable parameters. Even the `vt=30` setting trained far fewer parameters than LoRA's `1.34M` trainable parameters. However, the tuned prompt methods still had lower accuracy than LoRA, so they are best interpreted as ultra-lightweight alternatives rather than the strongest PEFT method.

One interesting observation is that LoRA did not train faster than full fine-tuning in wall-clock time. This can happen because PEFT wrappers introduce implementation overhead, and runtime is not determined only by the number of trainable parameters. Therefore, LoRA’s main efficiency advantage in this notebook is:

```text
far fewer trainable parameters
lower memory usage
competitive accuracy
better calibration
```

rather than faster training time.

---

## 6. LoRA Ablation Analysis

LoRA ablations were used to study the effect of:

- LoRA rank: `r=8` vs `r=16`
- Learning rate: `3e-5` vs `2e-5`

| LoRA Setting | SST-2 Dev Accuracy | Cross-Domain Avg Accuracy | ECE ↓ | Trainable Parameters |
|---|---:|---:|---:|---:|
| r=8, lr=3e-5 | 0.9037 | 0.8610 | **0.0142** | 1.34M |
| r=8, lr=2e-5 | 0.8972 | 0.8532 | 0.0188 | 1.34M |
| r=16, lr=2e-5 | 0.9006 | 0.8592 | 0.0151 | 2.68M |
| r=16, lr=3e-5 | **0.9052** | **0.8617** | 0.0204 | 2.68M |

### Rank 8 vs Rank 16

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

### Learning Rate 2e-5 vs 3e-5

For LoRA, the learning rate `3e-5` performed better than `2e-5`.

The setting `r=8, lr=2e-5` performed worse than `r=8, lr=3e-5` on both SST-2 dev accuracy and cross-domain average accuracy. This suggests that the lower learning rate under-trained the LoRA adapters in this setup.

Overall, the original LoRA baseline setting was already a strong choice:

```text
LoRA r=8, lr=3e-5
```

---

## 7. Exploratory Prompt-Tuning Follow-Up

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

### Exploratory Prompt-Tuning Results

| Method | SST-2 Dev Accuracy | Cross-Domain Avg Accuracy | ECE ↓ | Trainable Parameters | CUDA Memory |
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

### Virtual Tokens 20 vs 30

The `vt=30` setting performed slightly better than `vt=20`.

| Setting | SST-2 Dev Accuracy | Cross-Domain Avg Accuracy | ECE ↓ |
|---|---:|---:|---:|
| vt=20 | 0.8486 | 0.7804 | 0.0277 |
| vt=30 | **0.8490** | **0.7974** | **0.0234** |

The SST-2 development accuracy was almost the same for both settings. However, `vt=30` gave better cross-domain performance, improving the cross-domain average by about `1.7` percentage points over `vt=20`.

This suggests that increasing the number of virtual tokens gave the prompt more capacity, which helped transfer to Yelp, IMDB, and Amazon. However, it also increased the number of trainable parameters from about `16.9K` to about `24.6K`.

### Tuned Prompt-Tuning vs LoRA

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

### Interpretation

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

---

## 8. Conceptual Interpretation of the Methods

### Full Fine-Tuning

Full fine-tuning updates all or most of BERT’s parameters. This gives the model the maximum flexibility to adapt to SST-2 sentiment classification.

```text
Full fine-tuning = change the model itself
```

This explains why full fine-tuning achieved the highest overall accuracy.

### LoRA

LoRA freezes the original BERT weights and adds small trainable low-rank matrices inside the model’s linear layers. These adapters allow the model to adjust internal representations without updating all BERT parameters.

```text
LoRA = add small trainable internal updates
```

This explains why LoRA performed close to full fine-tuning while training far fewer parameters.

### Prompt-Tuning

Prompt-tuning freezes BERT and learns a small set of virtual prompt tokens at the input level. It does not modify the internal transformer layers.

```text
Prompt-tuning = add trainable input prompts
```

This explains why prompt-tuning was weak under the shared baseline setup: it had less control over the internal model layers and required a more suitable prompt-specific training configuration. The exploratory tuned prompt runs showed that prompt-tuning can improve substantially with a higher learning rate, longer training, and more virtual tokens, although it still remained below LoRA and full fine-tuning.

---

## 9. Accuracy and Macro-F1

The notebook reports both accuracy and macro-F1.

Macro-F1 is computed separately from accuracy using scikit-learn’s macro-F1 calculation. It is not copied from accuracy.

The values are similar for full fine-tuning and LoRA because the datasets are binary and mostly balanced, and the stronger models perform similarly on both sentiment classes. In this situation, accuracy and macro-F1 naturally become very close.

However, macro-F1 is still useful because it reveals class imbalance or class-specific weakness. For example, prompt-tuning sometimes has accuracy around `0.51`, but macro-F1 around `0.35`, showing that it is not learning both classes well.

Micro-F1 was not used because, for single-label binary classification, micro-F1 is usually almost identical to accuracy. Since accuracy is already reported, micro-F1 would add little additional information. Macro-F1 is more informative for this comparison.

---

## 10. Notes on Synonym Perturbation

The synonym perturbation produced negligible performance changes for all methods.

This likely means that the synonym replacement function was mild, replaced only a small number of words, or often returned the original sentence. Therefore, synonym robustness should not be over-interpreted.

A suitable limitation statement is:

```text
Synonym perturbation caused negligible change, suggesting that the implemented synonym replacement was mild or affected only a small portion of tokens. Future work could use stronger semantic perturbation methods or controlled paraphrase generation.
```

---

## 11. Final Conclusion

Overall, full fine-tuning produced the best raw accuracy, but LoRA provided the best performance-efficiency trade-off.

Full fine-tuning achieved the highest in-domain and cross-domain accuracy, but required updating all `109.48M` BERT parameters. LoRA achieved competitive performance with about `81.6×` fewer trainable parameters, lower GPU memory usage, better calibration, and smaller robustness drops under perturbations.

Baseline prompt-tuning was the most parameter-efficient main method, training only `16.9K` parameters, but it performed weakly under the shared baseline configuration. However, the exploratory tuned prompt runs showed that prompt-tuning can improve substantially with prompt-specific hyperparameters. The best tuned prompt setting, `vt=30`, reached `0.8490` SST-2 dev accuracy and `0.7974` cross-domain average accuracy while training only `24.6K` parameters. Despite this improvement, tuned prompt-tuning still remained below LoRA and full fine-tuning in accuracy.

The LoRA ablation results showed that increasing rank from `8` to `16` only slightly improved accuracy while doubling the number of trainable parameters. Therefore, LoRA with rank `8` offered the better efficiency balance in this experiment.

The final takeaway is:

```text
Full fine-tuning is best for maximum accuracy.
LoRA is best for the overall performance-efficiency trade-off.
Baseline prompt-tuning is weakest under the shared setup.
Tuned prompt-tuning shows that prompt methods are highly hyperparameter-sensitive and can become a useful ultra-lightweight alternative, but still remain below LoRA in this experiment.
```
