CS5143 NLP Project - Rationale and Experiment Design Notes

Title
- Robust Cross-Domain Sentiment Analysis with BERT: PEFT vs Full Fine-Tuning

Purpose of this document
- Explain what we are doing in the project, why each component is included, and what we expect to learn.
- Provide a common technical understanding for the supervisor and team members.

Project motivation
- Sentiment models often perform well on in-domain data but degrade on unseen domains.
- In real applications, data distribution shifts are common (movies vs products vs restaurant reviews).
- We want to study both model quality and efficiency: not only accuracy, but also computational cost.

Main research questions
- Can PEFT methods (LoRA, prompt-tuning) approach full fine-tuning performance on SST-2?
- Which method generalizes better across domains (Yelp, IMDB, Amazon)?
- How robust are methods to input perturbations (punctuation changes, character noise, synonym swaps)?
- How calibrated are model confidences, and can temperature scaling improve reliability?
- What is the performance vs cost trade-off across methods and ablations?

Overall methodology
- Train on SST-2 (GLUE) and validate on SST-2 dev.
- Evaluate cross-domain generalization on Yelp/IMDB/Amazon.
- Compare three training regimes:
  - Full fine-tuning
  - LoRA-based PEFT
  - Prompt-tuning PEFT
- Apply early stopping on dev accuracy and restore best checkpoint.
- Log computational cost: training time, GPU memory, trainable parameter count.

Why each component is included

1) Full fine-tuning baseline
- Why: establishes a strong reference point.
- Benefit: helps interpret whether PEFT is competitive.
- Expected outcome: strong in-domain performance, potentially higher compute cost.

2) LoRA
- Why: reduces trainable parameters by adapting low-rank updates.
- Benefit: lower memory/computation, often strong transfer performance.
- Expected outcome: near-baseline in-domain accuracy with better efficiency.

3) Prompt-tuning
- Why: tests another PEFT family with minimal parameter updates.
- Benefit: very parameter-efficient adaptation.
- Expected outcome: useful comparison to LoRA and full fine-tuning; may trade some accuracy for efficiency.

4) Cross-domain evaluation (Yelp/IMDB/Amazon)
- Why: in-domain scores alone can be misleading for practical deployment.
- Benefit: measures robustness to distribution shift.
- Expected outcome: identify methods that are not only accurate on SST-2 but also transfer better.

5) Robustness perturbations
- Included perturbations:
  - punctuation removal
  - character-level noise
  - simple synonym replacement
- Why: simulate noisy or naturally varied user text.
- Benefit: reveals sensitivity of each method to small surface-form changes.
- Expected outcome: robustness ranking across methods and potential failure patterns.

6) Calibration (ECE + temperature scaling)
- Why: confidence quality matters for real decision systems.
- Benefit: better understanding of overconfidence/underconfidence behavior.
- Expected outcome: lower ECE after temperature scaling, with minimal impact on accuracy.

7) Early stopping + best-checkpoint restore
- Why: prevent overfitting and reduce unnecessary training.
- Benefit: more stable and reproducible final checkpoints.
- Expected outcome: efficient training and more reliable dev/generalization behavior.

8) Targeted ablations
- Ablations:
  - LoRA rank: r=8 vs r=16
  - Learning rate: 2e-5 vs 3e-5
- Why: quantify sensitivity to key controllable settings.
- Benefit: supports principled hyperparameter choice and better scientific reporting.
- Expected outcome: explicit performance/cost trade-offs, not just single-point results.

What we expect to achieve
- A reproducible notebook-based pipeline with clear experimental comparisons.
- Evidence-backed recommendation of method under compute constraints.
- A concise story of trade-offs:
  - quality (accuracy/F1, cross-domain behavior, robustness, calibration)
  - cost (train time, memory, trainable parameters)
- Results that are credible for course evaluation and useful for future extension.

Success criteria
- Competitive SST-2 dev performance across methods.
- Clear cross-domain comparison with interpretable differences.
- Completed robustness and calibration analysis.
- Ablation results summarized with both performance and cost metrics.
- Clean documentation and runnable notebook artifacts.

Potential risks and mitigations
- Runtime can be high for multi-seed + ablations.
  - Mitigation: sanity mode for quick checks, full mode for final runs.
- GPU memory constraints.
  - Mitigation: mixed precision, tuned batch size, PEFT methods.
- Variance across random seeds.
  - Mitigation: fixed multi-seed evaluation and mean/std reporting.

Expected deliverables for supervisor review
- `proposal.md` - project proposal aligned with implementation.
- `sentiment_cross_domain.ipynb` - full experiment notebook.
- `outputs/results.json` or `outputs/results_summary.json` - core metrics.
- `outputs/ablations.json` - ablation metrics for performance vs cost analysis.

