CS5143 – Natural Language Processing
Spring 2026 – Class Project Proposal

Title: Robust Cross‑Domain Sentiment Analysis with BERT: PEFT vs Full Fine‑Tuning

Group Members
- Lead: <Full Name> — <Student ID>
- Member: <Full Name> — <Student ID>

1) Project Scope, Problem, and Motivation
Sentiment models trained on one domain (e.g., movie reviews) often degrade on other domains (e.g., product or restaurant reviews). This project studies cross‑domain generalization for sentiment analysis using modern transfer techniques. We will fine‑tune a pre‑trained BERT model and compare full fine‑tuning to parameter‑efficient fine‑tuning (PEFT, e.g., LoRA and prompt‑tuning). We will evaluate robustness under domain shift, simple text perturbations, and assess calibration quality.

Goal: Achieve competitive in‑domain accuracy while improving out‑of‑domain performance/cost trade‑offs and providing a clear analysis of robustness and confidence calibration.

2) Seminal Paper, Datasets, and Reference Code
- Seminal paper: Devlin, J., Chang, M.‑W., Lee, K., & Toutanova, K. (2019). BERT: Pre‑training of Deep Bidirectional Transformers for Language Understanding. NAACL‑HLT 2019. [arXiv](`https://arxiv.org/abs/1810.04805`)
- Primary train/eval dataset (in‑domain): SST‑2 (GLUE). [Dataset card](`https://huggingface.co/datasets/glue/viewer/sst2`) • [GLUE site](`https://gluebenchmark.com/`)
- Cross‑domain evaluation sets (test‑only, no label leakage):
  - Yelp Polarity (subset for eval) — reviews domain. [HF](`https://huggingface.co/datasets/yelp_polarity`)
  - IMDB (binary) — movies domain. [HF](`https://huggingface.co/datasets/imdb`)
  - Amazon Reviews (polarity subset) — products domain. [HF](`https://huggingface.co/datasets/amazon_polarity`)
- Reference GitHub: HF Transformers text‑classification examples (run_glue). [Repo](`https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification`)

3) Methods and Implementation Details
- Models: `bert-base-uncased` with classification head (num_labels=2)
- Fine‑tuning regimes:
  - Full fine‑tuning (all parameters)
  - PEFT: LoRA (r ∈ {8, 16}; α tuned), and prompt‑tuning (soft prompts)
- Tokenization: WordPiece (uncased), max_seq_length 128
- Optimization: AdamW, linear warmup/decay; primary lr = 3e‑5 (ablation compares 2e‑5 vs 3e‑5); weight decay 0.01; gradient clipping
- Training schedule: batch size tuned to VRAM (target 16), baseline/max epochs = 3 (training may stop earlier via patience-based early stopping on SST‑2 dev with best-checkpoint restore)
- Robustness tests: synonym replacement, random character noise, punctuation/no‑punct variants
- Calibration: expected calibration error (ECE), reliability diagrams; temperature scaling on dev
- Evaluation metrics:
  - In‑domain (SST‑2 dev/test): accuracy (primary), F1 macro (secondary)
  - Cross‑domain (Yelp/IMDB/Amazon held‑out): accuracy/F1; report delta vs in‑domain
  - Cost metrics: trainable parameter count, wall‑clock time/epoch, and VRAM footprint
- Reproducibility: 3 fixed seeds (`7, 42, 2026`); log package versions; self‑contained notebook with thorough markdown

Tools and Environment
- Python 3.10+; PyTorch; Transformers; Datasets; Accelerate; PEFT; scikit‑learn; matplotlib/seaborn
- Optional: Weights & Biases for experiment tracking (if permitted)

Notebook Structure (Planned)
1. Setup and environment validation
2. Load SST‑2; exploratory analysis (class balance, length distribution)
3. Tokenization and dataloaders (SST‑2)
4. Baseline training (full fine‑tune) with dev eval
5. PEFT implementations (LoRA, prompt‑tuning) and training
6. Cross‑domain evaluation on Yelp/IMDB/Amazon (no training on these)
7. Robustness: perturbation tests; compare sensitivity across methods
8. Calibration: ECE, reliability diagrams; temperature scaling
9. Targeted ablations: LoRA rank (`r=8` vs `r=16`) and one optimization setting (learning rate `2e‑5` vs `3e‑5`) vs performance/cost
10. Conclusions and limitations

4) Expected Results and How We Will Validate Them
- We expect PEFT to approach full fine‑tuning’s in‑domain accuracy while using far fewer trainable parameters and lower VRAM, with comparable or slightly better cross‑domain robustness.
- We validate via: (i) matching or closely approaching strong SST‑2 baselines; (ii) reporting mean ± std over seeds; (iii) demonstrating improved cost‑performance trade‑offs; and (iv) quantitative robustness and calibration assessments.

5) Deliverables
- A well‑commented Python notebook implementing full and PEFT training, cross‑domain/robustness evaluation, and calibration
- Trained model artifacts (adapters for PEFT), logs, and plots (learning curves, reliability diagrams)
- A concise results write‑up inside the notebook, highlighting trade‑offs and key findings

6) Timeline (2‑Person Team, 4 Weeks + buffer)
- Week 1: Environment, SST‑2 baseline full fine‑tune; set evaluation harness
- Week 2: Implement PEFT (LoRA, prompt‑tuning); run controlled sweeps; pick best configs
- Week 3: Cross‑domain evaluation; robustness perturbations; calibration analysis
- Week 4: Error analysis; ablations; finalize documentation and figures
- Buffer: Re‑runs for reproducibility; polish and submission

7) Risks and Mitigations
- Compute limits: Use FP16, batch 16–32, gradient accumulation as needed; PEFT reduces memory/compute
- Domain mismatch: Avoid any training on out‑of‑domain sets; ensure clean test‑only usage
- Variance across seeds: Use 3 seeds and report mean ± std; fix seeds consistently

8) References
- Devlin et al., 2019 (BERT): [arXiv](`https://arxiv.org/abs/1810.04805`)
- HF Transformers text‑classification examples: [GitHub](`https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification`)
- SST‑2: [HF Datasets](`https://huggingface.co/datasets/glue/viewer/sst2`) • Yelp Polarity: [HF](`https://huggingface.co/datasets/yelp_polarity`) • IMDB: [HF](`https://huggingface.co/datasets/imdb`) • Amazon Polarity: [HF](`https://huggingface.co/datasets/amazon_polarity`)


Submission Note
Per instructions: only the group lead submits at the GCR site. Fill in names/IDs above and upload the notebook and proposal as required.
