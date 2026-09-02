# From Classification to Generation: A Comparative Evaluation of VQA Architectures

Visual Question Answering (VQA) is a multimodal task at the intersection of Computer Vision and Natural Language Processing: given an image *I* and a natural language question *Q*, a system must produce a natural language answer *A*.

This project empirically compares **one representative model per architectural generation** of the VQA literature — a CNN+LSTM baseline implemented from scratch, LXMERT, and BLIP — on two benchmarks, **VQA v2** and **GQA**, to study how architecture and training data jointly determine cross-dataset generalization.

Full write-up: [`report/acl2015.pdf`](report/acl2015.pdf) (ACL-style report, see [`report/acl2015.tex`](report/acl2015.tex) and [`report/sections/`](report/sections/) for source).

## Task

Given an image and a question, produce an answer. Answers fall into three types:

| Type | Example question | Example answer |
|---|---|---|
| Yes/No | "Is the dog sitting on the couch?" | "yes" |
| Number | "How many people are in the image?" | "3" |
| Other | "What color is the car?" | "red" |

## Models compared

| Generation | Model | Paradigm | VQA v2 checkpoint | GQA checkpoint |
|---|---|---|---|---|
| I — CNN+RNN fusion (2015-2018) | **Vanilla VQA** | closed-vocabulary classification | trained from scratch (`notebooks/vanilla-vqa.ipynb`) | trained from scratch |
| II — Cross-modal Transformers (2019-2021) | **LXMERT** | closed-vocabulary classification | `unc-nlp/lxmert-vqa-uncased` | `unc-nlp/lxmert-gqa-uncased` |
| III — Generative pretraining (2022-present) | **BLIP** | open-vocabulary generation | `ybelkada/blip-vqa-base` | same checkpoint (zero-shot, no GQA-specific model exists) |

**Vanilla VQA** (`notebooks/vanilla-vqa.ipynb`): images encoded offline with a frozen ResNet-101 (ImageNet-pretrained, 2048-d features), questions tokenized with NLTK and encoded with a 2-layer LSTM initialized with GloVe embeddings. Image and question features are projected to 1024-d (Linear + LayerNorm + Tanh), fused via Hadamard product, and classified over a fixed answer vocabulary (3129 answers for VQA v2, 1842 for GQA).

**LXMERT** (`notebooks/lxmert.ipynb`) and **BLIP** (`notebooks/blip.ipynb`) are evaluated zero-shot via their pretrained HuggingFace checkpoints, without any additional fine-tuning.

## Datasets

- **[VQA v2](https://visualqa.org/)** (primary benchmark): >1.1M questions on COCO images, 10 answers/question, soft accuracy metric.
- **[GQA](https://cs.stanford.edu/people/dorarad/gqa/)** (secondary benchmark): questions generated from scene graphs, used to test compositional reasoning and generalization beyond VQA v2 training.

## Results

**VQA v2** (% soft accuracy, test-dev):

| Model | Y/N | Number | Other | Overall |
|---|---|---|---|---|
| Vanilla | 75.23 | 35.70 | 46.03 | 56.88 |
| LXMERT | 86.76 | 52.95 | 60.17 | 70.30 |
| BLIP | 92.25 | 54.55 | 66.91 | **75.95** |

**GQA** (% exact-match accuracy, test-dev balanced):

| Model | Binary | Open | Overall |
|---|---|---|---|
| Vanilla | 64.46 | 37.95 | 47.49 |
| LXMERT | 77.17 | 49.25 | **59.29** |
| BLIP | 65.66 | 36.69 | 47.11 |

On VQA v2, accuracy improves monotonically with architectural generation. On GQA the ranking **reverses**: LXMERT (fine-tuned specifically on GQA) outperforms BLIP, whose open-vocabulary generation is applied zero-shot with no GQA-specific adaptation. This suggests that **dataset-specific training may contribute more to cross-dataset generalization than architectural generation alone** — see the report's Discussion section for the full analysis.

Raw evaluation outputs: [`results/`](results/).

## Repository structure

```
notebooks/        Model implementation, training and evaluation (Vanilla VQA, LXMERT, BLIP)
report/           ACL-style LaTeX report (source in sections/, compiled PDF)
results/          Accuracy metrics per model/dataset (JSON)
```

## Use of GenAI tools

Disclosed in full in the report ([`report/sections/genai.tex`](report/sections/genai.tex)): ChatGPT assisted with writing/debugging the model notebooks, Claude Code assisted with drafting and revising the report.
