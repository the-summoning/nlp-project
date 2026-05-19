# NLP Homework as a Project


Visual Question Answering (VQA) is a multimodal task at the intersection of Computer Vision and Natural Language Processing. Given an image I and a natural language question Q, the system must produce a natural language answer A.

Answering a visual question correctly requires at minimum: (i) visual feature extraction (objects, attributes, spatial relations); (ii) natural language understanding; (iii) cross-modal alignment and reasoning. Examples:
- Type: Y/N, Q: Is the dog sitting on the couch? A: yes  (Object detection + spatial relation)
- Type: Counting, Q: How many people are in the image? A: 3 (Instance counting)

VQA has direct applications in several domains, from assistive technology (image-based question answering for visually impaired users) to medical imaging (automated query interfaces for radiology and pathology images).

The VQA literature can be organised into three architectural generations, reflecting the broader evolution of deep learning for multimodal understanding:
1) Era I — CNN + RNN models (2015–2018): Antol et al. (2015) introduced the VQA task and baseline, encoding images with VGGNet and questions with a LSTM, achieving  approximately 54% accuracy on VQA v2. Anderson et al. (2018) proposed Bottom-Up and Top-Down Attention, combining Faster R-CNN region proposals with a top-down attention mechanism, reaching ~65% and establishing the dominant pre-Transformer paradigm.

2) Era II — Transformer-based vision-language models (2019–2021): ViLBERT (Lu et al., 2019) introduced dual-stream Transformer encoders with cross-modal attention, achieving approximately 73% on VQA v2. Tan and Bansal (2019) proposed LXMERT with separate Transformers for object relationships, language, and cross-modality attention, reaching ~72% on VQA v2.

3) Era III — Large-scale pretrained vision-language models (2022–present): this era is characterised by large-scale multimodal pretraining on image-text pairs, which significantly improves cross-modal alignment without relying solely on task-specific supervision. Among the models in this generation, BLIP represents a well-suited candidate for this project: it achieves competitive performance while remaining reproducible.


Main reference paper: Visual Question Answering: A Survey of Methods, Datasets, Evaluation, and Challenges (ACM Computing Surveys, 2025)


Datasets considered in this project: VQA v2 (primary benchmark), GQA (secondary benchmark, to evaluate generalization beyond the training distribution). Evaluation will be conducted using the standard VQA accuracy metrics, with a possible additional breakdown by question type (yes/no, counting, other). I would like also to include random and majority baselines.

Models (expected performance based on reported results in the literature):
1) Baseline: Vanilla VQA (VGGNet/ResNet + 2-layer LSTM). Implemented from scratch. Expected VQA v2 acc.: ~54%.
2) Intermediate: LXMERT. Usage of HuggingFace model 'unc-nlp/lxmert-vqa-uncased'. Expected VQA v2 acc.: ~72%.
3) SOTA: BLIP (ViT-B/L + BERT encoder-decoder). Usage of HuggingFace model 'Salesforce/blip-vqa-base'. Expected VQA v2 acc.: ~78%.

Vanilla VQA and LXMERT operate under a closed-vocabulary classification paradigm, producing answers via argmax over a fixed answer set. BLIP (Salesforce/blip-vqa-base) instead produces answers via token-by-token generation with no fixed vocabulary. 