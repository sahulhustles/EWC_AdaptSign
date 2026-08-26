# AdaptSign — Lifelong Learning for Indian Traffic Sign Recognition

**Phase 1 | Sem 6 | Sri Krishna College of Technology**

> A traffic sign recognition system that learns new weather conditions without forgetting old ones.
> Built specifically for Indian roads.

---

## Project Structure

```
adaptsign/
├── data/
│   ├── download_gtsrb.py     # Dataset loader (auto-downloads GTSRB)
│   └── augment.py            # Weather augmentation (Rain/Fog/Night/Dust)
├── models/
│   ├── backbone.py           # ResNet18 backbone
│   ├── ewc.py                # EWC implementation (core innovation)
│   ├── drift_detector.py     # Fisher Drift Detector (YOUR innovation)
│   └── confidence_gate.py   # Dharma Gate (IKS: ethical prediction)
├── train/
│   └── train_adaptsign.py   # Full training pipeline
├── evaluate/
│   └── metrics.py            # Lifelong learning metrics
├── demo/
│   └── app.py               # Streamlit interactive demo
├── results/                  # Saved models and plots
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train/train_adaptsign.py
```

This will:
- Download GTSRB dataset automatically (~300MB)
- Train BASELINE model (proves forgetting problem)
- Train AdaptSign WITH EWC (proves your solution)
- Save `comparison_plot.png` to `./results/`

### 3. Run interactive demo
```bash
streamlit run demo/app.py
```

---

## Key Files to Understand

| File | What It Does | IKS Principle |
|------|-------------|---------------|
| `models/ewc.py` | Protects important weights | Gurukul |
| `models/drift_detector.py` | Detects domain shift | Viveka |
| `models/confidence_gate.py` | Refuses uncertain predictions | Dharma |
| `data/augment.py` | Weather simulation | Panchang |
| `data/download_gtsrb.py` | Indian sign extension | Lok Vigyan |

---

## Metrics to Report

| Metric | Target | Meaning |
|--------|--------|---------|
| Accuracy per task | >90% | Correct sign recognition |
| Forgetting Score | <10% | Accuracy drop on old tasks |
| BWT | >-0.10 | Backward transfer |
| Inference Time | <50ms | Real-time on Raspberry Pi |

---

## Google Colab (Recommended for Training)

```python
!git clone <your-repo-url>
%cd adaptsign
!pip install -r requirements.txt
!python train/train_adaptsign.py
```

---

## IKS Integration

- **Gurukul** — EWC retains old knowledge while learning new tasks
- **Viveka** — Drift Detector decides WHEN to protect knowledge
- **Dharma** — Confidence Gate refuses uncertain predictions
- **Lok Vigyan** — Extended with 8 Indian-specific sign classes
- **Panchang** — Weather augmentation for all Indian seasons

---

## SDG Alignment

- **SDG 3** — Reduce road deaths (4.6 lakh/year in India)
- **SDG 9** — Indigenous AI for Atmanirbhar Bharat
- **SDG 11** — Safer smart city transportation

---

## Phase Roadmap

| Semester | Feature | Status |
|----------|---------|--------|
| Sem 6 | EWC + Drift Detector | This code |
| Sem 7 | + Generative Replay for rare signs | Next |
| Sem 8 | + MAML Meta-learning for regions | Final |

---

*Sri Krishna College of Technology | CSE (IoT) | Team 9*
