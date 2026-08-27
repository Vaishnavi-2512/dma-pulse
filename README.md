# DMA-PULSE

### Scenario-Driven Insider Threat Detection Using Dual Modelling Architecture

> Collaborative research project by Pullela Vaishnavi and Pullela Giridhar.

DMA-PULSE is a research implementation of a **Dual Modelling Architecture (DMA)** for scenario-driven insider-threat detection from temporal user-behavior sequences. The framework combines a self-attention Transformer with sequential global-to-local metaheuristic optimization using the Archimedes Optimization Algorithm (AOA) and Red Fox Optimization (RFO).

## Overview

The pipeline operates on the PULSE (Profile-based User Logs for Synthetic Evaluation) scenario-driven dataset and supports both binary insider-threat detection and multiclass discrimination across four threat scenarios.

```mermaid
flowchart LR
    A[PULSE activity sequences] --> B[Preprocessing]
    B --> C[Temporal representation: 5 x 11]
    C --> D[Self-Attention Transformer]
    D --> E[AOA global search]
    E --> F[RFO local refinement]
    F --> G[Optimized classifier]
    G --> H[Binary / Multiclass evaluation]
```

## Technical Components

- **Temporal behavioral modeling:** self-attention Transformer for sequential user-activity representations.
- **Global-to-local optimization:** AOA for coarse exploration followed by RFO for local refinement of sensitive model hyperparameters.
- **Scenario-driven detection:** binary benign/insider classification and multiclass classification across four activity scenarios.
- **Research implementation:** compact PyTorch modules for preprocessing, modeling, optimization, training, and testing.

## Repository Structure

```text
src/dma_pulse/
├── data.py          # Data preparation and preprocessing
├── model.py         # Transformer architecture
└── optimizers.py    # AOA/RFO optimization components
experiments/
└── train.py         # Training entry point
tests/
├── test_model.py
└── test_optimizers.py
docs/
└── REPRODUCIBILITY.md
figures/
└── architecture.mmd
```

## Reported Results

The following results are **reported in the associated publication** and are not presented as newly reproduced benchmarks by this repository:

| Evaluation | Reported result |
|---|---:|
| Binary accuracy | **97.62%** |
| Binary precision | **98.84%** |
| Binary recall / detection rate | **90.90%** |
| Binary false-positive rate | **0.33%** |
| Binary false-negative rate | **9.10%** |
| Multiclass accuracy | **97.67%** |
| Multiclass precision | **97.70%** |
| Multiclass recall / detection rate | **97.67%** |

## Dataset

The published workflow uses **14,010 synthetic activity sequences**, represented as five sequential time steps with 11 feature dimensions. The raw dataset is not included in this repository.

## Research Scope

DMA-PULSE sits at the intersection of:

- Insider-threat detection
- Behavioral cybersecurity analytics
- Temporal sequence modeling
- Self-attention Transformers
- Metaheuristic optimization
- Scenario-driven synthetic security data

## Reproducibility and Scope

The repository is intended as a transparent engineering/research implementation. The optimizer implementation is documented as an engineering realization of the described global-to-local search strategy rather than a claim of exact reproduction of unpublished implementation details. See `docs/REPRODUCIBILITY.md` for setup and experiment guidance.

## Publication

**Scenario Driven Insider Threat Detection Using Dual Modelling Architecture**  
P. Lavanya, Pullela Vaishnavi, Pullela Giridhar, H. Anila Glory, V. S. Shankar Sriram.

*Artificial Intelligence and Sustainable Computing — Proceedings of ICSISCET 2025, Volume 2*, Springer Lecture Notes in Networks and Systems, vol. 1938, pp. 129–143, 2026.

DOI: `10.1007/978-3-032-23945-7_11`

## Attribution

This is a **collaborative project**. The repository is maintained under Pullela Vaishnavi's GitHub account to document her contribution and provide a reproducible public portfolio artifact. Original collaborative work and publication authorship are retained in the repository metadata and citation information.

## License

See the repository files for licensing and citation metadata.
