# Reproduction and implementation notes

## Scope

This repository provides a clean research implementation of the DMA-PULSE pipeline described in the published paper. It is intended to make the architecture and experimental workflow understandable and executable without distributing the original PULSE dataset.

The repository is **not the original authors' experimental codebase**, and the current implementation should not be described as an exact reproduction of the published benchmark.

## Published experimental setup

- 14,010 synthetic PULSE activity sequences
- 5 sequential time steps
- 11 final feature dimensions
- binary and four-class evaluation
- minority oversampling and majority undersampling
- 80:20 train-test split
- self-attention Transformer
- sequential AOA global search followed by RFO local refinement

## Local dataset format

Training utilities accept a NumPy `.npz` file containing `X` shaped `[N, 5, 11]` and `y` shaped `[N]`. The original PULSE data should not be committed unless distribution rights and provenance are established.

## Reproducing the published numbers

To claim an independent reproduction, record exact dataset provenance, preprocessing, split, seeds, Transformer hyperparameters, AOA/RFO configuration, training budget, evaluation script, and generated metrics. Until then, results generated here should be reported as **repository experiments**, not reproduced paper results.
