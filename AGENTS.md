# Agent Instructions

This repository is a research-grade Python project for Parkinson's freezing-of-gait detection from wearable accelerometer time-series.

## Core Goal

Build a reproducible baseline pipeline for detecting Parkinson's freezing-of-gait events from wearable accelerometer signals, including preprocessing, windowing, feature extraction, subject-aware validation, class-imbalance analysis, calibration metrics, and cautious non-clinical digital-biomarker interpretation.

## Non-Negotiable Rules

1. Do not move to the next milestone unless all tests pass.
2. Every new source module must have tests.
3. Never use random train/test split as the main validation strategy.
4. The main validation strategy must be subject-aware using GroupKFold or Leave-One-Subject-Out.
5. Prevent subject leakage between training and test sets.
6. Do not commit raw data or processed data.
7. Keep all clinical language cautious and non-diagnostic.
8. All pipeline steps must be reproducible from the command line.
9. Before every commit, run:
   - ruff check .
   - pytest
10. If any quality gate fails, fix it before continuing.
11. Do not push directly to main.
12. Use a feature branch for agent-generated work.

## Project Standards

- Use Python 3.11+.
- Prefer simple, interpretable baselines before deep learning.
- Use deterministic random seeds where applicable.
- Prefer clear functions with type hints.
- Avoid hidden state and notebook-only logic.
- Keep notebooks exploratory only; production logic belongs in src/ and scripts/.

## Validation Requirements

The project must include tests for:

- data schema validation
- windowing correctness
- overlap behavior
- subject-aware splitting
- leakage prevention
- feature extraction output shape
- NaN handling
- metric correctness
- calibration metric behavior

## Interpretation Boundary

This project is not a diagnostic medical device and must not claim clinical readiness. It evaluates whether wearable-sensor models can provide reproducible, calibrated, subject-aware digital-biomarker candidates for Parkinson's freezing-of-gait detection.
