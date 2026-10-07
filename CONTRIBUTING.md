# Contributing to Parkinson Wearable Digital Biomarkers

Thank you for your interest in contributing to this research baseline! We welcome contributions that improve reproducibility, documentation, testing, and scientific rigor.

## Code of Conduct

This project is committed to fostering an inclusive community. Please treat all contributors with respect and refer to common open-source codes of conduct.

## What We Accept

We welcome contributions in these areas:

### 🔬 **Research & Methods**
- New feature engineering approaches with proper validation
- Improved subject-aware validation strategies
- Additional performance metrics or calibration methods
- Documentation of dataset characteristics or limitations
- Reproducibility improvements

### 📝 **Documentation**
- Clarifications to README or API documentation
- Examples of using the library with custom data
- Additions to methodology or ethical considerations
- Improved comments in complex code sections

### 🧪 **Testing & Quality**
- Additional unit or integration tests
- Edge-case coverage improvements
- Performance or memory usage optimizations
- Type hints and documentation strings

### 🐛 **Bug Fixes**
- Fixes for reported issues or edge cases
- Improvements to error messages and validation

## What We Don't Accept

- **Clinical claims or diagnostic applications**: This is research-grade software, not a medical device
- **Removal of safety warnings**: We maintain cautions about FoG detection limitations
- **Modifications that weaken subject-aware validation**: Subject leakage prevention is non-negotiable
- **Undocumented or untested code**: All changes require tests and clear explanation

## Getting Started

1. **Fork and clone** the repository:
   ```bash
   git clone https://github.com/your-username/parkinson-wearable-digital-biomarkers.git
   cd parkinson-wearable-digital-biomarkers
   ```

2. **Set up development environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install -e .
   ```

3. **Run tests and quality checks**:
   ```bash
   python -m pytest
   python -m ruff check .
   python -m ruff format .
   python scripts/smoke_test.py
   ```

## Pull Request Process

1. **Create a descriptive branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear commits:
   ```bash
   git commit -m "Add feature: description of what and why"
   ```

3. **Ensure all tests pass**:
   ```bash
   python -m pytest --verbose
   ```

4. **Format and lint your code**:
   ```bash
   python -m ruff format .
   python -m ruff check . --fix
   ```

5. **Update documentation** if needed (README, MODEL_CARD, docstrings)

6. **Push and create a Pull Request** with:
   - Clear title describing the change
   - Description of what changed and why
   - Reference to any related issues
   - Confirmation that tests pass
   - For research changes: reference to methodology or literature

## PR Review Guidelines

All PRs will be reviewed for:

- ✅ **Correctness**: Does it work as intended? Are edge cases handled?
- ✅ **Testing**: Are there tests? Do they pass?
- ✅ **Documentation**: Are changes documented clearly?
- ✅ **Reproducibility**: Are random seeds set? Are results deterministic?
- ✅ **Subject Safety**: Does it maintain subject-aware validation integrity?
- ✅ **Code Quality**: Does it follow the project's style and patterns?
- ✅ **Research Ethics**: Is the non-diagnostic boundary respected?

## Reporting Issues

Please use GitHub Issues to report:

- **Bugs**: Include Python version, error message, minimal reproducible example
- **Questions**: Ask about methodology, usage, or scientific design
- **Feature requests**: Describe what you'd like and why (with reference to research needs if applicable)

## Development Notes

### Project Structure

```
src/parkinson_wearable_biomarkers/
  ├── data.py          # CSV schema validation and loading
  ├── preprocessing.py # Subject-safe windowing
  ├── features.py      # Signal feature extraction
  ├── validation.py    # GroupKFold and LOSO splitting
  ├── models.py        # Class-weighted classifiers
  ├── evaluate.py      # Discrimination and threshold metrics
  └── calibration.py   # Probability calibration assessment
```

### Key Principles

1. **Subject-aware validation is mandatory**: Windows from the same subject must not split between train/test
2. **Reproducibility via seeds**: All random operations must respect `random_state` parameters
3. **Interpretability first**: Handcrafted features and logistic regression preferred over black-box methods
4. **Class imbalance awareness**: Always report AUPRC, class counts, and prevalence
5. **Non-diagnostic boundary**: Never remove or weaken warnings about research-use limitations

### Testing

Add tests for:
- **Unit tests**: Individual functions with synthetic data
- **Integration tests**: End-to-end pipelines (see `test_scripts.py`)
- **Determinism**: Verify results are identical with the same seed
- **Edge cases**: Empty data, single subject, perfect separation, etc.

Example test structure:
```python
def test_my_feature():
    # Setup
    data = create_synthetic_data()
    
    # Action
    result = my_function(data)
    
    # Assert
    assert result.shape[0] == expected_rows
    assert not np.isnan(result).any()
```

## Questions?

- **Research methodology**: Open an Issue to discuss approach
- **Usage questions**: Check README or existing Issues first
- **CI/CD issues**: See `.github/workflows/ci.yml` for current checks

Thank you for contributing to making this baseline more robust and transparent!
