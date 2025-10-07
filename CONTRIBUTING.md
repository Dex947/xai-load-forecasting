# Contributing to XAI Load Forecasting

Thank you for your interest in contributing to the XAI Load Forecasting project! This document provides guidelines for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

---

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive

---

## Getting Started

1. **Fork the repository** on GitHub: https://github.com/Dex947/xai-load-forecasting/fork
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/xai-load-forecasting.git
   cd xai-load-forecasting
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

{{ ... }}

---

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, package versions)
- Error messages or logs

### Suggesting Enhancements

For feature requests or enhancements:
- Describe the feature and its benefits
- Provide use cases
- Suggest implementation approach if possible

### Code Contributions

We welcome contributions in these areas:
- **Bug fixes**
- **New features** (feature engineering, models, explainability methods)
- **Performance improvements**
- **Documentation improvements**
- **Test coverage**
- **Examples and tutorials**

---

## Development Setup

### Prerequisites

- Python 3.9 or higher
- pip or conda

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install development dependencies:
   ```bash
   pip install pytest pytest-cov black flake8 mypy
   ```

---

## Coding Standards

### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **type hints** for function signatures
- Write **comprehensive docstrings** (Google style)
- Keep functions focused and modular
- Maximum line length: 100 characters

### Code Formatting

Use `black` for automatic formatting:
```bash
black src/ scripts/
```

### Linting

Run `flake8` to check code quality:
```bash
flake8 src/ scripts/
```

### Type Checking

Use `mypy` for type checking:
```bash
mypy src/
```

### Example Code Style

```python
def calculate_feature_importance(
    model: GradientBoostingModel,
    X: pd.DataFrame,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Calculate and return feature importance.
    
    Args:
        model: Trained gradient boosting model
        X: Feature DataFrame
        top_n: Number of top features to return
    
    Returns:
        DataFrame with feature importance sorted by value
    
    Raises:
        ValueError: If model is not fitted
    """
    if not model.is_fitted():
        raise ValueError("Model must be fitted before calculating importance")
    
    importance = model.get_feature_importance(top_n=top_n)
    return importance
```

---

## Testing

### Running Tests

Run all tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

### Writing Tests

- Place tests in `tests/` directory
- Mirror the structure of `src/`
- Use descriptive test names
- Test edge cases and error conditions
- Aim for >80% code coverage

Example test:
```python
def test_temporal_feature_engineer():
    """Test temporal feature creation."""
    # Arrange
    df = create_sample_dataframe()
    engineer = TemporalFeatureEngineer()
    
    # Act
    features = engineer.create_features(df)
    
    # Assert
    assert 'hour' in features.columns
    assert 'day_of_week' in features.columns
    assert len(features) == len(df)
```

---

## Documentation

### Docstrings

Use Google-style docstrings:
```python
def function_name(arg1: type, arg2: type) -> return_type:
    """
    Brief description.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Raises:
        ExceptionType: When this exception is raised
    """
```

### README Updates

If your contribution affects usage:
- Update README.md with new features
- Add examples for new functionality
- Update configuration documentation

### Changelog

Add your changes to CHANGELOG.md under "Unreleased":
```markdown
### Added
- New feature description

### Changed
- Modified behavior description

### Fixed
- Bug fix description
```

---

## Submitting Changes

### Commit Messages

Write clear, descriptive commit messages:
```
Add SHAP time-varying analysis feature

- Implement hourly/daily/monthly SHAP aggregation
- Add visualization for time-varying importance
- Update documentation with usage examples
```

Format:
- First line: Brief summary (50 chars max)
- Blank line
- Detailed description (wrap at 72 chars)

### Pull Request Process

1. **Update your branch** with latest main:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests** and ensure they pass:
   ```bash
   pytest tests/
   black src/ scripts/
   flake8 src/ scripts/
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what changed and why
   - Reference to related issues (e.g., "Fixes #123")
   - Screenshots for UI changes
   - Test results

5. **Address review feedback**:
   - Respond to comments
   - Make requested changes
   - Push updates to the same branch

### PR Checklist

Before submitting, ensure:
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts with main
- [ ] Commit messages are clear

---

## Areas for Contribution

### High Priority

- **Additional feature engineering methods**
- **Alternative model implementations** (e.g., neural networks)
- **Enhanced explainability visualizations**
- **Performance optimization**
- **Additional validation metrics**

### Medium Priority

- **More baseline models**
- **Hyperparameter tuning automation**
- **Data quality checks**
- **Additional weather data sources**
- **Regional holiday calendars**

### Documentation

- **Tutorials and examples**
- **API documentation**
- **Best practices guide**
- **Case studies**

---

## Questions?

If you have questions about contributing:
- Open a [Discussion](https://github.com/Dex947/xai-load-forecasting/discussions)
- Check existing [Issues](https://github.com/Dex947/xai-load-forecasting/issues)
- Review this guide README and documentation

---

## Recognition

Contributors will be recognized in:
- README.md acknowledgments section
- Release notes
- Project documentation

Thank you for contributing to XAI Load Forecasting!
