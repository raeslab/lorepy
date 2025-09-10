# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lorepy is a Python package for creating Logistic Regression plots. It visualizes the distribution of categorical dependent variables as a function of continuous independent variables using logistic regression models. The package consists of two main modules:

- `src/lorepy/lorepy.py`: Core plotting functionality with the `loreplot` function
- `src/lorepy/uncertainty.py`: Uncertainty analysis with the `uncertainty_plot` function using resampling/jackknifing

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage and detailed output
pytest --exitfirst --verbose --failed-first --cov=src

# Run tests with coverage report (matching CI setup)
pytest --exitfirst --verbose --failed-first --cov=src tests/ --cov-report=term-missing --cov-report=xml
```

### Package Installation and Building
```bash
# Install package in development mode
pip install -e .

# Install package from source
pip install .

# Build source distribution and wheel
python setup.py sdist bdist_wheel
```

### Development Dependencies
Development requirements are pinned in `docs/dev/requirements.txt`:
```bash
pip install -r docs/dev/requirements.txt
```

## Code Architecture

### Core Components
- **Main plotting function**: `loreplot()` in `src/lorepy/lorepy.py` handles the primary visualization logic
- **Uncertainty analysis**: `uncertainty_plot()` in `src/lorepy/uncertainty.py` provides confidence interval visualization
- **Data preparation**: `_prepare_data()` handles input validation and preprocessing
- **Area calculation**: `_get_area_df()` generates probability areas for visualization
- **Scatter dots**: `_get_dots_df()` positions sample points within probability bands

### Key Dependencies
- `matplotlib>=3.4.1` for plotting
- `pandas>=1.2.4` for data manipulation  
- `scikit-learn>=1.5.0` for logistic regression models
- `numpy>=1.20.2` for numerical operations

### Configuration
- **Python version**: Supports Python >=3.9
- **Test configuration**: `pytest.ini` sets `pythonpath = src` for module imports
- **Package structure**: Uses `src/` layout with packages defined in `setup.py`

## Testing Structure

Tests are located in `tests/` directory:
- `tests/test_plot.py`: Tests for main plotting functionality
- `tests/test_uncertainty.py`: Tests for uncertainty analysis features

Tests use pytest fixtures for sample data and model setup. The CI workflow runs tests across Python versions 3.9-3.12.

## License and Usage
- Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
- For commercial use, contact is required
- Developed by Sebastian Proost at RaesLab based on R code by Sara Vieira-Silva

## Key Features to Understand
- Supports custom classifiers beyond logistic regression (SVM, Random Forest, etc.)
- Handles confounders through additional model features
- Provides jitter option for discrete x-axis values
- Uncertainty visualization through bootstrap resampling or jackknifing
- Matplotlib integration with customizable styling options