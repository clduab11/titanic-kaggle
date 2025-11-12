# Helios ML Framework - Installation & Setup Guide

## Quick Start

### 1. Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (for cloning the repository)

### 2. Clone Repository

```bash
git clone https://github.com/clduab11/titanic-kaggle.git
cd titanic-kaggle
```

### 3. Install Dependencies

#### Option A: Using pip (recommended)
```bash
pip install -r requirements.txt
```

#### Option B: Using virtual environment (best practice)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Download Titanic Dataset

1. Go to [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data)
2. Download `train.csv` and `test.csv`
3. Place them in the `data/` directory:
   ```
   titanic-kaggle/
   └── data/
       ├── train.csv
       └── test.csv
   ```

### 5. Run Framework

#### Option A: Run validation test (no data required)
```bash
python3 test_framework.py
```

This will run with synthetic data to validate all components work correctly.

#### Option B: Run with real Titanic data
```bash
python3 example_usage.py
```

This requires `train.csv` and `test.csv` in the `data/` directory.

#### Option C: Use Jupyter Notebook
```bash
# Install Jupyter if not already installed
pip install jupyter matplotlib seaborn

# Start Jupyter
jupyter notebook

# Open notebooks/helios_demo.ipynb
```

## Verification

After installation, verify everything works:

```bash
python3 -c "
import sys
sys.path.insert(0, 'src')
from helios import ISRValidator, QMVMonitor, FeatureEngineer, EnsembleOrchestrator
print('✓ Helios ML Framework installed successfully!')
"
```

## Expected Output

When you run `test_framework.py`, you should see:

```
======================================================================
HELIOS ML FRAMEWORK - VALIDATION TEST
======================================================================

1. Creating synthetic Titanic-like data...
   Train: (400, 11), Test: (200, 10)
   Survival rate: XX.XX%

[... training progress ...]

======================================================================
FRAMEWORK VALIDATION: ✓ ALL COMPONENTS WORKING
======================================================================

✓ Test completed successfully!
```

## Directory Structure After Setup

```
titanic-kaggle/
├── src/
│   └── helios/              # Core framework modules
├── notebooks/               # Jupyter notebooks
├── data/                    # Data files (add train.csv, test.csv)
├── configs/                 # Configuration files
├── audit_trails/            # Generated audit trails (created automatically)
├── requirements.txt         # Python dependencies
├── test_framework.py        # Validation test
├── example_usage.py         # Example usage script
└── README.md               # Documentation
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution**: Make sure you're in the project root directory and dependencies are installed:
```bash
cd titanic-kaggle
pip install -r requirements.txt
```

### Issue: FileNotFoundError for train.csv

**Solution**: Download the Titanic dataset from Kaggle and place in `data/` directory.

### Issue: Import errors

**Solution**: Ensure you're running from the project root or adjust Python path:
```python
import sys
sys.path.insert(0, 'src')
```

## Dependencies

Core dependencies (installed automatically):
- scikit-learn ≥ 1.3.0 - Machine learning algorithms
- xgboost ≥ 2.0.0 - Gradient boosting
- lightgbm ≥ 4.0.0 - Light gradient boosting
- pandas ≥ 2.0.0 - Data manipulation
- numpy ≥ 1.24.0 - Numerical computing
- scipy ≥ 1.11.0 - Scientific computing

Optional (for notebooks):
- jupyter ≥ 1.0.0 - Notebook interface
- matplotlib ≥ 3.7.0 - Plotting
- seaborn ≥ 0.12.0 - Statistical visualization

## Next Steps

1. Run `test_framework.py` to validate installation
2. Download Titanic dataset from Kaggle
3. Run `example_usage.py` or explore `notebooks/helios_demo.ipynb`
4. Review generated audit trails in `audit_trails/`
5. Customize configuration in `configs/config.py`

## Support

For issues or questions:
- Check the troubleshooting section above
- Review the main README.md
- Open an issue on GitHub

## License

See LICENSE file for details.
