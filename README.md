# Assignment 2 - Birds Classification (Perceptron + Adaline)

This project implements Task 1 requirements using the provided dataset `birds(in).csv`.

## What is implemented

- Binary classification between any pair of classes (`A`, `B`, `C`) using:
  - Perceptron (signum activation)
  - Adaline (linear output in training, sign decision in testing)
- GUI inputs:
  - Select exactly 2 features
  - Select exactly 2 classes
  - Learning rate `eta`
  - Number of epochs `m`
  - MSE threshold
  - Use bias or not
  - Choose algorithm (Perceptron / Adaline)
- Data split rule:
  - 30 random (non-repeated) samples for training from each selected class
  - 20 remaining samples for testing from each selected class
- Preprocessing:
  - `gender` encoded as: male = `1`, female = `-1`, `NA` = `0`
  - No rows dropped
- Outputs:
  - Decision boundary visualization
  - Manual confusion matrix (without sklearn)
  - Overall accuracy
  - Single-sample prediction

## Files

- `app.py`: main GUI application
- `run_experiments.py`: runs all class/feature combinations for both algorithms and saves top results/plots
- `birds(in).csv`: dataset
- `requirements.txt`: dependencies

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run GUI

```bash
python app.py
```

## Run report helper (recommended)

```bash
python run_experiments.py
```

This generates:
- `outputs/all_results.csv`
- `outputs/top5_results.csv`
- Top 5 plots in `outputs/`

You can use these outputs directly in your report screenshots and analysis.

## GitHub push steps

```bash
git init
git add .
git commit -m "Implement birds classification GUI with perceptron and adaline"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Submission checklist (for your `.rar`)

Include:
- Python code files (`app.py`, `run_experiments.py`)
- Dataset file (`birds(in).csv`)
- Generated visualizations (at least 5 combinations across algorithms/classes/features)
- Analysis report (PDF/DOC) with:
  - confusion matrix + accuracy per shown experiment
  - interpretation of good and bad performance cases
  - final statement of best feature combination(s)

Notes respected:
- No sklearn confusion matrix usage
- No row dropped
- Gender preprocessing applied
