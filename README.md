# AutoGluon Assignment

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![AutoGluon](https://img.shields.io/badge/AutoGluon-latest-brightgreen.svg)](#)
[![Colab](https://img.shields.io/badge/Notebook-Colab-black.svg)](#)

This repository contains end-to-end AutoGluon experiments and demos for both tabular ML and Kaggle-style workflows. It includes Colab notebooks, docs, and saved artifacts to reproduce experiments easily.

---

## 📂 Repository Structure

```
Autogluon/
├── artifacts/                  # submissions
│   ├── ieee_fraud/             # IEEE-CIS Fraud Detection outputs
│   └── cal_housing/            # California Housing outputs
├── colabs/                     # Reproducible Colab notebooks
│   ├── AutoGluon_Tabular.ipynb
│   ├── AutoGluon_Tabular_Feature_Engineering.ipynb
│   ├── Autogluon_Kaggle_IEEE_fraud.ipynb
│   ├── Autogluon_California_Housing_Price.ipynb
│   └── AutoGluon_Multimodal.ipynb
├── Video
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

### Option 1 — Run on Colab
- Open any notebook in `colabs/`  
- Go to **Runtime → Change runtime type → GPU (optional)**  
- No local setup needed

### Option 2 — Run Locally
```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -U "autogluon>=1.1.0"
pip install -U kaggle  # optional for Kaggle competitions
```

---

## 📦 Data

- Place raw CSVs inside `data/` (ignored by git).
- Most notebooks already contain Kaggle download cells.

### Kaggle API Setup
1. Create API token on Kaggle → download `kaggle.json`
2. In Colab:
   ```python
   from pathlib import Path
   Path('/root/.kaggle').mkdir(exist_ok=True)
   with open('/root/.kaggle/kaggle.json','w') as f:
       f.write('{"username":"YOUR_NAME","key":"YOUR_KEY"}')
   !chmod 600 /root/.kaggle/kaggle.json
   !kaggle competitions download -c ieee-fraud-detection -p /content/data
   !unzip -o /content/data/ieee-fraud-detection.zip -d /content/data
   ```

---

## 🧠 Notebooks Overview

| Notebook | Description |
|-----------|--------------|
| **AutoGluon_Tabular.ipynb** | Quick start for tabular classification/regression using `TabularPredictor` |
| **AutoGluon_Tabular_Feature_Engineering.ipynb** | Adds preprocessing, encoding, feature importance, and optimization |
| **Autogluon_Kaggle_IEEE_fraud.ipynb** | Complete IEEE-CIS Fraud Detection pipeline — data loading, AutoGluon training, leaderboard, submission |
| **Autogluon_California_Housing_Price.ipynb** | Complete California House Prices pipeline — data loading, AutoGluon training, leaderboard, submission |
| **AutoGluon_Multimodal.ipynb** |  Demonstrates text/image/tabular multimodal learning |

---

## ▶️ Example Run

```python
from autogluon.tabular import TabularPredictor

train_path = "data/train.csv"
label = "target"

predictor = TabularPredictor(label=label).fit(
    train_path,
    presets="best_quality",
    time_limit=3600
)

print(predictor.leaderboard(silent=True))
preds = predictor.predict("data/test.csv")
preds.to_csv("artifacts/submission.csv", index=False)
```

---

## 📁 Artifacts

- All submissions are saved under `artifacts/`.
- Organized by project name (`ieee_fraud/`, `cal_housing/`, etc.).

---

## 💡 Tips

- Use `presets="good or high"` for faster tests.  
- Set `random_state` for reproducibility.  
- Use `time_limit` for fair comparisons.  
- GPUs are optional but help with multimodal or large datasets.  
- Keep raw datasets in `data/` (not tracked in Git).

---

## 🙏 Acknowledgements

Built using [AutoGluon](https://github.com/autogluon/autogluon) by AWS AI.  
Special thanks to the open-source community for datasets and support.


---

## 📬 Contact

Feel free to open an issue or reach out for collaboration or clarifications.
