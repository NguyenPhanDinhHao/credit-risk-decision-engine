# Data

Raw data is **never committed** (see `.gitignore`) and this project never downloads it automatically.

## Home Credit Default Risk (required: `application_train.csv`)

1. Sign in to Kaggle and accept the competition rules: <https://www.kaggle.com/competitions/home-credit-default-risk/data>
2. Download `home-credit-default-risk.zip` from the website (or with your own Kaggle CLI configuration).
3. Unzip into `data/raw/` (or any directory; set `paths.data_dir` in `configs/config.yaml` or `CREDIT_RISK_DATA_DIR`).

```
data/raw/
├── application_train.csv      # required
├── application_test.csv       # optional (unlabeled; not used for evaluation)
├── bureau.csv                 # optional - used with --supplementary
├── previous_application.csv   # optional - used with --supplementary
└── ...                        # bureau_balance, POS_CASH_balance, credit_card_balance, installments_payments (not used yet)
```

Check what the project sees: `python scripts/run_eda.py` prints a file-by-file status report.

## Synthetic data (tests / smoke runs only)

`python scripts/generate_demo_data.py` writes a schema-compatible **artificial** `application_train.csv` to `data/synthetic/`.
Models trained on it say nothing about real credit risk. The training CLI refuses to use synthetic data unless `--demo` is passed,
and every report generated from it carries a warning banner.

Dataset terms: use of the Home Credit data is governed by the Kaggle competition rules; do not redistribute it.
