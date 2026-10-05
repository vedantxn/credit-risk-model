# Credit Risk Model

Predicts the probability that a loan applicant will have payment difficulties, using only information available at application time. Built on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset (`application_train.csv`, 307k applications, 8% defaults) and served through a FastAPI endpoint.

## Results

Test set, 61,503 held-out applications. XGBoost settings were chosen by 5-fold cross-validation.

| Model | ROC-AUC |
|---|---|
| Always "no default" (91.9% accuracy, catches 0 defaulters) | 0.500 |
| Logistic regression, 8 features | 0.723 |
| Random forest, 8 features | 0.734 |
| XGBoost, 8 features | 0.741 |
| **XGBoost, 127 features incl. engineered ratios, tuned** | **0.768** |

Final model: PR-AUC 0.258 (random: 0.081), log loss 0.243 (baseline: 0.281). Predicted probabilities are well calibrated, within about one percentage point per decile.

A cutoff from expected profit, `p < margin / (margin + LGD)`, gives 0.182 under assumed economics (10% margin, 45% loss given default). It beats approving everyone by about 8% in a backtest.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download application_train.csv from Kaggle into data/raw/

python -m src.train              # train and save to models/
uvicorn src.api:app --reload     # API docs at http://127.0.0.1:8000/docs
```

`POST /predict` takes one application (same field names as the CSV) and returns `pd_default` and `decision`. Fields that were never blank in training are required (see `GET /model-info`), because the model can't score them reliably when they're missing.

## Layout

- `notebooks/`: exploration, model comparison, final model
- `src/features.py`: cleaning, feature engineering, category encoder
- `src/train.py`: reproducible training
- `src/predict.py`: inference
- `src/api.py`: FastAPI service

## Limitations

Uses only the main application table. Includes sensitive attributes (gender, age) that a real lender would need to review. The cutoff economics are illustrative. The split is random, not by date.
