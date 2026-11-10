# Credit Card Fraud Prediction

Detecting fraudulent credit card transactions in a highly imbalanced dataset, comparing logistic regression, random forest and XGBoost, each as an untuned baseline and a tuned model.

**Result:** tuned XGBoost was the best model. It had a cross-validated PR-AUC of 0.852 and a test PR-AUC of 0.824. At its chosen threshold, it caught 73 of 95 test frauds with 8 false alarms out of 56,651 legitimate transactions.

## Data

The Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset contains 284,807 transactions from two days. After removing 1,081 duplicate rows, 283,726 remain, of which 473 (0.167%) are fraud. Features V1–V28 are anonymized PCA components; the other features are `Time` and `Amount`.

The CSV is not included in this repository (150 MB). Download `creditcard.csv` from Kaggle and place it in `data/`.

## Approach

- **EDA:** class balance, feature distributions by class (KS statistic), VIF for multicollinearity.
- **Features:** removed duplicates before splitting; converted `Time` to hour of day with sine/cosine encoding; log-transformed `Amount`.
- **Split:** stratified 80/20 train/test split. Scaling inside a pipeline, for logistic regression only.
- **Imbalance:** class weights / `scale_pos_weight` plus threshold tuning (no SMOTE).
- **Model selection:** PR-AUC (average precision), since accuracy is meaningless at a 0.17% fraud rate. Hyperparameters were tuned with stratified cross-validation on the training set. The decision threshold was chosen to maximize F1 on out-of-fold predictions. The test set was used once, at the end.
- **Uncertainty:** stratified bootstrap (1,000 resamples) of the test set for 95% confidence intervals, plus paired comparisons between models.
- **Interpretation:** XGBoost gain, permutation importance and SHAP.

## Results (test set, 95 frauds)

| Model | CV PR-AUC | Test PR-AUC [95% CI] | Precision | Recall | F1 | Caught | False alarms |
|---|---|---|---|---|---|---|---|
| LR baseline | – | 0.699 [0.601, 0.794] | 0.846 | 0.579 | 0.688 | 55 | 10 |
| LR tuned | 0.762* | 0.700 [0.598, 0.798] | 0.830 | 0.768 | 0.798 | 73 | 15 |
| RF baseline | – | 0.783 [0.698, 0.862] | 0.973 | 0.747 | 0.845 | 71 | 2 |
| RF tuned | 0.832 | 0.807 [0.729, 0.878] | 0.921 | 0.737 | 0.819 | 70 | 6 |
| XGB baseline | – | 0.669 [0.577, 0.765] | 0.846 | 0.695 | 0.763 | 66 | 12 |
| **XGB tuned** | **0.852** | **0.824 [0.748, 0.894]** | 0.901 | 0.768 | 0.830 | 73 | 8 |

\*Logistic regression used 5-fold CV; random forest and XGBoost used the same 3 folds.

- In the paired bootstrap, tuned XGBoost's PR-AUC was higher than tuned random forest's (+0.017, 95% CI [+0.004, +0.033]) and the random forest baseline's (+0.041, 95% CI [+0.017, +0.070]).
- The random forest baseline's slightly higher F1 was not significant (−0.016, 95% CI [−0.046, +0.014]).
- **Top predictors:** V14 ranks in the top two by gain, permutation importance and SHAP. V4 ranks 1st by permutation importance and 2nd by SHAP, but 5th by gain. V10, V12 and V17 rank high by gain and SHAP but lower by permutation importance, since they carry overlapping fraud signal: when one is shuffled, the model relies on the others. SHAP shows that very negative values of V14, V10, V12 and V17, and high values of V4, push predictions toward fraud.

## Limitations

- V1–V28 are anonymized, so the predictors can't be explained in business terms.
- The test set has only 95 frauds, so test metrics are uncertain (about ±0.07 PR-AUC). XGBoost's PR-AUC lead holds across bootstrap resamples of this test set, but its F1 is tied with random forest.
- The data covers only two days, and a random split may overstate performance compared with an out-of-time split.
- The threshold maximizes F1, which treats missed fraud and false alarms as equally costly.

## Run it

```powershell
pip install -r requirements.txt
cd notebooks
jupyter notebook credit_card_fraud_detection.ipynb
```

The full notebook takes about an hour to run, mostly for the random forest and XGBoost hyperparameter searches.
