# Predicting Online Shoppers' Purchasing Intention

Machine Learning (AIMLZG565), Assignment 2 | Praveen Garg | BITS ID 2026AJ05478

## a. Problem statement

This project predicts whether a browsing session on an online shop
ends in a purchase, using 17 features that describe the session: how
many pages of each kind were visited and how long each took, three
Google Analytics measures of the pages visited (bounce rate,
exit rate and page value), how close it is to a
special day, and details of the visit itself such as month, browser
and visitor type. It is a binary classification problem, where the
positive class is a purchase (`Revenue = True`) and the negative
class is no purchase. A shop would want this prediction while a
visitor is still browsing, so it can act on sessions that look
unlikely to convert, for example with a discount or a product
recommendation, without handing out discounts to visitors who were
going to buy anyway.

## b. Dataset description

Online Shoppers Purchasing Intention Dataset, from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).
Creators: C. Sakar and Y. Kastro. DOI:
[10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q). Licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- Size: 12,330 sessions, 18 columns.
- Features: 17. Ten are numeric - page counts and durations,
  `BounceRates`, `ExitRates`, `PageValues` and `SpecialDay`. The
  other seven are categorical - `Month`, `OperatingSystems`,
  `Browser`, `Region`, `TrafficType`, `VisitorType` and `Weekend`.
- Target: `Revenue`, True/False. 10,422 sessions (84.5%) had no
  purchase and 1,908 (15.5%) had one.
- Data quality: no missing values. There are 125 duplicate rows, all
  bounce sessions, which are kept rather than dropped.
- Preprocessing, inside each saved pipeline: log1p plus scaling on
  the 7 skewed numeric columns, scaling only on the 3 bounded ones,
  and one-hot encoding on the 7 categorical columns.
- Split: 80/20, stratified on `Revenue`, seed 42 - 9,864 training
  sessions and 2,466 test sessions (382 buyers). The test part is
  saved as `test_data.csv`.

## c. GitHub repository link

[https://github.com/PraveenGarg09/2026aj05478-ml-assignment-2](https://github.com/PraveenGarg09/2026aj05478-ml-assignment-2)

## d. Models used

- Logistic Regression: defaults (lbfgs solver, L2 penalty, C = 1).
- Decision Tree: `max_depth = 5`, chosen by 5-fold CV on MCC.
- KNN: `k = 15`, chosen the same way.
- Gaussian Naive Bayes: `var_smoothing = 0.5`, chosen the same way.
- Support Vector Machine: RBF kernel, `C = 1`, `gamma = 'scale'`,
  `probability=True`; not grid-searched.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.894 | 0.911 | 0.705 | 0.545 | 0.614 | 0.561 |
| Decision Tree | 0.899 | 0.917 | 0.711 | 0.586 | 0.643 | 0.588 |
| KNN | 0.894 | 0.896 | 0.705 | 0.545 | 0.614 | 0.561 |
| Naive Bayes | 0.821 | 0.841 | 0.454 | 0.762 | 0.569 | 0.490 |
| Support Vector Machine | 0.894 | 0.900 | 0.700 | 0.550 | 0.616 | 0.561 |

All six metrics are measured on the 2,466 held-out test sessions,
with a purchase as the positive class. AUC uses the predicted
probabilities rather than the labels.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic regression reaches a test MCC of 0.561 and the second-highest AUC, 0.911. It finds 208 of the 382 buyers in the test set and misses 174. Its training MCC of 0.586 is close to the test score, so it barely overfits, and that is with every setting left at its default. A straight-line model on log-scaled features keeps up with the more flexible models here, which suggests much of the signal in this data is close to linear. |
| Decision Tree | The decision tree gives the best test results of the five, with an MCC of 0.588, and it leads on every metric except recall. In its depth search the training MCC climbs to 1.000 as the tree grows, while the mean CV MCC peaks at 0.589 at depth 5 and falls to 0.476 with no limit, so capping the depth at 5 is what keeps it from overfitting. One feature, PageValues, carries 0.782 of its feature importance, so the tree is mostly one strong rule with small corrections. It is also the least stable model across folds, with the widest CV spread (standard deviation 0.041). |
| KNN | KNN settles on k = 15. The CV curve is almost flat from 15 to 51 neighbours (0.560 to 0.568), and k = 3 is the worst at 0.524. Its confusion counts are exactly the same as logistic regression's, but only by coincidence: the two label 110 of the 2,466 test sessions differently. Its AUC is lower, 0.896 against 0.911, because its probability is just the share of the 15 neighbours that are buyers, so it takes only 16 values and ranks sessions more coarsely than a continuous score. Its saved file is also far larger than the other four, because the model keeps every training row to find neighbours at prediction time. |
| Naive Bayes | With default settings Gaussian Naive Bayes collapses, predicting 2,149 of 2,466 test sessions as buyers, for an MCC of only 0.131. Tuning var_smoothing fixes most of that, but it is still the weakest model on MCC at 0.490. It has the highest recall of the five, 0.762 (291 of the 382 buyers), but the lowest precision, 0.454, with 350 false alarms. Part of the reason it trails is that its independence assumption clearly does not hold here: ProductRelated correlates with its own duration at 0.861, and BounceRates with ExitRates at 0.913, so the model counts the same evidence twice. |
| Support Vector Machine | The SVM's test MCC of 0.561 ties with logistic regression and KNN, but it has the largest train-test gap of the five: 0.658 on the training data against 0.561 on test. In 5-fold CV, though, it has the best mean MCC of all five, 0.601, and its test score is below even its weakest fold (0.574), which points to this test split being a harder draw for it rather than to a weak model. It was not grid-searched like the tree, KNN and Naive Bayes, and it is the slowest to fit, because `probability=True` makes it fit five extra internal models for Platt scaling. |
| Overall, Winner for your dataset? | My pick is the decision tree. I judge the models mainly on MCC, because only 15.5% of sessions are purchases and MCC only goes up when a model actually finds them. The tree leads on it, 0.588 against 0.561 for logistic regression, KNN and the SVM, and it also has the best F1, 0.643. The lead comes with a caveat: the SVM has the best 5-fold CV mean at 0.601, and the tree is the most variable model across folds (standard deviation 0.041), so on another test split the gap could shrink or reverse. There is also a caveat about PageValues itself. The purchase rate is 0.563 when it is above 0 and only 0.039 when it is 0, and Google Analytics derives page value from completed transactions, so in real-time use, before a session ends, this feature may already partly contain the answer the model is meant to predict. |

## Live app

https://2026aj05478-ml-assignment-2.streamlit.app/

## Repository contents

- `app.py` - the Streamlit app.
- `model_training.ipynb` - loads the data, preprocesses it, trains
  and compares the five models, and saves the pipelines.
- `model/` - the five saved pipelines (`.joblib` files).
- `test_data.csv` - the 2,466 held-out test sessions, for the app.
- `data/online_shoppers_intention.csv` - the full dataset.
- `requirements.txt` - pinned package versions.
- `.streamlit/config.toml` - the app's colour theme.

## Running it locally

```
pip install -r requirements.txt
streamlit run app.py
```

To rebuild the models, Restart & Run All `model_training.ipynb`.
