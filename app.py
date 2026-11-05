from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# Paths start from this file's folder, so the app finds its files
# whichever folder Streamlit Cloud runs it from
APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / 'model'
TEST_DATA_PATH = APP_DIR / 'test_data.csv'

# Same five names and files as the saving step in the notebook
MODEL_FILES = {
    'Logistic Regression': MODEL_DIR / 'logistic_regression.joblib',
    'Decision Tree': MODEL_DIR / 'decision_tree.joblib',
    'KNN': MODEL_DIR / 'knn.joblib',
    'Naive Bayes': MODEL_DIR / 'naive_bayes.joblib',
    'Support Vector Machine': MODEL_DIR / 'svm.joblib',
}

# The 17 raw input columns the pipelines were trained on
FEATURE_COLUMNS = [
    'Administrative', 'Administrative_Duration', 'Informational',
    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
    'Month', 'OperatingSystems', 'Browser', 'Region',
    'TrafficType', 'VisitorType', 'Weekend',
]

TARGET_COLUMN = 'Revenue'

# Names for False and True in the confusion matrix and the report
CLASS_NAMES = ['No purchase', 'Purchase']

# Short meaning of each metric, shown when hovering over the ? icon
METRIC_HELP = {
    'Accuracy': "Share of all sessions classified correctly. Predicting "
                "'no purchase' every time already gets 84.5% here.",
    'AUC': "How well the predicted probabilities rank buyers above "
           "non-buyers.",
    'Precision': "Of the sessions the model calls a purchase, the share "
                 "that really were.",
    'Recall': "Of the real purchases, the share the model finds.",
    'F1': "Harmonic mean of precision and recall.",
    'MCC': "Matthews correlation, -1 to 1. Uses all four cells of the "
           "confusion matrix; the main score I compare models on.",
}

# st.dataframe drops trailing zeros (0.490 shows as 0.49), so the score
# columns in both tables get a fixed three-decimal format
THREE_DECIMALS = st.column_config.NumberColumn(format='%.3f')

st.set_page_config(
    page_title='Online Shoppers Purchasing Intention',
    layout='wide',
)

st.title('Online Shoppers Purchasing Intention')

# The intro sits in the wider column so its lines stay readable on a wide
# screen; the test file download sits next to it
intro_col, download_col = st.columns([3, 2])

with intro_col:
    st.write(
        "This app scores browsing sessions from the UCI Online Shoppers "
        "Purchasing Intention dataset, where each row is one visit to an "
        "online shop and `Revenue` marks whether that visit ended in a "
        "purchase. The five models were trained in "
        "`model_training.ipynb`, and this page only loads them to "
        "predict, never to retrain."
    )

# So someone who opens the app without the file can get it here
with download_col:
    if TEST_DATA_PATH.exists():
        st.caption(
            "No file to hand? This is the 20% test split the notebook "
            "held out."
        )
        with open(TEST_DATA_PATH, 'rb') as test_data_file:
            st.download_button(
                'Download test_data.csv',
                data=test_data_file,
                file_name='test_data.csv',
                mime='text/csv',
            )


# Streamlit re-runs the whole script on every click, so the models are
# loaded once and kept in memory
@st.cache_resource
def load_models():
    return {
        name: joblib.load(path) for name, path in MODEL_FILES.items()
    }


models = load_models()


# Same six metrics as the notebook, rounded the same way, so test_data.csv
# gives exactly the README table. AUC comes back as nan when the file
# holds only one class.
def compute_metrics(y_true, y_pred, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    return {
        'Accuracy': round(accuracy_score(y_true, y_pred), 3),
        'AUC': auc if np.isnan(auc) else round(auc, 3),
        'Precision': round(precision_score(y_true, y_pred), 3),
        'Recall': round(recall_score(y_true, y_pred), 3),
        'F1': round(f1_score(y_true, y_pred), 3),
        'MCC': round(matthews_corrcoef(y_true, y_pred), 3),
    }


st.header('Upload the test data and pick a model')

# Upload on the left, model choice on the right, so changing the model
# keeps the results just below in view
upload_col, model_col = st.columns([3, 2])

with upload_col:
    uploaded_file = st.file_uploader(
        'Test data (CSV with the same 18 columns as test_data.csv)',
        type=['csv'],
    )

with model_col:
    model_name = st.selectbox('Model', list(MODEL_FILES.keys()))

if uploaded_file is None:
    st.info(
        "Upload test_data.csv to see the six metrics, the confusion "
        "matrix and the classification report for the chosen model."
    )
    st.stop()

# An empty or broken file gets a message instead of a traceback
try:
    uploaded_df = pd.read_csv(uploaded_file)
except (pd.errors.EmptyDataError, pd.errors.ParserError,
        UnicodeDecodeError):
    st.error(
        "This file could not be read as a CSV. Check that it is a "
        "comma-separated file like test_data.csv."
    )
    st.stop()

# Check the file before any model sees it
required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
missing_columns = [
    col for col in required_columns if col not in uploaded_df.columns
]
if missing_columns:
    st.error(
        "The file is missing these columns: " + ", ".join(missing_columns)
        + ". It needs all 17 input columns plus Revenue, as in "
        "test_data.csv."
    )
    st.stop()

# Every model except the tree fails on empty cells
columns_with_gaps = [
    col for col in required_columns if uploaded_df[col].isna().any()
]
if columns_with_gaps:
    st.error(
        "These columns have empty cells: " + ", ".join(columns_with_gaps)
        + ". Fill them in or drop those rows, then upload again."
    )
    st.stop()

# True/False as in test_data.csv; 1/0 works the same way
if not uploaded_df[TARGET_COLUMN].isin([True, False]).all():
    st.error(f"{TARGET_COLUMN} must hold True/False (or 1/0) values.")
    st.stop()

session_count = len(uploaded_df)
buyer_count = int(uploaded_df[TARGET_COLUMN].sum())

with upload_col:
    st.caption(
        f"{session_count:,} sessions loaded, {buyer_count:,} of them "
        f"buyers ({buyer_count / session_count:.1%})."
    )
    with st.expander('Preview the uploaded data'):
        st.dataframe(uploaded_df.head(), width='stretch')

X_uploaded = uploaded_df[FEATURE_COLUMNS]
y_uploaded = uploaded_df[TARGET_COLUMN]
chosen_model = models[model_name]

# Each pipeline does the log, scaling and one-hot encoding itself, so it
# gets the raw columns. Nothing is fitted here. A value the model cannot
# read (text in a number column, say) gets a message too.
try:
    with st.spinner(f"Scoring {session_count:,} sessions..."):
        predictions = chosen_model.predict(X_uploaded)
        probabilities = chosen_model.predict_proba(X_uploaded)[:, 1]
except ValueError as err:
    st.error(f"The model could not score this file: {err}")
    st.stop()

# The model name in the heading says which results these are, even when
# the dropdown has scrolled out of view
st.header(f"Results: {model_name}")

metrics = compute_metrics(y_uploaded, predictions, probabilities)
# Three decimals as text, so 0.490 does not show up as 0.49
shown = {
    name: 'n/a' if np.isnan(value) else f"{value:.3f}"
    for name, value in metrics.items()
}

for metric_col, name in zip(st.columns(6), shown):
    metric_col.metric(name, shown[name], help=METRIC_HELP[name])

# Side by side, so the heatmap does not stretch across the wide layout
matrix_col, report_col = st.columns(2)

with matrix_col:
    st.subheader('Confusion matrix')

    # labels= keeps the order fixed even when the file has only one class
    confusion_counts = confusion_matrix(
        y_uploaded, predictions, labels=[False, True]
    )

    # The counts are written in the cells, so the colour bar adds nothing
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        confusion_counts,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    plt.close(fig)

with report_col:
    st.subheader('Classification report')

    # The accuracy row repeats one number across every column, and
    # accuracy is already shown above, so I drop it
    report = classification_report(
        y_uploaded,
        predictions,
        labels=[False, True],
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    report_table = pd.DataFrame(report).T.drop(index='accuracy').round(3)
    report_table['support'] = report_table['support'].astype(int)
    st.dataframe(
        report_table,
        width='stretch',
        column_config={
            col: THREE_DECIMALS
            for col in ['precision', 'recall', 'f1-score']
        },
    )
    st.caption(
        "The Purchase row's precision and recall are the same numbers "
        "as the metrics above."
    )

st.header('All five models on this file')

# Behind a checkbox because the SVM takes a second or two on this file
if st.checkbox('Run every model on this file (the SVM is the slow one)'):
    all_model_rows = {}
    with st.spinner('Scoring with all five models...'):
        for name, pipe in models.items():
            model_preds = pipe.predict(X_uploaded)
            model_probs = pipe.predict_proba(X_uploaded)[:, 1]
            all_model_rows[name] = compute_metrics(
                y_uploaded, model_preds, model_probs
            )
    comparison_table = pd.DataFrame(all_model_rows).T
    st.dataframe(
        comparison_table,
        width='stretch',
        column_config={col: THREE_DECIMALS for col in comparison_table},
    )
    st.caption(
        "On test_data.csv these are the numbers in the README comparison "
        "table."
    )
