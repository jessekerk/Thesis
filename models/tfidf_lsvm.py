from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC

from models.utilities import get_text_series


def run_tfidf_lsvm(train_df, eval_df):
    train_texts = get_text_series(train_df)
    y_train = train_df["label"]

    eval_texts = get_text_series(eval_df)
    y_eval = eval_df["label"]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=2000,
        ngram_range=(1, 2),
        min_df=2,
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_eval = vectorizer.transform(eval_texts)

    model = LinearSVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)

    return {
        "accuracy": accuracy_score(y_eval, y_pred),
        "report": classification_report(y_eval, y_pred),
        "confusion_matrix": confusion_matrix(y_eval, y_pred),
    }
