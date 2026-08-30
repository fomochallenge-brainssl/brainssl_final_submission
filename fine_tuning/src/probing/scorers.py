from .fomo_metrics import compute_macro_f1, compute_auroc, compute_ovr_auroc

def roc_auc_scorer(estimator, X, y):
        y_proba = estimator.predict_proba(X).tolist()
        if len(estimator.classes_) == 2:
            return compute_auroc(y.tolist(), [row[1] for row in y_proba])
        return compute_ovr_auroc(y.tolist(), y_proba)

def f1_scorer(estimator, X, y):
    y_pred = estimator.predict(X).tolist()
    return compute_macro_f1(y.tolist(), y_pred)