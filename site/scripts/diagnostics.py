import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
import json
def generate_diagnostics(x, y, current_best_model, label_type):
    print json.dumps({"status": "validating_model", "percent": 0.85, "best_model": [str(current_best_model[0]), current_best_model[1]]})
    if label_type == "Binary":
        return generate_binary_diagnostics(x, y, current_best_model)
    elif label_type == "Categorical":
        return generate_categorical_diagnostics(x, y, current_best_model)
    elif label_type == "Ordinal":
        return generate_ordinal_diagnostics(x, y, current_best_model)

def generate_ordinal_diagnostics(x, y, current_best_model):
    x = np.array(x)
    y = np.array(y)
    kf = KFold(n_splits=10, shuffle=True)
    guesses = []
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        model = current_best_model[0].fit(X_train, y_train)
        for guess in zip(y_test.tolist(), model.predict(X_test).tolist()):
            guesses.append(guess)
    return {"mse": mean_squared_error(*np.array(guesses).transpose()), "r2": r2_score(*np.array(guesses).transpose()), "mae": median_absolute_error(*np.array(guesses).transpose()), "evs": explained_variance_score(*np.array(guesses).transpose()), "rmse": np.sqrt(mean_squared_error(*np.array(guesses).transpose()))}

def generate_binary_diagnostics(x, y, current_best_model):
    x = np.array(x)
    y = np.array(y)
    kf = KFold(n_splits=10, shuffle=True)
    guesses = []
    proba_guesses = []
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = current_best_model[0].fit(X_train, y_train)
        predictions = []
        try:
            predictions = model.predict_proba(X_test).tolist()
        except:
            predictions = model.predict(X_test).tolist()
        for guess in zip(y_test.tolist(), predictions):
            guessed = 0
            if (type(guess[1]) != type([]) and type(guess[1]) != type(())):
                guessed = guess[1]
                proba_guesses.append([guess[0], guessed])
            else:
                if guess[1][0] < guess[1][1]:
                    guessed = 1
                    if guessed == 0:
                        proba_guesses.append([guess[0], 1-guess[1][np.round(guess[1]).tolist().index(1.0)]])
                    elif guessed == 1:
                        proba_guesses.append([guess[0], guess[1][np.round(guess[1]).tolist().index(1.0)]])
            guesses.append([guess[0], guessed])
    tn, fp, fn, tp = confusion_matrix(*np.array(guesses).transpose()).ravel()
    return {"accuracy": (tp+tn)/float(len(guesses)), "confusion_matrix": {"tn": tn, "tp": tp, "fn": fn, "fp": fp}, "auc": roc_auc_score(*np.array(proba_guesses).transpose())}

def generate_categorical_diagnostics(x, y, current_best_model, label_type):
    generate_binary_diagnostics(x, y, current_best_model)