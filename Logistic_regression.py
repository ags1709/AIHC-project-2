# Importing dependencies
from IPython.display import Image, Audio
import scipy.io
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn.base import clone
import pandas as pd

def load_data():
    features_files = ['data/features1.mat', 'data/features2.mat', 'data/features3.mat', 'data/features4.mat']
    label_files = ['data/y1.mat', 'data/y2.mat', 'data/y3.mat', 'data/y4.mat']

    features = []
    labels = []

    #load features and labels from all pairs
    #we store them in lists, because they have different number of trials
    for i,feature_f in enumerate(features_files):
        feature_map = scipy.io.loadmat(feature_f)
        y_mat = scipy.io.loadmat(label_files[i])
        idx = str(i+1)
        features.append(feature_map[f'features{idx}'])
        labels.append(np.array(y_mat[f'y{idx}'].T[0]))
        assert features[i].shape[0] == labels[i].shape[0]

    return features, labels

def run_logistic_regression():

    features, labels = load_data()

    K = 10
    tol = 1e-6
    CV = KFold(K, shuffle=True, random_state=42)

    n_pairs = len(features)
    test_MSE = np.zeros((n_pairs, K))
    train_MSE = np.zeros((n_pairs, K))
    sfs_features = [[] for _ in range(n_pairs)]

    base_lr = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)

    for p in tqdm(range(n_pairs), desc="Pairs"):
        X = features[p]
        y = labels[p]

        for k, (train_idx, test_idx) in enumerate(
            tqdm(CV.split(X, y), total=K, desc=f"CV Folds (Pair {p+1})", leave=False)
        ):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            best_train_mse = np.inf
            best_support = None

            for j in tqdm(range(1, X.shape[1] + 1), desc=f"Feature Selection (Pair {p+1}, Fold {k+1})", leave=False):
                sfs = SequentialFeatureSelector(
                    base_lr,
                    n_features_to_select=j,
                    direction="forward",
                    n_jobs=-1,
                    cv=5,
                )
                sfs.fit(X_train, y_train)
                support = sfs.get_support()

                model = clone(base_lr)
                model.fit(X_train[:, support], y_train)

            
                yhat_train = model.predict(X_train[:, support])
                mse_train = mean_squared_error(y_train, yhat_train)

                if best_train_mse - mse_train < tol:
                    break
                else:
                    best_train_mse = mse_train
                    best_support = support


            final_model = clone(base_lr)
            final_model.fit(X_train[:, best_support], y_train)

            yhat_train_final = final_model.predict(X_train[:, best_support])
            yhat_test_final = final_model.predict(X_test[:, best_support])

            train_MSE[p, k] = mean_squared_error(y_train, yhat_train_final)
            test_MSE[p, k] = mean_squared_error(y_test, yhat_test_final)

            sfs_features[p].append(best_support)

    results = []
    for p in range(n_pairs):
        for k in range(K):
            selected = np.where(sfs_features[p][k])[0].tolist()
            results.append({
                "pair": p + 1,
                "fold": k + 1,
                "selected_features": ",".join(map(str, selected)),
                "test_MSE": test_MSE[p, k],
            })
    df = pd.DataFrame(results)    

    return df

