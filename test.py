# Importing dependencies
import scipy.io
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from sklearn.base import clone

def run_logistic_regression():
    features_files = ['data/features1.mat', 'data/features2.mat', 'data/features3.mat', 'data/features4.mat']
    label_files    = ['data/y1.mat',       'data/y2.mat',       'data/y3.mat',       'data/y4.mat']

    # Load all pairs
    features, labels = [], []
    for i, feature_f in enumerate(features_files):
        feature_map = scipy.io.loadmat(feature_f)
        y_map = scipy.io.loadmat(label_files[i])
        idx = str(i + 1)
        X = feature_map[f'features{idx}']
        y = np.array(y_map[f'y{idx}'].T[0])
        assert X.shape[0] == y.shape[0]
        features.append(X)
        labels.append(y)

    # Hyperparameters
    K = 2
    tol = 1e-6
    CV = KFold(K, shuffle=True, random_state=42)

    n_pairs = len(features)
    test_MSE = np.zeros((n_pairs, K))
    train_MSE = np.zeros((n_pairs, K))
    sfs_features = [[] for _ in range(n_pairs)]

    base_lr = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)

    # Progress over pairs
    for p in tqdm(range(n_pairs), desc="Pairs", position=0):
        X = features[p]
        y = labels[p]
        n_features = X.shape[1]

        # Progress over CV folds
        for k, (train_idx, test_idx) in enumerate(tqdm(CV.split(X, y),
                                                       total=K,
                                                       desc=f"Pair {p+1} CV",
                                                       position=1,
                                                       leave=False)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            best_train_mse = np.inf
            best_support = None
            best_j = 0

            # Progress over feature counts
            for j in trange(1, n_features + 1,
                            desc=f"Pair {p+1} Fold {k+1} SFS",
                            position=2,
                            leave=False):
                sfs = SequentialFeatureSelector(
                    base_lr,
                    n_features_to_select=j,
                    direction="forward",
                    n_jobs=-1,
                    cv=5
                )
                sfs.fit(X_train, y_train)
                support = sfs.get_support()

                model = clone(base_lr)
                model.fit(X_train[:, support], y_train)

                yhat_train = model.predict(X_train[:, support])
                mse_train = mean_squared_error(y_train, yhat_train)

                # update progress bar postfix
                tr = tqdm._instances  # global set of active bars
                # safe postfix update
                try:
                    trange_bar = [b for b in tr if b.desc.startswith(f"Pair {p+1} Fold {k+1} SFS")][0]
                    trange_bar.set_postfix(best_mse=f"{mse_train:.4f}", j=j)
                except Exception:
                    pass

                if best_train_mse - mse_train < tol:
                    break
                else:
                    best_train_mse = mse_train
                    best_support = support
                    best_j = j

            # Edge case: if loop broke at j=1 without improvement, use the last computed support
            if best_support is None:
                # sfs holds the last fitted selector from the loop
                best_support = sfs.get_support()
                best_j = j

            # Final refit and evaluation
            final_model = clone(base_lr)
            final_model.fit(X_train[:, best_support], y_train)

            yhat_train_final = final_model.predict(X_train[:, best_support])
            yhat_test_final  = final_model.predict(X_test[:, best_support])

            train_MSE[p, k] = mean_squared_error(y_train, yhat_train_final)
            test_MSE[p, k]  = mean_squared_error(y_test,  yhat_test_final)
            sfs_features[p].append(best_support)

            # Fold-level summary line
            tqdm.write(f"Pair {p+1} Fold {k+1}: best j={best_j}, "
                       f"train MSE={train_MSE[p,k]:.4f}, test MSE={test_MSE[p,k]:.4f}")

    # Final summary
    tqdm.write("Mean TEST MSE per pair: " + ", ".join(f"{m:.4f}" for m in test_MSE.mean(axis=1)))
    tqdm.write(f"Overall mean TEST MSE: {test_MSE.mean():.4f}")

    # Return features for plotting
    return sfs_features, features, K, test_MSE, train_MSE

def plot_logistic_regression_results(sfs_features, features, K):
    n_pairs = len(features)
    for p in range(n_pairs):
        n_features = features[p].shape[1]
        assert len(sfs_features[p]) == K, f"Pair {p+1}: expected {K} folds, got {len(sfs_features[p])}"
        sel_mat = np.column_stack(sfs_features[p])  # (n_features, K)

        plt.figure(figsize=(8, 6))
        plt.imshow(sel_mat, aspect='auto', interpolation='nearest', cmap='Greys')
        plt.title(f'Pair {p+1}: Selected features per fold')
        plt.xlabel('CV fold')
        plt.ylabel('Feature index')
        plt.xticks(ticks=np.arange(K), labels=np.arange(1, K + 1))
        plt.yticks(ticks=np.arange(0, n_features, 5))
        if n_features == 60:
            plt.axhline(29.5, color='red', linewidth=1)
        cbar = plt.colorbar()
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Not selected', 'Selected'])
        plt.tight_layout()
        plt.show()

    # frequency plot
    for p in range(n_pairs):
        n_features = features[p].shape[1]
        sel_mat = np.column_stack(sfs_features[p])
        freq = sel_mat.sum(axis=1)
        plt.figure(figsize=(8, 3))
        plt.bar(np.arange(n_features), freq)
        plt.title(f'Pair {p+1}: Selection frequency across {K} folds')
        plt.xlabel('Feature index')
        plt.ylabel('Count')
        if n_features == 60:
            plt.axvline(29.5, color='red', linewidth=1)
        plt.tight_layout()
        plt.show()

# Run + plot
sfs_features, features, K, test_MSE, train_MSE = run_logistic_regression()
plot_logistic_regression_results(sfs_features, features, K)
