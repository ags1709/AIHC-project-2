# Importing dependencies
from IPython.display import Image, Audio
import scipy.io
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
import mne
from tqdm import tqdm
from sklearn.base import clone
import pandas as pd
from GraphResultOfLogisticRegression import plot_logistic_regression_results

def run_logistic_regression():
    # Loading features and labels
    feature_mat = scipy.io.loadmat('data/features1.mat')
    y1_mat = scipy.io.loadmat('data/y1.mat')
    features1 = feature_mat['features1']
    y1 = np.array(y1_mat['y1'].T[0])
    print('The shape of the X feature matrix is: ' + str(features1.shape))
    print('The shape of the y vector is: ' + str(y1.shape))

    features_files = ['data/features1.mat', 'data/features2.mat', 'data/features3.mat', 'data/features4.mat']
    label_files = ['data/y1.mat', 'data/y2.mat', 'data/y3.mat', 'data/y4.mat']

    n_pairs = len(features_files)

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
        assert features[i].shape[0] == labels[i].shape[0] #ensure same n of trials and labels


    # Logisitic regression classifier with sequential feature selection
    K = 10
    tol = 1e-6 # Minimum MSE improvement required to add an additional feature
    test_MSE = np.zeros((n_pairs, K))
    train_MSE = np.zeros((n_pairs, K))
    CV = KFold(K,shuffle=True, random_state=42) #Select Cross-validation strategy
    sfs_features = [[] for x in range(n_pairs)] #for saving the best features for each pair and each fold

    # Combine matrix features1 and vector y1
    # features_pair = np.hstack((features1, y1.reshape(-1, 1)))
    # features_pair = features1
    # features_pair = y1 @ features1


    #perform CV for each pair
    for p in tqdm(range(n_pairs), desc="Pairs"):
        # Use the correct pair's features and labels
        X_pair = features[p]
        y_pair = labels[p]
        
        for k, (train_index, test_index) in enumerate(
            tqdm(CV.split(X_pair), total=CV.get_n_splits(), desc=f"CV for Pair {p+1}", leave=False)
        ):    
            X_train, y_train = X_pair[train_index, :], y_pair[train_index] 
            X_test, y_test = X_pair[test_index, :], y_pair[test_index] 
            
            best_MSE = 1e3
            chosen_features = None   
            
            for i in tqdm(range(1, X_pair.shape[1] + 1), desc=f"Feature selection {p+1}-{k+1}", leave=False):
                model = LogisticRegression(max_iter=300, solver="liblinear")
                sfs = SequentialFeatureSelector(LogisticRegression(max_iter=1000), n_features_to_select=i, n_jobs = -1)
                sfs.fit(X_train, y_train)

                model.fit(X_train[:, sfs.get_support()], y_train)

                predictions = model.predict(X_test[:, sfs.get_support()]) 
                MSE = mean_squared_error(y_test, predictions)

                if MSE > best_MSE - tol:
                    break
                else:
                    best_MSE = MSE
                    best_support = sfs.get_support()

            sfs_features[p].append(best_support)
            test_MSE[p, k] = best_MSE
            
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
                



def run_logistic_regression_2():
    # Loading features and labels
    # feature_mat = scipy.io.loadmat('data/features1.mat')
    # y1_mat = scipy.io.loadmat('data/y1.mat')
    # features1 = feature_mat['features1']
    # y1 = np.array(y1_mat['y1'].T[0])
    # print('The shape of the X feature matrix is: ' + str(features1.shape))
    # print('The shape of the y vector is: ' + str(y1.shape))

    features_files = ['data/features1.mat', 'data/features2.mat', 'data/features3.mat', 'data/features4.mat']
    label_files = ['data/y1.mat', 'data/y2.mat', 'data/y3.mat', 'data/y4.mat']

    n_pairs = len(features_files)

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
        assert features[i].shape[0] == labels[i].shape[0] #ensure same n of trials and labels


    # Hyperparameters
    K = 10
    tol = 1e-6
    CV = KFold(K, shuffle=True, random_state=42)

    n_pairs = len(features)
    test_MSE = np.zeros((n_pairs, K))
    train_MSE = np.zeros((n_pairs, K))
    sfs_features = [[] for _ in range(n_pairs)]

    base_lr = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)

    for p in range(n_pairs):
        X = features[p]
        y = labels[p]

        for k, (train_idx, test_idx) in enumerate(CV.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            best_train_mse = np.inf
            best_support = None

            # Grow subset size j with SFS on TRAIN only; stop when improvement < tol
            for j in range(1, X.shape[1] + 1):
                sfs = SequentialFeatureSelector(
                    base_lr,
                    n_features_to_select=j,
                    direction="forward",
                    n_jobs=-1,
                    cv=5,  # inner CV on training data only
                )
                sfs.fit(X_train, y_train)
                support = sfs.get_support()

                model = clone(base_lr)
                model.fit(X_train[:, support], y_train)

            
                yhat_train = model.predict(X_train[:, support])
                mse_train = mean_squared_error(y_train, yhat_train)

                if best_train_mse - mse_train < tol:
                    # no meaningful improvement by increasing j further
                    break
                else:
                    best_train_mse = mse_train
                    best_support = support


            # Final refit on TRAIN with the selected features, then evaluate TRAIN and TEST
            final_model = clone(base_lr)
            final_model.fit(X_train[:, best_support], y_train)

            yhat_train_final = final_model.predict(X_train[:, best_support])
            yhat_test_final = final_model.predict(X_test[:, best_support])

            train_MSE[p, k] = mean_squared_error(y_train, yhat_train_final)
            test_MSE[p, k] = mean_squared_error(y_test, yhat_test_final)

            # Save boolean mask of selected features for later visualization
            sfs_features[p].append(best_support)

    #  Report simple summaries
    # print("Mean TEST MSE per pair:", test_MSE.mean(axis=1))
    # print("Overall mean TEST MSE:", test_MSE.mean())
    return sfs_features, K, test_MSE

# sfs_features, K, test_MSE = run_logistic_regression_2()
# plot_logistic_regression_results(sfs_features, K, test_MSE)
