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
features_pair = features1
# features_pair = y1 @ features1


#perform CV for each pair
for p in tqdm(range(n_pairs), desc="Pairs"):
    for k, (train_index, test_index) in enumerate(
        tqdm(CV.split(features_pair), total=CV.get_n_splits(), desc=f"CV for Pair {p+1}", leave=False)
    ):    
        X_train, y_train = features_pair[train_index, :], y1[train_index] 
        X_test, y_test = features_pair[test_index, :], y1[test_index] 
        
        best_MSE = 1e3
        chosen_features = None   
        
        for i in tqdm(range(1, features_pair.shape[1] + 1), desc=f"Feature selection {p+1}-{k+1}", leave=False):
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
            
