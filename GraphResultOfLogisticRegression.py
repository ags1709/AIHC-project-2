import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
import mne
import numpy as np


def plot_logistic_regression_results(selected_features_series, pair_series, fold_series, test_MSE_series):
    """
    Plot logistic regression results from DataFrame columns.
    
    Parameters:
    - selected_features_series: pandas Series with comma-separated feature indices
    - pair_series: pandas Series with pair numbers
    - fold_series: pandas Series with fold numbers
    - test_MSE_series: pandas Series with test MSE values
    """
    
    # Convert DataFrame columns to structured format
    pairs = sorted(pair_series.unique())
    n_pairs = len(pairs)
    K = int(fold_series.max())
    
    # First pass: determine the global max feature index across all pairs and folds
    max_feature_idx = 0
    for _, feature_str in selected_features_series.items():
        if isinstance(feature_str, str) and feature_str:
            selected_indices = [int(x) for x in feature_str.split(',')]
            if selected_indices:
                max_feature_idx = max(max_feature_idx, max(selected_indices))
    
    n_features = max_feature_idx + 1
    
    # Second pass: build structured data with uniform boolean mask size
    sfs_features = [[] for _ in range(n_pairs)]
    
    for pair_num in pairs:
        pair_idx = pair_num - 1  # Convert to 0-indexed
        
        for fold_num in range(1, K + 1):
            # Get the row for this pair and fold
            mask = (pair_series == pair_num) & (fold_series == fold_num)
            fold_data = selected_features_series[mask]
            
            if len(fold_data) > 0:
                # Parse comma-separated feature indices
                feature_str = fold_data.iloc[0]
                if isinstance(feature_str, str) and feature_str:
                    selected_indices = [int(x) for x in feature_str.split(',')]
                else:
                    selected_indices = []
            else:
                selected_indices = []
            
            # Create boolean mask with uniform size
            boolean_mask = np.zeros(n_features, dtype=bool)
            boolean_mask[selected_indices] = True
            sfs_features[pair_idx].append(boolean_mask)
    
    # Matrix plot of selected features per CV fold for each pair
    for p in range(n_pairs):
        if len(sfs_features[p]) == 0:
            continue
            
        # Stack boolean masks into a matrix: rows=features, cols=folds
        sel_mat = np.column_stack(sfs_features[p])  # shape (n_features, K)

        plt.figure(figsize=(8, 6))
        plt.imshow(sel_mat, aspect='auto', interpolation='nearest', cmap='Greys')
        plt.title(f'Pair {p+1}: Selected features per fold')
        plt.xlabel('CV fold')
        plt.ylabel('Feature index')
        plt.xticks(ticks=np.arange(K), labels=np.arange(1, K + 1))
        plt.yticks(ticks=np.arange(0, n_features, max(1, n_features // 10)))

        # Optional: separate the two participants if first 30 belong to A and last 30 to B
        if n_features == 60:
            plt.axhline(29.5, color='red', linewidth=1)

        cbar = plt.colorbar()
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Not selected', 'Selected'])
        plt.tight_layout()
        plt.show()

    # Optional: selection frequency plots (how often each feature was picked across folds)
    for p in range(n_pairs):
        if len(sfs_features[p]) == 0:
            continue
            
        sel_mat = np.column_stack(sfs_features[p])
        freq = sel_mat.sum(axis=1)  # 0..K

        plt.figure(figsize=(8, 3))
        plt.bar(np.arange(n_features), freq)
        plt.title(f'Pair {p+1}: Selection frequency across {K} folds')
        plt.xlabel('Feature index')
        plt.ylabel('Count')
        if n_features == 60:
            plt.axvline(29.5, color='red', linewidth=1)
        plt.tight_layout()
        plt.show()
