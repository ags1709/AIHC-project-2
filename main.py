import pandas as pd
import os
from Logistic_regression import run_logistic_regression
from GraphResultOfLogisticRegression import plot_logistic_regression_results

csv_file = "featuresData.csv"

def main():
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                print("CSV is empty. Training new model...")
                df = run_logistic_regression()
                df.to_csv(csv_file, index=False)
            else:
                print("Loaded existing data from CSV.")
        except pd.errors.EmptyDataError:
            print("CSV has no data. Training new model...")
            df = run_logistic_regression()
            df.to_csv(csv_file, index=False)
    else:
        print("CSV file not found. Training new model...")
        df = run_logistic_regression()
        df.to_csv(csv_file, index=False)

    plot_logistic_regression_results(df["selected_features"], df["pair"], df["fold"], df["test_MSE"])


if __name__ == "__main__":
    main()
