import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.neural_network import MLPClassifier

import pickle

# https://fairmlbook.org/code/adult.html

# Source: https://www.valentinmihov.com/2015/04/17/adult-income-data-set/

features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"]

# This will download 3.8M
original_train = pd.read_csv(filepath_or_buffer="data/adult.data", names=features, sep=r'\s*,\s*',
                             engine='python', na_values="?")
# This will download 1.9M
original_test = pd.read_csv(filepath_or_buffer="data/adult.test", names=features, sep=r'\s*,\s*',
                            engine='python', na_values="?", skiprows=1)

num_train = len(original_train)
original = pd.concat([original_train, original_test])
roc_original = original
labels = original['Target']
labels = labels.replace('<=50K', 0).replace('>50K', 1)
labels = labels.replace('<=50K.', 0).replace('>50K.', 1)

# Redundant column
del original["Education"]

# Remove target variable
del original["Target"]


def data_transform(df):
    binary_data = pd.get_dummies(df)
    feature_cols = binary_data[binary_data.columns[:-2]]
    scaler = preprocessing.StandardScaler()
    scaled_data = scaler.fit_transform(feature_cols)

    continuous_columns = ["Age", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week"]
    continuous_col_indices = [feature_cols.columns.get_loc(c) for c in continuous_columns if c in feature_cols.columns]

    continuous_means = scaler.mean_[continuous_col_indices]
    continuous_stdev = scaler.scale_[continuous_col_indices]

    scaled_data_df = pd.DataFrame(scaled_data, columns=feature_cols.columns)

    print(f"For continuous columns:")
    for (col, mean, std) in zip(continuous_columns, continuous_means, continuous_stdev):
        print(f"Column: {col}, Mean: {mean}, StdDev: {std}")
        print(f"[-0.4, 0.4] for {col} corresponds to [{mean - 0.4 * std}, {mean + 0.4 * std}] in the original dataset")

        within_range_count = np.sum((scaled_data_df[col] >= -0.4) & (scaled_data_df[col] <= 0.4))
        total_count = len(scaled_data_df[col])
        proportion = within_range_count / total_count

        print(f"Proportion of data for {col} that falls within [-0.4, 0.4]: {proportion:.4f}\n")

    mask = np.all(np.logical_and(scaled_data[:, continuous_col_indices] >= -0.4,
                                 scaled_data[:, continuous_col_indices] <= 0.4), axis=1)

    proportion = np.mean(mask)
    print(f"Proportion of data points where all five continuous columns fall within [-0.4, 0.4]: {proportion:.4f}")

    data = pd.DataFrame(scaled_data, columns=feature_cols.columns)
    return data


data = data_transform(original)

train_data = data[:num_train]
train_labels = labels.iloc[:num_train]
test_data = data[num_train:]
test_labels = labels.iloc[num_train:]

k = 10

fresh = False  # Training from scratch or load pre-saved model.

cls = MLPClassifier(solver='sgd', hidden_layer_sizes=(10, 10), activation='tanh', random_state=1, max_iter=250)

if fresh:
    cls.fit(train_data, train_labels)
    pickle.dump(cls, open('saved/cls.sav', 'wb'))
    print("Model saved!")
else:
    cls = pickle.load(open('saved/cls.sav', 'rb'))
    print("Model loaded!")

inputs = []
for i in range(100):
    inputs.append(test_data.iloc[i].values)

if fresh:
    pickle.dump(inputs, open('saved/inputs.sav', 'wb'))
