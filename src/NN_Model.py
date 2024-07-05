import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


X = merged_data.drop(columns=['Xs_Exper'])
y = merged_data['Xs_Exper']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Clustering on the training data
kmeans = KMeans(n_clusters=4, random_state=42)  # You can adjust the number of clusters
train_clusters = kmeans.fit_predict(X_train.drop(columns=['System No', 'Solute', 'Solvent 1', 'Solvent 2']))
test_clusters = kmeans.predict(X_test.drop(columns=['System No', 'Solute', 'Solvent 1', 'Solvent 2']))

# Add clusters to the training and testing data
X_train['Cluster'] = train_clusters
X_test['Cluster'] = test_clusters

# Define function to build and train neural network
def build_and_train_nn(X_train, y_train):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    return model

# Train a neural network for each cluster
models = {}
for cluster in np.unique(train_clusters):
    cluster_data = X_train[X_train['Cluster'] == cluster]
    X_cluster = cluster_data.drop(columns=['Cluster', 'System No', 'Solute', 'Solvent 1', 'Solvent 2']).values.astype(float)
    y_cluster = y_train[cluster_data.index].values

    model = build_and_train_nn(X_cluster, y_cluster)
    models[cluster] = model

# Predict using the appropriate model for each sample in the testing data
test_predictions = []
for index, row in X_test.iterrows():
    cluster = row['Cluster']
    X_sample = np.array(row.drop(['Cluster', 'System No', 'Solute', 'Solvent 1', 'Solvent 2'])).reshape(1, -1).astype(float)
    prediction = models[cluster].predict(X_sample)[0][0]
    test_predictions.append(prediction)

X_test['predicted_solubility_MM'] = test_predictions

# Evaluate the model
def evaluate_model(data, predicted_column, actual_column):
    valid_data = data.dropna(subset=[predicted_column])
    y_true = actual_column.loc[valid_data.index]
    y_pred = valid_data[predicted_column]

    # Calculate MPD
    mpd = np.mean(np.abs((y_pred - y_true) / y_true)) * 100

    # Calculate R-squared and ensure it is within [0, 1]
    r2 = r2_score(y_true, y_pred)
    r2 = min(r2, 1)

    return mpd, r2

# Calculate evaluation metrics for each system and store predictions
systems = X_test['System No'].unique()
evaluation_results = []
all_predictions = []

for system in systems:
    system_data = X_test[X_test['System No'] == system].copy()
    system_actual = y_test[system_data.index]
    mpd, r2 = evaluate_model(system_data, 'predicted_solubility_MM', system_actual)

    evaluation_results.append({
        'System No': system,
        'MPD': mpd,
        'R2': r2
    })

    system_data['System No'] = system
    all_predictions.append(system_data)

# Create DataFrames to save the results
evaluation_results_df = pd.DataFrame(evaluation_results)
all_predictions_df = pd.concat(all_predictions)

# Save the evaluation results and predictions to CSV files
evaluation_results_df.to_csv('nn_mml_evaluation_results_by_system.csv', index=False)
all_predictions_df.to_csv('nn_mml_predictions_by_system.csv', index=False)

# Display the evaluation results
print("\nEvaluation Results by System:")
print(evaluation_results_df)
