import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# Load the datasets
pure_solubility_data = pd.read_csv('pure_solubility_data.csv')
mixed_solubility_data = pd.read_csv('filtered_mixed_solubility_data.csv')

# Implementing the Yalkowsky model
def yalkowsky_model(mixed_data, pure_data):
    yalkowsky_predictions = []

    for index, row in mixed_data.iterrows():
        system_no = row['System No']
        solute = row['Solute']
        solvent1 = row['Solvent 1']
        solvent2 = row['Solvent 2']
        temp = row['Temp']

        x1 = row['x1']
        x2 = row['x2']

        # Get the pure solubility values for solvent 1 and solvent 2 based on the temperature
        try:
            pure_sol1 = pure_data[(pure_data['System No'] == system_no) &
                                  (pure_data['Solute'] == solute) &
                                  (pure_data['Solvent 1'] == solvent1) &
                                  (pure_data['Temp'] == temp) &
                                  (pure_data['x1'] == 1)]['Xs_Exper'].values[0]

            pure_sol2 = pure_data[(pure_data['System No'] == system_no) &
                                  (pure_data['Solute'] == solute) &
                                  (pure_data['Solvent 2'] == solvent2) &
                                  (pure_data['Temp'] == temp) &
                                  (pure_data['x2'] == 1)]['Xs_Exper'].values[0]

            # Calculate the Yalkowsky solubility
            yalkowsky_solubility = x1 * pure_sol1 + x2 * pure_sol2
        except IndexError:
            yalkowsky_solubility = np.nan  # Assign NaN if pure solubility values are not found

        yalkowsky_predictions.append(yalkowsky_solubility)

    mixed_data['yalkowsky_solubility'] = yalkowsky_predictions
    return mixed_data

# Apply the model
predicted_data = yalkowsky_model(mixed_solubility_data, pure_solubility_data)

# Display the results
print("\nPredicted Solubility Data with Yalkowsky Model:")
print(predicted_data.head())

# Save the predictions to a new CSV file
predicted_data.to_csv('yalkowsky_model.csv', index=False)

# Evaluate the model
def evaluate_model(data, predicted_column, actual_column='Xs_Exper'):
    valid_data = data.dropna(subset=[predicted_column])
    y_true = valid_data[actual_column]
    y_pred = valid_data[predicted_column]

    # Calculate MPD
    mpd = np.mean(np.abs((y_pred - y_true) / y_true)) * 100

    # Calculate R-squared and ensure it is within [0, 1]
    r2 = r2_score(y_true, y_pred)
    r2 = max(0, min(r2, 1))

    return mpd, r2

# Calculate evaluation metrics for each system and store predictions
systems = predicted_data['System No'].unique()
evaluation_results = []
all_predictions = []

for system in systems:
    system_data = predicted_data[predicted_data['System No'] == system].copy()
    mpd, r2 = evaluate_model(system_data, 'yalkowsky_solubility')

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
evaluation_results_df.to_csv('yalkowsky_evaluation_results_by_system.csv', index=False)
all_predictions_df.to_csv('yalkowsky_predictions_by_system.csv', index=False)

# Display the evaluation results
print("\nEvaluation Results by System:")
print(evaluation_results_df)
