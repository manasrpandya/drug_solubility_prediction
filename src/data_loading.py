# src/data_preprocessing.py

import pandas as pd

def load_datasets():
    solute_descriptors = pd.read_csv('data/scaled_solute_descriptors.csv')
    solvent_1_descriptors = pd.read_csv('data/scaled_solvent_1_descriptors.csv')
    solvent_2_descriptors = pd.read_csv('data/scaled_solvent_2_descriptors.csv')
    mixed_solubility_data = pd.read_csv('data/filtered_mixed_solubility_data.csv')
    pure_solubility_data = pd.read_csv('data/pure_solubility_data.csv')

    return solute_descriptors, solvent_1_descriptors, solvent_2_descriptors, mixed_solubility_data, pure_solubility_data

def merge_datasets(solute_descriptors, solvent_1_descriptors, solvent_2_descriptors, mixed_solubility_data):
    solute_descriptors.rename(columns={'System no.': 'System No'}, inplace=True)
    solvent_1_descriptors.rename(columns={'System no.': 'System No'}, inplace=True)
    solvent_2_descriptors.rename(columns={'System no.': 'System No'}, inplace=True)

    merged_data = pd.merge(mixed_solubility_data, solute_descriptors, on='System No', suffixes=('', '_solute'))
    merged_data = pd.merge(merged_data, solvent_1_descriptors, on='System No', suffixes=('', '_solvent1'))
    merged_data = pd.merge(merged_data, solvent_2_descriptors, on='System No', suffixes=('', '_solvent2'))

    return merged_data

def main():
    solute_descriptors, solvent_1_descriptors, solvent_2_descriptors, mixed_solubility_data, pure_solubility_data = load_datasets()
    merged_data = merge_datasets(solute_descriptors, solvent_1_descriptors, solvent_2_descriptors, mixed_solubility_data)
    
    print("Merged Data:")
    print(merged_data.head())

if __name__ == "__main__":
    main()
