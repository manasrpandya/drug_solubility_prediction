# Solubility Prediction in Binary Solvent Mixtures

## Project Overview

This project aims to replicate and extend the methodologies presented in the paper "[Machine Learning Derived Quantitative Structure Property Relationship Models to Predict Drug Solubility in Binary Solvent Mixtures](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00282)". The objective is to implement and evaluate different solubility prediction models, including the Yalkowsky model, Jouyban-Acree models, Multiple Model Learning (MML), and Neural Networks (NN).

## Data Preprocessing

The data preprocessing involved converting the raw data from the supporting data file into five CSV files for solute descriptors, solvent descriptors, and solubility data. These files were then merged to create a comprehensive dataset for model training and evaluation.

### Files:
- `scaled_solute_descriptors.csv`
- `scaled_solvent_1_descriptors.csv`
- `scaled_solvent_2_descriptors.csv`
- `filtered_mixed_solubility_data.csv`
- `pure_solubility_data.csv`

## Models Implemented

### Yalkowsky Model
The Yalkowsky model is used to predict drug solubility in binary solvent mixtures based on the solubility of the drug in pure solvents and their proportions in the mixture.

### Jouyban-Acree Models
Implemented both Ordinary Least Squares (OLS) and Weighted by Optimization (WBO) methods for solubility prediction. These models showed lower performance, indicated by low R-squared values and high MSE.

### Multiple Model Learning (MML)
MML approach was successfully implemented using clustering and Random Forest models. This method improved prediction accuracy.

### Neural Networks (NN)
A neural network model was attempted but faced implementation challenges due to coding inexperience.

## Results

- **Yalkowsky Model:** Achieved results consistent with those presented in the paper.
- **Multiple Model Learning (MML):**
  - MPD: 23.51%
  - R-squared: 0.9867
- **Neural Networks (NN):** Further refinement needed due to coding challenges.

## Challenges and Limitations

- **Genetic Algorithm (GA):** Faced overfitting issues and coding challenges.
- **Jouyban-Acree Models:** Lower R-squared values and higher MSE.
- **Neural Networks:** Implementation difficulties due to limited coding experience.

## Future Work

- Refine the implementation of the neural network model.
- Address overfitting issues in the Genetic Algorithm.
- Explore additional machine learning models to improve prediction accuracy.

## References

- [Machine Learning Derived Quantitative Structure Property Relationship Models to Predict Drug Solubility in Binary Solvent Mixtures](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00282)
