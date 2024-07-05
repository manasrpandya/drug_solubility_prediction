import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Assuming merged_data is already available from previous steps
# Extract features and target variable
features = merged_data.drop(columns=['System No', 'Solute', 'Solvent 1', 'Solvent 2', 'Xs_Exper', 'x1', 'x2', 'xs1p', 'xs2p', 'Temp'])
target = merged_data['Xs_Exper']

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Define evaluation function
def evaluate(individual, X_train, y_train):
    selected_features = [index for index, val in enumerate(individual) if val == 1]
    if len(selected_features) == 0:
        return 10000,  # Return a high error if no features are selected

    X_selected = X_train[:, selected_features]
    model = LinearRegression()
    scores = cross_val_score(model, X_selected, y_train, cv=5, scoring='neg_mean_squared_error')
    return -np.mean(scores),

# Initialize GA components
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=scaled_features.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# GA parameters
population_size = 50
num_generations = 70
crossover_prob = 0.8
mutation_prob = 0.2

kf = KFold(n_splits=5)
fold_results = {}
best_individuals = []

# Run GA for each fold
for fold, (train_index, test_index) in enumerate(kf.split(scaled_features)):
    X_train, X_test = scaled_features[train_index], scaled_features[test_index]
    y_train, y_test = target.values[train_index], target.values[test_index]

    # Track the mean and best fitness values for each fold
    mean_fitness_values = []
    best_fitness_values = []

    # Run GA and log stats for each generation
    def log_stats(population):
        fitnesses = [ind.fitness.values[0] for ind in population]
        mean_fitness_values.append(np.mean(fitnesses))
        best_fitness_values.append(np.min(fitnesses))

    # Evaluate function specific to this fold
    toolbox.register("evaluate", evaluate, X_train=X_train, y_train=y_train)

    population = toolbox.population(n=population_size)
    for gen in range(num_generations):
        population = algorithms.varAnd(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob)
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        population = toolbox.select(population, len(population))
        log_stats(population)

    best_individual = tools.selBest(population, k=1)[0]
    best_individuals.append(best_individual)
    fold_results[f'Fold {fold + 1}'] = {'mean': mean_fitness_values, 'best': best_fitness_values}

# Plot the mean and best fitness values for each fold
plt.figure(figsize=(10, 6))
for fold, results in fold_results.items():
    plt.plot(results['mean'], label=f'{fold} mean', linestyle='--')
    plt.plot(results['best'], label=f'{fold} best')

plt.xlabel('Generation')
plt.ylabel('Objective')
plt.title('GA Feature Selection Progress Across Folds')
plt.legend()
plt.show()

# Find the best overall individual
best_overall = tools.selBest(best_individuals, k=1)[0]
selected_features = [index for index, val in enumerate(best_overall) if val == 1]
print("Best overall feature selection:", selected_features)
