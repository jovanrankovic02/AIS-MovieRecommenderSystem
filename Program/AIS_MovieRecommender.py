import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Load MovieLens dataset
def load_movielens_data():
    df = pd.read_csv('Data/ml-1m/ratings.csv')  # Replace with actual path
    df = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    return df

# Calculate affinity (similarity) between antibodies and antigens
def calculate_affinity(user_profiles, user_ratings):
    return cosine_similarity(user_profiles, user_ratings)

# Clonal selection and mutation
def clonal_selection_and_mutation(affinity, user_profiles, mutation_rate=0.1):
    # Select best-performing antibodies
    best_indices = np.argsort(affinity.sum(axis=1))[-int(len(user_profiles) * 0.1):]
    best_profiles = user_profiles[best_indices]
    
    # Clone and mutate
    clones = np.repeat(best_profiles, repeats=5, axis=0)
    mutation = np.random.normal(0, mutation_rate, clones.shape)
    clones = clones + mutation
    
    return clones

# AIS-based recommender system
def ais_recommender_system(data, num_generations=10, mutation_rate=0.1):
    user_profiles = np.random.rand(data.shape[0], data.shape[1])  # Initialize antibodies
    
    for _ in range(num_generations):
        affinity = calculate_affinity(user_profiles, data)
        user_profiles = clonal_selection_and_mutation(affinity, user_profiles, mutation_rate)
        
        # Convergence check (optional): if no improvement, stop early
    
    # Generate recommendations based on final user_profiles
    recommendations = np.dot(user_profiles, data.T)
    
    return recommendations

def gaussian_noise_mutation(clones, mean=0, std=0.1):
    noise = np.random.normal(mean, std, clones.shape)
    mutated_clones = clones + noise
    return np.clip(mutated_clones, 0, 1)  # Ensure values are within valid range

def adaptive_mutation(clones, affinities, base_mutation_rate=0.1, scaling_factor=0.5):
    # Higher mutation for lower affinity
    mutation_rate = base_mutation_rate * (1 - scaling_factor * affinities)
    noise = np.random.normal(0, mutation_rate[:, np.newaxis], clones.shape)
    mutated_clones = clones + noise
    return np.clip(mutated_clones, 0, 1)

def crossover_mutation(clones, crossover_rate=0.5):
    num_clones = clones.shape[0]
    for i in range(0, num_clones, 2):
        if np.random.rand() < crossover_rate:
            # Select a crossover point
            crossover_point = np.random.randint(1, clones.shape[1])
            # Swap segments between two clones
            clones[i, crossover_point:], clones[i+1, crossover_point:] = clones[i+1, crossover_point:], clones[i, crossover_point:]
    return clones

def differential_mutation(clones, mutation_factor=0.8):
    num_clones = clones.shape[0]
    for i in range(num_clones):
        indices = np.random.choice(num_clones, 3, replace=False)
        diff = mutation_factor * (clones[indices[1]] - clones[indices[2]])
        clones[i] = np.clip(clones[indices[0]] + diff, 0, 1)
    return clones

def simulated_annealing_mutation(clones, initial_temp=1.0, cooling_rate=0.99, iteration=1):
    temp = initial_temp * (cooling_rate ** iteration)
    noise = np.random.normal(0, temp, clones.shape)
    mutated_clones = clones + noise
    return np.clip(mutated_clones, 0, 1)

def elite_mutation(clones, affinities, elite_fraction=0.1):
    elite_size = int(elite_fraction * len(clones))
    elite_indices = np.argsort(affinities)[-elite_size:]
    mutated_clones = np.copy(clones)
    
    # Apply mutation to non-elite clones
    non_elite_indices = np.setdiff1d(np.arange(len(clones)), elite_indices)
    mutated_clones[non_elite_indices] = gaussian_noise_mutation(clones[non_elite_indices])
    
    return mutated_clones

def clonal_selection_and_advanced_mutation(affinity, user_profiles, iteration, mutation_rate=0.1):
    best_indices = np.argsort(affinity.sum(axis=1))[-int(len(user_profiles) * 0.1):]
    best_profiles = user_profiles[best_indices]
    
    clones = np.repeat(best_profiles, repeats=5, axis=0)
    
    # Apply advanced mutations
    clones = gaussian_noise_mutation(clones)
    clones = adaptive_mutation(clones, affinity)
    clones = crossover_mutation(clones)
    clones = differential_mutation(clones)
    clones = simulated_annealing_mutation(clones, iteration=iteration)
    clones = elite_mutation(clones, affinity)
    
    return clones

def precision_at_k(recommendations, test_data, k=10):
    num_users = test_data.shape[0]
    precision_scores = []
    
    for user_id in range(num_users):
        # Get the top K recommended movie indices
        recommended_indices = recommendations[user_id].argsort()[::-1][:k]
        # Get the actual top K movie indices based on the test data
        relevant_indices = test_data[user_id].argsort()[::-1][:k]
        
        # Compute the intersection of recommended and relevant movies
        relevant_and_recommended = np.intersect1d(recommended_indices, relevant_indices)
        precision = len(relevant_and_recommended) / k
        precision_scores.append(precision)
    
    return np.mean(precision_scores)

def compute_rmse(predictions, ground_truth):
    # Flatten both matrices to compare the corresponding predicted and actual ratings
    predictions = predictions[test_data.nonzero()].flatten()
    ground_truth = test_data[test_data.nonzero()].flatten()
    
    return np.sqrt(mean_squared_error(ground_truth, predictions))


# Example usage
if __name__ == "__main__":
    data = load_movielens_data()
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    recommendations = ais_recommender_system(train_data)
    
    # Evaluate the recommendations (e.g., using RMSE or Precision@K)
    # Placeholder: print recommendations for a specific user
    user_id = 0
    print("Recommended movies for user:", recommendations[user_id].argsort()[::-1][:10])


#==================================================================
# Hypermparameter fitting code below
#==================================================================
def objective_function(hyperparameters, train_data, test_data):
    population_size = hyperparameters['population_size']
    cloning_rate = hyperparameters['cloning_rate']
    mutation_rate = hyperparameters['mutation_rate']
    num_generations = hyperparameters['num_generations']
    crossover_rate = hyperparameters['crossover_rate']
    differential_factor = hyperparameters['differential_factor']
    cooling_rate = hyperparameters['cooling_rate']
    
    # Train the AIS-based recommender system with these hyperparameters
    recommendations = ais_recommender_system(train_data, 
                                             population_size=population_size, 
                                             cloning_rate=cloning_rate, 
                                             mutation_rate=mutation_rate, 
                                             num_generations=num_generations, 
                                             crossover_rate=crossover_rate, 
                                             differential_factor=differential_factor, 
                                             cooling_rate=cooling_rate)
    
    # Evaluate the system on the test set

    score = precision_at_k(recommendations, test_data)
    
    return score  # Typically, lower is better if you're minimizing error

from sklearn.model_selection import train_test_split
import numpy as np

# Define the search space
param_space = {
    'population_size': [50, 100, 200],
    'cloning_rate': [0.1, 0.3, 0.5],
    'mutation_rate': [0.05, 0.1, 0.2],
    'num_generations': [10, 20, 50],
    'crossover_rate': [0.3, 0.5, 0.7],
    'differential_factor': [0.5, 0.8, 1.0],
    'cooling_rate': [0.95, 0.99, 0.999]
}

# Random Search
best_score = float('inf')
best_params = None
num_iterations = 50

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

for _ in range(num_iterations):
    # Randomly sample hyperparameters
    hyperparameters = {key: np.random.choice(values) for key, values in param_space.items()}
    
    # Evaluate the objective function
    score = objective_function(hyperparameters, train_data, test_data)
    
    # Update the best score and parameters if the current score is better
    if score < best_score:
        best_score = score
        best_params = hyperparameters

print("Best Score:", best_score)
print("Best Hyperparameters:", best_params)
    
