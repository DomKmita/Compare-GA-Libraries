import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pygad
import traceback
from utils.logger import logger

"""
    Callback function to track average & max fitness per generation.
    This is to replicate DEAPs stats functionality.
"""

# Store stats per generation
stats_log = []
def on_generation(ga_instance):
    try:
        population_fitness = ga_instance.last_generation_fitness
        avg_fitness = np.mean(population_fitness)
        max_fitness = np.max(population_fitness)
        nevals = len(population_fitness)  # Number of evaluations

        stats_log.append({"gen": ga_instance.generations_completed, "nevals": nevals, "avg": avg_fitness, "max": max_fitness})

        print(f"{ga_instance.generations_completed:<3}{nevals:<9}{avg_fitness:<10.6f}{max_fitness:<10.6f}")
    except Exception as e:
        logger.error(f"Error logging stats at generation {ga_instance.generations_completed}: {e}")
        print(f"Error logging pygad stats at generation {ga_instance.generations_completed}: {e}")


'''
*** From Docs: PyGAD has the following modules:
    The main module has the same name as the library pygad which is the main interface to build the genetic algorithm.
    The nn module builds artificial neural networks.
    The gann module optimizes neural networks (for classification and regression) using the genetic algorithm.
    The cnn module builds convolutional neural networks.
    The gacnn module optimizes convolutional neural networks using the genetic algorithm.
    The kerasga module to train Keras models using the genetic algorithm.
    The torchga module to train PyTorch models using the genetic algorithm.
    The visualize module to visualize the results.
    The utils module contains the operators (crossover, mutation, and parent selection) and the NSGA-II code.
    The helper module has some helper functions.
    ***

    Using main module below. Will also use visualize, utils and helper. Considering using others, however DEAP does not
    have functionality for training neural networks or training Keras or PyTorch. This might be something I look at 
    later down the line. 
'''

def run_ga(df):
    # Train-test split
    y = df["target"].values
    X = df.drop("target", axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=932024)

    # GA parameters
    num_generations = 50
    num_parents_mating = 15
    sol_per_pop = 100
    num_genes = X_train.shape[1]
    mutation_probability = 0.08

    # Fitness function
    def fitness_function(ga_instance, solution, solution_idx):
        try:
            predictions = np.dot(X_train, solution) > 0
            predictions = predictions.astype(int)
            accuracy = accuracy_score(y_train, predictions)
            return accuracy,
        except Exception as e:
            logger.error(f"Error in pygad fitness function at index {solution_idx}: {e}")
            return np.nan,

    # Attempting to have as similar to the DEAP implementation. Might refactor both to ensure consistency. The commented
    # sections require some revisiting as they differ slightly to the DEAP tester.
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        # This is a slight issue. DEAP doesn't seem to have an equivalent parameter.
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        init_range_low=-1.0,
        init_range_high=1.0,
        mutation_probability=mutation_probability,
        crossover_probability=0.7,
        mutation_type="inversion",
        # Current DEAP implementation uses Inversion mutation. Possible options here are: random,
        # swap, inversion, scramble and adaptive.
        crossover_type="single_point",
        # Current DEAP implementation uses cxBlend. Possible options here are: single-point,
        # two-points, uniform and scattered.
        parent_selection_type="tournament",
        # Changed to tournament as this way I can better control the behaviour of both
        # libraries
        K_tournament=3,
        on_generation=on_generation,
    )

    # Run the GA
    try:
        ga_instance.run()
    except Exception as e:
        logger.error(f"PyGAD run has failed: {e}")
        logger.error(traceback.format_exc())
        return None, None

    # Get the best solution
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()

    pygad_df = pd.DataFrame(stats_log)

    # Test the model on unseen data
    predictions_test = np.dot(X_test, best_solution) > 0
    predictions_test = predictions_test.astype(int)
    test_accuracy = accuracy_score(y_test, predictions_test)

    print("Best Weights:", best_solution)
    print("Test Accuracy:", test_accuracy)

    return best_solution, test_accuracy, pygad_df
