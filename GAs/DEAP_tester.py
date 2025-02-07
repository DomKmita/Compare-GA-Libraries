import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.logger import logger

# Load dataset
dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "small_dataset_default_version.csv"
try:
    df = pd.read_csv(dataset_path)
    y = df["target"].values
    X = df.drop("target", axis=1).values
except Exception as e:
    logger.error(f"Failed to load dataset from {dataset_path}: {e}")
    raise

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=932024)

""" 
*** From Docs: Fitness class is an abstract class that needs a weights attribute in order to be functional. A minimizing
   fitness is built using negatives weights, while a maximizing fitness has positive weights
 """
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximise objective (two values, potential for
# multi-objective)


""" 
*** From Docs: Simply by thinking about the different flavors of evolutionary algorithms (GA, GP, ES, PSO, DE, …), we
  notice that an extremely large variety of individuals are possible, reinforcing the assumption that all types cannot
  be made available by developers. ***

  The above means that this is an area with a lot of flexibility, will have to revisit later to test variety of types.
  for now a simple individual (a list containing floats)
  Some possible options are: A list
                             A permutation
                             An Arithmetic Expression
                             An Evolution Strategy *(ES)
                             A Particle *(Particle Swarm Optimisation)
                             A "Funky One" ergo custom
"""
creator.create("Individual", list, fitness=creator.FitnessMax)

# Number of features
IND_SIZE=len(X_train[0])

"""
 *** From Docs: The register() method takes at least two arguments; an alias and a function assigned to this alias. Any
  subsequent argument is passed to the function when called (à la functools.partial()). Thus, the below code creates two
  aliases in the toolbox; attr_float and individual. The first one redirects to the random.random() function. The second
  one is a shortcut to the initRepeat() function, fixing its container argument to the creator.Individual class, its
  func argument to the toolbox.attr_float() function, and its number of repetitions argument to IND_SIZE.

  Now, calling toolbox.individual() will call initRepeat() with the fixed arguments and return a complete 
  creator.Individual composed of IND_SIZE floating point numbers with a maximizing single objective fitness attribute.***
"""
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -1.0, 1.0)  # Random weights
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)

"""
*** From Docs: Populations are much like individuals. Instead of being initialized with attributes, they are filled with
 individuals, strategies or particles.***
 Some possible types of populations: A bag (Most commonly used, generally implemented using a list (Simple GA))
                                    A Grid (Special case of structured population where neighbouring individuals have a 
                                    direct effect on each other (Cellular Model GA))
                                    A Swarm (Contains communication network (PSO))
                                    A deme (sub-population within a population (island model GA))
                                    
Current population is random but they can be seeded. This will be changed for reproducibility)
"""
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

"""
*** From Docs: The evaluation is the most personal part of an evolutionary algorithm, it is the only part of the library
  that you must write yourself. A typical evaluation function takes one individual as argument and returns its fitness as
  a tuple. As shown in the Creating Types section, a fitness is a list of floating point values and has a property valid
  to know if this individual shall be re-evaluated. The fitness is set by setting the values to the associated tuple. 

  Dealing with single objective fitness is not different, the evaluation function must return a tuple because 
  single-objective is treated as a special case of multi-objective.
  ***
  
  Currently using a simple fitness function. Will most likely look into defining more complex fitness functions. This is 
  dependant on the direction I take. This is ultimately a comparison between libraries. I am not focusing on maximising
  algorithm effectiveness. I may attempt to write a GA to the best of my ability and gauge how well the libraries support
  me in the process.
"""
def fitness_function(individual):
    try:
        # Apply weights as a linear model
        predictions = np.dot(X_train, individual) > 0
        predictions = predictions.astype(int)
        accuracy = accuracy_score(y_train, predictions)
        return accuracy,
    except Exception as e:
        logger.error(f"Failed to evaluate {individual}: {e}")
        return np.nan,

# Register fitness function with toolbox
toolbox.register("evaluate", fitness_function)

"""
*** From Docs: Each crossover has its own characteristics and may be applied to different types of individuals.
    The general rule for crossover operators is that they only mate individuals, this means that an independent copies 
    must be made prior to mating the individuals if the original individuals have to be kept or are references to other 
    individuals.
    
    Must ensure two things: 
        1. I use the correct crossover for the type of algorithm I am using to avoid unwanted behaviour
        2. Create copies of individuals if I want to retain the individual.
        
    Changed from cxBlend to cxOnePoint for simplicity and consistency between PyGAD implementation that currently also
    uses single-point crossover. PyGAD does not have a cxBlend varient.
"""
toolbox.register("mate", tools.cxOnePoint)  # Crossover

"""
*** From Docs: There is a variety of mutation operators in the deap.tools module. Each mutation has its own 
    characteristics and may be applied to different types of individuals.

     The general rule for mutation operators is that they only mutate, this means that an independent copy 
    must be made prior to mutating the individual if the original individual has to be kept or is a reference to another 
    individual (see the selection operator). ***

    Must ensure two things: 
        1. I use the correct mutation for the type of algorithm I am using to avoid unwanted behaviour
        2. Create copies of individuals if I want to retain the individual.
        
    Changing mutation to inversion mutation as PyGAD offers that also.
"""
toolbox.register("mutate", tools.mutInversion)  # Current PyGAD implementation uses
# Inversion mutation possible options here are: Inversion, FlipBit, UniformInt, ShuffleIndexes, ESLogNormal,
# PolynomialBounded mutGaussian

"""
*** From Docs: Selection is made among a population by the selection operators that are available in the deap.tools 
    module. The selection operator usually takes as first argument an iterable container of individuals and the number 
    of individuals to select. It returns a list containing the references to the selected individuals. 
    
    

    Warning
    
    It is very important here to note that the selection operators does not duplicate any individual during the 
    selection process. If an individual is selected twice and one of either object is modified, the other will also be 
    modified. Only a reference to the individual is copied. Just like every other operator it selects and only selects.  
    ***

    Updated to use tournament selection, this way both DEAP and PyGAD will behave more similarly.
"""
toolbox.register("select", tools.selTournament, tournsize=3)  # Selection

# Run the genetic algorithm
def run_ga():
    population = toolbox.population(n=100)  # Population size (Will play around with this)
    n_generations = 50  # Number of generations

    """
    *** From Docs: Often, one wants to compile statistics on what is going on in the optimization. The Statistics are 
        able to compile such data on arbitrary attributes of any designated object. To do that, one needs to register 
        the desired statistic functions inside the stats object using the exact same syntax as in the toolbox.
    
        The statistics object is created using a key as first argument. This key must be supplied a function that will 
        later be applied to the data on which the statistics are computed. ***
        
        Currently just calculating mean average and maximum. Will look into other statistics that I would like.
    """
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # Run the GA (using DEAP eaSimple, their basic GA option)
    # logbook is where the statistics are stored.

    """
    These are the following algorithms provided. (Note: The DEAP Team encourage users to write their own for their 
    specific use case.):    eaSimple
                            varOr
                            varAnd
                            eaMuPlusLambda
                            eaGenerateUpdate
                            eaMuCommaLambda

    """
    try:
        population, logbook = algorithms.eaSimple(
            population,
            toolbox,
            cxpb=0.7,  # Crossover probability (Will play around with this)
            mutpb=0.08,  # Mutation probability (Will play around with this)
            ngen=n_generations,
            stats=stats,
            verbose=True,
        )
    except Exception as e:
        logger.error(f"DEAP run has failed: {e}")
        logger.error(traceback.format_exc())
        return None, None

    # Get the best individual
    best_ind = tools.selBest(population, k=1)[0]
    deap_stats = list(logbook)
    deap_df = pd.DataFrame(deap_stats)

    try:
        output_dir = Path(__file__).resolve().parent.parent / "usage_data"
        deap_df.to_csv(output_dir / "deap_stats_log.csv", index=False)
    except Exception as e:
        logger.error(f"Failed to save DEAP model memory and runtime data: {e}")

    # Test the model on unseen data
    predictions_test = np.dot(X_test, best_ind) > 0
    predictions_test = predictions_test.astype(int)
    test_accuracy = accuracy_score(y_test, predictions_test)

    print("Best Weights:", best_ind)
    print("Test Accuracy:", test_accuracy)

    return best_ind, test_accuracy
