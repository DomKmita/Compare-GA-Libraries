import time
import cProfile
import pstats
import io
import pandas as pd
from memory_profiler import memory_usage
from utils.logger import logger
from pathlib import Path

from GAs import DEAP_tester
from GAs import PyGAD_tester

def profile(GA_run_callback, data, ga_name):
    # Start logging cpu profile, runtime and memory usage
    try:
        start_time = time.time()
        profiler = cProfile.Profile()
        profiler.enable()

        # Run GA
        mem_usage_list, (best_solution, unseen_data_test_accuracy, ga_fitness_training_stats) = memory_usage(
            (GA_run_callback, (data,)),
            interval=0.1,
            retval=True
        )

        # End logging cpu profile, runtime and memory usage
        profiler.disable()
        end_time = time.time()
    except Exception as e:
        logger.error(f"Failed to gather profiling data for {ga_name}: {e}")
        return pd.DataFrame()

    try:
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()
    except Exception as e:
        logger.error(f"Failed to generate pstats from cpu profiling data for {ga_name}: {e}")
        return pd.DataFrame()

    runtime_and_memory_stats = pd.DataFrame([{
        "Algorithm": ga_name,
        "Runtime (s)": end_time - start_time,
        "Memory Usage (MB)": mem_usage_list,
    }])

    profiling_stats = s.getvalue()

    return runtime_and_memory_stats, ga_fitness_training_stats, profiling_stats, best_solution, unseen_data_test_accuracy

def load_data(dataset_name):
    # Load dataset
    dataset_path = Path(__file__).resolve().parent.parent / "datasets" / f"{dataset_name}.csv"
    try:
        return pd.read_csv(dataset_path)
    except FileNotFoundError as e:
        logger.error(f"Dataset not found at: {dataset_path}. Failed to run: {e}")
        return pd.DataFrame()

def save_data(ga_name, directory_label, profiling_stats, memory_and_runtime_stats, fitness_stats):
    # Saving these to separate files as their format isn't suitable for a df and require additional parsing
    dataset_size_directory = directory_label.split("_")[0]
    output_dir = Path(__file__).resolve().parent.parent / "usage_data" / dataset_size_directory / directory_label
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # profiling results
    profile_filename = f"{output_dir}/{ga_name}_cpu_profile.txt"
    try:
        with open(profile_filename, "w") as f:
            f.write(profiling_stats)
    except Exception as e:
        logger.error(f"Could not write profile file to {profile_filename}: {e}")

    # fitness results
    try:
        fitness_stats.to_csv(output_dir / f"{ga_name}_fitness_stats_log.csv", index=False)
    except Exception as e:
        logger.error(f"Failed to save {ga_name} model memory and runtime data: {e}")

    # memory results
    try:
        if memory_and_runtime_stats is not None:
            memory_and_runtime_stats.to_csv(output_dir / "ga_profiling_results.csv", index=False)
    except Exception as e:
        logger.error(f"Failed to save memory and runtime data: {e}")

def run_GAs_and_gen_data(data_set_name):
    # load dataset
    df = load_data(data_set_name)
    # Profile both algorithms
    deap_memory_and_runtime_stats, deap_fitness_stats, deap_profiling_stats, best_deap_ind, deap_unseen_data_test_accuracy = profile(DEAP_tester.run_ga, df, "DEAP")
    pygad_memory_and_runtime_stats_results, pygad_fitness_stats, pygad_profiling_stats, best_pygad_ind, pyad_unseen_data_test_accuracy = profile(PyGAD_tester.run_ga, df, "PyGAD")

    # Combine results (they are small and this makes them easier to visualise
    results_df = pd.concat([deap_memory_and_runtime_stats, pygad_memory_and_runtime_stats_results], ignore_index=True)

    save_data("DEAP", data_set_name, deap_profiling_stats, None, deap_fitness_stats)
    save_data("PyGAD", data_set_name, pygad_profiling_stats, results_df, pygad_fitness_stats)

    print("Profiling results saved to usage_data")



