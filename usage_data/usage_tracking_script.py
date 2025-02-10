import time
import cProfile
import pstats
import io
import numpy as np
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

def run_GAs_and_gen_data(data_set_name, num_runs=3):
    df = load_data(data_set_name)
    
    # Deap run data 
    deap_runs = []
    deap_fitness_frames = []
    deap_profiling_txt = [] 

    # PyGAD run data
    pygad_runs = []
    pygad_fitness_frames = []
    pygad_profiling_txt = []

    for _ in range(num_runs):
        # DEAP single run
        (memory_and_runtime_data_deap, fitness_data_deap, profiling_data_deap, best_ind_deap,
         unseen_data_accuracy_deap) = profile(DEAP_tester.run_ga, df, "DEAP")
        if not memory_and_runtime_data_deap.empty:
            deap_runs.append(memory_and_runtime_data_deap)
        if fitness_data_deap is not None and not fitness_data_deap.empty:
            deap_fitness_frames.append(fitness_data_deap)
        if profiling_data_deap:
            deap_profiling_txt.append(profiling_data_deap)

        # PyGAD single run
        (memory_and_runtime_data_pygad, fitness_data_pygad, profiling_data_pygad, best_ind_pygad,
         unseen_data_accuracy_pygad) = profile(PyGAD_tester.run_ga, df, "PyGAD")
        if not memory_and_runtime_data_pygad.empty:
            pygad_runs.append(memory_and_runtime_data_pygad)
        if fitness_data_pygad is not None and not fitness_data_pygad.empty:
            pygad_fitness_frames.append(fitness_data_pygad)
        if profiling_data_pygad:
            pygad_profiling_txt.append(profiling_data_pygad)

    # Combine/average DEAP stats 
    if deap_runs:
        deap_concat = pd.concat(deap_runs, ignore_index=True)
        # Store a single row with average runtime + average/peak memory usage
        # Because memory usage is a *list* of floats, can’t directly average them in a normal sense.
        # Instead, store peak memory usage per run, and then average that across runs.
        deap_concat["PeakMem"] = deap_concat["Memory Usage (MB)"].apply(np.max)
        deap_concat["Runtime (s)"] = deap_concat["Runtime (s)"]
        avg_runtime = deap_concat["Runtime (s)"].mean()
        avg_peak_mem = deap_concat["PeakMem"].mean()

        # Build a 1-row DataFrame for DEAP
        deap_final_df = pd.DataFrame([{
            "Algorithm": "DEAP",
            "Runtime (s)": avg_runtime,
            "Peak Memory (MB)": avg_peak_mem,
        }])
    else:
        deap_final_df = pd.DataFrame()

    # Merge all DEAP fitness logs into one, then average numeric columns
    if deap_fitness_frames:
        deap_fitness_concat = pd.concat(deap_fitness_frames, ignore_index=True)
        
        # A single averaged log, grouping by 'gen' and averaging the rest
        deap_fitness_agg = deap_fitness_concat.groupby("gen", as_index=False).mean()
    else:
        deap_fitness_agg = pd.DataFrame()

    # Combine DEAP cProfile texts. 
    deap_profile_str = "\n\n".join(deap_profiling_txt) if deap_profiling_txt else ""

    # Combine/average PyGAD stats
    if pygad_runs:
        pygad_concat = pd.concat(pygad_runs, ignore_index=True)
        pygad_concat["PeakMem"] = pygad_concat["Memory Usage (MB)"].apply(np.max)
        pygad_concat["Runtime (s)"] = pygad_concat["Runtime (s)"]
        avg_runtime = pygad_concat["Runtime (s)"].mean()
        avg_peak_mem = pygad_concat["PeakMem"].mean()

        pygad_final_df = pd.DataFrame([{
            "Algorithm": "PyGAD",
            "Runtime (s)": avg_runtime,
            "Peak Memory (MB)": avg_peak_mem,
        }])
    else:
        pygad_final_df = pd.DataFrame()

    # Merge all PyGAD fitness logs into one, then average numeric columns
    if pygad_fitness_frames:
        pygad_fitness_concat = pd.concat(pygad_fitness_frames, ignore_index=True)
        pygad_fitness_agg = pygad_fitness_concat.groupby("gen", as_index=False).mean()
    else:
        pygad_fitness_agg = pd.DataFrame()

    # Combine PyGAD cProfile texts
    pygad_profile_str = "\n\n".join(pygad_profiling_txt) if pygad_profiling_txt else ""

    # Combine final usage stats from DEAP & PyGAD into a single DataFrame
    results_df = pd.concat([deap_final_df, pygad_final_df], ignore_index=True)

    # Save everything
    save_data("DEAP", data_set_name, deap_profile_str, None, deap_fitness_agg)
    save_data("PyGAD", data_set_name, pygad_profile_str, results_df, pygad_fitness_agg)

    print(f"[{data_set_name}] Profiling results saved to usage_data")

