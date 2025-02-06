import time
import cProfile
import pstats
import io
import pandas as pd
from memory_profiler import memory_usage

from GAs import DEAP_tester
from GAs import PyGAD_tester

def profile(GA_run_callback, label):
    # Start logging cpu profile, runtime and memory usage
    mem_before = memory_usage()[0]
    start_time = time.time()
    profiler = cProfile.Profile()
    profiler.enable()

    # Run GA
    GA_run_callback()

    # End logging cpu profile, runtime and memory usage
    profiler.disable()
    end_time = time.time()
    mem_after = memory_usage()[0]

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats()

    df = pd.DataFrame([{
        "Algorithm": label,
        "Runtime (s)": end_time - start_time,
        "Memory Usage (MB)": mem_after - mem_before,
    }])

    # Saving these to separate files as their format isn't suitable for a df and require additional parsing
    profile_filename = f"usage_data/{label}_cpu_profile.txt"
    with open(profile_filename, "w") as f:
        f.write(s.getvalue())

    return df

def run_GAs_and_gen_data():
    # Profile both algorithms
    deap_results = profile(DEAP_tester.run_ga, "DEAP")
    pygad_results = profile(PyGAD_tester.run_ga, "PyGAD")

    # Combine results (they are small and this makes them easier to visualise
    results_df = pd.concat([deap_results, pygad_results], ignore_index=True)

    # Save results to CSV
    results_df.to_csv("usage_data/ga_profiling_results.csv", index=False)

    print("Profiling results saved to usage_data/ga_profiling_results.csv")