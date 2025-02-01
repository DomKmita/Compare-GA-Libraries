import time
import cProfile
import pstats
import io
from pathlib import Path
import pandas as pd
from memory_profiler import memory_usage

from GAs import DEAP_tester
from GAs import PyGAD_tester

def profile(GA_run_callback, label):
    mem_before = memory_usage()[0]
    start_time = time.time()

    profiler = cProfile.Profile()
    profiler.enable()

    # Run GA
    best_solution, test_accuracy = GA_run_callback()

    profiler.disable()
    end_time = time.time()
    mem_after = memory_usage()[0]

    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats()

    df = pd.DataFrame([{
        "Algorithm": label,
        "Runtime (s)": end_time - start_time,
        "Memory Usage (MB)": mem_after - mem_before,
        "CPU Profile": s.getvalue()
    }])

    return df

if __name__ == "__main__":
    # Profile both algorithms
    deap_results = profile(DEAP_tester.run_ga, "DEAP")
    pygad_results = profile(PyGAD_tester.run_ga, "PyGAD")

    # Combine results
    results_df = pd.concat([deap_results, pygad_results], ignore_index=True)

    # Save results to CSV
    output_dir = Path(__file__).resolve().parent / "usage_data"
    results_df.to_csv(output_dir / "ga_profiling_results.csv", index=False)

    print("Profiling results saved to results/ga_profiling_results.csv")