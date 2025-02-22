import os
from pathlib import Path
import pandas as pd
from utils.logger import logger

BASE_USAGE_PATH = Path(__file__).resolve().parent
def get_final_fitness(fitness_csv_path):
    final_fitness = []
    try:
        df = pd.read_csv(fitness_csv_path)
        if 'gen' in df.columns:
            # Sort by generation and take the last row's fitness value.
            df.sort_values('gen', inplace=True)
            avg_fitness = df.iloc[-1]['avg']
            max_fitness = df.iloc[-1]['max']
            final_fitness = [avg_fitness, max_fitness]
        else:
            logger.error(f'No gen column in fitness file: {fitness_csv_path}')

    except Exception as ex:
        print(f"Error reading fitness file {fitness_csv_path}: {ex}")
        final_fitness = None
    return final_fitness


def aggregate_results_to_xlsx(dataset_size="small", output_file="aggregated_results"):
    # Find all subdirectories
    usage_dirs = [
        d for d in (BASE_USAGE_PATH / dataset_size).iterdir()
        if d.is_dir()
    ]

    aggregated_data = []

    for d in usage_dirs:
        try:
            # Get usage data
            df_usage = pd.read_csv(d / "ga_profiling_results.csv")

            # split for algorithms
            deap_stats = df_usage[df_usage["Algorithm"].str.upper() == "DEAP"]
            pygad_stats = df_usage[df_usage["Algorithm"].str.upper() == "PYGAD"]

            # Get fitness data
            deap_fitness = get_final_fitness(d / "DEAP_fitness_stats_log.csv")
            pygad_fitness = get_final_fitness(d / "PyGAD_fitness_stats_log.csv")

            if deap_stats.empty or pygad_stats.empty or df_usage.empty:
                print(f"Missing algorithm stats in {d}")
                continue

            # split further into runtime and memory values
            deap_runtime = deap_stats.iloc[0]["Runtime (s)"]
            deap_memory = deap_stats.iloc[0]["Peak Memory (MB)"]

            pygad_runtime = pygad_stats.iloc[0]["Runtime (s)"]
            pygad_memory = pygad_stats.iloc[0]["Peak Memory (MB)"]

            # Combine
            aggregated_data.append({
                "Dataset": d.stem,
                "DEAP Runtime": deap_runtime,
                "DEAP Memory": deap_memory,
                "DEAP Fitness Avg": deap_fitness[0],
                "DEAP Fitness Max": deap_fitness[1],
                "PyGAD Runtime": pygad_runtime,
                "PyGAD Memory": pygad_memory,
                "PyGAD Fitness Avg": pygad_fitness[0],
                "PyGAD Fitness Max": pygad_fitness[1],
            })

        except Exception as e:
            logger.error(f"Error loading GA profiling results from {d.name}: {e}")

    if not aggregated_data:
        logger.error("No GA profiling data available for xlsx aggregation")
        return

    if aggregated_data:
        # Generate xlsx file. I am creating an excel file rather than a csv and matplotlib table to keep in line with my
        # thesis' table formatting. I will take the excel data and put it manually into a latex defined table at a
        # later date
        summary_df = pd.DataFrame(aggregated_data)
        summary_df = summary_df.set_index("Dataset")
        try:
            summary_df.to_excel(f"{BASE_USAGE_PATH}/table_data/{output_file}.xlsx")
            print(f"Aggregated results saved to {output_file}.xlsx")
        except Exception as e:
            print(f"Error writing Excel file: {e}")
    else:
        print("No aggregated data found.")

