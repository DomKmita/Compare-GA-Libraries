import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import ast
from utils.logger import logger

# Base path
BASE_USAGE_PATH = Path(__file__).resolve().parent.parent

def get_path(directory, dataset_size, dataset_name):
    return BASE_USAGE_PATH / directory / dataset_size / dataset_name

def gen_fitness_plots(dataset_size, dataset_name):
    dataset_path = get_path("usage_data", dataset_size, dataset_name)
    try:
        deap_df = pd.read_csv(dataset_path / "DEAP_fitness_stats_log.csv")
    except Exception as e:
        logger.error(f"Error loading DEAP fitness stats for {dataset_name}: {e}")
        return
    try:
        pygad_df = pd.read_csv(dataset_path / "PyGAD_fitness_stats_log.csv")
    except Exception as e:
        logger.error(f"Error loading PyGAD fitness stats for {dataset_name}: {e}")
        return

    plt.figure(figsize=(12, 5))
    # DEAP plot
    plt.subplot(1, 2, 1)
    plt.plot(deap_df["gen"], deap_df["avg"], label="Avg Fitness", marker="o")
    plt.plot(deap_df["gen"], deap_df["max"], label="Max Fitness", marker="x")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"DEAP Convergence - {dataset_name}")
    plt.legend()
    # PyGAD plot
    plt.subplot(1, 2, 2)
    plt.plot(pygad_df["gen"], pygad_df["avg"], label="Avg Fitness", marker="o")
    plt.plot(pygad_df["gen"], pygad_df["max"], label="Max Fitness", marker="x")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"PyGAD Convergence - {dataset_name}")
    plt.legend()

    plt.tight_layout()
    figure_dir = get_path("visualisations", dataset_size, dataset_name)
    figure_dir.mkdir(parents=True, exist_ok=True)
    filename = figure_dir / f"{dataset_name}_fitness.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def gen_usage_plots(dataset_size, dataset_name):
    dataset_path = get_path("usage_data", dataset_size, dataset_name)
    try:
        profile_df = pd.read_csv(dataset_path / "ga_profiling_results.csv")
        profile_df["Memory Usage (MB)"] = profile_df["Memory Usage (MB)"].apply(ast.literal_eval)
    except Exception as e:
        logger.error(f"Error loading GA profiling results for {dataset_name}: {e}")
        return

    # Compute average & peak memory for each algorithm row
    profile_df["AvgMem"] = profile_df["Memory Usage (MB)"].apply(lambda mem: np.mean(mem))
    profile_df["PeakMem"] = profile_df["Memory Usage (MB)"].apply(lambda mem: np.max(mem))

    plt.figure(figsize=(4, 4))
    plt.bar(profile_df["Algorithm"], profile_df["Runtime (s)"], color=["blue", "orange"])
    plt.ylabel("Runtime (s)")
    plt.title(f"Runtime Comparison \n {dataset_name}")
    figure_dir = get_path("visualisations", dataset_size, dataset_name)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(figure_dir / f"{dataset_name}_runtime.png", dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(4, 4))
    algs = profile_df["Algorithm"]
    avg_vals = profile_df["AvgMem"]
    peak_vals = profile_df["PeakMem"]

    yerr = peak_vals - avg_vals

    plt.bar(algs, avg_vals, yerr=yerr, capsize=5, color=["blue", "orange"])
    plt.ylabel("Memory (MB)")
    plt.title(f"Memory Usage - Avg + Peak \n {dataset_name}")
    plt.tight_layout()
    plt.savefig(figure_dir / f"{dataset_name}_memory.png", dpi=300)
    plt.show()
    plt.close()


def gen_cpu_profile_plots(dataset_size, dataset_name):
    dataset_path = get_path("usage_data", dataset_size, dataset_name)
    try:
        df_deap = parse_cprofile_txt(dataset_path / "DEAP_cpu_profile.txt")
    except Exception as e:
        logger.error(f"Error loading DEAP CPU profile for {dataset_name}: {e}")
        return
    try:
        df_pygad = parse_cprofile_txt(dataset_path / "PyGAD_cpu_profile.txt")
    except Exception as e:
        logger.error(f"Error loading PyGAD CPU profile for {dataset_name}: {e}")
        return

    plot_top_functions(df_deap, df_pygad, dataset_size, dataset_name)

def gen_aggregated_fitness_plots(dataset_size):
    usage_dirs = [d for d in (BASE_USAGE_PATH / "usage_data" / dataset_size).iterdir() if d.is_dir()]
    if not usage_dirs:
        logger.error("No dataset directories found for size: " + dataset_size)
        return

    # Load all fitnesses across datasets
    deap_list, pygad_list = [], []
    for d in usage_dirs:
        try:
            df_deap = pd.read_csv(d / "DEAP_fitness_stats_log.csv")
            deap_list.append(df_deap)
        except Exception as e:
            logger.error(f"Error loading DEAP stats from {d.name}: {e}")
        try:
            df_pygad = pd.read_csv(d / "PyGAD_fitness_stats_log.csv")
            pygad_list.append(df_pygad)
        except Exception as e:
            logger.error(f"Error loading PyGAD stats from {d.name}: {e}")

    if not deap_list or not pygad_list:
        logger.error("Insufficient data for averaged fitness plots.")
        return

    # combine all fitness results across datasets
    agg_deap = pd.concat(deap_list).groupby("gen", as_index=False).mean()
    agg_pygad = pd.concat(pygad_list).groupby("gen", as_index=False).mean()

    plt.figure(figsize=(12, 5))

    # Averaged DEAP
    plt.subplot(1, 2, 1)
    plt.plot(agg_deap["gen"], agg_deap["avg"], label="Avg Fitness", marker="o")
    plt.plot(agg_deap["gen"], agg_deap["max"], label="Max Fitness", marker="x")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Averaged DEAP Convergence")
    plt.legend()

    # Averaged PyGAD
    plt.subplot(1, 2, 2)
    plt.plot(agg_pygad["gen"], agg_pygad["avg"], label="Avg Fitness", marker="o")
    plt.plot(agg_pygad["gen"], agg_pygad["max"], label="Max Fitness", marker="x")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Averaged PyGAD Convergence")
    plt.legend()

    plt.tight_layout()
    figure_dir = get_path("visualisations", dataset_size, "aggregate")
    figure_dir.mkdir(parents=True, exist_ok=True)
    filename = figure_dir / "averaged_fitness.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def gen_aggregated_usage_plots(dataset_size):
    usage_dirs = [
        d for d in (BASE_USAGE_PATH / "usage_data" / dataset_size).iterdir()
        if d.is_dir()
    ]
    if not usage_dirs:
        logger.error("No dataset directories found for size: " + dataset_size)
        return

    usage_list = []
    for d in usage_dirs:
        try:
            df_usage = pd.read_csv(d / "ga_profiling_results.csv")
            # convert string list to regular list
            df_usage["Memory Usage (MB)"] = df_usage["Memory Usage (MB)"].apply(ast.literal_eval)
            # Compute average & peak memory for each run
            df_usage["AvgMem"] = df_usage["Memory Usage (MB)"].apply(np.mean)
            df_usage["PeakMem"] = df_usage["Memory Usage (MB)"].apply(np.max)
            usage_list.append(df_usage)
        except Exception as e:
            logger.error(f"Error loading GA profiling results from {d.name}: {e}")

    if not usage_list:
        logger.error("No GA profiling data available for aggregated usage plots.")
        return

    # combine all averages
    agg_usage = pd.concat(usage_list)

    agg_usage_bar = agg_usage.groupby("Algorithm", as_index=False).agg({
        "Runtime (s)": "mean",
        "AvgMem": "mean",
        "PeakMem": "mean"
    })

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].bar(agg_usage_bar["Algorithm"], agg_usage_bar["Runtime (s)"], color=["blue", "orange"])
    axs[0].set_ylabel("Runtime (s)")
    axs[0].set_title(f"Averaged Runtime ({dataset_size} Datasets)")

    avg_vals = agg_usage_bar["AvgMem"]
    peak_vals = agg_usage_bar["PeakMem"]
    yerr = peak_vals - avg_vals

    axs[1].bar(agg_usage_bar["Algorithm"], avg_vals, yerr=yerr, color=["blue", "orange"], capsize=5)
    axs[1].set_ylabel("Memory (MB)")
    axs[1].set_title(f"Averaged Memory + Peak ({dataset_size} Datasets)")

    plt.tight_layout()
    figure_dir = get_path("visualisations", dataset_size, "aggregate")
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_dir / "averaged_usage.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def gen_aggregated_cpu_profile_plots(dataset_size):
    usage_dirs = [d for d in (BASE_USAGE_PATH / "usage_data" / dataset_size).iterdir() if d.is_dir()]
    deap_dfs, pygad_dfs = [], []
    for d in usage_dirs:
        try:
            df_deap = parse_cprofile_txt(d / "DEAP_cpu_profile.txt")
            if not df_deap.empty:
                deap_dfs.append(df_deap)
        except Exception as e:
            logger.error(f"Error loading DEAP CPU profile from {d.name}: {e}")
        try:
            df_pygad = parse_cprofile_txt(d / "PyGAD_cpu_profile.txt")
            if not df_pygad.empty:
                pygad_dfs.append(df_pygad)
        except Exception as e:
            logger.error(f"Error loading PyGAD CPU profile from {d.name}: {e}")

    if not deap_dfs and not pygad_dfs:
        logger.error("No CPU profile data available for aggregated CPU plots.")
        return

    # Average each function’s cumtime across all runs, for DEAP and PyGAD separately
    if deap_dfs:
        agg_deap = pd.concat(deap_dfs).groupby("function", as_index=False)["cumtime"].mean()
        agg_deap.rename(columns={"cumtime": "cumtime_deap"}, inplace=True)
    else:
        agg_deap = pd.DataFrame(columns=["function", "cumtime_deap"])

    if pygad_dfs:
        agg_pygad = pd.concat(pygad_dfs).groupby("function", as_index=False)["cumtime"].mean()
        agg_pygad.rename(columns={"cumtime": "cumtime_pygad"}, inplace=True)
    else:
        agg_pygad = pd.DataFrame(columns=["function", "cumtime_pygad"])

    # Merge DEAP & PyGAD average data
    merged = pd.merge(agg_deap, agg_pygad, on="function", how="outer").fillna(0)

    # Compute total average cumtime across both algorithms (to decide which are "top")
    merged["total"] = merged["cumtime_deap"] + merged["cumtime_pygad"]

    # Keep only top N by total
    top_n = 20
    merged = merged.nlargest(top_n, "total")

    # Sort by DEAP’s cumtime
    merged = merged.sort_values("cumtime_deap", ascending=False)

    # Shorten function name
    merged["short_func"] = merged["function"].apply(lambda x: x[-40:])

    # Plot
    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    x = range(len(merged))
    plt.bar(x, merged["cumtime_deap"], width=bar_width, label="DEAP")
    plt.bar([i + bar_width for i in x], merged["cumtime_pygad"], width=bar_width, label="PyGAD")

    plt.xticks([i + bar_width/2 for i in x], merged["short_func"], rotation=45, ha="right")
    plt.ylabel("Average Cumulative Time (s)")
    plt.title(f"Top {top_n} Functions by Avg cumtime - {dataset_size}")
    plt.legend()
    plt.tight_layout()

    figure_dir = get_path("visualisations", dataset_size, "aggregate")
    figure_dir.mkdir(parents=True, exist_ok=True)
    filename = figure_dir / "averaged_cpu.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def parse_cprofile_txt(profile_file):
    """
    Parses a cProfile text file into a DataFrame.
    Expected line format:
      "  ncalls   tottime   percall   cumtime   percall  filename:lineno(function)"
    """
    line_regex = re.compile(r"^\s*(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.*)$")
    try:
        with profile_file.open() as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Could not open CPU profile file {profile_file}: {e}")
        return pd.DataFrame()

    header_idx = None
    for i, line in enumerate(lines):
        if "ncalls" in line and "cumtime" in line:
            header_idx = i
            break

    if header_idx is None:
        logger.error(f"No valid CPU profiling data found in {profile_file}")
        return pd.DataFrame()

    # Get all cpu profiling lines excluding headers
    data_lines = [line.strip() for line in lines[header_idx+1:] if line.strip() and not line.startswith("---")]
    records = []
    for line in data_lines:
        match = line_regex.match(line)
        if match:
            ncalls, tottime, percall1, cumtime, percall2, func = match.groups()
            records.append({
                "ncalls": ncalls,
                "tottime": float(tottime),
                "cumtime": float(cumtime),
                "function": func
            })
    return pd.DataFrame(records)

def plot_top_functions(df_deap: pd.DataFrame, df_pygad: pd.DataFrame, dataset_size, dataset_name, top=10):
    if df_deap.empty or df_pygad.empty:
        logger.error("One or both CPU profile dataframes are empty. Skipping CPU profile plots.")
        return

    df_deap_top = (df_deap.sort_values(by="cumtime", ascending=False)
                      .head(top)[["function", "cumtime"]]
                      .rename(columns={"cumtime": "cumtime_deap"}))
    df_pygad_top = (df_pygad.sort_values(by="cumtime", ascending=False)
                      .head(top)[["function", "cumtime"]]
                      .rename(columns={"cumtime": "cumtime_pygad"}))
    merged = pd.merge(df_deap_top, df_pygad_top, on="function", how="outer").fillna(0)
    merged["short_func"] = merged["function"].apply(lambda x: x[-40:])
    merged = merged.sort_values("cumtime_deap", ascending=False)

    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    x = range(len(merged))
    plt.bar(x, merged["cumtime_deap"], width=bar_width, label="DEAP")
    plt.bar([i + bar_width for i in x], merged["cumtime_pygad"], width=bar_width, label="PyGAD")
    plt.xticks([i + bar_width/2 for i in x], merged["short_func"], rotation=45, ha="right")
    plt.ylabel("Cumulative Time (s)")
    plt.title(f"Top {top} Functions by cumtime - {dataset_name}")
    plt.legend()
    plt.tight_layout()
    figure_dir = get_path("visualisations", dataset_size, dataset_name)
    figure_dir.mkdir(parents=True, exist_ok=True)
    filename = figure_dir / f"{dataset_name}_cpu.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
