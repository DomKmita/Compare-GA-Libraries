import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

usage_data_path = Path(__file__).resolve().parent.parent / "usage_data"
def gen_runtime_plots():
    # Load the stats logs
    deap_df = pd.read_csv(usage_data_path / "deap_stats_log.csv")
    pygad_df = pd.read_csv(usage_data_path / "pygad_stats_log.csv")

    # Plot convergence for DEAP
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(deap_df["gen"], deap_df["avg"], label="Avg Fitness", marker="o")
    plt.plot(deap_df["gen"], deap_df["max"], label="Max Fitness", marker="x")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("DEAP Convergence")
    plt.legend()

    # Plot convergence for PyGAD
    plt.subplot(1, 2, 2)
    plt.plot(pygad_df["gen"], pygad_df["avg"], label="Avg Fitness", marker="o")
    plt.plot(pygad_df["gen"], pygad_df["max"], label="Max Fitness", marker="x")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("PyGAD Convergence")
    plt.legend()
    plt.tight_layout()
    plt.show()

def gen_usage_plots():
    # Read the profiling results from the CSV file
    profile_df = pd.read_csv(usage_data_path / "ga_profiling_results.csv")

    plt.figure(figsize=(10, 4))

    # Runtime comparison
    plt.subplot(1, 2, 1)
    plt.bar(profile_df["Algorithm"], profile_df["Runtime (s)"], color=["blue", "orange"])
    plt.ylabel("Runtime (s)")
    plt.title("Runtime Comparison")

    # Memory comparison
    plt.subplot(1, 2, 2)
    plt.bar(profile_df["Algorithm"], profile_df["Memory Usage (MB)"], color=["blue", "orange"])
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage Comparison")

    plt.tight_layout()
    plt.show()

def gen_cpu_profile_plots():
    # Parse each algorithm's profile data
    df_deap = parse_cprofile_txt(usage_data_path / "DEAP_cpu_profile.txt")
    df_pygad = parse_cprofile_txt(usage_data_path / "PyGAD_cpu_profile.txt")

    # Plot the top 10 functions
    plot_top_functions(df_deap, df_pygad, top=10)

def parse_cprofile_txt(profile_file):
    # Regex to match lines like:
    # "     3714    0.045    0.000    3.124    0.001 /path/to/file.py:123(funcName)"
    line_regex = re.compile(
        r"^\s*(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.*)$"
    )

    # Find the table portion after "Ordered by: cumulative time"
    with profile_file.open() as f:
        lines = f.readlines()

    # cProfile prints a header line with "ncalls tottime percall ...", so we skip down to that
    header_idx = None
    for i, line in enumerate(lines):
        if "ncalls" in line and "cumtime" in line:
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame()  # If table never found

    data_lines = []
    for line in lines[header_idx+1:]:
        line = line.strip()
        # Stop if we reach a blank line or something that clearly isn't part of the table
        if not line or line.startswith("---"):
            break
        data_lines.append(line)

    # Parse each data line with regex
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

def plot_top_functions(df_deap, df_pygad, top=10):
    # Sort by descending cumtime and slice top rows
    df_deap_top = df_deap.sort_values(by="cumtime", ascending=False).head(top)
    df_pygad_top = df_pygad.sort_values(by="cumtime", ascending=False).head(top)

    # Merge them to plot side by side
    # Rename columns cumtime_deap / cumtime_pygad for clarity
    df_deap_top = df_deap_top[["function", "cumtime"]].rename(
        columns={"cumtime": "cumtime_deap"}
    )
    df_pygad_top = df_pygad_top[["function", "cumtime"]].rename(
        columns={"cumtime": "cumtime_pygad"}
    )

    # Do an outer join so we see top functions from both
    merged = pd.merge(df_deap_top, df_pygad_top, on="function", how="outer").fillna(0)

    # Because function strings can be long we shorten them
    merged["short_func"] = merged["function"].apply(lambda x: x[-40:])  # keep last 40 chars

    # Plot
    merged = merged.sort_values("cumtime_deap", ascending=False)  # or just sort by one
    plt.figure(figsize=(12, 6))
    bar_width = 0.4

    x = range(len(merged))
    plt.bar(x, merged["cumtime_deap"], width=bar_width, label="DEAP")
    plt.bar([i + bar_width for i in x], merged["cumtime_pygad"], width=bar_width,
            label="PyGAD")

    plt.xticks([i + bar_width/2 for i in x], merged["short_func"], rotation=45, ha="right")
    plt.ylabel("Cumulative Time (s)")
    plt.title(f"Top {top} Functions by cumtime")
    plt.legend()
    plt.tight_layout()
    plt.show()
