import time
import cProfile
import pstats
import io
import numpy as np
import pandas as pd
from memory_profiler import memory_usage
from utils.utils import logger, BASE_PATH, get_root, get_directory
from pathlib import Path

from deap import  tools
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

def run_GAs_and_gen_data(size="small", num_runs=3):
    for p in Path(BASE_PATH / f'datasets/{size}').iterdir():
        if p.is_file() and p.suffix == '.csv':
            try:
                df = pd.read_csv(p)
            except FileNotFoundError as e:
                logger.error(f"Dataset not found at: {p}. Failed to run: {e}")
                return pd.DataFrame()

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
            save_data("DEAP", p.name, deap_profile_str, None, deap_fitness_agg)
            save_data("PyGAD", p.name, pygad_profile_str, results_df, pygad_fitness_agg)

            print(f"[{p.name}] Profiling results saved to usage_data")

def test_mutation_experiments(dataset_size="small"):
    # Collected results
    deap_results = []
    pygad_results = []

    # Will probably not be playing around with the different values for mu, sigma, indp, low, high ect. due to time
    # constraints. This is something I can do if I have time but the variability is just too great.
    deap_mutations = {
        "mutInversion": {
            "mutation_func": tools.mutInversion,
            "mutation_kwargs": {}  # No extra arguments needed.
        },
        "mutFlipBit": {
            "mutation_func": tools.mutFlipBit,
            "mutation_kwargs": {"indpb": 0.2}  # 20% probability per gene.
        },
        "mutGaussian": {
            "mutation_func": tools.mutGaussian,
            "mutation_kwargs": {
                "mu": 0.0,
                "sigma": 0.2,
                "indpb": 0.2
            }
        },
        "mutShuffleIndexes": {
            "mutation_func": tools.mutShuffleIndexes,
            "mutation_kwargs": {"indpb": 0.2}
        },
        "mutPolynomialBounded": {
            "mutation_func": tools.mutPolynomialBounded,
            "mutation_kwargs": {
                "eta": 20,  # Distribution index.
                "low": -1.0,  # Lower bound for gene values.
                "up": 1.0,  # Upper bound for gene values.
                "indpb": 0.2  # Added: probability per gene.
            }
        },
        "mutUniformInt": {
            "mutation_func": tools.mutUniformInt,
            "mutation_kwargs": {
                "indpb": 0.2,  # Probability per gene.
                "low": -1,  # Lower bound (as integer).
                "up": 1  # Upper bound (as integer).
            }
        }
    }

    pygad_mutations = {
        "inversion": {"mutation_type": "inversion"},  # Inverts a section of the solution.
        "swap": {"mutation_type": "swap"},  # Swaps two genes.
        "scramble": {"mutation_type": "scramble"},  # Randomly shuffles a subset of genes.
        "random": {"mutation_type": "random"},  # Replaces genes with random values.
        "adaptive": {"mutation_type": "adaptive", "mutation_probability": [0.08, 0.08]}  # Uses an adaptive scheme for mutation rates.
    }

    for p in Path(BASE_PATH / f'datasets/{dataset_size}').iterdir():
        if p.is_file() and p.suffix == '.csv':
            df = pd.read_csv(p)
            dataset_name = p.stem

            # Test each DEAP mutation variant
            for mutation_name, params in deap_mutations.items():
                run_callback = lambda data, p=params: DEAP_tester.run_ga(data, **p)
                res = profile(run_callback, df, f"DEAP_mut_{mutation_name}")
                rt_mem, _, _, _, test_acc = res
                if not rt_mem.empty:
                    deap_results.append({
                        "Dataset": dataset_name,
                        "Experiment": "DEAP_Mutation",
                        "Parameter": mutation_name,
                        "Runtime": rt_mem["Runtime (s)"].mean(),
                        "PeakMemory": rt_mem["Memory Usage (MB)"].apply(np.max).mean(),
                        "TestAccuracy": test_acc,
                    })

            # Test each PyGAD mutation variant
            for mutation_name, params in pygad_mutations.items():
                run_callback = lambda data, p=params: PyGAD_tester.run_ga(data, **p)
                res = profile(run_callback, df, f"PyGAD_mut_{mutation_name}")
                rt_mem, _, _, _, test_acc = res
                if not rt_mem.empty:
                    pygad_results.append({
                        "Dataset": dataset_name,
                        "Experiment": "PyGAD_Mutation",
                        "Parameter": mutation_name,
                        "Runtime": rt_mem["Runtime (s)"].mean(),
                        "PeakMemory": rt_mem["Memory Usage (MB)"].apply(np.max).mean(),
                        "TestAccuracy": test_acc,
                    })

    # Aggregate & save
    deap_df = pd.DataFrame(deap_results)
    pygad_df = pd.DataFrame(pygad_results)
    deap_agg = deap_df.groupby(["Experiment", "Parameter"]).mean(numeric_only=True).reset_index()
    pygad_agg = pygad_df.groupby(["Experiment", "Parameter"]).mean(numeric_only=True).reset_index()

    deap_agg.to_excel(get_directory("table_data", dataset_size) / "mutation_deap_results.xlsx", index=False)
    pygad_agg.to_excel(get_directory("table_data", dataset_size) / "mutation_pygad_results.xlsx", index=False)
    print("Mutation experiments completed. Results saved to Excel.")

def test_crossover_experiments(dataset_size="small"):
    deap_results = []
    pygad_results = []

    # DEAP crossover operators.

    # Only testing crossover for real valued individuals, not using ES crossover operators or permutation based
    # crossover operators. This might be something that I get to at a later date but I would have to revise the
    # types of individuals I create.
    deap_crossovers = {
        "cxOnePoint": {
            "crossover_func": tools.cxOnePoint
        },
        "cxTwoPoint": {
            "crossover_func": tools.cxTwoPoint
        },
        "cxUniform": {
            "crossover_func": tools.cxUniform,
            "crossover_kwargs": {"indpb": 0.7}  # 70% chance for each gene to be exchanged.
        },
        "cxBlend": {
            "crossover_func": tools.cxBlend,
            "crossover_kwargs": {"alpha": 0.5}  # Blend factor for averaging parents.
        },
        "cxSimulatedBinary": {
            "crossover_func": tools.cxSimulatedBinary,
            "crossover_kwargs": {"eta": 15}  # Higher eta yields children closer to parents.
        },
        "cxSimulatedBinaryBounded": {
            "crossover_func": tools.cxSimulatedBinaryBounded,
            "crossover_kwargs": {"eta": 15, "low": -1.0, "up": 1.0}  # Bounded real-valued crossover.
        }
    }

    pygad_crossovers = {
        "single_point": {
            "crossover_type": "single_point"
        },
        "two_points": {
            "crossover_type": "two_points"
        },
        "uniform": {
            "crossover_type": "uniform"
        },
        "scattered": {
            "crossover_type": "scattered"
        },
    }

    for p in Path(BASE_PATH / f'datasets/{dataset_size}').iterdir():
        if p.is_file() and p.suffix == '.csv':
            df = pd.read_csv(p)
            dataset_name = p.stem

            # Test each DEAP crossover
            for cx_name, params in deap_crossovers.items():
                run_callback = lambda data, p=params: DEAP_tester.run_ga(data, **p)
                res = profile(run_callback, df, f"DEAP_cx_{cx_name}")
                rt_mem, _, _, _, test_acc = res
                if not rt_mem.empty:
                    deap_results.append({
                        "Dataset": dataset_name,
                        "Experiment": "DEAP_Crossover",
                        "Parameter": cx_name,
                        "Runtime": rt_mem["Runtime (s)"].mean(),
                        "PeakMemory": rt_mem["Memory Usage (MB)"].apply(np.max).mean(),
                        "TestAccuracy": test_acc,
                    })

            # Test each PyGAD crossover
            for cx_name, params in pygad_crossovers.items():
                run_callback = lambda data, p=params: PyGAD_tester.run_ga(data, **p)
                res = profile(run_callback, df, f"PyGAD_cx_{cx_name}")
                rt_mem, _, _, _, test_acc = res
                if not rt_mem.empty:
                    pygad_results.append({
                        "Dataset": dataset_name,
                        "Experiment": "PyGAD_Crossover",
                        "Parameter": cx_name,
                        "Runtime": rt_mem["Runtime (s)"].mean(),
                        "PeakMemory": rt_mem["Memory Usage (MB)"].apply(np.max).mean(),
                        "TestAccuracy": test_acc,
                    })

    # Aggregate & save
    deap_df = pd.DataFrame(deap_results)
    pygad_df = pd.DataFrame(pygad_results)
    deap_agg = deap_df.groupby(["Experiment", "Parameter"]).mean(numeric_only=True).reset_index()
    pygad_agg = pygad_df.groupby(["Experiment", "Parameter"]).mean(numeric_only=True).reset_index()

    deap_agg.to_excel(get_directory("table_data", dataset_size) / "crossover_deap_results.xlsx", index=False)
    pygad_agg.to_excel(get_directory("table_data", dataset_size) / "crossover_pygad_results.xlsx", index=False)
    print("Crossover experiments completed. Results saved to Excel.")

def test_selection_experiments(dataset_size="small"):
    deap_results = []
    pygad_results = []

    # DEAP selection operators
    deap_selections = {
        "selTournament": {"selection_func": tools.selTournament, "selection_kwargs": {"tournsize": 3}},
        "selRoulette": {"selection_func": tools.selRoulette, "selection_kwargs": {}},
        "selRandom": {"selection_func": tools.selRandom, "selection_kwargs": {}},
        "selStochasticUniversalSampling": {"selection_func": tools.selStochasticUniversalSampling,
                                           "selection_kwargs": {}},
        "selDoubleTournament": {"selection_func": tools.selDoubleTournament,
                                "selection_kwargs": {"fitness_size": 3, "parsimony_size": 2, "fitness_first": True}},
        "selBest": {"selection_func": tools.selBest, "selection_kwargs": {}},
        "selWorst": {"selection_func": tools.selWorst, "selection_kwargs": {}},
    }

    # PyGAD selection operators
    pygad_selections = {
        "tournament": {"parent_selection_type": "tournament", "K_tournament": 3},
        "rank": {"parent_selection_type": "rank"},
        "sss": {"parent_selection_type": "sss"},  # steady state selection
        "rws": {"parent_selection_type": "rws"},  # roulette wheel selection
        "random": {"parent_selection_type": "random"},
        "sus": {"parent_selection_type": "sus"},  # stochastic universal sampling
    }

    for p in Path(BASE_PATH / f'datasets/{dataset_size}').iterdir():
        if p.is_file() and p.suffix == '.csv':
            df = pd.read_csv(p)
            dataset_name = p.stem

            # Base DEAP run
            base_deap = profile(DEAP_tester.run_ga, df, "DEAP_baseSelection")
            rt_mem_base, _, _, _, test_acc_base = base_deap
            if not rt_mem_base.empty:
                deap_results.append({
                    "Dataset": dataset_name,
                    "Experiment": "DEAP_Selection",
                    "Parameter": "BASE",
                    "Runtime": rt_mem_base["Runtime (s)"].mean(),
                    "PeakMemory": rt_mem_base["Memory Usage (MB)"].apply(np.max).mean(),
                    "TestAccuracy": test_acc_base,
                })

            # Test each DEAP selection operator
            for sel_name, params in deap_selections.items():
                run_callback = lambda data, p=params: DEAP_tester.run_ga(data, **p)
                res = profile(run_callback, df, f"DEAP_sel_{sel_name}")
                rt_mem, _, _, _, test_acc = res
                if not rt_mem.empty:
                    deap_results.append({
                        "Dataset": dataset_name,
                        "Experiment": "DEAP_Selection",
                        "Parameter": sel_name,
                        "Runtime": rt_mem["Runtime (s)"].mean(),
                        "PeakMemory": rt_mem["Memory Usage (MB)"].apply(np.max).mean(),
                        "TestAccuracy": test_acc,
                    })

            # Base PyGAD run
            base_pygad = profile(PyGAD_tester.run_ga, df, "PyGAD_baseSelection")
            rt_mem_base, _, _, _, test_acc_base = base_pygad
            if not rt_mem_base.empty:
                pygad_results.append({
                    "Dataset": dataset_name,
                    "Experiment": "PyGAD_Selection",
                    "Parameter": "BASE",
                    "Runtime": rt_mem_base["Runtime (s)"].mean(),
                    "PeakMemory": rt_mem_base["Memory Usage (MB)"].apply(np.max).mean(),
                    "TestAccuracy": test_acc_base,
                })

            # Test each PyGAD selection operator
            for sel_name, params in pygad_selections.items():
                run_callback = lambda data, p=params: PyGAD_tester.run_ga(data, **p)
                res = profile(run_callback, df, f"PyGAD_sel_{sel_name}")
                rt_mem, _, _, _, test_acc = res
                if not rt_mem.empty:
                    pygad_results.append({
                        "Dataset": dataset_name,
                        "Experiment": "PyGAD_Selection",
                        "Parameter": sel_name,
                        "Runtime": rt_mem["Runtime (s)"].mean(),
                        "PeakMemory": rt_mem["Memory Usage (MB)"].apply(np.max).mean(),
                        "TestAccuracy": test_acc,
                    })

    # Aggregate & save
    deap_df = pd.DataFrame(deap_results)
    pygad_df = pd.DataFrame(pygad_results)
    deap_agg = deap_df.groupby(["Experiment", "Parameter"]).mean(numeric_only=True).reset_index()
    pygad_agg = pygad_df.groupby(["Experiment", "Parameter"]).mean(numeric_only=True).reset_index()

    deap_agg.to_excel(get_directory("table_data", dataset_size) / "selection_deap_results.xlsx", index=False)
    pygad_agg.to_excel(get_directory("table_data", dataset_size) / "selection_pygad_results.xlsx", index=False)
    print("Selection experiments completed. Results saved to Excel.")

def test_deap_algorithm_experiments(dataset_size="small"):
    # Only DEAP results matter here
    deap_results = []

    # The different DEAP algorithms to test
    deap_algorithms = {
        "eaSimple": {
            "algorithm": "eaSimple"
        },
        "eaMuPlusLambda": {
            "algorithm": "eaMuPlusLambda",
            "number_of_parents": 20,  # corresponds to mu
            "number_of_offspring": 20  # corresponds to lambda_
        },
        "eaMuCommaLambda": {
            "algorithm": "eaMuCommaLambda",
            "number_of_parents": 20,  # corresponds to mu
            "number_of_offspring": 20  # corresponds to lambda_
        },
        "varAnd": {
            "algorithm": "varAnd"
            # This algorithm applies variation operators over ngen generations.
        },
        "varOr": {
            "algorithm": "varOr",
            "number_of_offspring": 20  # corresponds to lambda_
            # Similar to varAnd, but uses a different strategy for generating offspring.
        },
        "eaGenerateUpdate": {
            "algorithm": "eaGenerateUpdate"
            # This algorithm generates a new population at each generation.
        },
    }

    for p in Path(BASE_PATH / f'datasets/{dataset_size}').iterdir():
        if p.is_file() and p.suffix == '.csv':
            df = pd.read_csv(p)
            dataset_name = p.stem

            # Test each DEAP algorithm
            for algo_name, params in deap_algorithms.items():
                run_callback = lambda data, p=params: DEAP_tester.run_ga(data, **p)
                res = profile(run_callback, df, f"DEAP_{algo_name}")
                rt_mem, _, _, _, test_acc = res
                if not rt_mem.empty:
                    deap_results.append({
                        "Dataset": dataset_name,
                        "Experiment": "DEAP_Algorithms",
                        "Parameter": algo_name,
                        "Runtime": rt_mem["Runtime (s)"].mean(),
                        "PeakMemory": rt_mem["Memory Usage (MB)"].apply(np.max).mean(),
                        "TestAccuracy": test_acc,
                    })

    # Aggregate & save
    deap_df = pd.DataFrame(deap_results)
    deap_agg = deap_df.groupby(["Experiment", "Parameter"]).mean(numeric_only=True).reset_index()

    deap_agg.to_excel(get_directory("table_data", dataset_size) / "deap_algorithms_results.xlsx", index=False)
    print("DEAP algorithm experiments completed. Results saved to Excel.")
