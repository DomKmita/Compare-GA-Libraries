from datasets import dataset_generator
from usage_data import usage_tracking_script as usage
from visualisations import visualisation_generator as vg
from usage_data import xlsx_table_generator as tg
import pathlib

def main():
    # Run algorithms and save data
    size = "small"

    # Generate usage data
    usage.run_GAs_and_gen_data(dataset_size=size, num_runs=3)

    #Create visualisations from the data
    vg.gen_usage_plots(size)
    vg.gen_fitness_plots(size)
    vg.gen_cpu_profile_plots(size)

    vg.gen_aggregated_usage_plots(size)
    vg.gen_aggregated_fitness_plots(size)
    vg.gen_aggregated_cpu_profile_plots(size)

    # create tables
    tg.aggregate_results_to_xlsx(size, "base_algorithm_xlsx_aggregate_data")
    usage.test_mutation_experiments()
    usage.test_selection_experiments()
    usage.test_crossover_experiments()
    usage.test_deap_algorithm_experiments()

if __name__ == "__main__":
    main()



