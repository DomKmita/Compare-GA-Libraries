from usage_data import usage_tracking_script as usage
from visualisations import visualisation_generator as vg
import pathlib

def main():
    # Run for each dataset in datasets (currently just small)
    size = "small"
    for p in pathlib.Path('datasets').iterdir():
        if p.is_file() and p.suffix == '.csv':
            # Run algorithms and save data
            usage.run_GAs_and_gen_data(data_set_name=p.stem)

            # Create visualisations from the data
            vg.gen_usage_plots(size, p.stem)
            vg.gen_fitness_plots(size, p.stem)
            vg.gen_cpu_profile_plots(size, p.stem)

    vg.gen_aggregated_usage_plots(size)
    vg.gen_aggregated_fitness_plots(size)
    vg.gen_aggregated_cpu_profile_plots(size)

if __name__ == "__main__":
    main()



