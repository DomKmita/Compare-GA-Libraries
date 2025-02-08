from usage_data import usage_tracking_script as usage
from visualisations import visualisation_generator as vg

def main():
    # Run algorithms and save data
    usage.run_GAs_and_gen_data()

    # Create visualisations from the data
    # vg.gen_usage_plots()
    # vg.gen_runtime_plots()
    # vg.gen_cpu_profile_plots()

if __name__ == "__main__":
    main()



