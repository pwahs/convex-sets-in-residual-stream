import sys
import os

# Add the root directory of the project to the Python path, to avoid needing to config
# VSCode correctly on every device.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from algorithms.greedy import greedy
from algorithms.smarter_greedy import smarter_greedy
from data.datasets import get_datasets, summarize

from convex_point_cover.utils.visualize import input_to_ascii, output_to_ascii

from convex_point_cover.algorithms.kruskal import kruskal
from convex_point_cover.algorithms.fast_kruskal import fast_kruskal


def compare_algos(datasets):
    algos = [greedy, smarter_greedy]
    print(
        f"{'Dataset':<25}{'Optimum-Bound':<15}{'Greedy':<10}{'Smarter Greedy':<15}{'Kruskal':<10}{'Fast Kruskal':<15}]"
    )
    print("=" * 75)
    for dataset in datasets:
        optimal_size = dataset["optimal"]
        # greedy_result = len(greedy(dataset["include"], dataset["exclude"]))
        # smarter_greedy_result = len(
        #     smarter_greedy(dataset["include"], dataset["exclude"])
        # )
        # kruskal_result = len(kruskal(dataset["include"], dataset["exclude"]))

        greedy_prob_result = len(
            greedy(dataset["include"], dataset["exclude"], probabilistic=True)
        )
        print(".")
        # smarter_greedy_prob_result = len(
        #     smarter_greedy(dataset["include"], dataset["exclude"], probabilistic=True)
        # )
        # print(".")
        # kruskal_prob_result = len(
        #     kruskal(
        #         dataset["include"],
        #         dataset["exclude"],
        #         probabilistic=True,
        #         num_rays=1000,
        #     )
        # )
        print(".")

        fast_kruskal_result = len(
            fast_kruskal(dataset["include"], dataset["exclude"], epsilon=0.01)
        )

        print(
            # f"{summarize(dataset):<25}{optimal_size:<15}{greedy_result:<10}{smarter_greedy_result:<15}{kruskal_result:<10}"
        )
        print(
            f"{'':<25}{'Probabilistic':<15}{greedy_prob_result:<10}{-1:<10}{-1:<10}{fast_kruskal_result:<15}"
        )
        print()


if __name__ == "__main__":
    datasets = get_datasets()
    compare_algos(datasets)
    # test_nr = -2
    # print(input_to_ascii(datasets[test_nr]))
    # print()
    # print(
    #     output_to_ascii(
    #         datasets[test_nr],
    #         greedy(datasets[test_nr]["include"], datasets[test_nr]["exclude"]),
    #     )
    # )
    # print()
    # print(
    #     output_to_ascii(
    #         datasets[test_nr],
    #         smarter_greedy(datasets[test_nr]["include"], datasets[test_nr]["exclude"]),
    #     )
    # )
    # print()
    # print(
    #     output_to_ascii(
    #         datasets[test_nr],
    #         kruskal(datasets[test_nr]["include"], datasets[test_nr]["exclude"]),
    #     )
    # )
    # print()
