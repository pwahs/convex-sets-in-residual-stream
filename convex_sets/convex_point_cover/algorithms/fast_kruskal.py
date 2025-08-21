import math
import random
import numpy as np

from scipy.spatial import ConvexHull
from itertools import combinations

from convex_point_cover.utils.point_set_with_convex_hull import PointSetWithConvexHull
from convex_point_cover.utils.probabilistic_point_set_with_convex_hull import (
    ProbabilisticPointSetWithConvexHull,
)

from convex_point_cover.utils.visualize import output_to_ascii
import sys

from convex_point_cover.utils.kalantari_convex_hull import (
    KalantariConvexHull,
)


# Inspired by Kruskal's algorithm: Try to add edges in increasing order of distance,
# but only if the new larger component does not contain an excluded point in it's convex hull
# Utilizes Kalantari's triangle algorithm
# If delta is not none, a point is considered "inside" if it's within delta of the convex hull.
def fast_kruskal(points, exclude, epsilon=0.1, debug=True, delta=None):
    # Find root point for each connected component
    def find(parent, i):
        if parent[i] == i:
            return i
        return find(parent, parent[i])

    # Merge two connected components
    def union(parent, rank, x, y):
        root_x = find(parent, x)
        root_y = find(parent, y)
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

    # Convert parent-trees to partitioning of points into connected components
    def group_points_by_component(parent, points):
        components = {}
        for i in range(len(points)):
            root = find(parent, i)
            if root not in components:
                components[root] = []
            components[root].append(points[i])

        return list(components.values())

    # Start by checking if one convex set is enough:
    hull = KalantariConvexHull(points, epsilon=epsilon, delta=delta)
    if not any(hull.is_inside_hull(point) for point in exclude):
        print("All points in one convex set, returning the set...")
        return [points]
    print("Not a single convex set, proceeding with Kruskal's algorithm...")

    # Generate all edges, sort by length
    edges = []
    maximum = 0
    minimum = math.inf
    for i, j in combinations(range(len(points)), 2):
        dist = np.linalg.norm(
            points[i].astype(np.float64) - points[j].astype(np.float64)
        )
        if dist > maximum:
            maximum = dist
        if dist < minimum:
            minimum = dist
        edges.append((dist, i, j))

    # ratio = 0.4
    # # Filter edges based on distance threshold
    # threshold = (1 - ratio) * minimum + ratio * maximum
    # edges = [edge for edge in edges if edge[0] <= threshold]

    edges.sort()
    parent = np.array(list(range(len(points))))
    rank = [0] * len(points)

    if debug:
        print("sorted...", file=sys.stderr)

    incompatible_pairs = {}

    for e, edge in enumerate(edges):
        if (e % 10000) == 0 and debug:
            print(f"Processing edge {e} of {len(edges)}", file=sys.stderr)
        dist, u, v = edge
        set_u = find(parent, u)
        set_v = find(parent, v)
        if (
            set_u != set_v
            and frozenset([min(set_u, set_v), max(set_u, set_v)])
            not in incompatible_pairs
        ):
            new_points = np.array(
                [
                    points[i]
                    for i in range(len(points))
                    if find(parent, i) in [set_u, set_v]
                ]
            )

            hull = KalantariConvexHull(new_points, epsilon=epsilon, delta=delta)
            if not any(hull.is_inside_hull(point) for point in exclude):
                union(parent, rank, set_u, set_v)
                # print()
                # print(
                #     output_to_ascii(
                #         {"include": points, "exclude": exclude},
                #         group_points_by_component(parent, points),
                #     )
                # )
            else:
                # Memorize incompatible component pair
                incompatible_pairs[
                    frozenset([min(set_u, set_v), max(set_u, set_v)])
                ] = True

    return group_points_by_component(parent, points)
