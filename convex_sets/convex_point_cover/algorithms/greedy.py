import numpy as np

from convex_point_cover.utils.probabilistic_point_set_with_convex_hull import ProbabilisticPointSetWithConvexHull
from convex_point_cover.utils.point_set_with_convex_hull import PointSetWithConvexHull


def greedy(points, exclude, probabilistic = False):
    points = np.array(points)
    exclude = np.array(exclude)
    convex_sets = []
    current_set = np.array([])
    for p in points:
        current_set = (
            np.append(current_set, [p], axis=0) if current_set.size else np.array([p])
        )
        if probabilistic:
            point_set = ProbabilisticPointSetWithConvexHull(current_set)
        else:
            point_set = PointSetWithConvexHull(current_set)
        for e in exclude:
            if (probabilistic and point_set.is_inside_hull(e)) or (not probabilistic and point_set.is_inside_hull_2d(e)):
                current_set = current_set[:-1]
                convex_sets.append(current_set)
                current_set = np.array([p])
                break
    convex_sets.append(current_set)
    return convex_sets
