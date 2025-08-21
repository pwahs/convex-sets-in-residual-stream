import numpy as np


class KalantariConvexHull:
    # Implements triangle algorithm for finding separating hyperplanes
    # as described in https://arxiv.org/pdf/1204.1873
    # If there is no such plane, it finds a point in the convex hull, which is by a factor
    # of epsilon closer to the point than the current pivot, which is one of the input points of the convex hull.
    def __init__(self, points, epsilon=0.1):
        self.points = [point.astype(np.float64) for point in points]
        self.epsilon = epsilon
        # If previous queries generated separating hyperplanes, store them here
        self.separating_hyperplanes = []

    def plane_separates(self, plane, point):
        normal, offset = plane
        # Check if the new point is on the negative side of the hyperplane
        return np.dot(normal, point.astype(np.float64)) + offset < 0

    def add_point(self, point):
        # Convert point to 32-bit precision and add to points list
        self.points.append(point.astype(np.float64))
        # Only keep planes that also have point on the right side
        self.separating_hyperplanes = [
            plane
            for plane in self.separating_hyperplanes
            if self.plane_separates(plane, point) == False
        ]

    def is_inside_hull(self, point16b):
        point = point16b.astype(np.float64)
        for plane in self.separating_hyperplanes:
            if self.plane_separates(plane, point):
                return False
        # Initialize Triangle Algorithm
        # pivot point, v is always one of self.points
        v = min(self.points, key=lambda p: np.linalg.norm(p - point))
        # Approximating the witness of separation, or point
        p_prime = v

        while True:
            if np.linalg.norm(p_prime - point) < self.epsilon * np.linalg.norm(
                v - point
            ):
                # Found approximation, point is likely inside the convex hull
                return True
            # Find a new pivot v with d(p', v) >= d(p, v)
            found = False
            # Project all points onto the p_prime - point line
            direction = point - p_prime
            direction = direction / np.linalg.norm(direction)
            max_projection = float("-inf")
            best_pivot = None

            for candidate in self.points:
                projection = np.dot(candidate - p_prime, direction)
                if projection > max_projection:
                    max_projection = projection
                    best_pivot = candidate

            if np.linalg.norm(p_prime - best_pivot) >= np.linalg.norm(
                point - best_pivot
            ):
                v = best_pivot
                found = True
            # Just any, not necessarily the best pivot:
            # for pivot in self.points:
            #     if np.linalg.norm(p_prime - pivot) >= np.linalg.norm(point - pivot):
            #         v = pivot
            #         found = True
            #         break
            if not found:
                # p' is a witness of separation, the bisecting hyperplane between p' and p separates
                # the point from the convex hull
                # Calculate the line direction from point to p_prime
                normal = p_prime - point
                normal = normal / np.linalg.norm(normal)

                # Find the point in self.points that is closest to the point along this line
                min_distance = float("inf")
                closest_hull_point = None

                for hull_point in self.points:
                    # Project hull_point onto the line
                    projection = np.dot(hull_point - point, normal)
                    if projection < min_distance:
                        min_distance = projection
                        closest_hull_point = hull_point
                if min_distance <= 0:
                    original_options = np.get_printoptions()

                    # Set options to print full arrays
                    np.set_printoptions(threshold=np.inf, precision=4)

                    print(
                        "Error, hyperplane is not separating for unknown reason. Data:"
                    )
                    print(
                        f"{len(self.points)} points, distance p_prime to point {np.linalg.norm(p_prime - point)}, min_distance {min_distance}."
                    )
                    print(f"Point {point}")
                    print(f"p_prime {p_prime}")
                    print(f"Closest hull point {closest_hull_point}")
                    print(f"v {v}")
                    print(f"Normal {normal}")

                    for i, hull_point in enumerate(self.points[:5]):
                        print(f"Hull point {i} of {len(self.points)}: {hull_point}")

                    # Restore original options
                    np.set_printoptions(**original_options)

                    assert (
                        min_distance > 0
                    )  # because p_prime is a witness of separation
                # Create separating hyperplane that goes through closest_hull_point
                offset = -np.dot(normal, closest_hull_point)
                self.separating_hyperplanes.append((normal, offset))

                return False
            # We have a new pivot, calculate the new p_prime, by projecting point onto the line
            # through p_prime and v
            direction = v - p_prime
            direction /= np.linalg.norm(direction)
            # Project point onto the line through p_prime along direction
            t = np.dot(point - p_prime, direction)
            p_prime = p_prime + t * direction

        return True
