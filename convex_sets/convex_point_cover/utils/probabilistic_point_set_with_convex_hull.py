import numpy as np


class ProbabilisticPointSetWithConvexHull:
    def __init__(self, points, num_rays=100):
        self.points = points
        self.num_rays = num_rays
        # If previous queries generated separating hyperplanes, store them here
        self.separating_hyperplanes = []

    def plane_separates(self, plane, point):
        normal, offset = plane
        # Check if the new point is on the negative side of the hyperplane
        return np.dot(normal, point) + offset < 0

    def add_point(self, point):
        self.points.append(point)
        # Only keep planes that also have point on the right side
        self.separating_hyperplanes = [
            plane
            for plane in self.separating_hyperplanes
            if self.plane_separates(plane, point) == False
        ]

    def is_inside_hull(self, point):
        for plane in self.separating_hyperplanes:
            if self.plane_separates(plane, point):
                return False
        dim = len(point)
        for _ in range(self.num_rays):
            direction = np.random.randn(dim)
            direction /= np.linalg.norm(direction)
            projections = [np.dot(p - point, direction) for p in self.points]
            if all(x >= 0 for x in projections) or all(x <= 0 for x in projections):
                # Find the closest point to the separating plane
                if all(x >= 0 for x in projections):
                    min_proj = min(projections)
                else:
                    min_proj = max(projections)
                # The plane normal is direction, offset so that it passes through the closest point
                closest_idx = projections.index(min_proj)
                closest_point = self.points[closest_idx]
                normal = direction
                # Adjust normal direction so all points have non-negative dot product
                if any(x < 0 for x in projections):
                    normal = -normal
                offset = -np.dot(normal, closest_point)
                self.separating_hyperplanes.append((normal, offset))
                return False
        return True
