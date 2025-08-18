from scipy.spatial import ConvexHull, Delaunay
import numpy as np

class PointSetWithConvexHull:
    def __init__(self, points):
        self.points = points
        self.delaunay = None
        self.subspace = None

    def add_point(self, point):
        self.points.append(point)
        self.delaunay = None
        self.subspace = None

    def is_point_on_line_segment(self, a, b, c):
        assert a.shape == (2,), "Points must be 2D vectors"
        # Check if point c is on the line segment between points a and b
        # Check is c is collinear with a and b
        cross_product = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(cross_product) > 1e-10:
            return False

        # Check is c is on the other side of a compared to b
        dot_product = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1]) * (b[1] - a[1])
        if dot_product < 0:
            return False

        # Check is c is further away from a than b
        squared_length_ab = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
        if dot_product > squared_length_ab:
            return False

        return True

    def is_inside_hull_2d(self, point):
        assert point.shape == (2,), "Point must be a 2D vector"
        if self.delaunay is None:
            if len(self.points) == 0:
                return False
            if len(self.points) == 1:
                return (self.points[0] == point).all()
            if len(self.points) == 2:
                return self.is_point_on_line_segment(
                    self.points[0], self.points[1], point
                )
            self.delaunay = Delaunay(ConvexHull(self.points).points)
        return self.delaunay.find_simplex(point) != -1

#-------------------------------------High-dimensional implementation-------------------------------
    def find_affine_subspace(self, points, tolerance=1e-10):
        """
        Find the k-dimensional affine subspace spanned by N-dimensional points.
        
        Args:
            points: array-like of shape (n_points, n_dimensions)
            tolerance: threshold for determining effective rank
        
        Returns:
            dict containing:
            - k: dimension of the affine subspace
            - origin: point on the affine subspace (centroid)
            - basis: orthonormal basis vectors for the subspace
            - projected_points: points projected onto the subspace
            - projections: The projected_points in terms of the new basis (k-dimensional vectors)
        """
        points = np.array(points)
        n_points, n_dims = points.shape
        
        if n_points == 0:
            return {"k": -1, "origin": None, "basis": None, "projected_points": None, "projections": None}
        
        # Center the points (translate to origin)
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        # Use SVD to find the span
        U, s, Vt = np.linalg.svd(centered_points.T, full_matrices=False)
        
        # Determine effective rank (dimension of subspace)
        k = np.sum(s > tolerance)
        
        if k == 0:
            # All points are the same
            return {
                "k": 0,
                "origin": centroid,
                "basis": np.zeros((0, n_dims)),
                "projected_points": np.tile(centroid, (n_points, 1)),
                "projections": np.zeros((n_points, 0))
            }
        
        # Basis vectors for the k-dimensional subspace
        basis = U[:, :k].T  # Shape: (k, n_dims)
        
        # Project points onto the subspace
        projections = centered_points @ basis.T  # Shape: (n_points, k)
        projected_points = projections @ basis + centroid
        
        return {
            "k": k,
            "origin": centroid,
            "basis": basis,
            "projected_points": projected_points,
            "projections": projections  # coordinates in subspace
        }
    
    def is_point_in_subspace(self, point, atol=1e-10):
        # Project point onto subspace in subspace coordinates, then convert back
        # Afterwards, check if we are within tolerance of the original point
        origin = self.subspace["origin"]
        basis = self.subspace["basis"]
        if basis is None or basis.shape[0] == 0:
            # Subspace is a single point
            return np.allclose(point, origin, atol=atol)
        # Compute coordinates of point in subspace
        coords = (point - origin) @ basis.T
        reconstructed = coords @ basis + origin
        return np.allclose(point, reconstructed, atol=atol)
    
    def is_inside_hull(self, point):
        if self.delaunay is None:
            if self.subspace is None:
                self.subspace = self.find_affine_subspace(self.points)
            if not self.is_point_in_subspace(point):
                return False
            self.delaunay = Delaunay(self.subspace["projections"])
        origin = self.subspace["origin"]
        basis = self.subspace["basis"]
        coords = (point - origin) @ basis.T
        return self.delaunay.find_simplex(coords) != -1