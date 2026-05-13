from itertools import product
from typing import List
import numpy as np
import open3d as o3d

from .utils import dist

corner_indices = np.array([v for v in list(product([-1, 0, 1], repeat=3)) if 0 not in v])
NEIGHBOR_INDICES = np.array(list(product([-1, 0, 1], repeat=3)))
NEIGHBOR_INDICES = np.delete(NEIGHBOR_INDICES, 13, axis=0)


class Octree:
    def __init__(self, cloud=None, center=None, normals=None) -> None:
        if cloud is None:
            return
        self.cloud = cloud
        self.root = self
        self.parent = None
        self.level = 0
        self.leaves: List[Octree] = []
        self.indices = []
        self.is_leaf = False
        self.normal = np.zeros(3)
        self.residual = float('inf')
        self.d = 0.0
        self.num_nb = 0
        self.is_unallocated = False
        self.children = [None] * 8

        self.normals = normals

        pts = np.array(cloud)
        minimum = pts.min(axis=0)
        maximum = pts.max(axis=0)

        if center is not None:
            self.center = np.array(center, dtype=float)
        else:
            self.center = (minimum + maximum) / 2

        self.size = (maximum - minimum).max()
        self.indices = list(range(len(cloud)))

        self._grid = None
        self._d_min = None
        self._all_uniform = []  # all uniform-level nodes including empties

    def __hash__(self) -> int:
        return hash(tuple(self.center * 100))

    def __eq__(self, __o: object) -> bool:
        return id(self) == id(__o)

    # ── paper-style construction ──────────────────────────────────────

    def create(self, d=0.4, r_th=0.08, d_min=0.05):
        """Phase 1: uniform → size ≈ d.  Phase 2: adaptive for high residual."""
        self._d = d
        self._uniform_subdivide(d)
        print(f'Uniform leaves: {len(self.leaves)}')

        for leaf in self.leaves:
            if len(leaf.indices) >= 3:
                leaf._calc_n_r()

        self._adaptive_subdivide(r_th, d_min)
        print(f'After adaptive: {len(self.leaves)} leaves')

        self._build_grid(d_min)

    def _uniform_subdivide(self, d):
        if len(self.indices) < 3:
            self.is_leaf = True
            self.root._all_uniform.append(self)
            if len(self.indices) > 0:
                self.root.leaves.append(self)
            return
        if self.size <= d:
            self.is_leaf = True
            self.root._all_uniform.append(self)
            self.root.leaves.append(self)
            return
        self._split()
        for child in self.children:
            child._uniform_subdivide(d)

    def _adaptive_subdivide(self, r_th, d_min):
        changed = True
        while changed:
            changed = False
            for leaf in list(self.leaves):
                if leaf.residual > r_th and leaf.size > d_min * 2:
                    self.leaves.remove(leaf)
                    if leaf in self._all_uniform:
                        self._all_uniform.remove(leaf)
                    leaf.is_leaf = False
                    leaf._split()
                    for child in leaf.children:
                        child.is_leaf = True
                        self._all_uniform.append(child)
                        if len(child.indices) >= 3:
                            child._calc_n_r()
                            self.leaves.append(child)
                            changed = True
                        elif len(child.indices) > 0:
                            self.leaves.append(child)

    def _split(self):
        new_size = self.size / 2
        c = self.center
        new_centers = [
            [c[0] - new_size, c[1] - new_size, c[2] - new_size],
            [c[0] - new_size, c[1] - new_size, c[2] + new_size],
            [c[0] - new_size, c[1] + new_size, c[2] - new_size],
            [c[0] - new_size, c[1] + new_size, c[2] + new_size],
            [c[0] + new_size, c[1] - new_size, c[2] - new_size],
            [c[0] + new_size, c[1] - new_size, c[2] + new_size],
            [c[0] + new_size, c[1] + new_size, c[2] - new_size],
            [c[0] + new_size, c[1] + new_size, c[2] + new_size],
        ]
        self.children = [Octree._make_child(self, np.array(nc), new_size)
                         for nc in new_centers]

        for i in self.indices:
            pt = self.cloud[i]
            idx = ((pt[0] > c[0]) << 2) | ((pt[1] > c[1]) << 1) | (pt[2] > c[2])
            self.children[idx].indices.append(i)

    @staticmethod
    def _make_child(parent, center, size):
        child = Octree()
        child.cloud = parent.cloud
        child.parent = parent
        child.root = parent.root
        child.level = parent.level + 1
        child.center = center
        child.size = size
        child.is_leaf = False
        child.indices = []
        child.num_nb = 0
        child.is_unallocated = False
        child.normals = parent.normals
        child.children = [None] * 8
        child.normal = np.zeros(3)
        child.residual = float('inf')
        child.d = 0.0
        return child

    # ── PCA normal + RMS residual ─────────────────────────────────────

    def _calc_n_r(self):
        inliers = np.array([self.cloud[i] for i in self.indices])
        centroid = inliers.mean(axis=0)
        centered = inliers - centroid
        cov = (centered.T @ centered) / len(inliers)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        self.normal = eigenvectors[:, 0]
        # orient towards room centre so adjacent voxels share sign
        if np.dot(self.normal, self.root.center - self.center) < 0:
            self.normal = -self.normal
        self.d = float(np.dot(self.normal, centroid))
        dists = np.abs(np.dot(centered, self.normal))
        self.residual = float(np.sqrt(np.mean(dists ** 2)))

    # alias kept for backward compat
    calc_n_r = _calc_n_r

    # ── neighbour lookup via fine grid ─────────────────────────────────

    def _build_grid(self, d_min):
        self._d_min = d_min
        self._grid = {}
        for node in self._all_uniform:
            half = node.size / 2
            lo = np.floor((node.center - half) / d_min).astype(int)
            hi = np.floor((node.center + half) / d_min).astype(int)
            for ix in range(lo[0], hi[0] + 1):
                for iy in range(lo[1], hi[1] + 1):
                    for iz in range(lo[2], hi[2] + 1):
                        self._grid[(ix, iy, iz)] = node

    def get_neighbors(self):
        assert self.is_leaf and self.root._grid is not None
        d_min = self.root._d_min
        d = self.root._d  # uniform voxel size, max gap between filled leaves
        half = self.size / 2
        lo = np.floor((self.center - half) / d_min).astype(int)
        hi = np.floor((self.center + half) / d_min).astype(int)
        g = self.root._grid
        padding = int(np.ceil(self.size / d_min))

        nbs = set()
        for ix in range(lo[0] - padding, hi[0] + padding + 1):
            for iy in range(lo[1] - padding, hi[1] + padding + 1):
                for iz in range(lo[2] - padding, hi[2] + padding + 1):
                    nb = g.get((ix, iy, iz))
                    if nb is None or nb is self or len(nb.indices) < 3:
                        continue
                    dist = np.abs(self.center - nb.center)
                    # allow gap of up to 1 uniform voxel between thin surfaces
                    if np.all(dist <= half + nb.size / 2 + d + 1e-6):
                        nbs.add(nb)
        self.num_nb = len(nbs)
        return list(nbs)

    # ── buffer zone (used by general_refinement) ──────────────────────

    def get_buffer_zone_points(self, kdtree):
        buffer_points = {}
        for nb in self.get_neighbors():
            if not nb.is_unallocated:
                continue
            buffer_points[nb] = set(nb.indices)
        return buffer_points

    def find_leaf_is_allocated(self, index):
        for leaf in self.root.leaves:
            if index in leaf.indices:
                return leaf.is_unallocated
        return False

    def draw(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            [self.cloud[p] for p in self.indices])
        o3d.visualization.draw_geometries([pcd])


# ── helper used by extract_boundary_voxels ────────────────────────────

def get_neighbor_count_same_cluster(leaf, cluster_centers):
    count = 0
    for nb in leaf.get_neighbors():
        if tuple(np.around(nb.center, decimals=6)) in cluster_centers:
            count += 1
    return count
