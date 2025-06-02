import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from rasterio.plot import show
from scipy.ndimage import label, binary_dilation
from collections import deque
import os


class FloodSimulator:
    def __init__(self, elevation_data, cell_size=1.0, nodata_value=-9999):
        """
        Initialize flood simulator with elevation data.

        Parameters:
        elevation_data: 2D numpy array of elevation values
        cell_size: size of each cell in meters
        nodata_value: value representing no data in elevation
        """
        self.elevation = np.array(elevation_data, dtype=np.float32)
        self.cell_size = cell_size
        self.nodata_value = nodata_value
        self.rows, self.cols = self.elevation.shape

        # Create mask for valid data
        self.valid_mask = self.elevation != nodata_value

    @classmethod
    def from_geotiff(cls, filepath):
        """Load elevation data from GeoTIFF file."""
        with rasterio.open(filepath) as src:
            elevation = src.read(1)
            transform = src.transform
            cell_size = abs(transform[0])  # assuming square pixels
            nodata = src.nodata
        return cls(elevation, cell_size, nodata)

    def simple_inundation(self, water_level):
        """
        Simple flood inundation: all areas below water level are flooded.

        Parameters:
        water_level: height of water surface above datum

        Returns:
        flooded: boolean array where True indicates flooded areas
        """
        flooded = (self.elevation <= water_level) & self.valid_mask
        return flooded

    def connected_flood_fill(self, seed_points, water_level):
        """
        Flood fill algorithm starting from seed points.
        Only connected areas below water level are flooded.

        Parameters:
        seed_points: list of (row, col) tuples for flood starting points
        water_level: maximum water surface elevation

        Returns:
        flooded: boolean array where True indicates flooded areas
        """
        flooded = np.zeros_like(self.elevation, dtype=bool)

        # Check each seed point
        for seed_row, seed_col in seed_points:
            if (seed_row < 0 or seed_row >= self.rows or
                    seed_col < 0 or seed_col >= self.cols or
                    not self.valid_mask[seed_row, seed_col] or
                    self.elevation[seed_row, seed_col] > water_level):
                continue

            # Flood fill from this seed point
            self._flood_fill_recursive(flooded, seed_row, seed_col, water_level)

        return flooded

    def _flood_fill_recursive(self, flooded, row, col, water_level):
        """Recursive flood fill implementation."""
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()

            # Check bounds and conditions
            if (r < 0 or r >= self.rows or c < 0 or c >= self.cols or
                    flooded[r, c] or not self.valid_mask[r, c] or
                    self.elevation[r, c] > water_level):
                continue

            # Mark as flooded
            flooded[r, c] = True

            # Add adjacent cells (4-connectivity)
            stack.extend([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])

    def breach_fill_algorithm(self, water_level, iterations=10):
        """
        Implements a simplified breach-fill algorithm to handle depressions.

        Parameters:
        water_level: target water level
        iterations: number of iterations for depression filling

        Returns:
        flooded: boolean array of flooded areas
        """
        # Create a copy of elevation for modification
        filled_dem = self.elevation.copy()

        # Fill depressions iteratively
        for _ in range(iterations):
            # Find local minima that could be filled
            kernel = np.ones((3, 3), dtype=bool)
            kernel[1, 1] = False  # Exclude center

            for i in range(1, self.rows - 1):
                for j in range(1, self.cols - 1):
                    if not self.valid_mask[i, j]:
                        continue

                    # Get neighboring elevations
                    neighbors = filled_dem[i - 1:i + 2, j - 1:j + 2][kernel]
                    valid_neighbors = neighbors[neighbors != self.nodata_value]

                    if len(valid_neighbors) > 0:
                        min_neighbor = np.min(valid_neighbors)
                        if filled_dem[i, j] < min_neighbor:
                            # Fill to lowest neighbor if below water level
                            fill_level = min(min_neighbor, water_level)
                            filled_dem[i, j] = fill_level

        # Create flood mask
        flooded = (filled_dem <= water_level) & self.valid_mask
        return flooded

    def calculate_flood_depth(self, water_level, flooded_mask):
        """Calculate flood depth for flooded areas."""
        depth = np.zeros_like(self.elevation)
        depth[flooded_mask] = water_level - self.elevation[flooded_mask]
        depth[depth < 0] = 0  # Ensure non-negative depths
        return depth

    def calculate_flood_volume(self, flooded_mask, water_level):
        """Calculate total flood volume in cubic units."""
        depth = self.calculate_flood_depth(water_level, flooded_mask)
        volume = np.sum(depth) * (self.cell_size ** 2)
        return volume

    def visualize_flood(self, flooded_mask, water_level, depth_map=None,
                        title="Flood Inundation", save_path=None):
        """
        Visualize flood inundation results.

        Parameters:
        flooded_mask: boolean array of flooded areas
        water_level: water surface elevation
        depth_map: optional flood depth array
        title: plot title
        save_path: optional path to save figure
        """
        fig, axes = plt.subplots(1, 2 if depth_map is not None else 1,
                                 figsize=(15, 6))

        if depth_map is None:
            axes = [axes]

        # Plot 1: Elevation with flood overlay
        ax1 = axes[0]

        # Create elevation plot
        elevation_plot = self.elevation.copy()
        elevation_plot[~self.valid_mask] = np.nan

        im1 = ax1.imshow(elevation_plot, cmap='terrain', aspect='equal')

        # Overlay flooded areas
        flood_overlay = np.ma.masked_where(~flooded_mask, flooded_mask)
        ax1.imshow(flood_overlay, cmap='Blues', alpha=0.6, aspect='equal')

        ax1.set_title(f'{title}\nWater Level: {water_level:.1f}m')
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Row')

        # Add colorbar for elevation
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Elevation (m)')

        # Plot 2: Flood depth (if provided)
        if depth_map is not None:
            ax2 = axes[1]
            depth_plot = depth_map.copy()
            depth_plot[depth_plot == 0] = np.nan

            im2 = ax2.imshow(depth_plot, cmap='Blues', aspect='equal')
            ax2.set_title('Flood Depth')
            ax2.set_xlabel('Column')
            ax2.set_ylabel('Row')

            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
            cbar2.set_label('Depth (m)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def multi_level_analysis(self, water_levels, seed_points=None, method='simple'):
        """
        Analyze flooding for multiple water levels.

        Parameters:
        water_levels: list of water levels to analyze
        seed_points: seed points for connected flood fill
        method: 'simple', 'connected', or 'breach_fill'

        Returns:
        results: dictionary with results for each water level
        """
        results = {}

        for level in water_levels:
            if method == 'simple':
                flooded = self.simple_inundation(level)
            elif method == 'connected' and seed_points:
                flooded = self.connected_flood_fill(seed_points, level)
            elif method == 'breach_fill':
                flooded = self.breach_fill_algorithm(level)
            else:
                raise ValueError("Invalid method or missing seed_points")

            depth = self.calculate_flood_depth(level, flooded)
            volume = self.calculate_flood_volume(flooded, level)
            area = np.sum(flooded) * (self.cell_size ** 2)

            results[level] = {
                'flooded_mask': flooded,
                'depth_map': depth,
                'volume': volume,
                'area': area,
                'max_depth': np.max(depth) if np.any(flooded) else 0
            }

        return results


# Example usage and demonstration
def create_sample_terrain(rows=100, cols=100):
    """Create sample terrain for testing."""
    # Create a simple valley terrain
    x = np.linspace(-5, 5, cols)
    y = np.linspace(-5, 5, rows)
    X, Y = np.meshgrid(x, y)

    # Create a valley with some hills
    terrain = (X ** 2 + Y ** 2) * 0.1 + np.sin(X) * 2 + np.cos(Y) * 1.5
    terrain += np.random.normal(0, 0.2, terrain.shape)  # Add noise

    # Add a river channel
    river_mask = np.abs(Y) < 0.5
    terrain[river_mask] -= 2

    return terrain


def main():
    """Demonstrate flood simulation capabilities."""
    print("Creating sample terrain...")
    terrain = create_sample_terrain(80, 80)

    # Initialize simulator
    # simulator = FloodSimulator(terrain, cell_size=10.0)
    simulator = FloodSimulator.from_geotiff('new york city.tif')

    # Define water levels to test
    water_levels = [0, 1, 2, 3, 4]

    print("Running multi-level flood analysis...")
    results = simulator.multi_level_analysis(water_levels, method='simple')

    # Print summary statistics
    print("\nFlood Analysis Results:")
    print("-" * 50)
    for level, result in results.items():
        print(f"Water Level {level}m:")
        print(f"  Flooded Area: {result['area']:.0f} m²")
        print(f"  Flood Volume: {result['volume']:.0f} m³")
        print(f"  Max Depth: {result['max_depth']:.1f} m")
        print()

    # Visualize results for a specific water level
    test_level = 2.0
    if test_level in results:
        print(f"Visualizing flood for water level {test_level}m...")
        simulator.visualize_flood(
            results[test_level]['flooded_mask'],
            test_level,
            results[test_level]['depth_map'],
            title=f"Flood Simulation - Water Level {test_level}m"
        )

    # Demonstrate connected flood fill
    print("Demonstrating connected flood fill...")
    seed_points = [(40, 40), (30, 50)]  # Start flooding from these points
    connected_flood = simulator.connected_flood_fill(seed_points, 2.0)

    simulator.visualize_flood(
        connected_flood,
        2.0,
        title="Connected Flood Fill"
    )


if __name__ == "__main__":
    main()