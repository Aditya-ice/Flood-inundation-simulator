import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os

from flood_sim import FloodSimulator  # Assuming your main simulator is saved as flood_simulator.py

def load_nyc_terrain(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Terrain file not found: {file_path}")
    print(f"Loading NYC terrain from: {file_path}")
    simulator = FloodSimulator.from_geotiff(file_path)
    return simulator

def main():
    dem_path = "new york city.tif"
    simulator = load_nyc_terrain(dem_path)

    print("Running multi-level flood analysis on NYC DEM...")
    water_levels = [0, 1, 2, 3, 4]
    results = simulator.multi_level_analysis(water_levels, method='simple')

    print("\nFlood Analysis Results:")
    print("-" * 50)
    for level, result in results.items():
        print(f"Water Level {level}m:")
        print(f"  Flooded Area: {result['area']:.0f} m²")
        print(f"  Flood Volume: {result['volume']:.0f} m³")
        print(f"  Max Depth: {result['max_depth']:.1f} m\n")

    test_level = 2.0
    if test_level in results:
        print(f"Visualizing flood for water level {test_level}m...")
        simulator.visualize_flood(
            results[test_level]['flooded_mask'],
            test_level,
            results[test_level]['depth_map'],
            title=f"NYC Flood Simulation - Water Level {test_level}m"
        )

if __name__ == "__main__":
    main()
