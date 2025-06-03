import tempfile

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import Affine
import collections
import heapq  # For priority queue in depression filling
import logging
from typing import List, Tuple, Optional, Dict, Any

# Imports for DEM downloading and geocoding
import os
import requests
from pathlib import Path
import elevation  # For SRTM data download
import tempfile  # For creating temporary files
import shutil  # For cleaning up temporary directories (if needed by other parts not directly used here)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Geocoding Function (from imp_sim2.py) ---
def geocode_city(city_name: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Geocodes a city name to obtain its bounding box using Nominatim (OpenStreetMap).

    Parameters:
    city_name: The name of the city (e.g., "London, UK", "New York City").

    Returns:
    Tuple (west, south, east, north) bounding box in WGS84 degrees, or None if not found.
    """
    # Use a more general endpoint for Nominatim search
    url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(city_name)}&format=json&limit=1&polygon_geojson=1"
    headers = {
        'User-Agent': 'FloodSimulationApp/1.0 (flood.simulator@example.com)'  # Provide a custom user agent
    }
    logging.info(f"Attempting to geocode '{city_name}'...")
    try:
        response = requests.get(url, headers=headers, timeout=10)  # Added timeout
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            # Nominatim returns 'boundingbox' as [south, north, west, east] (strings)
            # Convert to float and then to (west, south, east, north)
            bbox_str = data[0].get('boundingbox')
            if bbox_str and len(bbox_str) == 4:
                # south, north, west, east
                s, n, w, e = map(float, bbox_str)
                logging.info(f"Geocoded bounds for '{city_name}': West={w}, South={s}, East={e}, North={n}")
                return (w, s, e, n)  # west, south, east, north
            else:
                logging.warning(
                    f"Bounding box not found or in unexpected format for '{city_name}' in API response: {data[0]}")
                return None
        else:
            logging.warning(f"Could not find geocoding data for '{city_name}'. Response: {data}")
            return None
    except requests.exceptions.Timeout:
        logging.error(f"Timeout during geocoding API request for '{city_name}'.")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error during geocoding API request for '{city_name}': {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during geocoding for '{city_name}': {e}")
        return None

class DEMDataDownloaderOriginal: # Renamed from DEMDataDownloader1
    """
    Download and process Digital Elevation Model data from various sources.
    """

    def __init__(self, data_dir="dem_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Attempting to create/use data directory: {self.data_dir.absolute()}")

    def download_srtm_data(self, bounds, output_file="srtm_dem.tif"):
        """
        Download SRTM 30m resolution data using the elevation library.

        Parameters:
        bounds: tuple (west, south, east, north) in WGS84 degrees
        output_file: output filename

        Returns:
        str: path to downloaded file
        """
        west, south, east, north = bounds
        print(self.data_dir)
        output_path = (self.data_dir / output_file).absolute()

        print(f"Downloading SRTM data for bounds: {bounds}")
        print("This may take several minutes...")
        print(f"DEBUG: Final output path for elevation.clip: {str(output_path)}")

        try:
            # Download SRTM data
            elevation.clip(bounds=bounds, output=str(output_path))
            print(f"Successfully downloaded SRTM data to {output_path}")
            return str(output_path)

        except Exception as e:
            print(f"Error downloading SRTM data: {e}")
            print(
                "Note: Ensure you have the 'elevation' library installed (`pip install elevation`) and a working internet connection.")
            return None

    def download_usgs_data(self, bounds, dataset="SRTM1", output_dir="usgs_data"):
        """
        Instructions for downloading USGS data (manual process).

        Parameters:
        bounds: tuple (west, south, east, north) in WGS84 degrees
        dataset: dataset type ("SRTM1", "SRTM3", "NED")
        """
        west, south, east, north = bounds

        print("To download USGS data:")
        print("1. Go to: https://earthexplorer.usgs.gov/")
        print("2. Create a free account")
        print("3. Set search criteria:")
        print(f"   - Coordinates: {west}, {south}, {east}, {north}")
        print(f"   - Dataset: Digital Elevation > {dataset}")
        print("4. Download the GeoTIFF files")
        print("5. Place files in: {self.data_dir / output_dir}")

        return self.data_dir / output_dir

    def download_copernicus_data(self, bounds):
        """
        Instructions for downloading Copernicus DEM data.
        """
        print("To download Copernicus DEM data (30m global):")
        print("1. Go to: https://spacedata.copernicus.eu/")
        print("2. Register for free account")
        print("3. Browse: Copernicus DEM GLO-30")
        print(f"4. Search area: {bounds}")
        print("5. Download tiles covering your area")
        print("Note: Copernicus DEM is very high quality and free!")

    def get_sample_coordinates(self):
        """Return sample coordinates for different regions."""
        regions = {
            "miami_florida": (-80.3, 25.6, -80.1, 25.8),  # Miami Beach area
            "houston_texas": (-95.5, 29.6, -95.2, 29.9),  # Houston flood-prone area
            "netherlands": (4.0, 51.8, 4.5, 52.2),  # Low-lying coastal area
            "new_orleans": (-90.2, 29.8, -89.9, 30.1),  # Below sea level city
            "bangladesh": (90.0, 23.0, 91.0, 24.0),  # Flood-prone river delta
            "venice_italy": (12.2, 45.3, 12.5, 45.5),  # Sea level rise concern
            "maldives": (73.0, 3.0, 73.5, 4.0),  # Sea level rise impact
            "san_francisco": (-122.6, 37.6, -122.3, 37.9),  # Bay area
        }

        print("Available sample regions:")
        for name, coords in regions.items():
            print(f"  {name}: {coords}")

        return regions


# --- Helper function for temporary file management with WhiteboxTools ---
def _save_array_to_temp_geotiff(data, transform, crs, nodata=None):
    """Saves a numpy array to a temporary GeoTIFF file."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / "temp_dem.tif"

    profile = {
        'driver': 'GTiff',
        'height': data.shape[0],
        'width': data.shape[1],
        'count': 1,
        'dtype': data.dtype,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    if nodata is not None:
        profile['nodata'] = nodata

    with rasterio.open(temp_path, 'w', **profile) as dst:
        dst.write(data, 1)
    return str(temp_path), temp_dir


def _load_array_from_geotiff(filepath):
    """Loads a numpy array and its profile from a GeoTIFF file."""
    with rasterio.open(filepath) as src:
        data = src.read(1)
        profile = src.profile
    return data, profile


# --- DEMDataDownloader Class (simplified for SRTM, from imp_sim2.py) ---
# Keeping this class definition here, but it won't be used in main_flood_analysis_for_city
# as per your request to use DEMDataDownloaderOriginal instead.
class DEMDataDownloader:
    """
    Download and process Digital Elevation Model data, focusing on SRTM.
    """

    def __init__(self, data_dir: str = "dem_data"):
        self.data_dir = Path(data_dir)
        # Try to create if it doesn't exist, but proceed if it's already there.
        # If creation fails due to permissions, rasterio/elevation will likely fail later.
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"DEM data will be stored in/read from: {self.data_dir.absolute()}")
        except OSError as e:
            logging.warning(f"Could not create DEM data directory {self.data_dir.absolute()}: {e}. "
                            "Script will attempt to proceed, assuming it exists or download will handle it.")

    def download_srtm_data(self, bounds: Tuple[float, float, float, float], output_filename: str = "srtm_dem.tif") -> \
    Optional[str]:
        """
        Download SRTM 30m resolution data using the elevation library.

        Parameters:
        bounds: tuple (west, south, east, north) in WGS84 degrees.
        output_filename: Name for the output GeoTIFF file (will be placed in self.data_dir).

        Returns:
        Path to downloaded GeoTIFF file, or None if download fails.
        """
        output_path = self.data_dir / output_filename
        logging.info(f"Attempting to download SRTM data for bounds: {bounds} to {output_path.absolute()}")

        cache_dir = Path.home() / ".cache/elevation"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Elevation library cache directory: {cache_dir.absolute()}")
        except OSError as e:
            logging.warning(f"Could not create elevation cache directory {cache_dir.absolute()}: {e}. "
                            "Download might still work if cache is not strictly needed or already exists.")

        try:
            elevation.clip(bounds=bounds, output=str(output_path.absolute()), product='SRTM1')  # SRTM1 is 30m
            if output_path.exists() and output_path.stat().st_size > 0:
                logging.info(f"Successfully downloaded SRTM data to {output_path.absolute()}")
                return str(output_path.absolute())
            else:
                logging.error(f"SRTM data download failed or resulted in an empty file for {output_path.absolute()}.")
                if output_path.exists():
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass
                return None
        except Exception as e:
            logging.error(f"Error downloading SRTM data using elevation library: {e}")
            logging.error(
                "Ensure 'elevation' library is installed (`pip install elevation`) and GDAL is correctly installed and in your system PATH.")
            if output_path.exists():
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            return None


class FloodSimulatorAdvanced:
    def __init__(self, elevation_data: np.ndarray, cell_size: float = 1.0,
                 nodata_value: float = -9999.0,
                 transform: Optional[Affine] = None,
                 crs: Optional[Any] = None):
        self.elevation = np.array(elevation_data, dtype=np.float32)
        self.cell_size = cell_size
        self.nodata_value = nodata_value
        self.transform = transform
        self.crs = crs
        self.rows, self.cols = self.elevation.shape
        self.valid_mask = (self.elevation != self.nodata_value) & np.isfinite(self.elevation)
        self.elevation_nan = self.elevation.copy()
        self.elevation_nan[~self.valid_mask] = np.nan
        logging.info(
            f"Initialized FloodSimulatorAdvanced with DEM of shape {self.elevation.shape}, cell size {self.cell_size}")

    @classmethod
    def from_geotiff(cls, filepath: str) -> Optional['FloodSimulatorAdvanced']:
        logging.info(f"Loading GeoTIFF from: {filepath}")
        try:
            with rasterio.open(filepath) as src:
                elevation_data = src.read(1).astype(np.float32)
                transform = src.transform
                if abs(transform.a) != abs(transform.e):
                    logging.warning(
                        f"Non-square pixels detected: pixel width={transform.a}, pixel height={transform.e}. Using width for cell_size.")
                cell_size = abs(transform.a)
                nodata = src.nodata if src.nodata is not None else -9999.0
                crs = src.crs
            logging.info(
                f"GeoTIFF loaded. Shape: {elevation_data.shape}, Cell size: {cell_size}, NoData: {nodata}, CRS: {crs}")
            return cls(elevation_data, cell_size, nodata, transform, crs)
        except rasterio.errors.RasterioIOError as e:
            logging.error(f"Could not read GeoTIFF file at {filepath}: {e}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading GeoTIFF {filepath}: {e}")
            return None

    def fill_depressions_priority_queue(self, dem_to_fill: Optional[np.ndarray] = None) -> np.ndarray:
        if dem_to_fill is None:
            dem = self.elevation.copy()
            valid_mask_local = self.valid_mask.copy()
        else:
            dem = dem_to_fill.copy()
            valid_mask_local = (dem != self.nodata_value) & np.isfinite(dem)

        logging.info("Starting depression filling using priority queue.")
        filled_dem = dem.copy()
        pq: List[Tuple[float, int, int]] = []
        processed = np.zeros_like(dem, dtype=bool)

        for r in range(self.rows):
            for c in range(self.cols):
                if not valid_mask_local[r, c]:
                    processed[r, c] = True
                    continue
                if r == 0 or r == self.rows - 1 or c == 0 or c == self.cols - 1:
                    heapq.heappush(pq, (filled_dem[r, c], r, c))
                    processed[r, c] = True

        iteration_count = 0
        while pq:
            iteration_count += 1
            if iteration_count % 50000 == 0:
                logging.debug(f"Depression filling iteration: {iteration_count}, PQ size: {len(pq)}")
            elev, r, c = heapq.heappop(pq)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and \
                        valid_mask_local[nr, nc] and not processed[nr, nc]:
                    processed[nr, nc] = True
                    new_elev = max(elev, dem[nr, nc])
                    filled_dem[nr, nc] = new_elev
                    heapq.heappush(pq, (new_elev, nr, nc))

        filled_dem[~valid_mask_local] = self.nodata_value
        logging.info(f"Depression filling completed after {iteration_count} iterations.")
        return filled_dem

    def simple_inundation(self, water_level: float, dem_to_use: Optional[np.ndarray] = None) -> np.ndarray:
        source_dem = dem_to_use if dem_to_use is not None else self.elevation
        valid_mask_local = (source_dem != self.nodata_value) & np.isfinite(source_dem)
        flooded = (source_dem <= water_level) & valid_mask_local
        logging.debug(f"Simple inundation for water level {water_level}: {np.sum(flooded)} cells flooded.")
        return flooded

    def _flood_fill_iterative(self, flooded: np.ndarray, start_row: int, start_col: int,
                              water_level: float, connectivity: int = 4) -> None:
        stack = collections.deque()
        if not (0 <= start_row < self.rows and 0 <= start_col < self.cols and
                not flooded[start_row, start_col] and self.valid_mask[start_row, start_col] and
                self.elevation[start_row, start_col] <= water_level):
            return
        stack.append((start_row, start_col))
        flooded[start_row, start_col] = True
        if connectivity == 4:
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        elif connectivity == 8:
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            raise ValueError("Connectivity must be 4 or 8.")
        while stack:
            r, c = stack.popleft()
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.rows and 0 <= nc < self.cols and
                        not flooded[nr, nc] and self.valid_mask[nr, nc] and
                        self.elevation[nr, nc] <= water_level):
                    flooded[nr, nc] = True
                    stack.append((nr, nc))

    def connected_flood_fill(self, seed_points: List[Tuple[int, int]], water_level: float,
                             connectivity: int = 4) -> np.ndarray:
        flooded = np.zeros_like(self.elevation, dtype=bool)
        valid_seeds_processed = 0
        for seed_row, seed_col in seed_points:
            if not (0 <= seed_row < self.rows and 0 <= seed_col < self.cols and
                    self.valid_mask[seed_row, seed_col] and
                    self.elevation[seed_row, seed_col] <= water_level):
                logging.warning(f"Seed point ({seed_row},{seed_col}) is invalid or above water level. Skipping.")
                continue
            if not flooded[seed_row, seed_col]:
                self._flood_fill_iterative(flooded, seed_row, seed_col, water_level, connectivity)
                valid_seeds_processed += 1
        logging.info(
            f"Connected flood fill from {valid_seeds_processed} valid seed(s) for water level {water_level}: {np.sum(flooded)} cells flooded.")
        return flooded

    def calculate_flood_depth(self, water_level: float, flooded_mask: np.ndarray,
                              dem_to_use: Optional[np.ndarray] = None) -> np.ndarray:
        source_dem = dem_to_use if dem_to_use is not None else self.elevation
        depth = np.zeros_like(source_dem, dtype=np.float32)
        valid_flooded_mask = flooded_mask & self.valid_mask
        depth[valid_flooded_mask] = water_level - source_dem[valid_flooded_mask]
        depth[depth < 0] = 0
        depth[~valid_flooded_mask] = 0
        return depth

    def calculate_flood_volume(self, flood_depth_map: np.ndarray) -> float:
        volume = np.sum(flood_depth_map[flood_depth_map > 0]) * (self.cell_size ** 2)
        return volume

    def visualize_flood(self, flooded_mask: np.ndarray, water_level: float,
                        depth_map: Optional[np.ndarray] = None,
                        title_prefix: str = "Flood Inundation", save_path: Optional[str] = None) -> None:
        num_plots = 1
        if depth_map is not None and np.any(depth_map > 0): num_plots = 2
        fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 7), squeeze=False)
        axes = axes.flatten()
        ax1 = axes[0]
        elevation_display = self.elevation_nan.copy()
        im1 = ax1.imshow(elevation_display, cmap='terrain', aspect='equal')
        flood_overlay_display = np.full(self.elevation.shape, np.nan, dtype=np.float32)
        flood_overlay_display[flooded_mask] = 1
        ax1.imshow(flood_overlay_display, cmap='Blues', alpha=0.6, aspect='equal', vmin=0, vmax=1)
        ax1.set_title(
            f'{title_prefix}\nWater Level: {water_level:.2f} ({self.get_units_label()})\nFlooded Area: {np.sum(flooded_mask) * self.cell_size ** 2:.0f} (sq. units)')
        ax1.set_xlabel('Column Index')
        ax1.set_ylabel('Row Index')
        plt.colorbar(im1, ax=ax1, shrink=0.7, label=f'Elevation ({self.get_units_label()})')
        if num_plots == 2:
            ax2 = axes[1]
            depth_plot_display = depth_map.copy()
            depth_plot_display[~flooded_mask | (depth_plot_display <= 0)] = np.nan
            if np.any(~np.isnan(depth_plot_display)):
                im2 = ax2.imshow(depth_plot_display, cmap='Blues_r', aspect='equal',
                                 vmin=0,
                                 vmax=np.nanmax(depth_plot_display) if np.sum(~np.isnan(depth_plot_display)) > 0 else 1)
                plt.colorbar(im2, ax=ax2, shrink=0.7, label=f'Depth ({self.get_units_label()})')
                max_depth_val = np.nanmax(depth_plot_display) if np.sum(~np.isnan(depth_plot_display)) > 0 else 0
                ax2.set_title(f'Flood Depth\nMax Depth: {max_depth_val:.2f} ({self.get_units_label()})')
            else:
                ax2.imshow(np.zeros_like(depth_plot_display), cmap='Greys', aspect='equal')
                ax2.set_title('Flood Depth (No significant depth)')
            ax2.set_xlabel('Column Index');
            ax2.set_ylabel('Row Index')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.suptitle(f"Flood Simulation Results", fontsize=16)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Visualization saved to {save_path}")
        plt.show(block=False);
        plt.pause(1)

    def get_units_label(self) -> str:
        if self.crs:
            if self.crs.is_projected:
                try:
                    return self.crs.linear_units
                except AttributeError:
                    return "units (projected)"
            elif self.crs.is_geographic:
                return "degrees (approx.)"
        return "m"

    def save_geotiff(self, filepath: str, data: np.ndarray,
                     custom_nodata_value: Optional[float] = None) -> None:
        output_nodata = custom_nodata_value if custom_nodata_value is not None else self.nodata_value
        if data.dtype != np.float32: data = data.astype(np.float32)
        profile = {'driver': 'GTiff', 'height': data.shape[0], 'width': data.shape[1],
                   'count': 1, 'dtype': data.dtype, 'nodata': output_nodata, 'compress': 'lzw'}
        if self.transform is None or self.crs is None:
            logging.warning(f"Cannot save full GeoTIFF to {filepath}: Missing transform or CRS. Saving as simple TIFF.")
        else:
            profile['crs'] = self.crs;
            profile['transform'] = self.transform
            logging.info(f"Saving data as GeoTIFF to: {filepath}")
        try:
            with rasterio.open(filepath, 'w', **profile) as dst:
                dst.write(data, 1)
            logging.info(f"Successfully saved TIFF/GeoTIFF to: {filepath}")
        except Exception as e:
            logging.error(f"Failed to save TIFF/GeoTIFF to {filepath}: {e}")

    def multi_level_analysis(self, water_levels: List[float],
                             method: str = 'simple_on_filled_dem',
                             seed_points: Optional[List[Tuple[int, int]]] = None,
                             connectivity_fill: int = 4) -> Dict[
        float, Dict[str, Any]]:  # Removed use_depression_filled_dem
        results = {}
        dem_for_inundation = self.elevation.copy()
        original_dem_for_depth_calc = self.elevation.copy()
        if method == 'simple_on_filled_dem':
            logging.info("Preparing depression-filled DEM for multi-level analysis.")
            dem_for_inundation = self.fill_depressions_priority_queue(dem_to_fill=self.elevation.copy())

        for level in water_levels:
            logging.info(f"Processing water level: {level} using method: {method}")
            flooded_mask: Optional[np.ndarray] = None
            if method == 'simple' or method == 'simple_on_filled_dem':  # simple will use original or filled based on above
                flooded_mask = self.simple_inundation(level, dem_to_use=dem_for_inundation)
            elif method == 'connected':
                if not seed_points:  # Auto-detect seed points if not provided
                    if np.any(self.valid_mask):
                        min_elev_val = np.min(self.elevation[self.valid_mask])
                        potential_seeds = np.argwhere(self.elevation == min_elev_val)
                        seed_points = [tuple(p) for p in potential_seeds[:min(3,
                                                                              len(potential_seeds))]] if potential_seeds.size > 0 else []
                        if seed_points:
                            logging.info(f"Auto-detected seed points for 'connected' method: {seed_points}")
                        else:
                            raise ValueError("Seed points required for 'connected' and auto-detection failed.")
                    else:
                        raise ValueError("DEM has no valid data for auto-detecting seed points for 'connected' method.")
                flooded_mask = self.connected_flood_fill(seed_points, level, connectivity_fill)
            else:
                raise ValueError(
                    f"Invalid method: {method}. Choose from 'simple', 'simple_on_filled_dem', 'connected'.")

            depth_map = self.calculate_flood_depth(level, flooded_mask, dem_to_use=original_dem_for_depth_calc)
            volume = np.sum(depth_map[depth_map > 0]) * (self.cell_size ** 2) # Corrected volume calculation to use depth_map directly
            area = np.sum(flooded_mask) * (self.cell_size ** 2)
            results[level] = {'flooded_mask': flooded_mask, 'depth_map': depth_map, 'volume': volume, 'area': area,
                              'max_depth': np.max(depth_map) if np.any(depth_map) else 0}
            logging.info(
                f"Level {level}{self.get_units_label()}: Area={area:.0f} sq.{self.get_units_label()}, Volume={volume:.0f} cu.{self.get_units_label()}, Max Depth={results[level]['max_depth']:.2f} {self.get_units_label()}")
        return results


# --- Main Execution Logic ---
def main_flood_analysis_for_city():
    city_name = input(
        "Enter the city name for flood analysis (e.g., 'Miami, USA', 'London, UK', 'Dhaka, Bangladesh'): ").strip()
    if not city_name:
        logging.error("No city name entered. Exiting.")
        return

    safe_city_name = "".join(c if c.isalnum() else "_" for c in city_name).lower()

    master_dem_download_dir = Path("dem_data")
    downloader = DEMDataDownloaderOriginal(data_dir=str(master_dem_download_dir))
    geocoded_bounds = geocode_city(city_name)

    dem_filepath = None
    if geocoded_bounds:
        dem_filename = f"{safe_city_name}_srtm_dem.tif"
        dem_filepath = downloader.download_srtm_data(geocoded_bounds, output_file=dem_filename) # Changed 'output_filename' to 'output_file'
    else:
        logging.error(f"Could not geocode '{city_name}'. Cannot download DEM. Exiting.")
        return

    simulator: Optional[FloodSimulatorAdvanced] = None
    if dem_filepath and Path(dem_filepath).exists():
        logging.info(f"Attempting to load downloaded DEM: {dem_filepath}")
        simulator = FloodSimulatorAdvanced.from_geotiff(dem_filepath)
        if not simulator:
            logging.error(
                f"Failed to load the DEM from {dem_filepath}, even though it was downloaded. The file might be corrupted or invalid. Exiting.")
            return
    else:
        # This case means download failed or the file path is None
        logging.error(
            f"DEM for '{city_name}' was not successfully downloaded or found at {dem_filepath if dem_filepath else 'N/A'}. Cannot proceed. Exiting.")
        return  # Exit if DEM download/existence check fails

    # If simulator is successfully initialized, proceed with analysis
    output_dir_base = "flood_outputs"
    output_dir = Path(f"{output_dir_base}_{safe_city_name}")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Outputs will be saved to: {output_dir.absolute()}")
    except OSError as e:
        logging.error(f"Could not create output directory {output_dir.absolute()}: {e}. Exiting.")
        return

    # --- 1. Demonstrate Depression Filling ---
    logging.info("\n--- Demonstrating Depression Filling ---")
    original_dem_viz = simulator.elevation_nan.copy()
    filled_dem = simulator.fill_depressions_priority_queue(dem_to_fill=simulator.elevation.copy())
    fig_dep, axes_dep = plt.subplots(1, 3, figsize=(18, 6))
    axes_dep[0].imshow(original_dem_viz, cmap='terrain', aspect='equal');
    axes_dep[0].set_title('Original DEM')
    im_filled = axes_dep[1].imshow(filled_dem, cmap='terrain', aspect='equal', vmin=np.nanmin(original_dem_viz),
                                   vmax=np.nanmax(original_dem_viz))
    axes_dep[1].set_title('Depression-Filled DEM')
    diff_map = filled_dem - simulator.elevation;
    diff_map[~simulator.valid_mask] = np.nan
    im_diff = axes_dep[2].imshow(diff_map, cmap='coolwarm_r', aspect='equal')
    axes_dep[2].set_title('Difference (Filled - Original)')
    plt.colorbar(im_filled, ax=axes_dep[1], shrink=0.6, label=f'Elevation ({simulator.get_units_label()})')
    plt.colorbar(im_diff, ax=axes_dep[2], shrink=0.6, label=f'Fill Depth ({simulator.get_units_label()})')
    plt.tight_layout();
    plt.savefig(output_dir / "depression_filling_comparison.png", dpi=300)
    plt.show(block=False);
    plt.pause(1)
    simulator.save_geotiff(str(output_dir / "filled_dem_output.tif"), filled_dem)

    # --- 2. Multi-Level Analysis ---
    logging.info("\n--- Multi-Level Analysis (Simple Inundation on Depression-Filled DEM) ---")
    dem_min = np.nanmin(simulator.elevation_nan);
    dem_max = np.nanmax(simulator.elevation_nan)
    logging.info(f"DEM Elevation Range: {dem_min:.2f} to {dem_max:.2f} {simulator.get_units_label()}")
    water_levels_analysis = []
    if dem_min is not np.nan and dem_max is not np.nan and dem_max > dem_min:
        water_levels_analysis = [round(dem_min + (dem_max - dem_min) * p, 2) for p in [0.1, 0.2, 0.3, 0.4, 0.5]]
        water_levels_analysis = sorted(list(set(wl for wl in water_levels_analysis if wl > dem_min)))
        if not water_levels_analysis: water_levels_analysis = [round(dem_min + (dem_max - dem_min) * 0.25, 2)]
    elif dem_min is not np.nan:
        water_levels_analysis = [round(dem_min + h, 1) for h in [1.0, 2.0, 3.0]]
    else:
        logging.warning("DEM min elevation is NaN, cannot determine relative water levels."); water_levels_analysis = [
            1.0, 3.0, 5.0]
    if not water_levels_analysis: logging.warning("Using default absolute water levels."); water_levels_analysis = [1.0,
                                                                                                                    3.0,
                                                                                                                    5.0]
    logging.info(f"Using water levels for analysis: {water_levels_analysis}")

    results_analysis = simulator.multi_level_analysis(water_levels_analysis, method='simple_on_filled_dem')
    print("\nFlood Analysis Results (Simple on Filled DEM):");
    print("-" * 70)
    for level, result in results_analysis.items():
        print(
            f"Water Level {level:.2f} ({simulator.get_units_label()}): Area={result['area']:.0f} (sq. {simulator.get_units_label()}), Volume={result['volume']:.0f} (cu. {simulator.get_units_label()}), Max Depth={result['max_depth']:.2f} ({simulator.get_units_label()})")
        simulator.visualize_flood(result['flooded_mask'], level, result['depth_map'],
                                  title_prefix=f"Flood on Filled DEM ({safe_city_name})",
                                  save_path=str(output_dir / f"flood_filled_dem_wl{level:.2f}.png"))
        simulator.save_geotiff(str(output_dir / f"flood_depth_filled_dem_wl{level:.2f}.tif"), result['depth_map'],
                               custom_nodata_value=0)

    # --- 3. Connected Flood Fill (Optional) ---
    if np.any(simulator.valid_mask):
        min_elev_val = np.min(simulator.elevation[simulator.valid_mask])
        potential_seeds = np.argwhere(simulator.elevation == min_elev_val)
        if potential_seeds.size > 0:
            seed_points = [tuple(p) for p in potential_seeds[:min(3, len(potential_seeds))]]
            connected_water_level = water_levels_analysis[
                len(water_levels_analysis) // 2] if water_levels_analysis else (
                dem_min + 2.0 if dem_min is not np.nan else 2.0)
            logging.info(
                f"\n--- Demonstrating Connected Flood Fill (Original DEM) seeds: {seed_points}, WL: {connected_water_level:.2f} ---")
            connected_results = simulator.multi_level_analysis([connected_water_level], method='connected',
                                                               seed_points=seed_points)
            for level, result in connected_results.items():  # Should be only one level
                simulator.visualize_flood(result['flooded_mask'], level, result['depth_map'],
                                          title_prefix=f"Connected Flood ({safe_city_name})",
                                          save_path=str(output_dir / f"connected_flood_wl{level:.2f}.png"))
        else:
            logging.info("\nSkipping connected flood fill: No suitable seed points found.")
    else:
        logging.info("\nSkipping connected flood fill: DEM has no valid data.")

    logging.info(f"\nFlood simulation for '{safe_city_name}' complete. Outputs in {output_dir.absolute()}")
    plt.show(block=True)


if __name__ == "__main__":
    main_flood_analysis_for_city()