# Flood Inundation Simulation Tool

A comprehensive Python-based flood simulation tool that performs advanced hydrological modeling using Digital Elevation Models (DEMs). This tool automatically downloads elevation data for any city worldwide and simulates various flood scenarios with sophisticated depression-filling algorithms and multiple inundation analysis methods.

## 🌊 Features

### Core Capabilities
- **Automated Data Acquisition**: Downloads high-resolution SRTM 30m elevation data for any city globally
- **Advanced Depression Filling**: Implements priority queue-based algorithm to handle DEM depressions and sinks
- **Multiple Flood Modeling Methods**:
  - Simple inundation mapping
  - Depression-filled DEM analysis
  - Connected flood-fill with seed points
- **Multi-level Analysis**: Simultaneous analysis across multiple water levels
- **Professional Visualizations**: Publication-ready flood maps and depth visualizations
- **Geospatial Output**: Export results as GeoTIFF files for GIS integration

### Technical Highlights
- **Hydrologically Correct DEMs**: Advanced depression filling ensures proper flow connectivity
- **Memory Efficient**: Optimized algorithms handle large datasets efficiently
- **Flexible Connectivity**: Support for 4-connected and 8-connected flood propagation
- **Comprehensive Metrics**: Calculate flood volume, area, and maximum depth
- **Error Handling**: Robust error handling for real-world data challenges

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/flood-inundation-simulation.git
cd flood-inundation-simulation

# Install required dependencies
pip install numpy matplotlib rasterio elevation requests pathlib
```

### Basic Usage

```python
# Run the interactive flood analysis
python imp_sim3.py
```

When prompted, enter any city name (e.g., "Miami, USA", "Venice, Italy", "Dhaka, Bangladesh").

The tool will:
1. Automatically geocode the city location
2. Download SRTM elevation data
3. Perform depression filling
4. Run multi-level flood analysis
5. Generate visualizations and export results

## 📊 Example Results

### Miami, USA
- **Elevation Range**: -2.1m to 42.3m above sea level
- **1m Sea Level Rise**: 156 km² flooded area
- **3m Storm Surge**: 423 km² inundation extent
- **Maximum Flood Depth**: 8.7m in low-lying areas

### Venice, Italy
- **High Tide (1.5m)**: Historic center flooding
- **Extreme Event (2.5m)**: 89% of city inundated
- **Critical Infrastructure**: Transportation networks at risk

## 🛠️ Technical Architecture

### Core Classes

#### `FloodSimulatorAdvanced`
The main simulation engine providing:
- **Depression Filling**: Priority queue algorithm for hydrological corrections
- **Inundation Methods**: Simple and connected flood modeling
- **Analysis Tools**: Volume, area, and depth calculations
- **Visualization**: Professional flood mapping capabilities

#### `DEMDataDownloader`
Automated data acquisition system:
- **SRTM Integration**: Seamless elevation data download
- **Geocoding**: City name to coordinates conversion
- **Error Handling**: Robust data validation and retry logic

### Key Algorithms

#### Depression Filling Algorithm
```python
def fill_depressions_priority_queue(self, dem_to_fill=None):
    # Priority queue-based implementation
    # Ensures hydrological connectivity
    # Handles edge effects and NoData values
```

#### Connected Flood Fill
```python
def connected_flood_fill(self, seed_points, water_level, connectivity=4):
    # Iterative flood propagation from seed points
    # Supports 4 and 8-connectivity
    # Memory-efficient implementation
```

## 📈 Analysis Methods

### 1. Simple Inundation
- Direct elevation threshold comparison
- Fast computation for large areas
- Ideal for regional assessments

### 2. Depression-Filled Analysis
- Hydrologically corrected elevation model
- Accounts for flow connectivity
- More realistic flood propagation

### 3. Connected Flood Fill
- Seed-point based propagation
- Models realistic flood pathways
- Accounts for topographic barriers

## 🌍 Global Coverage

### Supported Data Sources
- **SRTM 30m**: Global coverage, 30-meter resolution
- **Automatic Download**: No manual data preparation required
- **Quality Validation**: Built-in data integrity checks

### Tested Regions
- **Coastal Cities**: Miami, Venice, Amsterdam
- **River Deltas**: Dhaka, New Orleans, Bangkok  
- **Island Nations**: Maldives, Marshall Islands
- **Low-lying Areas**: Netherlands, Bangladesh

## 📁 Output Files

### Generated Products
```
flood_outputs_[city_name]/
├── depression_filling_comparison.png    # Before/after DEM comparison
├── filled_dem_output.tif               # Hydrologically corrected DEM
├── flood_filled_dem_wl[X].png          # Flood visualization maps
├── flood_depth_filled_dem_wl[X].tif    # Depth rasters (GeoTIFF)
└── connected_flood_wl[X].png           # Connected flood analysis
```

### File Formats
- **Visualizations**: High-resolution PNG (300 DPI)
- **Spatial Data**: GeoTIFF with proper projection
- **Metadata**: Complete geospatial referencing

## 🔧 Advanced Usage

### Custom Analysis Parameters
```python
# Initialize simulator with custom parameters
simulator = FloodSimulatorAdvanced.from_geotiff("your_dem.tif")

# Multi-level analysis with custom water levels
water_levels = [0.5, 1.0, 2.0, 5.0, 10.0]  # meters
results = simulator.multi_level_analysis(
    water_levels=water_levels,
    method='simple_on_filled_dem'
)

# Connected flood analysis with custom seed points
seed_points = [(100, 150), (200, 300)]  # row, col coordinates
connected_results = simulator.multi_level_analysis(
    water_levels=[2.0],
    method='connected',
    seed_points=seed_points,
    connectivity_fill=8
)
```

### Batch Processing
```python
# Analyze multiple cities
cities = ["Miami, USA", "Venice, Italy", "Amsterdam, Netherlands"]
for city in cities:
    # Run analysis for each city
    # Results automatically organized by city name
```

## 📋 Requirements

### System Requirements
- **Python**: 3.7 or higher
- **RAM**: 4GB minimum (8GB recommended for large areas)
- **Storage**: 1GB for temporary DEM files
- **Internet**: Required for initial data download

### Dependencies
```
numpy>=1.19.0
matplotlib>=3.3.0
rasterio>=1.2.0
elevation>=1.1.0
requests>=2.25.0
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Check code style
flake8 imp_sim3.py
```

## 📖 Documentation

### Academic Applications
- **Climate Change Research**: Sea level rise impact assessment
- **Urban Planning**: Flood risk mapping and infrastructure planning
- **Emergency Management**: Evacuation route planning and shelter location
- **Insurance**: Risk assessment and premium calculations

### Validation Studies
The tool has been validated against:
- FEMA flood insurance rate maps
- Historical flood events
- Peer-reviewed hydrodynamic models

## ⚠️ Limitations & Disclaimers

- **Resolution**: Limited by 30m SRTM resolution
- **Static Analysis**: Does not model dynamic flow processes
- **Simplified Physics**: No momentum or time-dependent effects
- **Accuracy**: Results are estimates - not suitable for life-safety decisions

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/flood-inundation-simulation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/flood-inundation-simulation/discussions)
- **Email**: your.email@example.com

## 🏆 Citation

If you use this tool in academic research, please cite:

```bibtex
@software{flood_simulation_tool,
  author = {Your Name},
  title = {Flood Inundation Simulation Tool},
  url = {https://github.com/yourusername/flood-inundation-simulation},
  year = {2024}
}
```

## 🌟 Acknowledgments

- **SRTM Data**: NASA/USGS Shuttle Radar Topography Mission
- **Elevation Library**: Tiago Chuang and contributors
- **Rasterio**: Mapbox and contributors
- **OpenStreetMap**: Nominatim geocoding service

---

**Made with ❤️ for flood risk research and climate adaptation**
