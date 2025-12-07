# 🚆 Train Comparison & Analysis Dashboard

A comprehensive Streamlit web application for comparing and analyzing Indian Railway trains between stations.

## Features

### 🔍 Route Comparison
- **Multi-train comparison**: Compare multiple trains running between any two stations
- **Speed profiles**: Visualize speed vs distance and speed vs station graphs
- **Dense train alignment**: Automatically aligns graphs to the train with the most stops
- **Route detection**: Identifies different physical routes (e.g., Main Line vs Cord Line) and groups trains accordingly
- **Comprehensive metrics**: 
  - Number of stops
  - Total distance
  - Journey duration
  - Average speed
  - Departure and arrival times

### 🚉 Train Meeting Analysis
- **Meeting point calculation**: Find where and when two trains (e.g., UP and DOWN) meet
- **Multiple crossing detection**: For daily trains, checks previous/next day departures
- **Visual timeline**: Interactive graph showing train movements and crossing points
- **Detailed meeting info**: Displays meeting time, location, nearest station, and track segment

### 📊 Visualization Features
- **Speed vs Distance graphs**: Track speed changes throughout the journey
- **Speed vs Station graphs**: Station-by-station speed comparison aligned to the densest route
- **Route grouping tabs**: Separate visualizations for different routes between the same stations
- **PDF export**: Download comprehensive comparison reports

## Live Demo

🌐 **[Try it on Streamlit Cloud](https://railapp.streamlit.app/)**

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/Proj_rail.git
cd Proj_rail
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run railapp.py
```

The app will open in your browser at `http://localhost:8501`

## Data Sources

The application uses three JSON files containing train data:
- `Raw_data/EXP-TRAINS.json` - Express trains
- `Raw_data/PASS-TRAINS.json` - Passenger trains
- `Raw_data/SF-TRAINS.json` - Superfast trains

Each train record includes:
- Train number and name
- Complete route with station codes
- Timing information (arrivals/departures)
- Running days
- Distance information

## Usage

### Comparing Trains

1. Navigate to the **Route Comparison** tab
2. Enter source station code (e.g., `DHN`)
3. Enter destination station code (e.g., `HWH`)
4. Click "Find Trains"
5. Select trains from the list to compare
6. View comparison summary, graphs, and metrics
7. Export results as PDF if needed

### Finding Train Meetings

1. Navigate to the **Train Meeting** tab
2. Enter train numbers for two trains (e.g., `18014` and `18013`)
3. Enable "Daily Train" option to check multiple crossings
4. Click "Calculate Meeting"
5. View meeting points on the graph and in the detailed table

## Project Structure

```
Proj_rail/
├── railapp.py              # Main Streamlit application
├── data_handler.py         # Data loading and processing utilities
├── requirements.txt        # Python dependencies
├── Raw_data/              # Train data JSON files
│   ├── EXP-TRAINS.json
│   ├── PASS-TRAINS.json
│   └── SF-TRAINS.json
└── README.md              # This file
```

## Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Graph plotting and visualization
- **NumPy**: Numerical computations
- **Python**: Core programming language

## Features in Detail

### Smart Route Alignment
The application uses the **Jaccard similarity coefficient** to detect when trains take different physical routes between the same stations. When multiple routes are detected, trains are automatically grouped and displayed in separate tabs.

### Dense Train Methodology
Graphs are aligned to the train with the most stops (the "dense train"), ensuring all available station data is visible. Other trains are plotted only at their matching stations, avoiding interpolation and maintaining data accuracy.

### Meeting Point Calculation
Uses linear interpolation between station timing data to calculate precise meeting points. The algorithm:
1. Maps both trains' routes to a common distance reference
2. Interpolates position over time
3. Detects sign changes in position difference
4. Calculates exact meeting time and location

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Author

Created by Abantika

## Acknowledgments

- Indian Railways for the inspiration
- Streamlit team for the excellent framework
- All contributors and users of this application

---

**Note**: Update the live demo URL after deploying to Streamlit Cloud, and replace `YOUR_USERNAME` with your actual GitHub username.
