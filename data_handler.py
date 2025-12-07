import json
import pandas as pd
import os
from datetime import datetime, timedelta

DATA_DIR = "/home/abantika/Documents/abantika_projects/Proj_rail/Raw_data"
FILES = ["EXP-TRAINS.json", "PASS-TRAINS.json", "SF-TRAINS.json"]

def load_data():
    """Loads all train data from JSON files and tags them with source type."""
    all_trains = []
    for file in FILES:
        path = os.path.join(DATA_DIR, file)
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    data = json.load(f)
                    # Tag trains with their type based on source file
                    train_type = file.replace("-TRAINS.json", "")  # EXP, PASS, or SF
                    for train in data:
                        train["trainType"] = train_type
                    all_trains.extend(data)
                except json.JSONDecodeError:
                    print(f"Error reading {file}")
    return all_trains

# Cache the data to avoid reloading on every call (Streamlit will handle caching, but good to have a global var here if needed, 
# though for Streamlit it's better to use st.cache_data in the main app)
# For now, we'll just provide the function.

def get_station_code(station_name):
    """Extracts station code from 'Name - Code' string."""
    if " - " in station_name:
        return station_name.split(" - ")[-1].strip()
    return station_name.strip()

def find_trains(source_code, dest_code, trains):
    """
    Finds trains running between source and destination.
    Returns a list of train objects.
    """
    valid_trains = []
    source_code = source_code.upper()
    dest_code = dest_code.upper()

    for train in trains:
        route = train.get("trainRoute", [])
        
        # Extract station codes from the route
        station_codes = [get_station_code(stop["stationName"]) for stop in route]
        
        if source_code in station_codes and dest_code in station_codes:
            src_idx = station_codes.index(source_code)
            dest_idx = station_codes.index(dest_code)
            
            if src_idx < dest_idx:
                valid_trains.append(train)
                
    return valid_trains

def get_train(train_number, trains):
    """
    Finds a specific train by its number.
    """
    for train in trains:
        if str(train.get("trainNumber")) == str(train_number):
            return train
    return None

def parse_time(t, day):
    if t in ["Source", "Destination"]:
        return None
    # Handle cases where time might be missing or malformed if any
    try:
        return datetime.strptime(t, "%H:%M") + timedelta(days=int(day) - 1)
    except:
        return None

def process_route(train, source_code=None, dest_code=None):
    """
    Processes the route to calculate speed, distance, etc.
    If source_code and dest_code are provided, it slices the route to that segment.
    """
    route = train["trainRoute"]
    df = pd.DataFrame(route)
    
    # Extract station codes
    df["code"] = df["stationName"].apply(get_station_code)
    
    # Filter for specific segment if requested
    if source_code and dest_code:
        try:
            src_idx = df[df["code"] == source_code].index[0]
            dest_idx = df[df["code"] == dest_code].index[0]
            df = df.loc[src_idx:dest_idx].reset_index(drop=True)
        except IndexError:
            pass # Should not happen if filtered correctly before

    df["distance"] = df["distance"].str.replace(" kms", "").astype(float)
    
    # Calculate times
    # For the first station in the slice, if it's not the original source, 'arrives' might be relevant, 
    # but usually we care about departure from source and arrival at destination.
    
    # Fix 'Source' and 'Destination' strings in time columns for the *original* route
    # If we sliced, the first station might have an arrival time, but we treat it as start point.
    
    times = []
    for i in range(len(df)):
        arr = df.loc[i, "arrives"]
        dep = df.loc[i, "departs"]
        day = df.loc[i, "day"]
        
        # If it's the very first station of the train, arr is "Source"
        # If it's the very last, dep is "Destination"
        
        arr_dt = parse_time(arr, day)
        dep_dt = parse_time(dep, day)
        
        # For plotting/calculation, we need a single time point.
        # Usually departure is good for progress, but for speed calc we need intervals.
        times.append((arr_dt, dep_dt))

    df["arr_dt"] = [t[0] for t in times]
    df["dep_dt"] = [t[1] for t in times]
    
    # Effective time for graph (Departure preferred, else Arrival)
    df["t"] = df["dep_dt"].fillna(df["arr_dt"])
    
    # Calculate duration and speed
    df["dt"] = df["t"].diff().dt.total_seconds() / 3600
    df["dd"] = df["distance"].diff()
    df["speed"] = df["dd"] / df["dt"]
    df.loc[0, "speed"] = 0
    
    # Clean station name for display
    df["station"] = df["code"]
    
    return df
