import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import data_handler as dh
from datetime import datetime
import matplotlib.dates as mdates

# =======================
# Page Config
# =======================
st.set_page_config(page_title="Train Comparison", layout="wide")

# =======================
# Load Data
# =======================
@st.cache_data
def get_data():
    return dh.load_data()

TRAINS = get_data()

# =======================
# STREAMLIT UI
# =======================
st.title("🚆 Train Analysis Dashboard")

# Create Tabs
tab1, tab2 = st.tabs(["Route Comparison", "Train Meeting"])

# =======================
# TAB 1: Route Comparison
# =======================
with tab1:
    st.header("Route Comparison")
    st.markdown("Compare trains between two stations to find the best option for your journey.")
    
    # Inputs in columns
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        source_station = st.text_input("Source Station Code", value="DHN", help="e.g., DHN").upper()
    with c2:
        dest_station = st.text_input("Destination Station Code", value="HWH", help="e.g., HWH").upper()
    with c3:
        st.write("") # Spacer
        st.write("") # Spacer
        find_btn = st.button("Find Trains", use_container_width=True)

    # State Management for Search
    if find_btn:
        st.session_state["search_clicked"] = True
        st.session_state["source"] = source_station
        st.session_state["dest"] = dest_station
    else:
        if "search_clicked" not in st.session_state:
            st.session_state["search_clicked"] = False

    # Main Logic
    if st.session_state["search_clicked"]:
        src = st.session_state["source"]
        dst = st.session_state["dest"]
        
        if not src or not dst:
            st.error("Please enter both Source and Destination station codes.")
        else:
            available_trains = dh.find_trains(src, dst, TRAINS)
            
            if not available_trains:
                st.warning(f"No trains found running directly between {src} and {dst}.")
            else:
                st.success(f"Found {len(available_trains)} trains between {src} and {dst}.")
                
                # Train Selection
                train_options = [f"{t['trainNumber']} - {t['trainName']}" for t in available_trains]
                selected_train_strs = st.multiselect(
                    "Select trains to compare:",
                    train_options,
                    default=train_options[:5] if len(train_options) > 0 else []
                )
                
                if len(selected_train_strs) < 1:
                    st.info("Select at least one train to see details.")
                else:
                    # Extract selected train objects
                    selected_nums = [s.split(" - ")[0] for s in selected_train_strs]
                    selected_train_objs = [t for t in available_trains if t["trainNumber"] in selected_nums]
                    
                    # Process & Compare
                    dfs = {}
                    summary_data = []
                    
                    for t in selected_train_objs:
                        # Process route specifically for the segment
                        df = dh.process_route(t, src, dst)
                        dfs[t["trainNumber"]] = df
                        
                        # Calculate metrics for the segment
                        distance = df["distance"].max() - df["distance"].min()
                        stops = len(df) - 1 
                        
                        # Duration
                        start_time = df["dep_dt"].iloc[0] if pd.notnull(df["dep_dt"].iloc[0]) else df["arr_dt"].iloc[0]
                        end_time = df["arr_dt"].iloc[-1] if pd.notnull(df["arr_dt"].iloc[-1]) else df["dep_dt"].iloc[-1]
                        
                        duration_hr = 0
                        if start_time and end_time:
                            duration_sec = (end_time - start_time).total_seconds()
                            duration_hr = duration_sec / 3600
                        
                        avg_speed = distance / duration_hr if duration_hr > 0 else 0
                        
                        summary_data.append({
                            "Train Number": t["trainNumber"],
                            "Train Name": t["trainName"],
                            "Type": t.get("trainType", "UNK"),
                            "Stops": stops,
                            "Distance (km)": round(distance, 1),
                            "Duration (hr)": round(duration_hr, 2),
                            "Avg Speed (km/h)": round(avg_speed, 1),
                            "Departs": start_time.strftime("%H:%M") if start_time else "N/A",
                            "Arrives": end_time.strftime("%H:%M") if end_time else "N/A"
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    
                    # Display Summary Table
                    st.subheader("📋 Comparison Summary")
                    st.dataframe(summary_df.style.highlight_min(subset=["Duration (hr)", "Stops"], color="lightgreen")
                                 .highlight_max(subset=["Avg Speed (km/h)"], color="lightgreen"))
                    
                    # Visualizations
                    st.subheader("📊 Visual Comparison")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.caption("Stops Count")
                        st.bar_chart(summary_df.set_index("Train Number")["Stops"])
                        
                    with col2:
                        st.caption("Duration (Hours)")
                        st.bar_chart(summary_df.set_index("Train Number")["Duration (hr)"])

                    # ---------------------------------------------------------
                    # Route Grouping & Dense Train Analysis
                    # ---------------------------------------------------------
                    st.subheader("🔍 Route Analysis")
                    
                    # Group trains by route similarity (using Jaccard similarity of station sets)
                    route_groups = []  # List of lists of train_numbers
                    
                    for t_num in dfs.keys():
                        t_stations = set(dfs[t_num]['code'])
                        added = False
                        for group in route_groups:
                            # Compare with first train in the group
                            rep_t_num = group[0]
                            rep_stations = set(dfs[rep_t_num]['code'])
                            
                            intersection = len(t_stations.intersection(rep_stations))
                            union = len(t_stations.union(rep_stations))
                            jaccard = intersection / union if union > 0 else 0
                            
                            # Threshold: 0.5 means at least 50% overlap
                            if jaccard >= 0.5:
                                group.append(t_num)
                                added = True
                                break
                        
                        if not added:
                            route_groups.append([t_num])
                    
                    # Display route grouping if multiple routes detected
                    if len(route_groups) > 1:
                        st.info(f"🔀 Detected **{len(route_groups)} different routes** between {src} and {dst}")
                        
                        group_data = []
                        for i, group in enumerate(route_groups):
                            # Get sample stations from middle of route
                            sample_df = dfs[group[0]]
                            mid_stations = []
                            if len(sample_df) > 4:
                                mid_stations = list(sample_df['code'])[1:4]
                            else:
                                mid_stations = list(sample_df['code'])[1:min(len(sample_df), 3)]
                            
                            group_data.append({
                                "Route Group": f"Group {i+1}",
                                "Train Count": len(group),
                                "Trains": ", ".join(group),
                                "Sample Via": " → ".join(mid_stations) + "..." if mid_stations else "N/A",
                                "Total Stops": f"{min([len(dfs[t]) for t in group])}-{max([len(dfs[t]) for t in group])}"
                            })
                        
                        st.table(pd.DataFrame(group_data))
                    else:
                        st.success("✓ All trains follow the same route")
                    
                    # ---------------------------------------------------------
                    # Create tabs ONLY if multiple routes exist
                    # ---------------------------------------------------------
                    if len(route_groups) > 1:
                        route_tabs = st.tabs([f"Route Group {i+1}" for i in range(len(route_groups))])
                    else:
                        # Single route - no tabs needed, show graphs directly
                        route_tabs = [st.container()]
                    
                    figs_for_pdf = []
                    
                    for route_idx, (tab_container, group_train_nums) in enumerate(zip(route_tabs, route_groups)):
                        with tab_container:
                            # Find the dense train (most stops)
                            dense_t_num = max(group_train_nums, key=lambda x: len(dfs[x]))
                            dense_df = dfs[dense_t_num]
                            
                            if len(route_groups) > 1:
                                st.markdown(f"**Trains in this group:** {', '.join(group_train_nums)}")
                            
                            # ---------------------------------------------------------
                            # GRAPH 1: Speed vs Distance (for this route group)
                            # ---------------------------------------------------------
                            st.subheader("📈 Speed vs Distance Profile")
                            
                            fig_dist, ax_dist = plt.subplots(figsize=(12, 6))
                            
                            for t_num in group_train_nums:
                                df = dfs[t_num]
                                dist_norm = df["distance"] - df["distance"].iloc[0]
                                speed_smooth = df["speed"].replace([np.inf, -np.inf], np.nan).interpolate()
                                
                                if t_num == dense_t_num:
                                    ax_dist.plot(dist_norm, speed_smooth, marker='o', markersize=4, 
                                               label=f"{t_num} (Ref - {len(df)} stations)", 
                                               linewidth=2, color='black')
                                else:
                                    ax_dist.plot(dist_norm, speed_smooth, marker='o', markersize=4, 
                                               label=f"{t_num} ({len(df)} stations)", linestyle='--')
                            
                            ax_dist.set_xlabel("Distance from Source (km)", fontweight='bold')
                            ax_dist.set_ylabel("Speed (km/h)", fontweight='bold')
                            ax_dist.set_title(f"Speed Profile: {src} to {dst}" + 
                                            (f" - Route Group {route_idx + 1}" if len(route_groups) > 1 else ""))
                            ax_dist.legend()
                            ax_dist.grid(True, alpha=0.3)
                            
                            st.pyplot(fig_dist)
                            figs_for_pdf.append(fig_dist)
                            
                            # ---------------------------------------------------------
                            # GRAPH 2: Speed vs Station (Aligned to Dense Train)
                            # ---------------------------------------------------------
                            st.subheader("🚉 Speed vs Station (Aligned)")
                            st.caption(f"📍 Graph aligned to train **{dense_t_num}** ({len(dense_df)} stations - most stops in {'this group' if len(route_groups) > 1 else 'selection'})")
                            
                            fig_station, ax_station = plt.subplots(figsize=(14, 7))
                            
                            # Master X-axis based on dense train's stations
                            master_stations = dense_df["code"].tolist()
                            master_station_names = dense_df["stationName"].tolist()
                            x_indices = range(len(master_stations))
                            
                            # Plot dense train first (as reference, emphasized)
                            ax_station.plot(x_indices, dense_df["speed"], marker='o', 
                                    label=f"{dense_t_num} (Reference - {len(dense_df)} stations)", 
                                    linewidth=2.5, color='black', zorder=10)
                            
                            # Plot other trains in the group
                            for t_num in group_train_nums:
                                if t_num == dense_t_num:
                                    continue
                                
                                other_df = dfs[t_num]
                                
                                # Map other train's data to master train's station indices
                                matched_indices = []
                                matched_speeds = []
                                
                                for _, row in other_df.iterrows():
                                    if row["code"] in master_stations:
                                        idx = master_stations.index(row["code"])
                                        matched_indices.append(idx)
                                        matched_speeds.append(row["speed"])
                                
                                if matched_indices:
                                    ax_station.plot(matched_indices, matched_speeds, 
                                            marker='o', label=f"{t_num} ({len(other_df)} stations)", 
                                            linestyle='--', alpha=0.8)
                            
                            # Set x-axis labels using master train's stations
                            ax_station.set_xticks(x_indices)
                            station_labels = [f"{name}\n({code})" for name, code in zip(master_station_names, master_stations)]
                            ax_station.set_xticklabels(station_labels, rotation=45, fontsize=7, ha='right')
                            
                            ax_station.set_xlabel("Station (Aligned to Dense Train)", fontweight='bold')
                            ax_station.set_ylabel("Speed (km/h)", fontweight='bold')
                            
                            if len(route_groups) > 1:
                                ax_station.set_title(f"Speed Comparison - Route Group {route_idx + 1}")
                            else:
                                ax_station.set_title(f"Speed Comparison by Station")
                            
                            ax_station.legend(loc='best')
                            ax_station.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
                            plt.tight_layout()
                            
                            st.pyplot(fig_station)
                            figs_for_pdf.append(fig_station)
                    
                    # Export
                    if st.button("Download Report PDF"):
                        pdf_path = "train_comparison_report.pdf"
                        with PdfPages(pdf_path) as pdf:
                            # Save all graphs (both distance and station graphs for all route groups)
                            for f in figs_for_pdf:
                                pdf.savefig(f)
                            d = pdf.infodict()
                            d['Title'] = f'Train Comparison {src}-{dst}'
                            
                        with open(pdf_path, "rb") as f:
                            st.download_button("Click to Download PDF", f, file_name=pdf_path)

# =======================
# TAB 2: Train Meeting
# =======================
with tab2:
    st.header("Train Meeting Point")
    st.markdown("Calculate when and where two trains (e.g., UP and DOWN) will meet.")
    
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        t1_num = st.text_input("Train 1 Number", value="18014")
    with c2:
        t2_num = st.text_input("Train 2 Number", value="18013")
    with c3:
        st.write("")
        st.write("")
        calc_btn = st.button("Calculate Meeting", use_container_width=True)
    
    check_daily = st.checkbox("Daily Train (Check multiple crossings)", value=True, help="Check if trains meet multiple times (e.g. previous/next day departures)")

    if calc_btn:
        t1 = dh.get_train(t1_num, TRAINS)
        t2 = dh.get_train(t2_num, TRAINS)
        
        if not t1 or not t2:
            st.error("Could not find one or both trains. Please check the train numbers.")
        else:
            st.success(f"Found {t1['trainName']} and {t2['trainName']}")
            
            # Process Routes
            df1 = dh.process_route(t1)
            df2 = dh.process_route(t2)
            
            # Map Train 2 to Train 1's distance
            station_map = df1.set_index('code')['distance'].to_dict()
            df2['mapped_dist'] = df2['code'].map(station_map)
            df2_common = df2.dropna(subset=['mapped_dist']).copy()
            
            if df2_common.empty:
                st.error("No common stations found between these trains.")
            else:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot Train 1 (Reference)
                p1, = ax.plot(df1['t'], df1['distance'], label=f"{t1['trainNumber']} (Ref)", color='blue', linewidth=2)
                # Add arrow at the end
                ax.plot(df1['t'].iloc[-1], df1['distance'].iloc[-1], marker='^', color='blue', markersize=8)
                
                # Offsets to check
                offsets = range(-2, 3) if check_daily else [0]
                
                meetings = []
                
                for offset in offsets:
                    # Shift Train 2 time
                    df2_shifted = df2_common.copy()
                    df2_shifted['t'] = df2_shifted['t'] + pd.Timedelta(days=offset)
                    
                    # Label
                    label = f"{t2['trainNumber']}"
                    if offset != 0:
                        label += f" (Day {offset:+d})"
                    
                    # Plot
                    # Only plot if it overlaps with Train 1's time range roughly
                    t1_start, t1_end = df1['t'].min(), df1['t'].max()
                    t2_start, t2_end = df2_shifted['t'].min(), df2_shifted['t'].max()
                    
                    if t2_end > t1_start and t2_start < t1_end:
                         line_style = '--' if offset != 0 else '-'
                         ax.plot(df2_shifted['t'], df2_shifted['mapped_dist'], label=label, linestyle=line_style)
                         # Add arrow at the end (pointing down usually for T2 if it's return)
                         # Check direction
                         if df2_shifted['mapped_dist'].iloc[-1] < df2_shifted['mapped_dist'].iloc[0]:
                             marker = 'v' # Down
                         else:
                             marker = '^' # Up
                         ax.plot(df2_shifted['t'].iloc[-1], df2_shifted['mapped_dist'].iloc[-1], marker=marker, color=ax.get_lines()[-1].get_color(), markersize=8)
                    
                         # Calculate Intersection
                         all_times = sorted(list(set(df1['t'].tolist() + df2_shifted['t'].tolist())))
                         
                         def to_timestamp(dt_series):
                             return dt_series.astype('int64') // 10**9

                         t1_times = to_timestamp(df1['t'])
                         t2_times = to_timestamp(df2_shifted['t'])
                         all_times_ts = [t.timestamp() for t in all_times]
                         
                         d1_interp = np.interp(all_times_ts, t1_times, df1['distance'])
                         d2_interp = np.interp(all_times_ts, t2_times, df2_shifted['mapped_dist'])
                         
                         diff = d1_interp - d2_interp
                         crossings = np.where(np.diff(np.sign(diff)))[0]
                         
                         for idx in crossings:
                             t_start = all_times_ts[idx]
                             t_end = all_times_ts[idx+1]
                             d_diff_start = diff[idx]
                             d_diff_end = diff[idx+1]
                             
                             if d_diff_end != d_diff_start:
                                 frac = -d_diff_start / (d_diff_end - d_diff_start)
                                 meet_ts = t_start + frac * (t_end - t_start)
                                 meet_time = datetime.fromtimestamp(meet_ts)
                                 meet_dist = np.interp(meet_ts, t1_times, df1['distance'])
                                 
                                 # Find stations
                                 df1['dist_diff'] = abs(df1['distance'] - meet_dist)
                                 nearest_station = df1.loc[df1['dist_diff'].idxmin()]
                                 
                                 prev_station = df1[df1['distance'] <= meet_dist].iloc[-1] if not df1[df1['distance'] <= meet_dist].empty else df1.iloc[0]
                                 next_station = df1[df1['distance'] >= meet_dist].iloc[0] if not df1[df1['distance'] >= meet_dist].empty else df1.iloc[-1]

                                 meetings.append({
                                     "Train 2 Instance": f"Day {offset:+d}",
                                     "Meeting Time": meet_time.strftime('%H:%M'),
                                     "Distance": f"{meet_dist:.1f} km",
                                     "Nearest Station": f"{nearest_station['stationName']} ({nearest_station['code']})",
                                     "Segment": f"{prev_station['code']} - {next_station['code']}"
                                 })
                                 
                                 ax.plot(meet_time, meet_dist, 'go', markersize=8)

                ax.set_xlabel("Time")
                ax.set_ylabel(f"Distance (relative to {t1['trainNumber']})")
                ax.set_title("Train Meeting Points")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Y-axis labels
                # Ensure last station is included
                stations_to_show = df1
                if len(df1) > 40:
                    indices = list(range(0, len(df1), 3))
                elif len(df1) > 20:
                    indices = list(range(0, len(df1), 2))
                else:
                    indices = list(range(len(df1)))
                
                if (len(df1) - 1) not in indices:
                    indices.append(len(df1) - 1)
                
                stations_to_show = df1.iloc[indices]
                
                ax.set_yticks(stations_to_show['distance'])
                # Format: "CODE (123)"
                labels = [f"{row['code']} ({int(row['distance'])})" for _, row in stations_to_show.iterrows()]
                ax.set_yticklabels(labels, fontsize=8)
                
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                st.pyplot(fig)
                
                if meetings:
                    st.success(f"Found {len(meetings)} meeting point(s).")
                    st.table(pd.DataFrame(meetings))
                    
                    st.subheader("📝 Meeting Summary")
                    for m in meetings:
                        day_str = "the same day"
                        if "Day -1" in m["Train 2 Instance"]:
                            day_str = "the previous day"
                        elif "Day +1" in m["Train 2 Instance"]:
                            day_str = "the next day"
                        elif "Day" in m["Train 2 Instance"]:
                            day_str = f"{m['Train 2 Instance']} relative"
                            
                        st.markdown(f"- The **{t1['trainName']}** meets the **{t2['trainName']}** (departing {day_str}) at **{m['Meeting Time']}**. "
                                    f"The meeting occurs near **{m['Nearest Station']}**, specifically in the section **{m['Segment']}**.")
                else:
                    st.warning("No meeting points found within the checked range.")
