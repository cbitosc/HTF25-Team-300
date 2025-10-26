import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import tempfile
import matplotlib.pyplot as plt
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import pow
import time
import os
from dotenv import load_dotenv
import streamlit.components.v1 as components

# Load environment variables
load_dotenv()

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Crowd Safety AI Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header { 
        font-size: 2.8rem !important; 
        color: #FF4B4B; 
        text-align: center; 
        font-weight: bold; 
        margin-bottom: 1rem;
    }
    .module-header { 
        font-size: 1.8rem !important; 
        color: #1f77b4; 
        border-left: 5px solid #FF4B4B; 
        padding-left: 1rem; 
        margin: 2rem 0 1rem 0;
    }
    .metric-card { 
        background-color: #f0f2f6; 
        padding: 1rem; 
        border-radius: 10px; 
        border-left: 4px solid #1f77b4; 
        margin: 0.5rem 0;
    }
    .risk-high { 
        background-color: #f44336; 
        border-left: 5px solid #b71c1c;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    .risk-medium { 
        background-color: #ff9800; 
        border-left: 5px solid #f57c00;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    .risk-low { 
        background-color: #4caf50; 
        border-left: 5px solid #1b5e20;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'video_running' not in st.session_state:
    st.session_state.video_running = False
if 'incident_history' not in st.session_state:
    st.session_state.incident_history = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'people_history' not in st.session_state:
    st.session_state.people_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Dashboard Overview"
if 'processed_frames' not in st.session_state:
    st.session_state.processed_frames = 0
if 'heatmap_accum' not in st.session_state:
    st.session_state.heatmap_accum = None
if 'zones' not in st.session_state:
    st.session_state.zones = []

# ==================== CITIES & CONFIG ====================
CITIES = {
    "🇮🇳 Hyderabad": (17.384, 78.4564),
    "🇮🇳 Mumbai": (19.0760, 72.8777),
    "🇮🇳 Delhi": (28.6139, 77.2090),
    "🇮🇳 Bangalore": (12.9716, 77.5946),
    "🇮🇳 Chennai": (13.0827, 80.2707)
}



import os

# API Keys from environment variables (set in .env file)
WAQI_TOKEN = os.getenv("WAQI_TOKEN", "demo")  # Replace with actual WAQI token
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "demo")  # Replace with actual TomTom API key

# Constants
MAX_PEOPLE_THRESHOLD = 100
FRAME_SKIP = 3
scale = 0.5
NUM_ROWS, NUM_COLS = 2, 3
ZONE_COLOR_LOW = (0, 255, 0)
ZONE_COLOR_MED = (0, 165, 255)
ZONE_COLOR_HIGH = (255, 0, 0)

# ==================== SIDEBAR NAVIGATION ====================
st.sidebar.image("https://img.icons8.com/fluency/96/crowd.png", width=80)
st.sidebar.title("🚨 Crowd Safety AI")

page = st.sidebar.radio("Navigation", [
    "🏠 Dashboard Overview",
    "🎥 Video Analytics",
    "🌡️ Sensors Monitoring", 
    "📍 GPS View",
    "📊 Integrated Analytics",
    "🤖 Safety Assistant"
])

# ==================== UTILITY FUNCTIONS ====================
def calculate_heat_index(temp, humidity):
    """Calculate heat index using NOAA formula"""
    if temp < 27 or humidity < 40:
        return temp
    
    HI = (-8.784695 + 1.61139411 * temp + 2.338549 * humidity 
          - 0.14611605 * temp * humidity - 0.012308094 * pow(temp, 2)
          - 0.016424828 * pow(humidity, 2) + 0.002211732 * pow(temp, 2) * humidity
          + 0.00072546 * temp * pow(humidity, 2) - 0.000003582 * pow(temp, 2) * pow(humidity, 2))
    return round(HI, 1)

@st.cache_data(ttl=600)
def fetch_weather_data(lat, lon):
    """Fetch weather data from Open-Meteo API"""
    url = (f"https://api.open-meteo.com/v1/forecast?"
           f"latitude={lat}&longitude={lon}&"
           f"hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&"
           f"current=temperature_2m,relative_humidity_2m,wind_speed_10m,is_day&"
           f"timezone=auto")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        # Return mock data if API fails
        return {
            "current": {
                "temperature_2m": 32.5,
                "relative_humidity_2m": 65,
                "wind_speed_10m": 3.2,
                "is_day": 1
            },
            "hourly": {
                "time": [datetime.now().strftime("%Y-%m-%dT%H:%M")],
                "temperature_2m": [32.5],
                "relative_humidity_2m": [65],
                "wind_speed_10m": [3.2]
            }
        }

@st.cache_data(ttl=300)
def fetch_aqi_data(city_name):
    """Fetch AQI data from WAQI API"""
    # Remove spaces and special characters for API call
    clean_city = city_name.split()[-1].strip()
    url = f"https://api.waqi.info/feed/{clean_city}/?token={WAQI_TOKEN}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "ok":
            aqi_data = data["data"]["aqi"]
            iaqi = data["data"]["iaqi"]
            return {
                "aqi": aqi_data,
                "pm25": iaqi.get("pm25", {}).get("v", 0),
                "pm10": iaqi.get("pm10", {}).get("v", 0),
                "no2": iaqi.get("no2", {}).get("v", 0),
                "o3": iaqi.get("o3", {}).get("v", 0),
                "co": iaqi.get("co", {}).get("v", 0)
            }
        else:
            # Return mock data if API fails
            st.warning("⚠️ Using mock AQI data due to API error")
            return {
                "aqi": 124,
                "pm25": 45,
                "pm10": 68,
                "no2": 32,
                "o3": 28,
                "co": 1.2
            }
    except Exception as e:
        # Return mock data if API fails
        st.warning(f"⚠️ Using mock AQI data due to error: {e}")
        return {
            "aqi": 124,
            "pm25": 45,
            "pm10": 68,
            "no2": 32,
            "o3": 28,
            "co": 1.2
        }



@st.cache_data(ttl=300)
def fetch_traffic_incidents(lat, lon):
    """Fetch traffic incidents from TomTom API"""
    # Define a bounding box around the location
    bbox = f"{lon-0.02},{lat-0.02},{lon+0.02},{lat+0.02}"
    url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
    
    params = {
        "key": TOMTOM_API_KEY,
        "bbox": bbox,
        "fields": "{incidents{type,geometry{type,coordinates},properties{id,iconCategory,magnitudeOfDelay,events{description,code},from,to,delay,length,startTime,endTime}}}",
        "language": "en-GB",
        "timeValidityFilter": "present"
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        incidents_data = data.get("incidents", [])
        
        # Process the incidents data
        processed_incidents = []
        for incident in incidents_data[:5]:  # Limit to first 5 incidents
            props = incident.get("properties", {})
            events = props.get("events", [{}])
            
            processed_incidents.append({
                "type": events[0].get("description", "Traffic incident"),
                "location": [incident["geometry"]["coordinates"][1], incident["geometry"]["coordinates"][0]],
                "description": events[0].get("description", "Traffic incident"),
                "delay": props.get("delay", 0),
                "severity": "High" if props.get("magnitudeOfDelay", "MINOR") in ["MAJOR", "CRITICAL"] else "Medium"
            })
        
        # If API returns demo data or fails, return mock data
        if len(processed_incidents) == 0 or TOMTOM_API_KEY == "demo":
            # Mock traffic data
            processed_incidents = [
                {
                    "type": "Accident",
                    "location": [lat + 0.01, lon + 0.01],
                    "description": "Multi-vehicle collision",
                    "delay": 25,
                    "severity": "High"
                },
                {
                    "type": "Congestion", 
                    "location": [lat - 0.01, lon - 0.01],
                    "description": "Heavy traffic jam",
                    "delay": 15,
                    "severity": "Medium"
                }
            ]
        
        return processed_incidents
    except Exception as e:
        # Return mock data if API fails
        st.warning(f"⚠️ Using mock traffic data due to error: {e}")
        return [
            {
                "type": "Accident",
                "location": [lat + 0.01, lon + 0.01],
                "description": "Multi-vehicle collision",
                "delay": 25,
                "severity": "High"
            },
            {
                "type": "Congestion", 
                "location": [lat - 0.01, lon - 0.01],
                "description": "Heavy traffic jam",
                "delay": 15,
                "severity": "Medium"
            }
        ]

# ==================== DASHBOARD OVERVIEW ====================
if page == "🏠 Dashboard Overview":
    st.markdown('<h1 class="main-header">Crowd Safety AI Platform</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time monitoring and predictive analytics for crowd safety management")
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏙️ Active City", "Hyderabad")
    with col2:
        st.metric("👥 Current Crowd", "87")
    with col3:
        st.metric("🌡️ Comfort Level", "Moderate")
    with col4:
        st.metric("🚦 Traffic Status", "Moderate")
    
    st.markdown("---")
    
    # Modules overview
    st.markdown('<h2 class="module-header">Platform Modules</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("""
        **🎥 Video Analytics**
        - Real-time crowd detection
        - Density heatmaps  
        - Zone-wise analysis
        - Risk assessment
        """)
    with col2:
        st.info("""
        **🌡️ Sensors Monitoring**
        - Weather conditions
        - Air quality index
        - Heat index calculation
        - Comfort level analysis
        """)
    with col3:
        st.info("""
        **📍 GPS View**
        - Real-time traffic incidents
        - Congestion monitoring
        - Route impact analysis
        - Historical trends
        """)
    
    # Recent alerts
    st.markdown('<h2 class="module-header">Live Safety Alerts</h2>', unsafe_allow_html=True)
    alert_col1, alert_col2 = st.columns(2)
    with alert_col1:
        st.error("**🚨 High Crowd Density** - Zone 3: 45 people detected")
        st.warning("**⚠️ Elevated Temperature** - 36°C with high humidity")
    with alert_col2:
        st.warning("**🚧 Traffic Congestion** - 12 min delay on main route")
        st.info("**🌫️ Air Quality** - Moderate pollution levels")
    
    # Quick actions
    st.markdown('<h2 class="module-header">Quick Actions</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    if col1.button("📹 Start Video Analysis", use_container_width=True):
        st.session_state.current_page = "🎥 Video Analytics"
        st.rerun()
    if col2.button("🌡️ Check Sensors", use_container_width=True):
        st.session_state.current_page = "🌡️ Sensors Monitoring"
        st.rerun()
    if col3.button("📍 View GPS", use_container_width=True):
        st.session_state.current_page = "📍 GPS View"
        st.rerun()

# ==================== VIDEO ANALYTICS MODULE ====================
elif page == "🎥 Video Analytics":
    st.markdown('<h1 class="main-header">Video Analytics Module</h1>', unsafe_allow_html=True)
    
    # Video source selection
    video_source = st.radio("Select Video Source", ["Upload Video", "Sample Video", "Webcam"], horizontal=True)
    
    video_path = None
    cap = None
    
    if video_source == "Upload Video":
        uploaded_file = st.file_uploader("Upload surveillance video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            st.success(f"✅ Video uploaded successfully: {uploaded_file.name}")
    
    elif video_source == "Sample Video":
        st.info("Using sample video for demonstration")
        # Using your specific video file
        video_path = "855564-hd_1920_1080_24fps.mp4"
        if os.path.exists(video_path):
            st.success("✅ Sample video loaded: 855564-hd_1920_1080_24fps.mp4")
        else:
            # Check in video folder
            video_path = "video/855564-hd_1920_1080_24fps.mp4"
            if os.path.exists(video_path):
                st.success("✅ Sample video loaded: video/855564-hd_1920_1080_24fps.mp4")
            else:
                st.warning("⚠️ Sample video file not found. Please upload a video or use webcam.")
                video_path = None
    
    else:  # Webcam
        st.info("Webcam access - Make sure your camera is connected")
        video_path = 0
    
    if video_path is not None:
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("❌ Cannot access video source. Please try another option.")
        else:
            # Control buttons
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            with col1:
                if st.button("▶️ Start Processing", use_container_width=True, type="primary"):
                    st.session_state.video_running = True
            with col2:
                if st.button("⏸️ Pause", use_container_width=True):
                    st.session_state.video_running = False
            with col3:
                if st.button("⏹️ Stop", use_container_width=True):
                    st.session_state.video_running = False
                    st.session_state.processed_frames = 0
                    st.session_state.people_history = []
                    st.session_state.heatmap_accum = None
                    st.session_state.zones = []
                    if cap:
                        cap.release()
                    st.rerun()
            with col4:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.video_running = False
                    st.session_state.processed_frames = 0
                    st.session_state.people_history = []
                    st.session_state.heatmap_accum = None
                    st.session_state.zones = []
                    if cap:
                        cap.release()
                    st.rerun()
            
            if st.session_state.video_running:
                # Initialize YOLO model
                try:
                    # Check if model file exists
                    model_path = 'models/yolov8n.pt'
                    if not os.path.exists(model_path):
                        # Try alternative path
                        model_path = 'yolov8n.pt'
                        if not os.path.exists(model_path):
                            st.error("❌ YOLO model file not found. Please ensure 'yolov8n.pt' exists in the project root or in the 'models' folder.")
                            st.session_state.video_running = False
                            st.stop()
                    
                    model = YOLO(model_path)

                    if torch.cuda.is_available():
                        model.to('cuda')
                        device = "GPU 🚀"
                    else:
                        device = "CPU"
                    st.sidebar.info(f"Model loaded on: {device}")
                except Exception as e:
                    st.error(f"❌ Failed to load YOLO model: {e}")
                    st.session_state.video_running = False
                    st.stop()
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                
                st.sidebar.info(f"📊 Video Info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
                
                # Initialize variables on first run
                if st.session_state.heatmap_accum is None:
                    ret, first_frame = cap.read()
                    if ret:
                        st.session_state.heatmap_accum = np.zeros((first_frame.shape[0], first_frame.shape[1]), dtype=np.float32)
                        # Initialize zones
                        h, w = first_frame.shape[:2]
                        st.session_state.zones = []
                        for r in range(NUM_ROWS):
                            for c in range(NUM_COLS):
                                x1 = c * w // NUM_COLS
                                y1 = r * h // NUM_ROWS
                                x2 = (c + 1) * w // NUM_COLS
                                y2 = (r + 1) * h // NUM_ROWS
                                st.session_state.zones.append((x1, y1, x2, y2))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                    else:
                        st.error("❌ Cannot read video frames")
                        st.session_state.video_running = False
                        st.stop()
                
                # Create placeholders for real-time display
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                col_metrics = st.columns(3)
                metric1_placeholder = col_metrics[0].empty()
                metric2_placeholder = col_metrics[1].empty()
                metric3_placeholder = col_metrics[2].empty()
                
                alert_placeholder = st.empty()
                
                # Create three columns for video outputs
                col1, col2, col3 = st.columns(3)
                frame_placeholder = col1.empty()
                heatmap_placeholder = col2.empty()
                zone_placeholder = col3.empty()
                
                # Graph placeholder
                graph_placeholder = st.empty()
                
                frame_count = 0
                start_time = time.time()
                
                while st.session_state.video_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.success("🎉 Video processing completed!")
                        st.session_state.video_running = False
                        break
                    
                    # Update progress
                    if total_frames > 0:
                        progress = min(100, int((frame_count / total_frames) * 100))
                        progress_bar.progress(progress)
                    
                    # Calculate processing speed
                    elapsed_time = time.time() - start_time
                    fps_actual = frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    status_text.text(f"📊 Frame: {frame_count}/{total_frames} | Speed: {fps_actual:.1f} FPS | Progress: {progress}%")
                    
                    if frame_count % FRAME_SKIP != 0:
                        frame_count += 1
                        continue
                    
                    # Resize frame for faster processing
                    frame_small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                    
                    # Run YOLO inference
                    try:
                        results = model(frame_small, verbose=False)[0]
                        
                        # Count people
                        people_count = 0
                        if results.boxes is not None and results.boxes.cls is not None:
                            people_count = sum(1 for cls in results.boxes.cls if int(cls) == 0)
                        
                        st.session_state.people_history.append(people_count)
                        st.session_state.processed_frames += 1
                        
                        # Create display frame with bounding boxes
                        display_frame = frame.copy()
                        if results.boxes is not None:
                            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                                x1, y1, x2, y2 = map(int, np.array(box) / scale)
                                label = model.names[int(cls)]
                                color = (0, 255, 0) if label == "person" else (0, 0, 255)
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(display_frame, f"{label}", (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Update heatmap
                        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
                        if results.boxes is not None:
                            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                                if int(cls) == 0:  # person class
                                    x1, y1, x2, y2 = map(int, np.array(box) / scale)
                                    center_x = (x1 + x2) // 2
                                    center_y = (y1 + y2) // 2
                                    # Add Gaussian blob at person center
                                    cv2.circle(heatmap, (center_x, center_y), 30, 1, -1)
                        
                        st.session_state.heatmap_accum = cv2.addWeighted(st.session_state.heatmap_accum, 0.9, heatmap, 0.1, 0)
                        heatmap_normalized = cv2.normalize(st.session_state.heatmap_accum, None, 0, 255, cv2.NORM_MINMAX)
                        heatmap_color = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
                        
                        # Overlay heatmap on frame
                        overlay_frame = cv2.addWeighted(display_frame, 0.7, heatmap_color, 0.3, 0)
                        
                        # Calculate density and risk
                        density_score = min(1.0, people_count / MAX_PEOPLE_THRESHOLD)
                        if people_count > 60 or density_score > 0.6:
                            risk_level = "HIGH"
                            risk_color = (255, 0, 0)
                            risk_class = "risk-high"
                        elif people_count > 30 or density_score > 0.3:
                            risk_level = "MEDIUM" 
                            risk_color = (0, 165, 255)
                            risk_class = "risk-medium"
                        else:
                            risk_level = "LOW"
                            risk_color = (0, 255, 0)
                            risk_class = "risk-low"
                        
                        # Update metrics
                        metric1_placeholder.metric("👥 People Count", people_count)
                        metric2_placeholder.metric("📊 Density Score", f"{density_score:.2f}")
                        metric3_placeholder.metric("⚠️ Risk Level", risk_level)
                        
                        # Update alert
                        alert_placeholder.markdown(
                            f'<div class="{risk_class}">🚨 <b>RISK ALERT:</b> {risk_level} - {people_count} people detected</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Display frames
                        frame_placeholder.image(cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB), 
                                              caption="Live Detection + Heatmap", use_column_width=True)
                        heatmap_placeholder.image(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB),
                                                caption="Crowd Density Heatmap", use_column_width=True)
                        
                        # Zone analysis
                        zone_img = np.zeros_like(frame)
                        zone_counts = []
                        
                        for idx, (x1, y1, x2, y2) in enumerate(st.session_state.zones):
                            count_in_zone = 0
                            if results.boxes is not None:
                                for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                                    if int(cls) == 0:
                                        bx1, by1, bx2, by2 = map(int, np.array(box) / scale)
                                        center_x = (bx1 + bx2) // 2
                                        center_y = (by1 + by2) // 2
                                        if x1 <= center_x < x2 and y1 <= center_y < y2:
                                            count_in_zone += 1
                            
                            zone_counts.append(count_in_zone)
                            # Color zones based on density
                            if count_in_zone > 10:
                                color = ZONE_COLOR_HIGH
                            elif count_in_zone > 5:
                                color = ZONE_COLOR_MED
                            else:
                                color = ZONE_COLOR_LOW
                            
                            cv2.rectangle(zone_img, (x1, y1), (x2, y2), color, -1)
                            cv2.putText(zone_img, f"Zone {idx+1}: {count_in_zone}", (x1 + 5, y1 + 25),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        zone_overlay = cv2.addWeighted(frame, 0.6, zone_img, 0.4, 0)
                        zone_placeholder.image(cv2.cvtColor(zone_overlay, cv2.COLOR_BGR2RGB),
                                             caption="Zone-wise Crowd Density", use_column_width=True)
                        
                        # Update people count graph
                        if len(st.session_state.people_history) > 1:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(st.session_state.people_history, color='lime', linewidth=2)
                            ax.set_title("People Count Over Time", color='white', fontsize=14)
                            ax.set_xlabel("Frame", color='white')
                            ax.set_ylabel("People Count", color='white')
                            ax.tick_params(colors='white')
                            ax.grid(True, alpha=0.3)
                            ax.set_facecolor('none')
                            fig.patch.set_alpha(0.0)
                            graph_placeholder.pyplot(fig, clear_figure=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing frame: {e}")
                        continue
                    
                    frame_count += 1
                    time.sleep(0.01)  # Small delay to prevent overwhelming
                
                # Cleanup
                if cap:
                    cap.release()
                progress_bar.empty()
                status_text.empty()
                
                # Show final statistics
                if st.session_state.processed_frames > 0:
                    st.success(f"""
                    **📈 Processing Complete!**
                    - Total frames processed: {st.session_state.processed_frames}
                    - Maximum people detected: {max(st.session_state.people_history) if st.session_state.people_history else 0}
                    - Average people count: {np.mean(st.session_state.people_history) if st.session_state.people_history else 0:.1f}
                    - Processing time: {time.time() - start_time:.1f} seconds
                    """)
                
    else:
        st.info("👆 Please select a video source to begin analysis")

# ==================== SENSORS MONITORING MODULE ====================
elif page == "🌡️ Sensors Monitoring":
    st.markdown('<h1 class="main-header">Environmental Sensors Monitoring</h1>', unsafe_allow_html=True)
    
    selected_city = st.selectbox("📍 Select City", list(CITIES.keys()))
    latitude, longitude = CITIES[selected_city]
    
    # Fetch data
    weather_data = fetch_weather_data(latitude, longitude)
    aqi_data = fetch_aqi_data(selected_city)
    
    if weather_data:
        current = weather_data["current"]
        temp = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]
        wind_speed = current["wind_speed_10m"]
        is_day = current["is_day"]
        
        # Calculate heat index
        heat_index = calculate_heat_index(temp, humidity)
        
        # Determine comfort level
        if 18 <= temp <= 26 and 40 <= humidity <= 70:
            comfort = "😊 Comfortable"
            comfort_color = "green"
        elif heat_index > 35:
            comfort = "😰 Hot"
            comfort_color = "red"
        elif temp < 10:
            comfort = "🥶 Cold" 
            comfort_color = "blue"
        else:
            comfort = "😐 Moderate"
            comfort_color = "orange"
        
        # Display current conditions
        st.subheader("📊 Current Environmental Conditions")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌡️ Temperature", f"{temp}°C")
        with col2:
            st.metric("💧 Humidity", f"{humidity}%")
        with col3:
            st.metric("🌬️ Wind Speed", f"{wind_speed} m/s")
        with col4:
            st.metric("🔥 Heat Index", f"{heat_index}°C")
        
        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric("🌫️ AQI", aqi_data["aqi"])
        with col6:
            st.metric("⏰ Time", "☀️ Day" if is_day else "🌙 Night")
        with col7:
            st.metric("😊 Comfort", comfort)
        
        # Environmental alerts
        st.subheader("🚨 Environmental Alerts")
        alerts = []
        
        if temp > 35:
            alerts.append(("🔥 High Temperature", "Temperature exceeds 35°C", "high"))
        if heat_index > 38:
            alerts.append(("🌡️ High Heat Index", f"Feels like {heat_index}°C", "high"))
        if humidity > 80:
            alerts.append(("💧 High Humidity", "Poor ventilation conditions", "medium"))
        if aqi_data["aqi"] > 150:
            alerts.append(("🌫️ Poor Air Quality", "AQI indicates unhealthy conditions", "high"))
        
        if alerts:
            for title, message, severity in alerts:
                if severity == "high":
                    st.error(f"**{title}**: {message}")
                else:
                    st.warning(f"**{title}**: {message}")
        else:
            st.success("✅ All environmental conditions are within normal ranges")
        
        # Forecast charts
        st.subheader("📈 12-Hour Forecast")
        
        # Generate forecast data
        hours = pd.date_range(start=datetime.now(), periods=12, freq='H')
        forecast_df = pd.DataFrame({
            'Time': hours,
            'Temperature': [temp + np.random.normal(0, 2) for _ in range(12)],
            'Humidity': [max(30, min(90, humidity + np.random.normal(0, 10))) for _ in range(12)],
            'Wind Speed': [max(0, wind_speed + np.random.normal(0, 1)) for _ in range(12)]
        })
        
        forecast_df['Heat Index'] = forecast_df.apply(
            lambda x: calculate_heat_index(x['Temperature'], x['Humidity']), axis=1
        )
        
        # Create forecast charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature Trend', 'Humidity Trend', 'Wind Speed', 'Heat Index'),
            vertical_spacing=0.12
        )
        
        fig.add_trace(
            go.Scatter(x=forecast_df['Time'], y=forecast_df['Temperature'], 
                      name='Temperature', line=dict(color='red')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=forecast_df['Time'], y=forecast_df['Humidity'],
                      name='Humidity', line=dict(color='blue'), fill='tozeroy'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=forecast_df['Time'], y=forecast_df['Wind Speed'],
                  name='Wind Speed', marker_color='green'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=forecast_df['Time'], y=forecast_df['Heat Index'],
                      name='Heat Index', line=dict(color='orange', dash='dash')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Environmental Forecast")
        st.plotly_chart(fig, use_container_width=True)
        
        # Air Quality Details
        st.subheader("🌫️ Air Quality Details")
        aq_col1, aq_col2, aq_col3, aq_col4 = st.columns(4)
        aq_col1.metric("PM2.5", f"{aqi_data['pm25']} μg/m³")
        aq_col2.metric("PM10", f"{aqi_data['pm10']} μg/m³")
        aq_col3.metric("NO₂", f"{aqi_data['no2']} ppb")
        aq_col4.metric("O₃", f"{aqi_data['o3']} ppb")
        
        # ---------- Live AQI Map ----------
        st.markdown("---")
        st.subheader("🌍 Live AQI Map")
        
        # Create a more detailed map with better controls
        m = folium.Map(
            location=[latitude, longitude], 
            zoom_start=11,
            tiles=None  # We'll add our own tiles
        )
        
        # Add OpenStreetMap as base layer
        folium.TileLayer(
            tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            attr='OpenStreetMap',
            name='OpenStreetMap',
            control=True
        ).add_to(m)
        
        # Add AQI overlay
        tile_url = f"https://tiles.aqicn.org/tiles/usepa-aqi/{{z}}/{{x}}/{{y}}.png?token={WAQI_TOKEN}"
        folium.TileLayer(
            tiles=tile_url,
            attr="Air Quality - WAQI",
            name="Air Quality Index",
            overlay=True,
            control=True,
            opacity=0.7
        ).add_to(m)
        
        # Add a circle marker for the selected city with AQI information
        aqi_value = "N/A"
        if aqi_data and isinstance(aqi_data, dict):
            aqi_value = aqi_data.get('aqi', 'N/A')
        else:
            aqi_value = "N/A"
        
        folium.CircleMarker(
            location=[latitude, longitude],
            radius=10,
            popup=f"{selected_city}<br>AQI: {aqi_value}",
            tooltip=f"{selected_city}",
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.8
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Display the map with error handling
        try:
            st_folium(m, width=700, height=500)
        except Exception as e:
            st.warning(f"⚠️ Map display error: {e}. Showing map HTML instead.")
            # Fallback: show the map as HTML
            st.components.v1.html(m._repr_html_(), height=500)
        
        # Add AQI legend
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px;">
        <strong>AQI Scale:</strong><br>
        <span style="color: #009966;">0-50: Good</span> | 
        <span style="color: #FFD700;">51-100: Moderate</span> | 
        <span style="color: #FF8C00;">101-150: Unhealthy for Sensitive Groups</span> | 
        <span style="color: #FF0000;">151-200: Unhealthy</span> | 
        <span style="color: #990066;">201-300: Very Unhealthy</span> | 
        <span style="color: #7E0023;">301+: Hazardous</span>
        </div>
        """, unsafe_allow_html=True)

# ==================== GPS VIEW MODULE ====================
elif page == "📍 GPS View":
    st.markdown('<h1 class="main-header">GPS Traffic Intelligence</h1>', unsafe_allow_html=True)
    
    selected_city = st.selectbox("📍 Select City", list(CITIES.keys()))
    latitude, longitude = CITIES[selected_city]
    
    # Fetch traffic incidents
    incidents = fetch_traffic_incidents(latitude, longitude)
    
    st.subheader("🗺️ Live Crowd GPS")
    
    # Create Folium map
    m = folium.Map(location=[latitude, longitude], zoom_start=12, control_scale=True)
    
    # Add OpenStreetMap base layer
    folium.TileLayer(
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='OpenStreetMap',
        name='OpenStreetMap',
        control=True
    ).add_to(m)
    
    # Extract coordinates for heatmap
    heat_data = []
    for incident in incidents:
        # Handle location which might be a single coordinate or a list of coordinates
        location = incident["location"]
        
        if isinstance(location, list) and len(location) > 0:
            if isinstance(location[0], list) and len(location[0]) >= 2:
                # This is a list of coordinates, use the first one
                coord = location[0]
                try:
                    lon = float(coord[0])
                    lat = float(coord[1])
                    location = [lat, lon]
                    # Add to heatmap data
                    severity_multiplier = 1.0 if incident.get('severity') == 'High' else 0.7
                    heat_data.append([lat, lon, severity_multiplier])
                except (ValueError, TypeError):
                    st.warning(f"⚠️ Invalid coordinate format: {coord}, skipping")
            elif len(location) >= 2:
                # This is a simple [lon, lat] coordinate
                try:
                    lon = float(location[0])
                    lat = float(location[1])
                    location = [lat, lon]
                    # Add to heatmap data
                    severity_multiplier = 1.0 if incident.get('severity') == 'High' else 0.7
                    heat_data.append([lat, lon, severity_multiplier])
                except (ValueError, TypeError):
                    st.warning(f"⚠️ Invalid location format: {location}, skipping")
    
    # Add heatmap layer if there's data
    if heat_data:
        # Add the heatmap
        try:
            from folium import plugins
            plugins.HeatMap(heat_data, min_opacity=0.2, max_zoom=18, radius=25, blur=15, max_val=1).add_to(m)
        except ImportError:
            st.warning("folium plugins not available. Install with: pip install branca")
            # Fallback: add markers if heatmap is not available
            for incident in incidents:
                location = incident["location"]
                
                # If location is a list of lists (like a line or polygon), use the first coordinate
                if isinstance(location, list) and len(location) > 0:
                    if isinstance(location[0], list) and len(location[0]) >= 2:
                        # This is a list of coordinates, use the first one
                        coord = location[0]
                        try:
                            lon = float(coord[0])
                            lat = float(coord[1])
                            location = [lat, lon]
                        except (ValueError, TypeError):
                            st.warning(f"⚠️ Invalid coordinate format: {coord}, using default")
                            location = [17.3850, 78.4867]  # Default to Hyderabad
                    elif len(location) >= 2:
                        # This is a simple [lon, lat] coordinate
                        try:
                            lon = float(location[0])
                            lat = float(location[1])
                            location = [lat, lon]
                        except (ValueError, TypeError):
                            st.warning(f"⚠️ Invalid location format: {location}, using default")
                            location = [17.3850, 78.4867]  # Default to Hyderabad
                    else:
                        location = [17.3850, 78.4867]  # Default to Hyderabad
                else:
                    location = [17.3850, 78.4867]  # Default to Hyderabad
                
                folium.Marker(
                    location=location,
                    popup=f"""
                    <b>{incident['type']}</b><br>
                    {incident['description']}<br>
                    Delay: {incident['delay']} min<br>
                    Severity: {incident['severity']}
                    """,
                    tooltip=incident["type"],
                    icon=folium.Icon(
                        color='red' if incident['severity'] == 'High' else 'orange',
                        icon='warning-sign'
                    )
                ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Display map with error handling
    try:
        st_folium(m, width=1200, height=500)
    except Exception as e:
        st.warning(f"⚠️ Map display error: {e}. Showing map HTML instead.")
        # Fallback: show the map as HTML
        st.components.v1.html(m._repr_html_(), height=500)
    
    # Traffic analytics
    st.subheader("📊 Traffic Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🚧 Total Incidents", len(incidents))
    # Calculate average delay, handling cases where delay might be None
    valid_delays = [i['delay'] for i in incidents if i['delay'] is not None]
    avg_delay = sum(valid_delays) / max(1, len(valid_delays)) if valid_delays else 0
    col2.metric("⏱️ Avg Delay", f"{avg_delay:.1f} min")
    col3.metric("🟥 High Severity", sum(1 for i in incidents if i['severity'] == 'High'))
    col4.metric("🟨 Medium Severity", sum(1 for i in incidents if i['severity'] == 'Medium'))
    
    # Incident analysis
    if incidents:
        incident_df = pd.DataFrame(incidents)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Incident type distribution
            type_counts = incident_df['type'].value_counts()
            fig1 = px.pie(values=type_counts.values, names=type_counts.index,
                         title="Incidents by Type", color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Delay by incident type
            delay_by_type = incident_df.groupby('type')['delay'].mean().sort_values(ascending=False)
            fig2 = px.bar(x=delay_by_type.index, y=delay_by_type.values,
                         title="Average Delay by Incident Type (min)",
                         labels={'x': 'Incident Type', 'y': 'Delay (min)'},
                         color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig2, use_container_width=True)
    
    else:
        st.info("No traffic incidents reported in the selected area.")

# ==================== INTEGRATED ANALYTICS ====================
elif page == "📊 Integrated Analytics":
    st.markdown('<h1 class="main-header">Integrated Safety Analytics</h1>', unsafe_allow_html=True)
    
    st.subheader("🔗 Cross-Module Correlation Analysis")
    
    # Define zones for later use
    zones = [f'Zone {i}' for i in range(1, 7)]
    
    # # Generate sample correlation data
    # correlation_data = pd.DataFrame({
    #     'Crowd Density': [0.8, 0.6, 0.9, 0.4, 0.7, 0.5],
    #     'Temperature': [0.4, 0.5, 0.6, 0.3, 0.7, 0.4],
    #     'Traffic Incidents': [0.7, 0.8, 0.4, 0.6, 0.9, 0.5],
    #     'AQI': [0.3, 0.4, 0.5, 0.2, 0.6, 0.3]
    # }, index=zones)
    # 
    # # Correlation matrix 
    # correlation_matrix = correlation_data.corr()
    # 
    # # Correlation matrix visualization
    # fig_corr = px.imshow(correlation_matrix,
    #                     text_auto=True, 
    #                     aspect="auto",
    #                     title="Safety Parameter Correlation Matrix",
    #                     color_continuous_scale='RdBu_r')
    # st.plotly_chart(fig_corr, use_container_width=True)
    # 
    # Instead, show a placeholder message
    st.info("Cross-module correlation analysis is currently under development.")
    
    st.subheader("🎯 Predictive Risk Assessment")
    
    # Risk assessment data
    risk_data = pd.DataFrame({
        'Zone': zones,
        'Crowd Risk': [0.8, 0.6, 0.9, 0.4, 0.7, 0.5],
        'Environmental Risk': [0.4, 0.7, 0.3, 0.6, 0.5, 0.8],
        'Traffic Risk': [0.6, 0.8, 0.4, 0.7, 0.3, 0.9],
        'Overall Risk': [0.72, 0.70, 0.68, 0.57, 0.60, 0.73]
    })
    
    # Stacked risk chart
    fig_risk = go.Figure(data=[
        go.Bar(name='Crowd Risk', x=risk_data['Zone'], y=risk_data['Crowd Risk']),
        go.Bar(name='Environmental Risk', x=risk_data['Zone'], y=risk_data['Environmental Risk']),
        go.Bar(name='Traffic Risk', x=risk_data['Zone'], y=risk_data['Traffic Risk'])
    ])
    
    fig_risk.update_layout(
        barmode='stack',
        title="Risk Distribution by Zone",
        xaxis_title="Zone",
        yaxis_title="Risk Score",
        height=400
    )
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Overall risk ranking
    st.subheader("🏆 Zone Risk Ranking")
    ranked_zones = risk_data.sort_values('Overall Risk', ascending=False)
    
    for idx, (_, zone_data) in enumerate(ranked_zones.iterrows(), 1):
        risk_score = zone_data['Overall Risk']
        if risk_score > 0.7:
            risk_color = "🔴"
            risk_class = "risk-high"
        elif risk_score > 0.5:
            risk_color = "🟡" 
            risk_class = "risk-medium"
        else:
            risk_color = "🟢"
            risk_class = "risk-low"
        
        st.markdown(f"""
        <div class="{risk_class}">
            {risk_color} <b>Rank #{idx}: {zone_data['Zone']}</b><br>
            Overall Risk: {risk_score:.2f} | Crowd: {zone_data['Crowd Risk']:.1f} | 
            Environment: {zone_data['Environmental Risk']:.1f} | Traffic: {zone_data['Traffic Risk']:.1f}
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("💡 Integrated Safety Recommendations")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.success("""
        **🚨 Immediate Actions Required:**
        • Deploy additional security staff to Zone 1 & 3
        • Activate emergency ventilation systems
        • Implement crowd control measures in high-density areas
        • Coordinate with traffic management for route optimization
        """)
    
    with rec_col2:
        st.info("""
        **🛡️ Preventive Measures:**
        • Schedule events during cooler hours (early morning/evening)
        • Implement staggered entry and exit times
        • Enhance medical support in high-risk zones
        • Prepare contingency plans for weather changes
        • Regular air quality monitoring in enclosed spaces
        """)

# ==================== SAFETY ASSISTANT ====================
elif page == "🤖 Safety Assistant":
    st.markdown('<h1 class=\"main-header\">🤖 Safety Assistant</h1>', unsafe_allow_html=True)
    st.markdown("Ask questions about the safety analytics and get intelligent responses")
    
    # Initialize chat history in session state if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Function to process user queries using simple keyword matching
    def get_bot_response(user_input):
        user_input = user_input.lower()
        
        # Keywords for different types of queries
        if any(keyword in user_input for keyword in ['hello', 'hi', 'hey', 'greetings']):
            return "Hello! I'm your Safety Assistant. I can help you understand the crowd safety analytics, traffic incidents, environmental conditions, and risk assessments. What would you like to know?"
        
        elif any(keyword in user_input for keyword in ['risk', 'danger', 'unsafe', 'hazard', 'threat']):
            # Sample response based on mock data
            return "Based on the current analytics data: Zone 1 and Zone 3 have the highest risk levels with crowd risk scores of 0.8. Zone 6 has the highest environmental risk (0.8) and Zone 2 has the highest traffic risk (0.8). I recommend focusing additional security measures on these zones."
        
        elif any(keyword in user_input for keyword in ['crowd', 'people', 'density', 'count']):
            return "Current crowd metrics show Zone 3 has the highest crowd density. The average crowd risk across zones is 0.67. For real-time crowd analysis, please check the Video Analytics module which provides live crowd detection and density heatmaps."
        
        elif any(keyword in user_input for keyword in ['traffic', 'incident', 'accident', 'jam', 'congestion']):
            return "Current traffic data shows several incidents in the monitored area. The GPS View module provides real-time traffic incident monitoring with detailed analytics on incident types and delays. Would you like to check the GPS View for more details?"
        
        elif any(keyword in user_input for keyword in ['weather', 'temperature', 'humidity', 'air', 'a.q.i', 'aqi', 'enviro']):
            return "Environmental sensors are monitoring temperature, humidity, wind speed, and air quality. The Sensors Monitoring module shows real-time environmental conditions. Currently, comfort levels vary by location and can be checked in the sensor dashboard."
        
        elif any(keyword in user_input for keyword in ['zone', 'area', 'location', 'section']):
            return "The system divides the surveillance area into 6 zones for targeted monitoring. Each zone has separate risk assessments for crowd density, environmental factors, and traffic impacts. Zone 1 and 3 currently show the highest crowd risk, while Zone 6 shows the highest environmental risk."
        
        elif any(keyword in user_input for keyword in ['help', 'assist', 'support', 'what can you do']):
            return "I can provide information about: 1) Current risk levels in different zones, 2) Crowd density and safety metrics, 3) Traffic incidents and their impact, 4) Environmental conditions and air quality, 5) Safety recommendations and alerts. Just ask a specific question!"
        
        elif any(keyword in user_input for keyword in ['recommend', 'suggest', 'advice', 'action']):
            return "Based on current analytics: 1) Deploy additional security staff to Zone 3, 2) Monitor air quality in enclosed spaces, 3) Implement crowd control measures during peak times, 4) Coordinate with traffic management for route optimization, 5) Enhance medical support in high-risk zones."
        
        elif any(keyword in user_input for keyword in ['data', 'analytics', 'metrics', 'stats', 'statistics']):
            return "The platform provides integrated analytics across video, sensor, and GPS data. Key metrics include crowd density, risk assessments per zone, environmental conditions, traffic incidents, and correlation analysis between different safety factors."
        
        elif any(keyword in user_input for keyword in ['thank', 'thanks', 'appreciate']):
            return "You're welcome! If you have any more questions about safety analytics, feel free to ask."
        
        else:
            return "I'm your Safety Assistant. I can help you understand the crowd safety analytics, traffic incidents, environmental conditions, and risk assessments. You can ask me about risks in different zones, crowd density, traffic incidents, environmental conditions, or safety recommendations."
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about safety analytics..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display bot response
        response = get_bot_response(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.markdown(response)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: gray;'>"
    "🚨 Crowd Safety AI Platform v2.0 | Integrated Monitoring & Analytics | "
    f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    "</div>", 
    unsafe_allow_html=True
)