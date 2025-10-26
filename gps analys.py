import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Attempt to import folium plugins for heatmap (install if needed)
try:
    from folium import plugins
    HEATMAP_AVAILABLE = True
except ImportError:
    HEATMAP_AVAILABLE = False
    st.warning("folium plugins not available. Install with: pip install branca")

# --------------------------
# CONFIGURATION & SETUP
# --------------------------
API_KEY = os.getenv("TOMTOM_API_KEY", "demo")
CITY_NAME = "Hyderabad"
BBOX = "78.223,17.215,78.602,17.600"
REFRESH_INTERVAL = 120  # seconds

st.set_page_config(
    page_title=f"{CITY_NAME} Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# SESSION STATE
# --------------------------
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'incident_history' not in st.session_state:
    st.session_state.incident_history = []
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

# --------------------------
# TRAFFIC SERVICE
# --------------------------
class TrafficIntelligenceService:
    def __init__(self, api_key):
        self.api_key = api_key
        self.severity_colors = {
            "Accident": "red",
            "Road Closed": "darkred", 
            "Jam": "orange",
            "Road Works": "purple",
            "Broken Vehicle": "lightred",
            "Dangerous Conditions": "beige",
            "Weather": "lightblue",
            "Other": "gray"
        }

    def get_icon(self, category):
        icons = {
            "Accident": "car-crash",
            "Road Closed": "ban-circle",
            "Jam": "time",
            "Road Works": "wrench",
            "Broken Vehicle": "cog",
            "Dangerous Conditions": "warning-sign",
            "Weather": "cloud",
            "Other": "info-sign"
        }
        return icons.get(category, "info-sign")

    def categorize_incident(self, code):
        mapping = {1:"Accident", 6:"Jam", 7:"Road Closed", 8:"Road Closed",
                   9:"Road Works", 14:"Broken Vehicle", 3:"Dangerous Conditions",
                   2:"Weather",4:"Weather",5:"Weather",10:"Weather",11:"Weather",15:"Weather"}
        return mapping.get(code, "Other")

    def fetch_incidents(self, bbox):
        url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
        params = {
            "key": self.api_key,
            "bbox": bbox,
            "fields": "{incidents{type,geometry{type,coordinates},properties{id,iconCategory,magnitudeOfDelay,events{description,code},from,to,delay,length,startTime,endTime}}}",
            "language": "en-GB",
            "timeValidityFilter": "present"
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            incidents = data.get("incidents", [])
            for inc in incidents:
                props = inc.get("properties", {})
                code = props.get("iconCategory", 0)
                inc['category'] = self.categorize_incident(code)
                inc['severity_color'] = self.severity_colors.get(inc['category'], "gray")
                inc['icon'] = self.get_icon(inc['category'])
            return incidents
        except Exception as e:
            st.error(f"Error fetching incidents: {e}")
            return []

# --------------------------
# MAP CREATION
# --------------------------
def create_map(incidents):
    m = folium.Map(location=[17.3850,78.4867], zoom_start=12, control_scale=True)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        name="OpenStreetMap",
        attr="© OpenStreetMap contributors"
    ).add_to(m)

    # Extract coordinates for heatmap
    heat_data = []
    for inc in incidents:
        geom = inc["geometry"]
        if geom["type"] == "Point":
            # TomTom API returns [lon, lat], so we need [lat, lon] for folium
            lon, lat = geom["coordinates"]
            # Add intensity based on incident severity
            severity_multiplier = 1.0 if inc.get('category') in ['Accident', 'Road Closed'] else 0.7
            heat_data.append([lat, lon, severity_multiplier])
        elif geom["type"] == "LineString":
            # For LineString, add all points in the line
            for coord in geom["coordinates"]:
                lon, lat = coord
                heat_data.append([lat, lon, 0.5])

    # Add heatmap layer if available
    if HEATMAP_AVAILABLE and heat_data:
        # Create a feature group for the heatmap
        heat_map = plugins.HeatMap(heat_data, min_opacity=0.2, max_zoom=18, radius=25, blur=15, max_val=1)
        heat_map.add_to(m)
        
        # Add a layer control for the heatmap
        folium.LayerControl().add_to(m)
    else:
        # Fallback: add markers if heatmap is not available
        for inc in incidents:
            geom = inc["geometry"]
            desc = inc.get("properties", {}).get("events", [{}])[0].get("description", "Traffic incident")
            popup_html = f"<b>{desc}</b><br>Type: {inc['category']}"

            if geom["type"] == "Point":
                lon, lat = geom["coordinates"]
                coords = [lat, lon]  # Convert [lon, lat] to [lat, lon]
                folium.Marker(
                    location=coords,
                    popup=popup_html,
                    tooltip=desc,
                    icon=folium.Icon(color=inc.get("severity_color", "gray"), icon=inc.get("icon", "info-sign"))
                ).add_to(m)

            elif geom["type"] == "LineString":
                coords = [(lat, lon) for lon, lat in geom["coordinates"]]
                folium.PolyLine(
                    coords,
                    color=inc.get("severity_color", "gray"),
                    weight=6,
                    opacity=0.8,
                    popup=popup_html
                ).add_to(m)

    return m

# --------------------------
# ANALYTICS
# --------------------------
def analytics(incidents):
    if not incidents:
        st.info("No incident data for analytics")
        return
    data=[]
    for inc in incidents:
        props = inc.get("properties",{})
        data.append({
            "category": inc['category'],
            "description": props.get("events",[{}])[0].get("description","Unknown"),
            "delay": props.get("delay",0),
            "length": props.get("length",0)/1000,
            "magnitude": props.get("magnitudeOfDelay","Unknown")
        })
    df = pd.DataFrame(data)
    
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Incidents", len(incidents))
    c2.metric("Total Delay (min)", f"{df['delay'].sum()/60:.0f}")
    c3.metric("Avg Incident Length (km)", f"{df['length'].mean():.1f}")
    
    c1,c2 = st.columns(2)
    cat_counts = df['category'].value_counts()
    fig1 = px.pie(values=cat_counts.values, names=cat_counts.index, 
                  title="Incidents by Category", color_discrete_sequence=px.colors.qualitative.Set3)
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig1, use_container_width=True)
    
    delay_by_cat = df.groupby('category')['delay'].sum().sort_values(ascending=False)
    fig2 = px.bar(x=delay_by_cat.index, y=delay_by_cat.values/60,
                  title="Total Delay by Category (min)", labels={'x':'Category','y':'Delay (min)'},
                  color=delay_by_cat.index, color_discrete_sequence=px.colors.qualitative.Set3)
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# --------------------------
# SIDEBAR
# --------------------------
def sidebar():
    with st.sidebar:
        st.header("Traffic Controls")
        st.session_state.auto_refresh = st.checkbox("Enable Auto-Refresh", value=st.session_state.auto_refresh)
        if st.button("Refresh Now"):
            st.experimental_rerun()
        st.markdown("---")
        st.header("Incident Legend")
        legend = {"🔴 Accident":"Collision","🟠 Traffic Jam":"Congestion",
                  "🟣 Road Works":"Construction","⚫ Road Closed":"Closed",
                  "🔵 Weather":"Weather issues","🟡 Other":"Miscellaneous"}
        for k,v in legend.items():
            st.caption(f"{k} - {v}")
        st.markdown("---")
        st.header("Statistics")
        st.caption(f"City: {CITY_NAME}")
        st.caption(f"Coverage Area: {BBOX}")
        st.caption(f"Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")

# --------------------------
# MAIN
# --------------------------
def main():
    traffic_service = TrafficIntelligenceService(API_KEY)
    st.title(f"{CITY_NAME} Traffic Intelligence Dashboard")
    st.markdown("Real-time traffic monitoring with analytics")
    
    sidebar()
    
    if st.session_state.auto_refresh:
        st_autorefresh(interval=REFRESH_INTERVAL*1000, key="auto_refresh")
    
    incidents = traffic_service.fetch_incidents(BBOX)
    st.session_state.incident_history.append({"timestamp":datetime.now(), "incident_count":len(incidents)})
    cutoff = datetime.now() - timedelta(hours=24)
    st.session_state.incident_history = [h for h in st.session_state.incident_history if h["timestamp"]>cutoff]
    
    tab1,tab2,tab3 = st.tabs(["Live Map","Analytics","Trends"])
    
    with tab1:
        st.subheader("Live Traffic Map")
        fol_map = create_map(incidents)
        st_folium(fol_map, width=1200, height=600)
    
    with tab2:
        st.subheader("Traffic Analytics")
        analytics(incidents)
    
    with tab3:
        st.subheader("Incident Trends")
        if len(st.session_state.incident_history) > 1:
            # Create dataframe ensuring proper data types
            df = pd.DataFrame(st.session_state.incident_history)
            
            # Ensure timestamp column is datetime type and incident_count is numeric
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["incident_count"] = pd.to_numeric(df["incident_count"], errors='coerce')
            
            # Drop any rows with invalid data
            df = df.dropna()
            
            if not df.empty:
                fig = px.line(df, x="timestamp", y="incident_count", title="Incident Trends Last 24h")
                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Number of Incidents",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional trend analysis
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Trend Summary")
                    if len(df) > 1:
                        first_count = df["incident_count"].iloc[0]
                        last_count = df["incident_count"].iloc[-1]
                        change = last_count - first_count
                        change_percent = (change / first_count * 100) if first_count != 0 else 0
                        trend = "📈 Increasing" if change > 0 else "📉 Decreasing" if change < 0 else "➡️ Stable"
                        st.metric("Trend", trend)
                        st.metric("Change", f"{change:+d}")
                        st.metric("Change %", f"{change_percent:+.1f}%")
                
                with col2:
                    # Create a more detailed chart with moving average
                    df_sorted = df.sort_values("timestamp")
                    df_sorted["moving_avg"] = df_sorted["incident_count"].rolling(window=3, min_periods=1).mean()
                    
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=df_sorted["timestamp"], y=df_sorted["incident_count"],
                                            mode='lines+markers', name='Actual', line=dict(color='blue')))
                    fig2.add_trace(go.Scatter(x=df_sorted["timestamp"], y=df_sorted["moving_avg"],
                                            mode='lines', name='Moving Average (3 pts)', line=dict(color='red', dash='dash')))
                    fig2.update_layout(title="Incidents with Moving Average", xaxis_title="Time", yaxis_title="Incidents")
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No valid historical data to display.")
        else:
            st.info("Collecting historical data...")
    
    st.markdown("---")
    st.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("Data provided by TomTom Traffic API")

if __name__=="__main__":
    main()
