# Crowd Safety AI Platform

This is a comprehensive crowd safety monitoring system built with Streamlit that provides real-time monitoring and predictive analytics for crowd safety management. It integrates video analytics, environmental sensors, and GPS traffic data.

## 🌟 Features

### **Dashboard Overview**
- Quick stats and platform modules overview
- Summary of active city, crowd density, comfort level, and traffic status
- Platform module descriptions and safety alerts
- Quick action buttons for direct access to modules

### **Video Analytics** 
- Real-time crowd detection using YOLOv8
- Density heatmaps for crowd visualization
- Zone-wise analysis with configurable grid layout
- Risk assessment based on density scores
- Live detection with bounding boxes
- People count tracking over time
- Performance metrics (FPS, processing speed)

### **Sensors Monitoring**
- Weather conditions (temperature, humidity, wind speed)
- Heat index calculation and comfort level assessment
- Air Quality Index (AQI) with detailed metrics (PM2.5, PM10, NO₂, O₃, CO)
- **Enhanced AQI Map** - Interactive map with AQI overlay and legend
- Environmental alerts and recommendations
- 12-hour forecast charts for all parameters
- Real-time air quality visualization

### **GPS Traffic Intelligence** 
- Live traffic incident monitoring
- **Interactive Heatmap** - Visual heatmap of traffic incidents by severity
- Multiple base map options with layer control
- Incident categorization (Accident, Road Closed, Jam, Road Works, etc.)
- Detailed analytics with incident distribution charts
- **Enhanced Trend Analysis** - 24-hour incident trends with moving averages
- Delay calculations and severity metrics
- City-specific traffic monitoring

### **Integrated Analytics**
- Cross-module correlation analysis
- Predictive risk assessment combining all data sources
- Zone-wise risk ranking and visualization
- Safety recommendations and action items
- Risk distribution charts

### **Safety Assistant** 
- Intelligent chatbot for answering safety-related questions
- Provides information about risk levels, crowd density, traffic incidents, and environmental conditions
- Offers safety recommendations based on current analytics
- Natural language interface for easy access to platform data

## 🛠️ Installation

1. **Clone the repository** (if applicable) or download the files

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root with your API keys:
   ```
   WAQI_TOKEN=your_waqi_token_here
   TOMTOM_API_KEY=your_tomtom_api_key_here
   ```
   
   For initial testing, you can use the defaults:
   ```
   WAQI_TOKEN=demo
   TOMTOM_API_KEY=demo
   ```

5. **Model and Video Files**:
   - Ensure `yolov8n.pt` exists in the project root or in the `models/` folder
   - Ensure video files are available in the `video/` folder

## 🚀 Usage

### Main Application
Run the main integrated application:
```bash
streamlit run full.py
```

The application now includes a **Safety Assistant** accessible from the sidebar navigation. Simply select "🤖 Safety Assistant" to interact with the chatbot and ask questions about safety analytics, risk levels, traffic incidents, and environmental conditions.

### Individual Modules (for testing/development)
```bash
streamlit run videoanalysis.py
streamlit run "sensors analys.py"
streamlit run "gps analys.py"
```

## 🔧 Configuration

### API Keys
- **WAQI Token**: For air quality data - get from https://aqicn.org/api/
- **TomTom API Key**: For traffic data - get from https://developer.tomtom.com/

### Settings
You can modify the following in the `.env` file:
- `WAQI_TOKEN`: Air quality API token
- `TOMTOM_API_KEY`: Traffic data API key

## 📁 Project Structure
```
├── full.py               # Main integrated application
├── videoanalysis.py      # Standalone video analytics
├── sensors analys.py     # Standalone environmental monitoring  
├── gps analys.py         # Standalone traffic intelligence
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── setup.py              # Setup/installation script
├── yolov8n.pt            # YOLO model
├── models/               # Alternative location for YOLO model
│   └── yolov8n.pt
└── video/                # Sample video files
    └── 855564-hd_1920_1080_24fps.mp4
```

## 📊 Enhanced Features

### **Improved Map Visualizations**
- **Traffic Heatmap** in GPS View: Shows traffic incident density with severity-based intensity
- **AQI Map** in Sensors: Interactive air quality overlay with legend and city markers
- Layer controls for better map navigation
- Error handling with fallback HTML display

### **Enhanced Analytics**
- Moving average trends in traffic analysis
- Detailed incident categorization with statistics
- Improved forecast charts with multiple parameters
- Risk correlation matrix

### **New Safety Assistant Feature**
- **Intelligent Chat Interface**: Natural language processing for safety-related queries
- **Context-Aware Responses**: Answers about risk levels, crowd density, traffic, and environmental conditions
- **Integrated Knowledge**: Access to all platform modules through a single conversational interface
- **Real-time Information**: Provides current safety metrics and recommendations

## 🚨 Troubleshooting

1. **Model not found error**:
   - Ensure `yolov8n.pt` exists in the project root or `models/` folder
   - Download from: https://github.com/ultralytics/assets/releases

2. **Map display errors**:
   - Install required plugins: `pip install branca`
   - Application has fallback HTML display for map components

3. **API errors**:
   - Check your API keys in the `.env` file
   - Verify API rate limits haven't been exceeded

4. **Streamlit performance**:
   - Video processing is intensive, ensure adequate hardware
   - Consider using GPU if CUDA is available

## ⚙️ Dependencies

Primary dependencies include:
- streamlit
- ultralytics
- torch (PyTorch)
- opencv-python
- numpy
- pandas
- plotly
- folium
- streamlit-folium
- requests
- matplotlib
- python-dotenv
- branca (for heatmaps)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add your changes to reflect the enhanced features
5. Commit your changes
6. Push to the branch
7. Create a Pull Request

## 📄 License

This project is licensed under the terms specified in the repository.

## 🐛 Issues

If you encounter any issues, please create an issue in the repository with detailed information about:
- Your environment
- Steps to reproduce the issue
- Expected vs actual behavior
- Any error messages received