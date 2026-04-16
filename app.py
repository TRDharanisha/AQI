import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime

# --------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------
st.set_page_config(
    page_title="Global AQI Intelligence Hub",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------
# PREMIUM CUSTOM STYLING
# --------------------------------------------------------
st.markdown("""
    <style>

    /* MAIN BACKGROUND – Light Green Gradient */
    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 40%, #a5d6a7 100%);
        background-attachment: fixed;
    }

    /* Glassmorphism Light Card */
    .main .block-container {
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    }

    /* Headings */
    h1 {
        color: #1b5e20 !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        padding: 1rem 0;
        text-shadow: none !important;
    }

    h2, h3 {
        color: #2e7d32 !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }

    /* Buttons – Green Gradient */
    .stButton>button {
        background: linear-gradient(135deg, #66bb6a, #43a047);
        color: white;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.75em 2em;
        border: none;
        transition: 0.25s ease;
        box-shadow: 0 4px 15px rgba(76,175,80,0.3);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(56,142,60,0.4);
    }

    /* Sidebar – Green Soft Gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #c8e6c9 0%, #e8f5e9 100%);
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #1b5e20 !important;
        font-weight: 600;
    }

    /* Metric Values */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #2e7d32 !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 12px;
        padding: 1rem;
    }

    /* Alerts */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #43a047 !important;
    }

    </style>
""", unsafe_allow_html=True)


# --------------------------------------------------------
# TITLE WITH ENHANCED HEADER
# --------------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center;'>🌍 Global AQI Intelligence Hub</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white; font-size: 18px;'>Advanced Air Quality Monitoring & Prediction System</p>", unsafe_allow_html=True)

# --------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎯 Navigation")
    page = st.radio(
        "",
        ["📊 Data Exploration", "🤖 AQI Prediction", "📈 Feature Insights", "🌐 Global Comparison"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📅 Dashboard Info")
    st.info(f"**Date:** {datetime.now().strftime('%B %d, %Y')}")
    st.markdown("**Version:** 2.0 Premium")
    # st.markdown("**Team:** Team 12")

# --------------------------------------------------------
# LOAD DATA FUNCTION WITH GEOCODING
# --------------------------------------------------------
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.replace('\t', '', regex=False)
    if "aqi_value" in df.columns:
        df.rename(columns={"aqi_value": "AQI"}, inplace=True)
    return df

# Comprehensive city coordinates database
CITY_COORDINATES = {
    # India - Major Cities
    'Delhi': (28.6139, 77.2090),
    'New Delhi': (28.6139, 77.2090),
    'Mumbai': (19.0760, 72.8777),
    'Kolkata': (22.5726, 88.3639),
    'Chennai': (13.0827, 80.2707),
    'Bangalore': (12.9716, 77.5946),
    'Bengaluru': (12.9716, 77.5946),
    'Hyderabad': (17.3850, 78.4867),
    'Ahmedabad': (23.0225, 72.5714),
    'Pune': (18.5204, 73.8567),
    'Surat': (21.1702, 72.8311),
    'Jaipur': (26.9124, 75.7873),
    'Lucknow': (26.8467, 80.9462),
    'Kanpur': (26.4499, 80.3319),
    'Nagpur': (21.1458, 79.0882),
    'Indore': (22.7196, 75.8577),
    'Thane': (19.2183, 72.9781),
    'Bhopal': (23.2599, 77.4126),
    'Visakhapatnam': (17.6868, 83.2185),
    'Pimpri-Chinchwad': (18.6298, 73.7997),
    'Patna': (25.5941, 85.1376),
    'Vadodara': (22.3072, 73.1812),
    'Ghaziabad': (28.6692, 77.4538),
    'Ludhiana': (30.9010, 75.8573),
    'Agra': (27.1767, 78.0081),
    'Nashik': (19.9975, 73.7898),
    'Faridabad': (28.4089, 77.3178),
    'Meerut': (28.9845, 77.7064),
    'Rajkot': (22.3039, 70.8022),
    'Varanasi': (25.3176, 82.9739),
    'Srinagar': (34.0836, 74.7973),
    'Amritsar': (31.6340, 74.8723),
    'Allahabad': (25.4358, 81.8463),
    'Prayagraj': (25.4358, 81.8463),
    'Ranchi': (23.3441, 85.3096),
    'Howrah': (22.5958, 88.2636),
    'Coimbatore': (11.0168, 76.9558),
    'Jabalpur': (23.1815, 79.9864),
    'Gwalior': (26.2183, 78.1828),
    'Vijayawada': (16.5062, 80.6480),
    'Jodhpur': (26.2389, 73.0243),
    'Madurai': (9.9252, 78.1198),
    'Raipur': (21.2514, 81.6296),
    'Kota': (25.2138, 75.8648),
    'Chandigarh': (30.7333, 76.7794),
    'Guwahati': (26.1445, 91.7362),
    'Tiruchirappalli': (10.7905, 78.7047),
    'Tiruchirapalli': (10.7905, 78.7047),
    'Solapur': (17.6599, 75.9064),
    'Hubballi': (15.3647, 75.1240),
    'Bareilly': (28.3670, 79.4304),
    'Moradabad': (28.8389, 78.7378),
    'Mysore': (12.2958, 76.6394),
    'Gurgaon': (28.4595, 77.0266),
    'Gurugram': (28.4595, 77.0266),
    'Aligarh': (27.8974, 78.0880),
    'Jalandhar': (31.3260, 75.5762),
    'Tiruchirappalli': (10.7905, 78.7047),
    'Bhubaneswar': (20.2961, 85.8245),
    'Salem': (11.6643, 78.1460),
    'Mira-Bhayandar': (19.2952, 72.8544),
    'Warangal': (17.9689, 79.5941),
    'Thiruvananthapuram': (8.5241, 76.9366),
    'Guntur': (16.3067, 80.4365),
    'Bhiwandi': (19.2969, 73.0635),
    'Saharanpur': (29.9680, 77.5460),
    'Gorakhpur': (26.7606, 83.3732),
    'Bikaner': (28.0229, 73.3119),
    'Amravati': (20.9320, 77.7523),
    'Noida': (28.5355, 77.3910),
    'Jamshedpur': (22.8046, 86.2029),
    'Bhilai': (21.2091, 81.3811),
    'Cuttack': (20.4625, 85.8830),
    'Firozabad': (27.1591, 78.3957),
    'Kochi': (9.9312, 76.2673),
    'Cochin': (9.9312, 76.2673),
    'Bhavnagar': (21.7645, 72.1519),
    'Dehradun': (30.3165, 78.0322),
    'Durgapur': (23.5204, 87.3119),
    'Asansol': (23.6833, 86.9833),
    'Nanded': (19.1383, 77.3210),
    'Kolhapur': (16.7050, 74.2433),
    'Ajmer': (26.4499, 74.6399),
    'Akola': (20.7002, 77.0082),
    'Gulbarga': (17.3297, 76.8343),
    'Jamnagar': (22.4707, 70.0577),
    'Ujjain': (23.1765, 75.7885),
    'Loni': (28.7521, 77.2867),
    'Siliguri': (26.7271, 88.3953),
    'Jhansi': (25.4484, 78.5685),
    'Ulhasnagar': (19.2183, 73.1382),
    'Jammu': (32.7266, 74.8570),
    'Sangli-Miraj & Kupwad': (16.8524, 74.5815),
    'Mangalore': (12.9141, 74.8560),
    'Erode': (11.3410, 77.7172),
    'Belgaum': (15.8497, 74.4977),
    'Ambattur': (13.1143, 80.1548),
    'Tirunelveli': (8.7139, 77.7567),
    'Malegaon': (20.5579, 74.5287),
    'Gaya': (24.7955, 85.0002),
    'Jalgaon': (21.0077, 75.5626),
    'Udaipur': (24.5854, 73.7125),
    'Maheshtala': (22.5095, 88.2475),
    
    # China
    'Beijing': (39.9042, 116.4074),
    'Shanghai': (31.2304, 121.4737),
    'Guangzhou': (23.1291, 113.2644),
    'Shenzhen': (22.5431, 114.0579),
    'Chengdu': (30.5728, 104.0668),
    'Chongqing': (29.4316, 106.9123),
    'Tianjin': (39.3434, 117.3616),
    'Wuhan': (30.5928, 114.3055),
    'Hangzhou': (30.2741, 120.1551),
    'Xi\'an': (34.3416, 108.9398),
    'Nanjing': (32.0603, 118.7969),
    'Shenyang': (41.8057, 123.4328),
    
    # Asia Pacific
    'Tokyo': (35.6762, 139.6503),
    'Seoul': (37.5665, 126.9780),
    'Singapore': (1.3521, 103.8198),
    'Hong Kong': (22.3193, 114.1694),
    'Bangkok': (13.7563, 100.5018),
    'Jakarta': (-6.2088, 106.8456),
    'Manila': (14.5995, 120.9842),
    'Kuala Lumpur': (3.1390, 101.6869),
    'Hanoi': (21.0285, 105.8542),
    'Ho Chi Minh City': (10.8231, 106.6297),
    'Taipei': (25.0330, 121.5654),
    'Osaka': (34.6937, 135.5023),
    'Yokohama': (35.4437, 139.6380),
    'Karachi': (24.8607, 67.0011),
    'Lahore': (31.5204, 74.3587),
    'Islamabad': (33.6844, 73.0479),
    'Dhaka': (23.8103, 90.4125),
    'Colombo': (6.9271, 79.8612),
    'Kathmandu': (27.7172, 85.3240),
    
    # Europe
    'London': (51.5074, -0.1278),
    'Paris': (48.8566, 2.3522),
    'Berlin': (52.5200, 13.4050),
    'Madrid': (40.4168, -3.7038),
    'Rome': (41.9028, 12.4964),
    'Amsterdam': (52.3676, 4.9041),
    'Brussels': (50.8503, 4.3517),
    'Vienna': (48.2082, 16.3738),
    'Athens': (37.9838, 23.7275),
    'Barcelona': (41.3851, 2.1734),
    'Munich': (48.1351, 11.5820),
    'Milan': (45.4642, 9.1900),
    'Prague': (50.0755, 14.4378),
    'Warsaw': (52.2297, 21.0122),
    'Budapest': (47.4979, 19.0402),
    'Stockholm': (59.3293, 18.0686),
    'Copenhagen': (55.6761, 12.5683),
    'Oslo': (59.9139, 10.7522),
    'Lisbon': (38.7223, -9.1393),
    'Dublin': (53.3498, -6.2603),
    'Zurich': (47.3769, 8.5417),
    'Geneva': (46.2044, 6.1432),
    
    # North America
    'New York': (40.7128, -74.0060),
    'Los Angeles': (34.0522, -118.2437),
    'Chicago': (41.8781, -87.6298),
    'Houston': (29.7604, -95.3698),
    'Phoenix': (33.4484, -112.0740),
    'Philadelphia': (39.9526, -75.1652),
    'San Antonio': (29.4241, -98.4936),
    'San Diego': (32.7157, -117.1611),
    'Dallas': (32.7767, -96.7970),
    'San Jose': (37.3382, -121.8863),
    'Austin': (30.2672, -97.7431),
    'Jacksonville': (30.3322, -81.6557),
    'San Francisco': (37.7749, -122.4194),
    'Seattle': (47.6062, -122.3321),
    'Denver': (39.7392, -104.9903),
    'Washington': (38.9072, -77.0369),
    'Boston': (42.3601, -71.0589),
    'Nashville': (36.1627, -86.7816),
    'Detroit': (42.3314, -83.0458),
    'Portland': (45.5152, -122.6784),
    'Las Vegas': (36.1699, -115.1398),
    'Miami': (25.7617, -80.1918),
    'Toronto': (43.6532, -79.3832),
    'Montreal': (45.5017, -73.5673),
    'Vancouver': (49.2827, -123.1207),
    'Calgary': (51.0447, -114.0719),
    'Ottawa': (45.4215, -75.6972),
    'Edmonton': (53.5461, -113.4938),
    'Mexico City': (19.4326, -99.1332),
    'Guadalajara': (20.6597, -103.3496),
    'Monterrey': (25.6866, -100.3161),
    
    # South America
    'São Paulo': (-23.5505, -46.6333),
    'Rio de Janeiro': (-22.9068, -43.1729),
    'Buenos Aires': (-34.6037, -58.3816),
    'Lima': (-12.0464, -77.0428),
    'Bogotá': (4.7110, -74.0721),
    'Santiago': (-33.4489, -70.6693),
    'Caracas': (10.4806, -66.9036),
    'Brasília': (-15.8267, -47.9218),
    'Fortaleza': (-3.7172, -38.5433),
    'Salvador': (-12.9714, -38.5014),
    
    # Middle East
    'Dubai': (25.2048, 55.2708),
    'Abu Dhabi': (24.4539, 54.3773),
    'Riyadh': (24.7136, 46.6753),
    'Jeddah': (21.2854, 39.2376),
    'Tehran': (35.6892, 51.3890),
    'Baghdad': (33.3152, 44.3661),
    'Istanbul': (41.0082, 28.9784),
    'Ankara': (39.9334, 32.8597),
    'Tel Aviv': (32.0853, 34.7818),
    'Jerusalem': (31.7683, 35.2137),
    'Doha': (25.2854, 51.5310),
    'Kuwait City': (29.3759, 47.9774),
    'Muscat': (23.5880, 58.3829),
    'Beirut': (33.8886, 35.4955),
    'Amman': (31.9454, 35.9284),
    
    # Africa
    'Cairo': (30.0444, 31.2357),
    'Lagos': (6.5244, 3.3792),
    'Johannesburg': (-26.2041, 28.0473),
    'Nairobi': (-1.2864, 36.8172),
    'Kinshasa': (-4.4419, 15.2663),
    'Accra': (5.6037, -0.1870),
    'Addis Ababa': (9.0320, 38.7469),
    'Casablanca': (33.5731, -7.5898),
    'Algiers': (36.7538, 3.0588),
    'Abuja': (9.0765, 7.3986),
    'Dar es Salaam': (-6.7924, 39.2083),
    'Khartoum': (15.5007, 32.5599),
    'Luanda': (-8.8383, 13.2344),
    'Cape Town': (-33.9249, 18.4241),
    'Durban': (-29.8587, 31.0218),
    
    # Oceania
    'Sydney': (-33.8688, 151.2093),
    'Melbourne': (-37.8136, 144.9631),
    'Brisbane': (-27.4698, 153.0251),
    'Perth': (-31.9505, 115.8605),
    'Auckland': (-36.8485, 174.7633),
    'Wellington': (-41.2865, 174.7762),
    'Adelaide': (-34.9285, 138.6007),
    'Canberra': (-35.2809, 149.1300),
}

def add_coordinates(df):
    """Add latitude and longitude to dataframe based on city names with smart matching"""
    if 'city_name' in df.columns:
        def get_coordinates(city_name):
            if pd.isna(city_name):
                return None, None
            
            # Clean the city name
            city_clean = str(city_name).strip()
            
            # Direct match
            if city_clean in CITY_COORDINATES:
                return CITY_COORDINATES[city_clean]
            
            # Case-insensitive match
            for key in CITY_COORDINATES:
                if key.lower() == city_clean.lower():
                    return CITY_COORDINATES[key]
            
            # Partial match (e.g., "New Delhi" matches "Delhi")
            for key in CITY_COORDINATES:
                if city_clean.lower() in key.lower() or key.lower() in city_clean.lower():
                    return CITY_COORDINATES[key]
            
            # No match found
            return None, None
        
        df[['lat', 'lon']] = df['city_name'].apply(lambda x: pd.Series(get_coordinates(x)))
        
        # Log cities without coordinates
        missing_coords = df[df['lat'].isna()]['city_name'].unique()
        if len(missing_coords) > 0:
            print(f"⚠️ Warning: {len(missing_coords)} cities don't have coordinates and will be excluded from map")
            print(f"Missing cities: {', '.join(missing_coords[:10])}")
            if len(missing_coords) > 10:
                print(f"... and {len(missing_coords) - 10} more")
    
    return df

def get_aqi_category(aqi):
    """Get AQI category and color"""
    if aqi <= 50:
        return "Good", "#00e400"
    elif aqi <= 100:
        return "Moderate", "#ffff00"
    elif aqi <= 150:
        return "Unhealthy (SG)", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"

# --------------------------------------------------------
# PAGE 1: ENHANCED DATA EXPLORATION
# --------------------------------------------------------
if page == "📊 Data Exploration":
    st.markdown("## 📊 Advanced Exploratory Data Analysis")
    
    uploaded_file = st.file_uploader("📁 Upload your AQI dataset (CSV format):", type=["csv"])
    
    if uploaded_file:
        df = load_csv(uploaded_file)
        df = add_coordinates(df)
        st.success("✅ Data uploaded and processed successfully!")
        
        # --- FILTER SECTION ---
        st.markdown("### 🔍 Data Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "country_name" in df.columns:
                countries = ["All"] + sorted(df['country_name'].dropna().unique().tolist())
                selected_country = st.selectbox("🌍 Select Country", countries)
            else:
                selected_country = "All"
        
        with col2:
            if selected_country != "All" and "country_name" in df.columns:
                filtered_cities = df[df['country_name'] == selected_country]['city_name'].dropna().unique()
                cities = ["All"] + sorted(filtered_cities.tolist())
            elif "city_name" in df.columns:
                cities = ["All"] + sorted(df['city_name'].dropna().unique().tolist())
            else:
                cities = ["All"]
            selected_city = st.selectbox("🏙️ Select City", cities)
        
        with col3:
            if "aqi_category" in df.columns:
                categories = ["All"] + sorted(df['aqi_category'].dropna().unique().tolist())
                selected_category = st.selectbox("📊 AQI Category", categories)
            else:
                selected_category = "All"
        
        # Apply filters
        filtered_df = df.copy()
        if selected_country != "All" and "country_name" in df.columns:
            filtered_df = filtered_df[filtered_df['country_name'] == selected_country]
        if selected_city != "All" and "city_name" in df.columns:
            filtered_df = filtered_df[filtered_df['city_name'] == selected_city]
        if selected_category != "All" and "aqi_category" in df.columns:
            filtered_df = filtered_df[filtered_df['aqi_category'] == selected_category]
        
        st.markdown("---")
        
        # --- KEY METRICS DASHBOARD ---
        st.markdown("### 📈 Key Metrics")
        metric1, metric2, metric3, metric4 = st.columns(4)
        
        with metric1:
            st.metric("🌍 Total Cities", len(filtered_df))
        with metric2:
            if "AQI" in filtered_df.columns:
                st.metric("📊 Avg AQI", f"{filtered_df['AQI'].mean():.1f}")
        with metric3:
            if "AQI" in filtered_df.columns:
                st.metric("🔴 Max AQI", f"{filtered_df['AQI'].max():.0f}")
        with metric4:
            if "AQI" in filtered_df.columns:
                st.metric("🟢 Min AQI", f"{filtered_df['AQI'].min():.0f}")
        
        st.markdown("---")
        
        # --- PREMIUM INTERACTIVE MAP ---
        st.markdown("### 🗺️ Interactive Global AQI Visualization")
        
        # Map visualization toggle
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            map_view = st.radio("🗺️ View Type", ["Bubble Map", "Heat Map", "Data Table"], horizontal=True)
        
        if {"city_name", "AQI", "lat", "lon"}.issubset(filtered_df.columns):
            # Filter to only cities with valid coordinates
            map_df = filtered_df.dropna(subset=['lat', 'lon', 'AQI']).copy()
            
            if len(map_df) == 0:
                st.warning("⚠️ No cities with valid coordinates found in the filtered data.")
            else:
                # Show info about mapped cities
                total_cities = len(filtered_df)
                mapped_cities = len(map_df)
                if mapped_cities < total_cities:
                    st.info(f"📍 Showing {mapped_cities} out of {total_cities} cities on map (cities with known coordinates)")
            
                if map_view == "Data Table":
                    # Clean table view
                    st.markdown("#### 📋 City Air Quality Data")
                    display_df = map_df[['city_name', 'country_name', 'AQI']].copy()
                    display_df['AQI Category'] = display_df['AQI'].apply(lambda x: get_aqi_category(x)[0])
                    display_df = display_df.sort_values('AQI', ascending=False)
                    
                    st.dataframe(
                        display_df.style.background_gradient(cmap='RdYlGn_r', subset=['AQI']),
                        use_container_width=True,
                        height=600
                    )
                    
                elif map_view == "Bubble Map":
                    # Cleaner bubble map with reduced size and better spacing
                    fig_map = px.scatter_mapbox(
                        map_df,
                        lat="lat",
                        lon="lon",
                        color="AQI",
                        size="AQI",
                        hover_name="city_name",
                        hover_data={
                            "country_name": True,
                            "AQI": ":.1f",
                            "lat": False,
                            "lon": False
                        },
                        color_continuous_scale=[
                            [0.0, "#00e400"],
                            [0.2, "#ffff00"],
                            [0.4, "#ff7e00"],
                            [0.6, "#ff0000"],
                            [0.8, "#8f3f97"],
                            [1.0, "#7e0023"]
                        ],
                        size_max=15,
                        zoom=1.5,
                        opacity=0.7
                    )
                    
                    fig_map.update_layout(
                        mapbox_style="carto-positron",
                        height=650,
                        margin={"r": 0, "t": 0, "l": 0, "b": 0},
                        font=dict(family="Poppins, sans-serif", size=12),
                        coloraxis_colorbar=dict(
                            title="AQI Level",
                            thickness=15,
                            len=0.6,
                            x=1.02
                        )
                    )
                    
                    st.plotly_chart(fig_map, use_container_width=True)
                
                else:  # Heat Map
                    # Density heatmap - cleanest visualization!
                    fig_map = px.density_mapbox(
                        map_df,
                        lat="lat",
                        lon="lon",
                        z="AQI",
                        hover_name="city_name",
                        hover_data={"country_name": True, "AQI": ":.1f"},
                        radius=20,
                        zoom=1.5,
                        color_continuous_scale=[
                            [0.0, "#00e400"],
                            [0.2, "#ffff00"],
                            [0.4, "#ff7e00"],
                            [0.6, "#ff0000"],
                            [0.8, "#8f3f97"],
                            [1.0, "#7e0023"]
                        ],
                        opacity=0.8
                    )
                    
                    fig_map.update_layout(
                        mapbox_style="carto-positron",
                        height=650,
                        margin={"r": 0, "t": 0, "l": 0, "b": 0},
                        font=dict(family="Poppins, sans-serif", size=12),
                        coloraxis_colorbar=dict(
                            title="AQI Density",
                            thickness=15,
                            len=0.6,
                            x=1.02
                        )
                    )
                    
                    st.plotly_chart(fig_map, use_container_width=True)
                
                # Map style selector (only for map views)
                if map_view != "Data Table":
                    with col2:
                        map_style = st.selectbox(
                            "🎨 Map Background",
                            ["Light", "Dark", "Satellite"]
                        )
                        
                        style_map = {
                            "Light": "carto-positron",
                            "Dark": "carto-darkmatter",
                            "Satellite": "stamen-terrain"
                        }
                        
                        fig_map.update_layout(mapbox_style=style_map[map_style])
        else:
            st.warning("⚠️ Map visualization requires city_name, AQI, latitude, and longitude columns.")
        
        st.markdown("---")
        
        # --- AQI DISTRIBUTION CHARTS ---
        col1, col2 = st.columns(2)
        
        with col1:
            if "aqi_category" in filtered_df.columns:
                st.markdown("### 🥧 AQI Category Distribution")
                category_counts = filtered_df['aqi_category'].value_counts()
                
                fig_donut = go.Figure(data=[go.Pie(
                    labels=category_counts.index,
                    values=category_counts.values,
                    hole=.4,
                    marker=dict(colors=px.colors.sequential.Plasma),
                    textinfo='label+percent',
                    textfont=dict(size=14)
                )])
                
                fig_donut.update_layout(
                    height=400,
                    showlegend=True,
                    font=dict(family="Poppins, sans-serif"),
                    annotations=[dict(text='AQI', x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                
                st.plotly_chart(fig_donut, use_container_width=True)
        
        with col2:
            if "AQI" in filtered_df.columns:
                st.markdown("### 📊 AQI Distribution Histogram")
                fig_hist = px.histogram(
                    filtered_df,
                    x="AQI",
                    nbins=30,
                    color_discrete_sequence=['#667eea'],
                    title="Frequency Distribution of AQI Values"
                )
                
                fig_hist.update_layout(
                    height=400,
                    font=dict(family="Poppins, sans-serif"),
                    showlegend=False,
                    xaxis_title="AQI Value",
                    yaxis_title="Frequency"
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("---")
        
        # --- TOP AND BOTTOM CITIES ---
        st.markdown("### 🏆 Top & Bottom Performing Cities")
        
        if "city_name" in filtered_df.columns and "AQI" in filtered_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔥 Most Polluted Cities (Top 10)")
                top_cities = filtered_df.nlargest(10, "AQI")[["city_name", "country_name", "AQI"]]
                
                # Create bar chart for top cities
                fig_top = px.bar(
                    top_cities,
                    y="city_name",
                    x="AQI",
                    color="AQI",
                    color_continuous_scale="Reds",
                    orientation='h',
                    text="AQI"
                )
                fig_top.update_layout(height=400, showlegend=False, yaxis={'categoryorder':'total ascending'})
                fig_top.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                st.plotly_chart(fig_top, use_container_width=True)
            
            with col2:
                st.markdown("#### 💨 Cleanest Air Cities (Top 10)")
                low_cities = filtered_df.nsmallest(10, "AQI")[["city_name", "country_name", "AQI"]]
                
                # Create bar chart for bottom cities
                fig_low = px.bar(
                    low_cities,
                    y="city_name",
                    x="AQI",
                    color="AQI",
                    color_continuous_scale="Greens_r",
                    orientation='h',
                    text="AQI"
                )
                fig_low.update_layout(height=400, showlegend=False, yaxis={'categoryorder':'total ascending'})
                fig_low.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                st.plotly_chart(fig_low, use_container_width=True)
        
        st.markdown("---")
        
        # --- CORRELATION HEATMAP ---
        st.markdown("### 🔥 Pollutant Correlation Matrix")
        numeric_df = filtered_df.select_dtypes(include=[np.number])
        
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            fig_corr = px.imshow(
                numeric_df.corr(),
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Correlation Between Air Quality Parameters"
            )
            fig_corr.update_layout(height=500, font=dict(family="Poppins, sans-serif"))
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # --- RAW DATA PREVIEW ---
        with st.expander("📄 View Raw Data"):
            st.dataframe(filtered_df, use_container_width=True, height=400)

# --------------------------------------------------------
# PAGE 2: ENHANCED AQI PREDICTION
# --------------------------------------------------------
elif page == "🤖 AQI Prediction":
    st.markdown("## 🤖 AI-Powered AQI Prediction Engine")
    
    uploaded_file = st.file_uploader("📁 Upload dataset for model training (CSV format):", type=["csv"])
    
    if uploaded_file:
        df = load_csv(uploaded_file)
        
        if "AQI" not in df.columns:
            st.error("❌ Dataset must contain a column named 'AQI' or 'aqi_value'.")
        else:
            with st.spinner("🔄 Training advanced LightGBM model..."):
                X = df.select_dtypes(include=[np.number]).drop("AQI", axis=1)
                y = df["AQI"]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train model with optimized parameters
                model = lgb.LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=7,
                    random_state=42,
                    verbose=-1
                )
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                accuracy = max(0, (1 - rmse / np.mean(y_test)) * 100)
            
            st.success("✅ Model trained successfully!")
            
            # --- MODEL PERFORMANCE DASHBOARD ---
            st.markdown("### 📊 Model Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 R² Score", f"{r2:.4f}", help="Coefficient of determination")
            col2.metric("📉 RMSE", f"{rmse:.2f}", help="Root Mean Square Error")
            col3.metric("📏 MAE", f"{mae:.2f}", help="Mean Absolute Error")
            col4.metric("✅ Accuracy", f"{accuracy:.2f}%", help="Model accuracy percentage")
            
            # --- PREDICTION VS ACTUAL PLOT ---
            st.markdown("### 📈 Prediction Performance Visualization")
            
            comparison_df = pd.DataFrame({
                'Actual AQI': y_test[:100],
                'Predicted AQI': y_pred[:100]
            }).reset_index(drop=True)
            
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Scatter(
                y=comparison_df['Actual AQI'],
                mode='lines+markers',
                name='Actual AQI',
                line=dict(color='#667eea', width=2),
                marker=dict(size=6)
            ))
            fig_comparison.add_trace(go.Scatter(
                y=comparison_df['Predicted AQI'],
                mode='lines+markers',
                name='Predicted AQI',
                line=dict(color='#f093fb', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            fig_comparison.update_layout(
                title="Actual vs Predicted AQI Values (First 100 Test Samples)",
                xaxis_title="Sample Index",
                yaxis_title="AQI Value",
                height=400,
                font=dict(family="Poppins, sans-serif"),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            st.markdown("---")
            
            # --- MANUAL PREDICTION INTERFACE ---
            st.markdown("### 🎯 Make Custom AQI Prediction")
            st.markdown("Enter pollutant values to predict the Air Quality Index")
            
            with st.form("aqi_prediction_form", clear_on_submit=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    pm25 = st.slider("🌫️ PM2.5 AQI Value", 0.0, 500.0, 60.0, 1.0,
                                    help="Fine particulate matter (≤2.5 micrometers)")
                    no2 = st.slider("💨 NO₂ AQI Value", 0.0, 500.0, 40.0, 1.0,
                                   help="Nitrogen Dioxide concentration")
                
                with col2:
                    co = st.slider("☁️ CO AQI Value", 0.0, 50.0, 1.0, 0.1,
                                  help="Carbon Monoxide concentration")
                    ozone = st.slider("🌡️ Ozone AQI Value", 0.0, 300.0, 20.0, 1.0,
                                     help="Ground-level Ozone concentration")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    submit_button = st.form_submit_button("🔮 Predict AQI", use_container_width=True)
            
            if submit_button:
                # Prepare input
                input_data = pd.DataFrame([[pm25, no2, co, ozone]],
                                        columns=["pm2.5_aqi_value", "no2_aqi_value", 
                                               "co_aqi_value", "ozone_aqi_value"])
                
                # Add missing columns with zeros
                missing_cols = [col for col in X.columns if col not in input_data.columns]
                for col in missing_cols:
                    input_data[col] = 0
                
                input_data = input_data[X.columns]
                
                # Make prediction
                predicted_aqi = model.predict(input_data)[0]
                category, color = get_aqi_category(predicted_aqi)
                
                # Display result with enhanced styling
                st.markdown("---")
                st.markdown("### 🎊 Prediction Result")
                
                result_col1, result_col2 = st.columns([1, 1])
                
                with result_col1:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {color}22 0%, {color}44 100%); 
                                padding: 2rem; border-radius: 15px; text-align: center;
                                border: 3px solid {color};'>
                        <h2 style='color: {color}; margin: 0;'>AQI: {predicted_aqi:.1f}</h2>
                        <h3 style='color: {color}; margin-top: 0.5rem;'>{category}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with result_col2:
                    # AQI Gauge Chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=predicted_aqi,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [None, 500]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 50], 'color': "rgba(0, 228, 0, 0.2)"},
                                {'range': [50, 100], 'color': "rgba(255, 255, 0, 0.2)"},
                                {'range': [100, 150], 'color': "rgba(255, 126, 0, 0.2)"},
                                {'range': [150, 200], 'color': "rgba(255, 0, 0, 0.2)"},
                                {'range': [200, 300], 'color': "rgba(143, 63, 151, 0.2)"},
                                {'range': [300, 500], 'color': "rgba(126, 0, 35, 0.2)"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 150
                            }
                        }
                    ))
                    
                    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Health recommendations
                st.markdown("### 💡 Health Recommendations")
                
                if predicted_aqi <= 50:
                    st.success("🟢 **Excellent!** Air quality is satisfactory. Perfect for outdoor activities!")
                elif predicted_aqi <= 100:
                    st.info("🟡 **Moderate** Air quality is acceptable. Sensitive individuals should consider limiting prolonged outdoor activities.")
                elif predicted_aqi <= 150:
                    st.warning("🟠 **Caution!** Sensitive groups may experience health effects. General public is less likely to be affected.")
                elif predicted_aqi <= 200:
                    st.error("🔴 **Unhealthy!** Everyone may begin to experience health effects. Sensitive groups should avoid outdoor activities.")
                elif predicted_aqi <= 300:
                    st.error("🟣 **Very Unhealthy!** Health alert. Everyone should avoid outdoor exertion.")
                else:
                    st.error("⚫ **Hazardous!** Emergency conditions. Everyone should remain indoors.")

# --------------------------------------------------------
# PAGE 3: ENHANCED FEATURE INSIGHTS
# --------------------------------------------------------
elif page == "📈 Feature Insights":
    st.markdown("## 📊 Advanced Feature Analysis & Insights")
    
    uploaded_file = st.file_uploader("📁 Upload dataset for feature analysis (CSV format):", type=["csv"])
    
    if uploaded_file:
        df = load_csv(uploaded_file)
        
        if "AQI" not in df.columns:
            st.error("❌ Dataset must contain 'AQI' column.")
        else:
            X = df.select_dtypes(include=[np.number]).drop("AQI", axis=1)
            y = df["AQI"]
            
            with st.spinner("🔄 Analyzing feature importance..."):
                model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
                model.fit(X, y)
                
                importance_df = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=False)
            
            # --- FEATURE IMPORTANCE VISUALIZATION ---
            st.markdown("### 🌟 LightGBM Feature Importance Analysis")
            
            fig_importance = px.bar(
                importance_df.head(15),
                x="Importance",
                y="Feature",
                orientation='h',
                color="Importance",
                color_continuous_scale="Viridis",
                title="Top 15 Most Important Features for AQI Prediction"
            )
            
            fig_importance.update_layout(
                height=500,
                font=dict(family="Poppins, sans-serif"),
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # --- FEATURE IMPORTANCE TABLE ---
            with st.expander("📋 View Complete Feature Importance Table"):
                st.dataframe(
                    importance_df.style.background_gradient(cmap="Blues", subset=['Importance']),
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # --- AQI CATEGORIES REFERENCE ---
            st.markdown("### 🌍 Global AQI Standards & Categories")
            
            aqi_info = pd.DataFrame({
                "AQI Range": ["0–50", "51–100", "101–150", "151–200", "201–300", "301+"],
                "Category": ["Good", "Moderate", "Unhealthy (Sensitive Groups)", 
                           "Unhealthy", "Very Unhealthy", "Hazardous"],
                "Health Impact": [
                    "Air quality is satisfactory",
                    "Acceptable for most, sensitive may feel effects",
                    "Sensitive groups experience health effects",
                    "Everyone may experience health effects",
                    "Health alert - significant health effects",
                    "Emergency conditions - serious health risks"
                ],
                "Color Code": ["🟢 Green", "🟡 Yellow", "🟠 Orange", "🔴 Red", "🟣 Purple", "⚫ Maroon"]
            })
            
            st.dataframe(aqi_info, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # --- POLLUTANT DISTRIBUTION ---
            st.markdown("### 📊 Pollutant Distribution Analysis")
            
            pollutant_cols = [col for col in X.columns if any(p in col.lower() 
                            for p in ['pm2.5', 'pm10', 'no2', 'co', 'ozone', 'so2'])]
            
            if pollutant_cols:
                fig_box = px.box(
                    df[pollutant_cols],
                    title="Distribution of Major Pollutants",
                    labels={"variable": "Pollutant", "value": "AQI Value"}
                )
                
                fig_box.update_layout(
                    height=400,
                    font=dict(family="Poppins, sans-serif"),
                    showlegend=True
                )
                
                st.plotly_chart(fig_box, use_container_width=True)

# --------------------------------------------------------
# PAGE 4: GLOBAL COMPARISON
# --------------------------------------------------------
elif page == "🌐 Global Comparison":
    st.markdown("## 🌐 Global AQI Comparison Dashboard")
    
    uploaded_file = st.file_uploader("📁 Upload dataset for comparison (CSV format):", type=["csv"])
    
    if uploaded_file:
        df = load_csv(uploaded_file)
        df = add_coordinates(df)
        
        if "country_name" in df.columns and "AQI" in df.columns:
            # --- COUNTRY COMPARISON ---
            st.markdown("### 🌍 Country-wise AQI Comparison")
            
            country_aqi = df.groupby('country_name')['AQI'].agg(['mean', 'min', 'max', 'count']).reset_index()
            country_aqi.columns = ['Country', 'Average AQI', 'Min AQI', 'Max AQI', 'Cities']
            country_aqi = country_aqi.sort_values('Average AQI', ascending=False)
            
            # Interactive bar chart
            fig_country = px.bar(
                country_aqi.head(20),
                x='Average AQI',
                y='Country',
                orientation='h',
                color='Average AQI',
                color_continuous_scale='RdYlGn_r',
                title="Top 20 Countries by Average AQI",
                hover_data=['Min AQI', 'Max AQI', 'Cities']
            )
            
            fig_country.update_layout(
                height=600,
                font=dict(family="Poppins, sans-serif"),
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig_country, use_container_width=True)
            
            st.markdown("---")
            
            # --- REGIONAL ANALYSIS ---
            st.markdown("### 📍 Regional Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Best performing countries
                st.markdown("#### 🥇 Best Air Quality Countries")
                best_countries = country_aqi.nsmallest(10, 'Average AQI')[['Country', 'Average AQI']]
                st.dataframe(
                    best_countries.style.background_gradient(cmap="Greens_r", subset=['Average AQI']),
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                # Worst performing countries
                st.markdown("#### 🔴 Countries Needing Attention")
                worst_countries = country_aqi.nlargest(10, 'Average AQI')[['Country', 'Average AQI']]
                st.dataframe(
                    worst_countries.style.background_gradient(cmap="Reds", subset=['Average AQI']),
                    use_container_width=True,
                    hide_index=True
                )
            
            st.markdown("---")
            
            # --- DETAILED STATISTICS ---
            with st.expander("📊 View Complete Country Statistics"):
                st.dataframe(
                    country_aqi.style.background_gradient(cmap="RdYlGn_r", subset=['Average AQI']),
                    use_container_width=True,
                    hide_index=True
                )

# --------------------------------------------------------
# FOOTER
# --------------------------------------------------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); 
                border-radius: 10px; backdrop-filter: blur(5px);'>
        <h4 style='color: white;'>🌍 Global AQI Intelligence Hub</h4>
        <p style='color: white;'>Developed with ❤️ by <b>Dharanisha</b></p>
        <p style='color: white;'>Powered by LightGBM | Streamlit | Plotly</p>
        <p style='color: white; font-size: 12px;'>Data visualization and predictive analytics for a cleaner tomorrow 🌱</p>
    </div>
""", unsafe_allow_html=True)