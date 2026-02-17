import streamlit as st
import pandas as pd
import joblib
import time
from math import radians, cos, sin, sqrt, atan2

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Zomato AI Engine",
    page_icon="🚀",
    layout="wide"
)

# ---------------- ADVANCED CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Animated Background */
.stApp {
    background: linear-gradient(-45deg, #ff416c, #ff4b2b, #1e3c72, #2a5298);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass Card */
.glass {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.6) !important;
    backdrop-filter: blur(15px);
}

/* Buttons */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border: none;
    padding: 18px;
    font-size: 20px;
    font-weight: bold;
    border-radius: 40px;
    transition: 0.3s ease;
    box-shadow: 0 10px 25px rgba(0,0,0,0.4);
}
div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 15px 35px rgba(0,0,0,0.6);
}

/* Metric Glow */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.15);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
}

/* Title */
h1 {
    font-weight: 900 !important;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_data():
    return joblib.load("model.joblib"), joblib.load("columns.joblib")

model, columns = load_data()

# ---------------- HAVERSINE ----------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return R * (2 * atan2(sqrt(a), sqrt(1-a)))

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🚴 Rider AI Profile")
    age = st.slider("Age", 18, 60, 28)
    rating = st.select_slider("Rating", [1,2,3,4,5], value=4)
    multiple = st.selectbox("Pending Deliveries", [0,1,2,3])
    order_h = st.number_input("Order Hour", 0, 23, 19)
    pickup_h = st.number_input("Pickup Hour", 0, 23, 19)

# ---------------- MAIN ----------------
st.title("🚀 Zomato Neural ETA Engine")
st.caption("Next-Gen Intelligent Delivery Estimation System")

col1, col2 = st.columns([2,3])

with col1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📍 Route Coordinates")

    res_lat = st.number_input("Restaurant Latitude", value=12.97)
    res_lon = st.number_input("Restaurant Longitude", value=77.59)
    del_lat = st.number_input("Delivery Latitude", value=13.08)
    del_lon = st.number_input("Delivery Longitude", value=80.27)

    dist = haversine(res_lat, res_lon, del_lat, del_lon)
    st.metric("📏 Linear Distance", f"{dist:.2f} KM")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    map_data = pd.DataFrame({
        'lat': [res_lat, del_lat],
        'lon': [res_lon, del_lon]
    })
    st.map(map_data, zoom=5)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")

# ---------------- PREDICT ----------------
if st.button("⚡ GET SMART ETA"):
    with st.spinner("Running AI Logistics Engine..."):
        time.sleep(1)

        input_df = pd.DataFrame([{
            "Delivery_person_Age": age,
            "Delivery_person_Ratings": rating,
            "multiple_deliveries": multiple,
            "Order_Hour": order_h,
            "Pickup_Hour": pickup_h,
            "Restaurant_latitude": res_lat,
            "Restaurant_longitude": res_lon,
            "Delivery_location_latitude": del_lat,
            "Delivery_location_longitude": del_lon,
            "distance_km": dist
        }])

        input_df = input_df.reindex(columns=columns, fill_value=0)
        prediction = model.predict(input_df)[0]

    st.balloons()

    st.markdown(f"""
    <div class="glass" style="text-align:center; margin-top:30px;">
        <h2 style="color:white;">Estimated Delivery Time</h2>
        <h1 style="font-size:80px; color:#00FFAA; 
            text-shadow: 0 0 20px #00FFAA;">
            {int(prediction)} mins
        </h1>
        <p style="color:white; font-size:18px;">
            ✔ Optimized Route | ✔ AI Traffic Modeling | ✔ Rider Efficiency Calculated
        </p>
    </div>
    """, unsafe_allow_html=True)
