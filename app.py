# ==============================================================================
# 🌾 KRISHI MITRA — MULTILINGUAL DIGITAL FARMING COMPANION STREAMLIT APPLICATION
# ==============================================================================
# Description: A full-stack Streamlit web application designed for Indian farmers.
# Features:
#   1. Multilingual Interface (English, Hindi, Marathi)
#   2. Real-time Weather Forecasting & 15-Day Historical Data via Open-Meteo API
#   3. Sub-Region / Taluka Selector for Pune District
#   4. Smart AI Predictions (XGBoost ML for Irrigation Need & Fungal Disease Risk)
#   5. MPKV Rahuri Standard Fertilizer Dosage Calculator (Acres, Hectares, Guntha)
#   6. Maha Seva Kendra (Aaple Sarkar / Abhishek Khedekar) Scheme Desk & Document Guidance
#   7. Offline Multilingual AI Assistant (Rule-based Chatbot)
#   8. Fertilizer & Crop Guides, Pest Management, Govt. Schemes, & Developer Support
# ==============================================================================

# ------------------------------------------------------------------------------
# SECTION 1: LIBRARIES & MODULE IMPORTS
# ------------------------------------------------------------------------------
import streamlit as st       # Core web framework for building interactive user interfaces
import pandas as pd          # Data analysis & manipulation library for DataFrames and tables
import numpy as np           # Numerical processing library for array calculations and random sampling
import os                    # Operating system utilities for checking file paths (e.g., image assets)
import requests              # HTTP library to make REST API requests to Open-Meteo weather servers
import datetime              # Date and time utilities to compute historical and forecast date ranges
from xgboost import XGBClassifier  # XGBoost Machine Learning algorithm for gradient boosted decision trees

# ------------------------------------------------------------------------------
# SECTION 2: STREAMLIT PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Krishi Mitra | Digital Farming Companion",  # Title displayed on browser tab
    page_icon="🌾",                                        # Favicon icon shown in browser tab
    layout="wide",                                         # Expand app layout across full screen width
    initial_sidebar_state="expanded"                       # Keep navigation sidebar expanded by default
)

# ------------------------------------------------------------------------------
# SECTION 3: CUSTOM CSS STYLING & BRAND THEMING
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, .brand-title {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background-color: #f8fafc;
    }

    .hero-card {
        background: linear-gradient(135deg, #15803d 0%, #166534 100%);
        color: white;
        padding: 2.2rem;
        border-radius: 1.25rem;
        box-shadow: 0 10px 25px -5px rgba(21, 128, 61, 0.3);
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #f0fdf4;
        opacity: 0.95;
        max-width: 800px;
        line-height: 1.6;
    }

    .fert-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 1rem;
        padding: 1.4rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    .fert-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(22, 101, 52, 0.15);
        border-color: #bbf7d0;
    }

    .fert-badge {
        display: inline-block;
        background-color: #dcfce7;
        color: #166534;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .fert-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #14532d;
        margin-bottom: 0.5rem;
    }

    .fert-desc {
        font-size: 0.92rem;
        color: #475569;
        line-height: 1.5;
        margin-bottom: 1rem;
    }

    .fert-note {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 0.65rem 0.85rem;
        border-radius: 0.4rem;
        font-size: 0.84rem;
        color: #166534;
        font-weight: 500;
    }

    .stat-box {
        background: white;
        border-radius: 1rem;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #15803d;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
    }

    .seva-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        border-left: 6px solid #f59e0b;
    }

    .seva-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 0.75rem;
        padding: 1.2rem;
        height: 100%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.04);
    }

    .footer-container {
        background: #14532d;
        color: #f0fdf4;
        padding: 2.5rem 2rem 1.5rem 2rem;
        border-radius: 1.25rem 1.25rem 0 0;
        margin-top: 3rem;
    }
    .footer-heading {
        color: #ffffff;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SECTION 4: GEOGRAPHICAL REGISTRY FOR PUNE TALUKAS / SUB-REGIONS
# ------------------------------------------------------------------------------
PUNE_REGIONS = {
    "Pune City / पुणे शहर": {"lat": 18.5204, "lon": 73.8567},
    "Baramati / बारामती": {"lat": 18.1517, "lon": 74.5770},
    "Haveli (Hadapsar/Pimpri) / हवेली": {"lat": 18.4984, "lon": 73.9298},
    "Junnar / जुन्नर": {"lat": 19.2084, "lon": 73.8763},
    "Khed (Rajgurunagar) / खेड": {"lat": 18.8550, "lon": 73.8860},
    "Maval (Lonavala/Talegaon) / मावळ": {"lat": 18.7557, "lon": 73.4091},
    "Mulshi (Paud/Lavasa) / मुळशी": {"lat": 18.5300, "lon": 73.6150},
    "Shirur / शिरूर": {"lat": 18.8284, "lon": 74.3773},
    "Indapur / इंदापूर": {"lat": 18.1154, "lon": 74.8080},
    "Daund / दौंड": {"lat": 18.4650, "lon": 74.5828},
    "Bhor / भोर": {"lat": 18.1478, "lon": 73.8447},
    "Purandar (Saswad) / पुरंदर": {"lat": 18.3444, "lon": 74.0308},
    "Ambegaon (Manchar) / आंबेगाव": {"lat": 19.0067, "lon": 73.9408},
    "Velhe (Torna/Rajgad) / वेल्हे": {"lat": 18.2975, "lon": 73.6360}
}

# ------------------------------------------------------------------------------
# TASK 3: SOIL TYPE MAPPING REGISTRY FOR SMART AI
# ------------------------------------------------------------------------------
# Automatically maps farmer soil type selection to estimated baseline agronomic parameters for XGBoost model inference.
SOIL_TYPES_MAPPING = {
    "Black Cotton Soil / भारी काळी माती": {"soil_N": 80.0, "soil_P": 40.0, "soil_K": 60.0, "soil_pH": 7.8, "desc": "High nutrient & moisture retention capacity, ideal for Cotton, Sugarcane & Gram."},
    "Medium Red / Loamy Soil / मध्यम तांबडी - दुमट माती": {"soil_N": 60.0, "soil_P": 30.0, "soil_K": 45.0, "soil_pH": 6.8, "desc": "Well-drained fertile loam, highly suitable for Wheat, Soybean, Onion & Vegetables."},
    "Light Murum / Shallow Soil / हलकी - मुरुमाड माती": {"soil_N": 40.0, "soil_P": 20.0, "soil_K": 30.0, "soil_pH": 6.2, "desc": "Shallow gravelly soil requiring split fertilizer dosing and frequent light irrigation."},
    "Alluvial / Riverbed Soil / गाळाची माती": {"soil_N": 70.0, "soil_P": 35.0, "soil_K": 50.0, "soil_pH": 7.2, "desc": "Rich riverbed deposits, ideal for Paddy (Rice), Sugarcane & cash crops."}
}

# ------------------------------------------------------------------------------
# SECTION 5: OPEN-METEO WEATHER FETCHING SERVICE (WITH IN-MEMORY CACHING)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_pune_weather(lat=18.5204, lon=73.8567):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_days=15&forecast_days=7&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean,soil_moisture_0_to_7cm_mean&timezone=Asia%2FKolkata"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            daily = data.get('daily', {})
            df = pd.DataFrame({
                'date': pd.to_datetime(daily.get('time', [])),
                'temp_max': daily.get('temperature_2m_max', []),
                'temp_min': daily.get('temperature_2m_min', []),
                'temp_mean': daily.get('temperature_2m_mean', []),
                'precipitation': daily.get('precipitation_sum', []),
                'soil_moisture': daily.get('soil_moisture_0_to_7cm_mean', [])
            })
            today = pd.to_datetime(datetime.date.today())
            hist_df = df[df['date'] < today]
            fore_df = df[df['date'] >= today]
            
            temp_past15_avg = float(hist_df['temp_mean'].mean()) if not hist_df.empty else 26.5
            rain_past15_sum = float(hist_df['precipitation'].sum()) if not hist_df.empty else 12.0
            temp_forecast_avg = float(fore_df['temp_mean'].mean()) if not fore_df.empty else 28.0
            rain_forecast_sum = float(fore_df['precipitation'].sum()) if not fore_df.empty else 5.0
            current_soil_moisture = float(df['soil_moisture'].iloc[-1] * 100) if not df.empty and df['soil_moisture'].iloc[-1] is not None else 45.0
            
            return {
                'df': df,
                'hist_df': hist_df,
                'fore_df': fore_df,
                'temp_past15_avg': temp_past15_avg,
                'rain_past15_sum': rain_past15_sum,
                'temp_forecast_avg': temp_forecast_avg,
                'rain_forecast_sum': rain_forecast_sum,
                'soil_moisture': current_soil_moisture,
                'success': True
            }
    except Exception as e:
        pass

    dates = [datetime.date.today() - datetime.timedelta(days=i) for i in range(15, 0, -1)] + [datetime.date.today() + datetime.timedelta(days=i) for i in range(7)]
    fallback_df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'temp_mean': [26.0 + (i%3)*0.5 for i in range(22)],
        'precipitation': [2.0 if i%4==0 else 0.0 for i in range(22)],
        'soil_moisture': [0.45 for _ in range(22)]
    })
    return {
        'df': fallback_df,
        'hist_df': fallback_df.iloc[:15],
        'fore_df': fallback_df.iloc[15:],
        'temp_past15_avg': 26.5,
        'rain_past15_sum': 12.0,
        'temp_forecast_avg': 28.0,
        'rain_forecast_sum': 5.0,
        'soil_moisture': 45.0,
        'success': False
    }

# ------------------------------------------------------------------------------
# SECTION 6: SYNTHETIC AGRONOMIC DATA GENERATION FOR MACHINE LEARNING
# ------------------------------------------------------------------------------
@st.cache_data
def generate_synthetic_agronomic_data(num_samples=2000):
    np.random.seed(42)
    
    temp_past15_avg = np.random.uniform(15.0, 40.0, num_samples)
    rain_past15_sum = np.random.uniform(0.0, 150.0, num_samples)
    temp_forecast_avg = np.random.uniform(15.0, 42.0, num_samples)
    rain_forecast_sum = np.random.uniform(0.0, 100.0, num_samples)
    soil_moisture = np.random.uniform(10.0, 90.0, num_samples)
    soil_N = np.random.uniform(10.0, 140.0, num_samples)
    soil_P = np.random.uniform(5.0, 80.0, num_samples)
    soil_K = np.random.uniform(10.0, 120.0, num_samples)
    soil_pH = np.random.uniform(5.5, 8.5, num_samples)
    
    df = pd.DataFrame({
        'temp_past15_avg': temp_past15_avg,
        'rain_past15_sum': rain_past15_sum,
        'temp_forecast_avg': temp_forecast_avg,
        'rain_forecast_sum': rain_forecast_sum,
        'soil_moisture': soil_moisture,
        'soil_N': soil_N,
        'soil_P': soil_P,
        'soil_K': soil_K,
        'soil_pH': soil_pH
    })
    
    # Irrigation Stress Index (ISI) Formula
    isi = (0.35 * df['temp_forecast_avg'] + 
           0.30 * (100.0 - df['soil_moisture']) - 
           0.25 * df['rain_past15_sum'] - 
           0.10 * df['rain_forecast_sum'])
    
    df['Irrigation_Needed'] = np.where(isi > 25.0, 2, np.where(isi >= 15.0, 1, 0))
    
    # Disease Risk Index (DRI) Condition Rules
    cond_disease = (
        (df['rain_past15_sum'] > 30.0) & 
        (df['temp_past15_avg'] >= 20.0) & 
        (df['temp_past15_avg'] <= 30.0) & 
        (df['soil_moisture'] > 60.0)
    )
    df['Disease_Risk'] = np.where(cond_disease, 1, 0)
    
    return df

# ------------------------------------------------------------------------------
# SECTION 7: XGBOOST ML MODEL TRAINING & CACHING
# ------------------------------------------------------------------------------
@st.cache_resource
def train_xgboost_models():
    df = generate_synthetic_agronomic_data(2000)
    features = ['temp_past15_avg', 'rain_past15_sum', 'temp_forecast_avg', 'rain_forecast_sum', 'soil_moisture', 'soil_N', 'soil_P', 'soil_K', 'soil_pH']
    X = df[features]
    
    model_irrigation = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
    model_irrigation.fit(X, df['Irrigation_Needed'])
    
    model_disease = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')
    model_disease.fit(X, df['Disease_Risk'])
    
    return model_irrigation, model_disease

# ------------------------------------------------------------------------------
# SECTION 8: CENTRAL MULTILINGUAL TRANSLATIONS DICTIONARY
# ------------------------------------------------------------------------------
TRANSLATIONS = {
    "English": {
        "app_title": "Krishi Mitra",
        "app_subtitle": "Digital Farming Companion",
        "nav_home": "🏠 Home",
        "nav_fert_guide": "🧪 Fertilizer Guide",
        "nav_calculator": "🧮 MPKV Fertilizer Calculator",
        "nav_crop_guide": "🌾 Crop Guide",
        "nav_pest": "🐛 Pest & Disease",
        "nav_schemes": "🏛️ Govt. Schemes & Maha Seva Kendra",
        "nav_contact": "📞 Contact & Support Desk",
        "nav_weather": "🌤️ Weather Forecast",
        "nav_ai_pred": "🤖 Smart AI Predictions",
        "nav_chat": "💬 Krishi AI Assistant",
        "helpline_title": "📍 Maha Seva Kendra Desk",
        "phone": "Operator Phone",
        "location": "Center Location",
        "email": "Developer Email",
        
        # Home
        "home_hero_title": "🌾 Welcome to Krishi Mitra",
        "home_hero_sub": "Your simple, smart digital farming companion helping Indian farmers grow healthier crops, optimize MPKV fertilizer dosage, access Maha Seva Kendra services, and increase yield.",
        "stat_crops": "Crops Covered",
        "stat_dosing": "MPKV Rahuri Dosing",
        "stat_schemes": "Maha Seva Desk",
        "stat_support": "Developer Support",
        "season_highlights": "💡 Seasonal Farming Quick Highlights",
        "kharif_title": "🌧️ Kharif Season (Monsoon)",
        "kharif_desc": "- Major Crops: Rice, Cotton, Soybean, Maize.\n- Focus: Basal DAP & Zinc application early.",
        "rabi_title": "🌾 Rabi Season (Winter)",
        "rabi_desc": "- Major Crops: Wheat, Gram, Mustard, Onion.\n- Focus: Balanced NPK & split Urea top dressing.",
        "zaid_title": "☀️ Zaid Season (Summer)",
        "zaid_desc": "- Major Crops: Watermelon, Cucumber, Pulses.\n- Focus: Micronutrient sprays & organic compost.",

        # Fert Guide
        "fert_hero_title": "🧪 Fertilizer Guide",
        "fert_hero_sub": "Know your Urea, DAP, Potash and NPK — which crop needs what nutrient, and the exact right time to apply it for peak yield.",
        "common_ferts": "🌿 Common Agricultural Fertilizers",
        "which_fert_title": "🌾 Which Fertilizer for Which Crop?",
        "search_crop_ph": "🔍 Search Crop (e.g. Wheat, Rice, Cotton, Soybean):",
        "tip_prefix": "💡 Tip:",

        # Calculator (MPKV Rahuri Standards)
        "calc_hero_title": "🧮 MPKV Rahuri Standard Fertilizer Calculator",
        "calc_hero_sub": "Calculate precise fertilizer dosage per acre based on Mahatma Phule Krishi Vidyapeeth (MPKV Rahuri) standards for Pune & Western Maharashtra crops.",
        "enter_field_details": "1. Field & Crop Selection",
        "select_crop": "Select Regional Crop (MPKV Standard)",
        "field_unit": "Field Measurement Unit",
        "enter_area": "Enter Field Area",
        "soil_fertility": "Soil Fertility Status",
        "compute_btn": "🧮 Compute MPKV Recommended Dosage",
        "rec_quantity": "2. Recommended MPKV Fertilizer Dosage",
        "urea_bags": "🌱 Urea (50kg Bags)",
        "dap_bags": "📦 DAP (50kg Bags)",
        "mop_bags": "🟠 Potash / MOP (50kg Bags)",
        "ssp_bags": "🌱 SSP (50kg Bags)",
        "zinc_kg": "⚡ Zinc Sulphate (kg)",
        "fym_tons": "🚜 Organic Compost (FYM Tons)",
        "sched_note": "📅 Recommended Schedule: Apply full DAP, Potash, SSP & FYM as basal dose during sowing. Apply Urea in 2-3 split doses at key growth stages!",

        # Crop Guide
        "crop_hero_title": "🌾 Comprehensive Crop Guide",
        "crop_hero_sub": "Explore optimal soil conditions, sowing windows, water needs, and yield expectations for major Indian crops.",
        "ideal_soil": "Ideal Soil",
        "temperature": "Temperature",
        "water_req": "Water Requirement",
        "exp_yield": "Expected Yield",
        "sowing_window": "Sowing Window",

        # Pest
        "pest_hero_title": "🐛 Pest & Disease Management",
        "pest_hero_sub": "Identify crop pests early, prevent infestations, and learn safe organic & chemical control methods.",
        "affected_crops": "Affected Crops",
        "symptoms": "Symptoms",
        "org_remedy": "🌿 Organic Remedy",
        "chem_control": "🧪 Recommended Chemical",

        # Task 1: Schemes & Maha Seva Kendra
        "schemes_hero_title": "🏛️ Govt. Schemes & Maha Seva Kendra",
        "schemes_hero_sub": "Key welfare schemes, subsidies, official government documents, and online form submission services.",
        "seva_center_title": "🏬 Aaple Sarkar Seva Kendra / CSIC Maha Seva Kendra",
        "seva_center_sub": "Official Government Service & Document Processing Center in Pune",
        "operator": "Authorized Operator",
        "seva_address": "Center Address",
        "contact_num": "Contact Number",
        "certificates_title": "📜 Government Certificates (दाखले)",
        "identity_docs_title": "🆔 Identity & Legal Documentation",
        "business_docs_title": "🏢 Business & Revenue Documents",
        "financial_schemes_title": "🏛️ Financial & Municipal Schemes",
        "call_to_action_title": "📌 Need assistance filing government scheme forms or obtaining official certificates?",
        "call_to_action_body": "Contact Abhishek Khedekar (+91 9960870182) at Shop No. 3, Trimurti Chowk, Near Relax Medical, Bharati Vidyapeeth, Pune - 411043.",
        "footer_tagline": "A simple digital farming companion helping Indian farmers grow better, every season.",
        "eligibility": "Eligibility",

        # Weather
        "weather_hero_title": "🌤️ Live & Historical Weather Forecast",
        "weather_hero_sub": "Real-time 15-day historical weather analysis & 7-day forecast for any Taluka / Sub-region in Pune District powered by Open-Meteo API.",
        "select_region": "📍 Select Sub-Region / Taluka in Pune District:",
        "past15_avg_temp": "Past 15-Day Avg Temp (°C)",
        "past15_rain": "Past 15-Day Rain (mm)",
        "forecast7_rain": "7-Day Forecast Rain (mm)",
        "soil_moisture": "Current Soil Moisture (%)",
        "temp_trend": "📈 22-Day Temperature Trend (Past 15 Days + 7 Days Forecast)",
        "rain_trend": "🌧️ 22-Day Precipitation Trend (Past 15 Days + 7 Days Forecast)",

        # Task 3: Smart AI
        "ai_hero_title": "🤖 Smart AI Irrigation & Pest Risk Prediction",
        "ai_hero_sub": "XGBoost Machine Learning models integrated with Soil Type mapping & live Open-Meteo weather parameters.",
        "soil_type_label": "🧪 Select Farm Soil Type",
        "predict_btn": "🤖 Run XGBoost AI Prediction",
        "irrigation_res": "💧 Irrigation Requirement Prediction",
        "disease_res": "🐛 Fungal & Disease Risk Prediction",

        # Chatbot
        "chat_hero_title": "💬 Krishi AI Assistant",
        "chat_hero_sub": "Ask any farming question regarding fertilizers, watering schedules, pest remedies, or government schemes.",
        "chat_input_ph": "Ask Krishi AI a question (e.g. When to apply Urea? Or how to get Income Certificate?)",
        "chat_welcome": "Hello Farmer! 🌾 How can I assist your farm today? You can ask me about Fertilizers, Irrigation, Pests & Diseases, or Govt. Schemes & Maha Seva Kendra services.",
        "contact_hero_title": "📞 Contact & Support Desk",
        "contact_hero_sub": "Reach out to our Website Developers Team or Maha Seva Kendra Desk for assistance.",
        "send_query": "💬 Send an Advisory Query",
        "farmer_name": "Farmer Name",
        "mobile_num": "Mobile Number",
        "district": "District / Village",
        "query_topic": "Query Topic",
        "message": "Write your message / crop issue",
        "submit_btn": "✉️ Submit Query",
        "success_msg": "Thank you {name}! Your query regarding '{topic}' has been received. Our Pune support team will call you back at {phone}.",
        "error_msg": "Please fill in your Name and Mobile Number."
    },

    "हिंदी": {
        "app_title": "कृषि मित्र",
        "app_subtitle": "डिजिटल किसान साथी",
        "nav_home": "🏠 होम",
        "nav_fert_guide": "🧪 उर्वरक मार्गदर्शिका",
        "nav_calculator": "🧮 एमपीकेवी उर्वरक कैलकुलेटर",
        "nav_crop_guide": "🌾 फसल मार्गदर्शिका",
        "nav_pest": "🐛 कीट एवं रोग प्रबंधन",
        "nav_schemes": "🏛️ सरकारी योजनाएं व महा सेवा केंद्र",
        "nav_contact": "📞 संपर्क व सहायता डेस्क",
        "nav_weather": "🌤️ मौसम पूर्वानुमान",
        "nav_ai_pred": "🤖 स्मार्ट AI भविष्यवाणी",
        "nav_chat": "💬 कृषि AI सहायक",
        "helpline_title": "📍 महा सेवा केंद्र डेस्क",
        "phone": "ऑपरेटर फोन",
        "location": "केंद्र स्थान",
        "email": "डेवलपर ईमेल",

        # Home
        "home_hero_title": "🌾 कृषि मित्र में आपका स्वागत है",
        "home_hero_sub": "आपका सरल और स्मार्ट डिजिटल किसान साथी जो भारतीय किसानों को स्वस्थ फसल उगाने, एमपीकेवी राहुरी उर्वरक मात्रा चुनने और महा सेवा केंद्र सेवाओं का लाभ उठाने में मदद करता है।",
        "stat_crops": "शामिल फसलें",
        "stat_dosing": "एमपीकेवी राहुरी मात्रा",
        "stat_schemes": "महा सेवा केंद्र डेस्क",
        "stat_support": "डेवलपर सहायता",
        "season_highlights": "💡 मौसमी खेती की मुख्य बातें",
        "kharif_title": "🌧️ खरीफ सीजन (मानसून)",
        "kharif_desc": "- मुख्य फसलें: चावल, कपास, सोयाबीन, मक्का।\n- मुख्य सुझाव: शुरुआती बुवाई पर डीएपी और जिंक दें।",
        "rabi_title": "🌾 रबी सीजन (सर्दियां)",
        "rabi_desc": "- मुख्य फसलें: गेहूं, चना, सरसों, प्याज।\n- मुख्य सुझाव: संतुलित एनपीके और यूरिया की टॉप ड्रेसिंग करें।",
        "zaid_title": "☀️ जायद सीजन (गर्मी)",
        "zaid_desc": "- मुख्य फसलें: तरबूज, खीरा, दालें।\n- मुख्य सुझाव: सूक्ष्म पोषक स्प्रे और जैविक खाद का प्रयोग करें।",

        # Fert Guide
        "fert_hero_title": "🧪 उर्वरक मार्गदर्शिका",
        "fert_hero_sub": "यूरिया, डीएपी, पोटाश और एनपीके को समझें — किस फसल को कौन सा पोषक तत्व चाहिए और कब सही मात्रा में प्रयोग करें।",
        "common_ferts": "🌿 प्रमुख कृषि उर्वरक",
        "which_fert_title": "🌾 किस फसल के लिए कौन सा उर्वरक?",
        "search_crop_ph": "🔍 फसल खोजें (जैसे गेहूं, धान, कपास, सोयाबीन):",
        "tip_prefix": "💡 सलाह:",

        # Calculator
        "calc_hero_title": "🧮 एमपीकेवी राहुरी मानक उर्वरक कैलकुलेटर",
        "calc_hero_sub": "पुणे और पश्चिमी महाराष्ट्र की फसलों के लिए महात्मा फुले कृषि विद्यापीठ (MPKV Rahuri) मानकों के आधार पर उर्वरक मात्रा की गणना करें।",
        "enter_field_details": "1. खेत व फसल चयन",
        "select_crop": "क्षेत्रीय फसल चुनें (MPKV मानक)",
        "field_unit": "खेत मापने की इकाई",
        "enter_area": "खेत का क्षेत्रफल दर्ज करें",
        "soil_fertility": "मिट्टी की उर्वरता स्तर",
        "compute_btn": "🧮 एमपीकेवी अनुशंसित मात्रा गणना करें",
        "rec_quantity": "2. अनुशंसित एमपीकेवी उर्वरक मात्रा",
        "urea_bags": "🌱 यूरिया (50 किग्रा बोरी)",
        "dap_bags": "📦 डीएपी (50 किग्रा बोरी)",
        "mop_bags": "🟠 पोटाश / एमओपी (50 किग्रा बोरी)",
        "ssp_bags": "🌱 एसएसपी (50 किग्रा बोरी)",
        "zinc_kg": "⚡ जिंक सल्फेट (किग्रा)",
        "fym_tons": "🚜 गोबर खाद (FYM टन)",
        "sched_note": "📅 सलाह: बुवाई के समय पूरी डीएपी, पोटाश, एसएसपी और गोबर खाद डालें। यूरिया को 2-3 बराबर भागों में फसल की वृद्धि के मुख्य चरणों में दें!",

        # Crop Guide
        "crop_hero_title": "🌾 विस्तृत फसल मार्गदर्शिका",
        "crop_hero_sub": "भारतीय फसलों के लिए उपयुक्त मिट्टी, बुवाई का समय, सिंचाई और अपेक्षित पैदावार देखें।",
        "ideal_soil": "उपयुक्त मिट्टी",
        "temperature": "तापमान",
        "water_req": "पानी की आवश्यकता",
        "exp_yield": "अनुमानित पैदावार",
        "sowing_window": "बुवाई का समय",

        # Pest
        "pest_hero_title": "🐛 कीट एवं रोग प्रबंधन",
        "pest_hero_sub": "फसल के कीटों की समय पर पहचान करें और जैविक व रासायनिक नियंत्रण के तरीके जानें।",
        "affected_crops": "प्रभावित फसलें",
        "symptoms": "लक्षण",
        "org_remedy": "🌿 जैविक उपचार",
        "chem_control": "🧪 अनुशंसित रासायनिक दवा",

        # Schemes & Seva Kendra
        "schemes_hero_title": "🏛️ सरकारी योजनाएं व महा सेवा केंद्र",
        "schemes_hero_sub": "प्रमुख कल्याणकारी योजनाएं, सब्सिडी, आधिकारिक सरकारी दस्तावेज और ऑनलाइन फॉर्म सेवाएं।",
        "seva_center_title": "🏬 आपले सरकार सेवा केंद्र / महा ई सेवा केंद्र",
        "seva_center_sub": "पुणे में आधिकारिक सरकारी सेवा व दस्तावेज केंद्र",
        "operator": "अधिकृत ऑपरेटर",
        "seva_address": "केंद्र का पता",
        "contact_num": "संपर्क नंबर",
        "certificates_title": "📜 सरकारी प्रमाण पत्र (दाखले)",
        "identity_docs_title": "🆔 पहचान व कानूनी दस्तावेज",
        "business_docs_title": "🏢 व्यवसाय व राजस्व दस्तावेज",
        "financial_schemes_title": "🏛️ वित्तीय व नगर निगम योजनाएं",
        "call_to_action_title": "📌 सरकारी योजनाओं के फॉर्म भरने या प्रमाण पत्र प्राप्त करने के लिए संपर्क करें:",
        "call_to_action_body": "अभिषेक खेडेकर (+91 9960870182), शॉप नं ३, त्रिमूर्ती चौक, रिलॅक्स मेडिकल जवळ, भारती विद्यापीठ, पुणे - ४११०४३।",
        "footer_tagline": "हर सीजन में भारतीय किसानों को बेहतर फसल उगाने में मदद करने वाला डिजिटल साथी।",
        "eligibility": "पात्रता",

        # Weather
        "weather_hero_title": "🌤️ लाइव और ऐतिहासिक मौसम पूर्वानुमान",
        "weather_hero_sub": "ओपन-मेटियो एपीआई द्वारा पुणे जिले के किसी भी तहसील / क्षेत्र के लिए पिछले 15 दिनों का विश्लेषण और 7 दिनों का पूर्वानुमान।",
        "select_region": "📍 पुणे जिले का तहसील / क्षेत्र चुनें:",
        "past15_avg_temp": "पिछले 15 दिन औसत तापमान (°C)",
        "past15_rain": "पिछले 15 दिन कुल वर्षा (मिमी)",
        "forecast7_rain": "7-दिन पूर्वानुमान वर्षा (मिमी)",
        "soil_moisture": "वर्तमान मिट्टी की नमी (%)",
        "temp_trend": "📈 22-दिवसीय तापमान प्रवृत्ति (पिछले 15 दिन + 7 दिन पूर्वानुमान)",
        "rain_trend": "🌧️ 22-दिवसीय वर्षा प्रवृत्ति (पिछले 15 दिन + 7 दिन पूर्वानुमान)",

        # Smart AI
        "ai_hero_title": "🤖 स्मार्ट AI सिंचाई एवं कीट जोखिम भविष्यवाणी",
        "ai_hero_sub": "मिट्टी के प्रकार की मैपिंग और लाइव मौसम डेटा पर आधारित एक्सजीबूस्ट (XGBoost) मॉडल।",
        "soil_type_label": "🧪 अपने खेत की मिट्टी का प्रकार चुनें",
        "predict_btn": "🤖 एक्सजीबूस्ट AI भविष्यवाणी चलाएं",
        "irrigation_res": "💧 सिंचाई आवश्यकता की भविष्यवाणी",
        "disease_res": "🐛 कवक एवं रोग जोखिम भविष्यवाणी",

        # Chatbot
        "chat_hero_title": "💬 कृषि AI सहायक",
        "chat_hero_sub": "उर्वरक, सिंचाई, कीट उपचार या सरकारी योजनाओं व महा सेवा केंद्र के बारे में प्रश्न पूछें।",
        "chat_input_ph": "कृषि एआई से प्रश्न पूछें (जैसे: यूरिया कब डालें? या आय प्रमाण पत्र कैसे बनाएं?)",
        "chat_welcome": "नमस्ते किसान भाई! 🌾 मैं आज आपकी क्या सहायता कर सकता हूँ? आप मुझसे उर्वरक, सिंचाई, कीट-रोग या सरकारी योजनाओं व महा सेवा केंद्र की सेवाओं के बारे में पूछ सकते हैं।",
        "contact_hero_title": "📞 संपर्क व सहायता डेस्क",
        "contact_hero_sub": "हमारी वेबसाइट डेवलपर टीम या महा सेवा केंद्र डेस्क से सहायता प्राप्त करें।",
        "send_query": "💬 अपना प्रश्न भेजें",
        "farmer_name": "किसान का नाम",
        "mobile_num": "मोबाइल नंबर",
        "district": "जिला / गांव",
        "query_topic": "प्रश्न का विषय",
        "message": "अपना संदेश या समस्या लिखें",
        "submit_btn": "✉️ प्रश्न जमा करें",
        "success_msg": "धन्यवाद {name}! आपका विषय '{topic}' पर प्रश्न प्राप्त हो गया है। हमारी पुणे टीम आपको {phone} पर कॉल करेगी।",
        "error_msg": "कृपया अपना नाम और मोबाइल नंबर दर्ज करें।"
    },

    "मराठी": {
        "app_title": "कृषी मित्र",
        "app_subtitle": "डिजिटल शेतकरी मित्र",
        "nav_home": "🏠 होम",
        "nav_fert_guide": "🧪 खत मार्गदर्शन",
        "nav_calculator": "🧮 MPKV खत कॅल्क्युलेटर",
        "nav_crop_guide": "🌾 पिके मार्गदर्शक",
        "nav_pest": "🐛 कीड व रोग नियंत्रण",
        "nav_schemes": "🏛️ शासकीय योजना व महा सेवा केंद्र",
        "nav_contact": "📞 संपर्क व मदत डेस्क",
        "nav_weather": "🌤️ हवामान अंदाज",
        "nav_ai_pred": "🤖 स्मार्ट AI अंदाज",
        "nav_chat": "💬 कृषी AI सहाय्यक",
        "helpline_title": "📍 महा सेवा केंद्र डेस्क",
        "phone": "ऑपरेटर फोन",
        "location": "केंद्र स्थान",
        "email": "डेव्हलपर ईमेल",

        # Home
        "home_hero_title": "🌾 कृषी मित्र मध्ये आपले स्वागत आहे",
        "home_hero_sub": "भारतीय शेतकऱ्यांना उत्तम पिके घेण्यास, महात्मा फुले कृषी विद्यापीठ (MPKV) खतांचा अचूक वापर करण्यास आणि महा ई सेवा केंद्र सेवांचा लाभ घेण्यास मदत करणारा मित्र.",
        "stat_crops": "समाविष्ट पिके",
        "stat_dosing": "MPKV राहुरी प्रमाण",
        "stat_schemes": "महा सेवा केंद्र डेस्क",
        "stat_support": "डेव्हलपर मदत",
        "season_highlights": "💡 हंगामी शेतीची मुख्य माहिती",
        "kharif_title": "🌧️ खरीप हंगाम (पावसाळा)",
        "kharif_desc": "- प्रमुख पिके: भात, कापूस, सोयाबीन, मका.\n- सल्ला: पेराताना डीएपी व झिंक खत नक्की द्या.",
        "rabi_title": "🌾 रब्बी हंगाम (हिवाळा)",
        "rabi_desc": "- प्रमुख पिके: गव्हा, हरभरा, मोहरी, कांदा.\n- सल्ला: एनपीके आणि युरियाचा हप्त्याने वापर करा.",
        "zaid_title": "☀️ उन्हाळी हंगाम (उन्हाळा)",
        "zaid_desc": "- प्रमुख पिके: कलिंगड, काकडी, डाळी.\n- सल्ला: सूक्ष्मअन्नद्रव्ये व सेंद्रिय खतांचा वापर करा.",

        # Fert Guide
        "fert_hero_title": "🧪 खत मार्गदर्शन",
        "fert_hero_sub": "युरिया, डीएपी, पोटॅश आणि एनपीके बद्दल सविस्तर माहिती — कोणत्या पिकाला कोणते खत हवे आणि ते देण्याची योग्य वेळ.",
        "common_ferts": "🌿 प्रमुख रासायनिक व सेंद्रिय खते",
        "which_fert_title": "🌾 कोणत्या पिकासाठी कोणते खत?",
        "search_crop_ph": "🔍 पीक शोधा (उदा. गहू, भात, कापूस, सोयाबीन):",
        "tip_prefix": "💡 टीप:",

        # Calculator
        "calc_hero_title": "🧮 MPKV राहुरी मानक खत कॅल्क्युलेटर",
        "calc_hero_sub": "महात्मा फुले कृषी विद्यापीठ (MPKV Rahuri) च्या मानकांनुसार पुणे व पश्चिम महाराष्ट्रातील पिकांसाठी प्रति एकर अचूक खत प्रमाणाची गणना करा.",
        "enter_field_details": "१. शेत व पीक निवड",
        "select_crop": "प्रादेशिक पीक निवडा (MPKV मानक)",
        "field_unit": "क्षेत्रफळाचे एकक",
        "enter_area": "क्षेत्रफळ टाका",
        "soil_fertility": "जमिनीची सुपीकता",
        "compute_btn": "🧮 MPKV शिफारसीनुसार खत प्रमाण काढा",
        "rec_quantity": "२. शिफारस केलेले MPKV खताचे प्रमाण",
        "urea_bags": "🌱 युरिया (५० किलो गोणी)",
        "dap_bags": "📦 डीएपी (५० किलो गोणी)",
        "mop_bags": "🟠 पोटॅश / एमओपी (५० किलो गोणी)",
        "ssp_bags": "🌱 एसएसपी (५० किलो गोणी)",
        "zinc_kg": "⚡ झिंक सल्फेट (किलो)",
        "fym_tons": "🚜 शेणखत (FYM टन)",
        "sched_note": "📅 सल्ला: पेरणीच्या वेळी संपूर्ण डीएपी, पोटॅश, एसएसपी आणि शेणखत द्या. युरिया २-३ समान हप्त्यांमध्ये पीक वाढीच्या टप्प्यावर द्या!",

        # Crop Guide
        "crop_hero_title": "🌾 सर्वसमावेशक पीक मार्गदर्शक",
        "crop_hero_sub": "जमिनीचा प्रकार, पेरणीचा काळ, पाण्याची गरज आणि अपेक्षित उत्पन्नाची सविस्तर माहिती.",
        "ideal_soil": "योग्य जमीन",
        "temperature": "तापमान",
        "water_req": "पाण्याची गरज",
        "exp_yield": "अपेक्षित उत्पादन",
        "sowing_window": "पेरणीचा काळ",

        # Pest
        "pest_hero_title": "🐛 कीड व रोग नियंत्रण",
        "pest_hero_sub": "पिकावरील कीड व रोगांची वेळीच ओळख करा आणि सेंद्रिय व रासायनिक उपाय शोधा.",
        "affected_crops": "बाधित पिके",
        "symptoms": "लक्षणे",
        "org_remedy": "🌿 सेंद्रिय उपाय",
        "chem_control": "🧪 शिफारस केलेले औषध",

        # Schemes & Seva Kendra
        "schemes_hero_title": "🏛️ शासकीय योजना व महा सेवा केंद्र",
        "schemes_hero_sub": "महत्त्वाच्या शासकीय योजना, अनुदाने, अधिकृत कागदपत्रे व ऑनलाइन फॉर्म भरण्याची सुविधा.",
        "seva_center_title": "🏬 आपले सरकार सेवा केंद्र / महा ई सेवा केंद्र",
        "seva_center_sub": "पुण्यातील अधिकृत शासकीय सेवा व कागदपत्र केंद्र",
        "operator": "अधिकृत ऑपरेटर",
        "seva_address": "केद्राचा पत्ता",
        "contact_num": "संपर्क क्रमांक",
        "certificates_title": "📜 शासकीय दाखले (Certificates)",
        "identity_docs_title": "🆔 ओळख व कायदेशीर कागदपत्रे",
        "business_docs_title": "🏢 व्यवसाय व महसूल कागदपत्रे",
        "financial_schemes_title": "🏛️ आर्थिक व मनपा योजना फॉर्म",
        "call_to_action_title": "📌 शासकीय योजनांचे फॉर्म भरण्यासाठी किंवा दाखले काढण्यासाठी संपर्क करा:",
        "call_to_action_body": "अभिषेक खेडेकर (+91 9960870182), शॉप नं ३, त्रिमूर्ती चौक, रिलॅक्स मेडिकल जवळ, भारती विद्यापीठ, पुणे - ४११०४३.",
        "footer_tagline": "प्रत्येक हंगामात भारतीय शेतकऱ्यांना उत्तम शेती करण्यासाठी मदत करणारा डिजिटल साथीदार.",
        "eligibility": "पात्रता",

        # Weather
        "weather_hero_title": "🌤️ थेट आणि ऐतिहासिक हवामान अंदाज",
        "weather_hero_sub": "ओपन-मेटिओ एपीआय द्वारे पुणे जिल्ह्यातील कोणत्याही तालुक्याचा / परिसराचा मागील १५ दिवसांचे विश्लेषण आणि ७ दिवसांचा हवामान अंदाज।",
        "select_region": "📍 पुणे जिल्ह्यातील तालुका / परिसर निवडा:",
        "past15_avg_temp": "मागील १५ दिवस सरासरी तापमान (°C)",
        "past15_rain": "मागील १५ दिवस एकूण पाऊस (मिमी)",
        "forecast7_rain": "७-दिवस अंदाज पाऊस (मिमी)",
        "soil_moisture": "सध्याची मातीतील ओलावा (%)",
        "temp_trend": "📈 २२-दिवसीय तापमान बदल (मागील १५ दिवस + ७ दिवस अंदाज)",
        "rain_trend": "🌧️ २२-दिवसीय पाऊस बदल (मागील १५ दिवस + ७ दिवस अंदाज)",

        # Smart AI
        "ai_hero_title": "🤖 स्मार्ट AI सिंचन व रोग धोका अंदाज",
        "ai_hero_sub": "जमिनीच्या प्रकाराचे मॅपिंग आणि थेट हवामान माहितीवर आधारित एक्सजीबूस्ट (XGBoost) मॉडेल.",
        "soil_type_label": "🧪 तुमच्या शेतातील जमिनीचा प्रकार निवडा",
        "predict_btn": "🤖 एक्सजीबूस्ट AI अंदाज मिळवा",
        "irrigation_res": "💧 सिंचन गरजेचा अंदाज",
        "disease_res": "🐛 बुरशी व रोग धोका अंदाज",

        # Chatbot
        "chat_hero_title": "💬 कृषी AI सहाय्यक",
        "chat_hero_sub": "खते, पाणी व्यवस्थापन, कीड नियंत्रण किंवा शासकीय योजना व महा सेवा केंद्राबद्दल प्रश्न विचारा.",
        "chat_input_ph": "कृषी AI ला प्रश्न विचारा (उदा. युरिया कधी द्यावा? किंवा उत्पन्न दाखला कसा काढायचा?)",
        "chat_welcome": "नमस्कार शेतकरी मित्रानो! 🌾 आज मी तुम्हाला कशी मदत करू शकतो? तुम्ही मला खते, सिंचन, कीड-रोग किंवा शासकीय योजना व महा ई सेवा केंद्राबद्दल विचारू शकता.",
        "contact_hero_title": "📞 संपर्क व मदत डेस्क",
        "contact_hero_sub": "आमच्या वेबसाइट डेव्हलपर टीम किंवा महा सेवा केंद्र डेस्ककडून मदत मिळवा.",
        "send_query": "💬 तुमचा प्रश्न पाठवा",
        "farmer_name": "शेतकऱ्याचे नाव",
        "mobile_num": "मोबाईल नंबर",
        "district": "जिल्हा / गाव",
        "query_topic": "प्रश्नाचा विषय",
        "message": "तुमचा संदेश किंवा समस्या लिहा",
        "submit_btn": "✉️ प्रश्न पाठवा",
        "success_msg": "धन्यवाद {name}! '{topic}' बद्दल तुमचा प्रश्न मिळाला आहे. आमची पुणे टीम तुम्हाला {phone} वर कॉल करेल.",
        "error_msg": "कृपया तुमचे नाव आणि मोबाईल नंबर टाका."
    }
}

# ------------------------------------------------------------------------------
# SECTION 9: SIDEBAR NAVIGATION & LANGUAGE SELECTION WIDGETS
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 0.5rem 0;">
            <span style="font-size: 2.5rem;">🌾</span>
            <h2 style="color: #15803d; margin: 0; font-size: 1.6rem; font-weight: 800;">Krishi Mitra</h2>
            <p style="color: #475569; font-size: 0.82rem; margin-top: 0.1rem;">Digital Farming Companion</p>
        </div>
    """, unsafe_allow_html=True)

    # 🌐 Language Selector Widget
    lang = st.selectbox(
        "🌐 Choose Language / भाषा / भाषा निवडा",
        ["English", "हिंदी", "मराठी"],
        index=0
    )
    
    t = TRANSLATIONS[lang]
    st.divider()

    menu = st.radio(
        "Navigation",
        [
            t["nav_home"],
            t["nav_fert_guide"],
            t["nav_calculator"],
            t["nav_crop_guide"],
            t["nav_pest"],
            t["nav_schemes"],
            t["nav_weather"],
            t["nav_ai_pred"],
            t["nav_chat"],
            t["nav_contact"]
        ],
        index=1
    )
    
    st.divider()
    
    # TASK 4: Updated Sidebar Helpline Card (Maha Seva Kendra & Developers)
    st.markdown(f"""
        <div style="background-color: #f0fdf4; border-radius: 0.75rem; padding: 1rem; border: 1px solid #bbf7d0;">
            <p style="font-weight: 700; color: #166534; font-size: 0.9rem; margin-bottom: 0.3rem;">📍 {t['helpline_title']}</p>
            <p style="font-size: 0.82rem; color: #15803d; margin: 0;"><b>{t['operator']}:</b> Abhishek Khedekar</p>
            <p style="font-size: 0.82rem; color: #15803d; margin: 0;"><b>{t['phone']}:</b> +91 9960870182</p>
            <p style="font-size: 0.82rem; color: #15803d; margin: 0;"><b>{t['location']}:</b> Bharati Vidyapeeth, Pune</p>
            <hr style="margin: 0.5rem 0; border-color: #bbf7d0;">
            <p style="font-size: 0.80rem; color: #166534; margin: 0;"><b>Dev Support:</b> Mriganka Bhattacharya</p>
            <p style="font-size: 0.80rem; color: #166534; margin: 0;">📞 +91 7709503474</p>
        </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SECTION 10: MULTILINGUAL DATASETS FOR EXISTING PAGES
# ------------------------------------------------------------------------------
fertilizers_multilingual = {
    "English": [
        {"name": "Urea", "tag": "Nitrogen (46% N)", "desc": "Provides Nitrogen (N) — boosts leaf growth, crop vigour, and deep green colour.", "note": "Apply in split doses; avoid using all at sowing time to prevent leaching.", "icon": "🍃"},
        {"name": "DAP (Di-Ammonium Phosphate)", "tag": "Nitrogen (18%) + Phosphorus (46%)", "desc": "Provides Nitrogen + Phosphorus for root development and early growth.", "note": "Best applied as a basal dose directly at sowing time.", "icon": "📦"},
        {"name": "Potash (MOP)", "tag": "Potassium (60% K₂O)", "desc": "Provides Potassium (K) — improves grain filling, drought tolerance, and disease resistance.", "note": "Apply at basal stage, and again at flowering for fruit & grain crops.", "icon": "🟠"},
        {"name": "NPK Complexes (10:26:26 / 12:32:16)", "tag": "Balanced N-P-K", "desc": "Combined mix of Nitrogen, Phosphorus, and Potassium in one balanced formulation.", "note": "Convenient for balanced feeding in vegetables and general farming.", "icon": "🧪"},
        {"name": "SSP (Single Super Phosphate)", "tag": "Phosphorus (16%) + Sulphur (11%)", "desc": "Promotes root expansion, seed formation, and oil content in oilseeds.", "note": "Great alternative to DAP for soybean, groundnut, and mustard.", "icon": "🌱"},
        {"name": "Zinc Sulphate", "tag": "Micronutrient (Zn 21% / 33%)", "desc": "Corrects zinc deficiency, prevents Khaira leaf yellowing, and boosts enzyme activity.", "note": "Apply during land preparation; do not mix directly with DAP.", "icon": "⚡"}
    ],
    "हिंदी": [
        {"name": "यूरिया (Urea)", "tag": "नाइट्रोजन (46% N)", "desc": "नाइट्रोजन (N) प्रदान करता है — पत्तियों की वृद्धि, पौधे का विकास और गहरा हरा रंग बढ़ाता है।", "note": "किस्तों में प्रयोग करें; बुवाई के समय पूरी मात्रा डालने से बचें।", "icon": "🍃"},
        {"name": "डीएपी (DAP)", "tag": "नाइट्रोजन (18%) + फास्फोरस (46%)", "desc": "नाइट्रोजन और फास्फोरस देता है जो जड़ों के विकास और शुरुआती वृद्धि के लिए आवश्यक है।", "note": "बुवाई के समय मुख्य (बेसल) खुराक के रूप में प्रयोग करें।", "icon": "📦"},
        {"name": "पोटाश (MOP)", "tag": "पोटेशियम (60% K₂O)", "desc": "पोटेशियम (K) देता है — दानों के भरने, सूखा सहनशीलता और रोग प्रतिरोधक क्षमता में सुधार करता है।", "note": "बुवाई के समय और फिर फूल आने पर प्रयोग करें।", "icon": "🟠"},
        {"name": "एनपीके (NPK Complexes)", "tag": "संतुलित N-P-K", "desc": "नाइट्रोजन, फास्फोरस और पोटेशियम का एक संतुलित मिश्रण।", "note": "सब्जियों और सामान्य फसलों में संतुलित पोषण के लिए सुविधाजनक।", "icon": "🧪"},
        {"name": "एसएसपी (Single Super Phosphate)", "tag": "फास्फोरस (16%) + सल्फर (11%)", "desc": "तिलहन और दलहन में जड़ों के फैलाव और तेल की मात्रा को बढ़ाता है।", "note": "सोयाबीन, मूंगफली और सरसों के लिए डीएपी का बेहतरीन विकल्प।", "icon": "🌱"},
        {"name": "जिंक सल्फेट (Zinc Sulphate)", "tag": "सूक्ष्म पोषक (Zn 21%)", "desc": "जिंक की कमी दूर करता है, खैरा रोग (पत्तियों का पीलापन) रोकता है।", "note": "खेत तैयार करते समय डालें; डीएपी के साथ सीधे न मिलाएं।", "icon": "⚡"}
    ],
    "मराठी": [
        {"name": "युरिया (Urea)", "tag": "नायट्रोजन (46% N)", "desc": "नायट्रोजन (N) पुरवते - पानांची वाढ, पिकाची ताकद आणि गडद हिरवा रंग वाढवते.", "note": "हप्त्याने वापरा; पेरणीच्या वेळी सर्व युरिया देणे टाळा.", "icon": "🍃"},
        {"name": "डी.ए.पी. (DAP)", "tag": "नायट्रोजन (18%) + फॉस्फरस (46%)", "desc": "मुळांच्या मजबूत वाढीसाठी आणि सुरुवातीच्या विकासासाठी नायट्रोजन व फॉस्फरस देते.", "note": "पेरणीच्या वेळी बेसल डोस म्हणून देणे उत्तम.", "icon": "📦"},
        {"name": "पोटॅश (MOP)", "tag": "पोटेशियम (60% K₂O)", "desc": "पोटेशियम (K) पुरवते - दाणे भरण्यास, रोगप्रतिकारशक्ती वाढवण्यास मदत करते.", "note": "पेरणीच्या वेळी आणि फुलोऱ्याच्या टप्प्यावर द्या.", "icon": "🟠"},
        {"name": "एन.पी.के. (NPK)", "tag": "संतुलित N-P-K", "desc": "नायट्रोजन, फॉस्फरस आणि पोटॅशियमचे एकत्रित संतुलित मिश्रण.", "note": "भाज्या आणि इतर पिकांसाठी संतुलित पोषण देण्यास सोपे.", "icon": "🧪"},
        {"name": "एस.एस.पी. (SSP)", "tag": "फॉस्फरस (16%) + गंधक (11%)", "desc": "मुळांची वाढ आणि तेलबिया पिकांमध्ये तेलाचे प्रमाण वाढवते.", "note": "सोयाबीन, भुईमूग आणि मोहरीसाठी डीएपी ऐवजी उत्तम पर्याय.", "icon": "🌱"},
        {"name": "झिंक सल्फेट (Zinc Sulphate)", "tag": "सूक्ष्मअन्नद्रव्य (Zn 21%)", "desc": "झिंकची कमतरता भरून काढते आणि पानांचा पिवळेपणा रोखते.", "note": "रानबांधणीच्या वेळी द्या; डीएपी सोबत थेट मिसळू नका.", "icon": "⚡"}
    ]
}

crop_table_multilingual = {
    "English": [
        {"Crop": "Sugarcane (MPKV Standard)", "Recommended Fertilizer": "Urea + DAP + Potash", "Application Timing": "Basal at planting; Urea in 3 split doses during tillering & growth"},
        {"Crop": "Onion (MPKV Standard)", "Recommended Fertilizer": "Urea + DAP + Potash + SSP", "Application Timing": "Basal at transplanting with SSP; Urea split at 30 & 45 days"},
        {"Crop": "Soybean (MPKV Standard)", "Recommended Fertilizer": "DAP + Potash + Low Urea", "Application Timing": "Full dose at sowing — soybean fixes its own nitrogen"},
        {"Crop": "Wheat (MPKV Standard)", "Recommended Fertilizer": "DAP + Urea + Potash", "Application Timing": "Basal at sowing; Urea split at 1st (21 days) & 2nd (42 days) irrigation"},
        {"Crop": "Rice / Paddy (MPKV Standard)", "Recommended Fertilizer": "DAP + Urea + Potash + Zinc", "Application Timing": "DAP & Potash at transplanting; Urea split at tillering & panicle initiation"},
        {"Crop": "Vegetables (Tomato/Onion/Chilli)", "Recommended Fertilizer": "NPK + Urea + Micronutrients", "Application Timing": "NPK as basal; Urea top-dressed every 2-3 weeks"},
    ],
    "हिंदी": [
        {"Crop": "गन्ना (MPKV मानक)", "Recommended Fertilizer": "यूरिया + डीएपी + पोटाश", "Application Timing": "रोपाई पर बेसल; बढ़वार के समय 3 किस्तों में यूरिया"},
        {"Crop": "प्याज (MPKV मानक)", "Recommended Fertilizer": "यूरिया + डीएपी + पोटाश + एसएसपी", "Application Timing": "रोपाई पर बेसल व एसएसपी; 30 और 45 दिनों में यूरिया की टॉप ड्रेसिंग"},
        {"Crop": "सोयाबीन (MPKV मानक)", "Recommended Fertilizer": "डीएपी + पोटाश + कम यूरिया", "Application Timing": "बुवाई के समय पूरी खुराक — सोयाबीन हवा से नाइट्रोजन लेती है"},
        {"Crop": "गेहूं (MPKV मानक)", "Recommended Fertilizer": "डीएपी + यूरिया + पोटाश", "Application Timing": "बुवाई के समय डीएपी व पोटाश; पहली (21 दिन) व दूसरी सिंचाई पर यूरिया"},
        {"Crop": "धान / चावल (MPKV मानक)", "Recommended Fertilizer": "डीएपी + यूरिया + पोटाश + जिंक", "Application Timing": "रोपाई पर डीएपी व पोटाश; कल्ले निकलने व बालियां बनने पर यूरिया"},
        {"Crop": "सब्जियां (टमाटर/बैंगन/मिर्च)", "Recommended Fertilizer": "एनपीके + यूरिया + सूक्ष्म पोषक", "Application Timing": "शुरुआत में एनपीके; हर 2-3 सप्ताह में यूरिया"},
    ],
    "मराठी": [
        {"Crop": "ऊस (MPKV मानक)", "Recommended Fertilizer": "युरिया + डीएपी + पोटॅश", "Application Timing": "लागवडीवेळी बेसल डोस; फूटवे येताना ३ हप्त्यांमध्ये युरिया"},
        {"Crop": "कांदा (MPKV मानक)", "Recommended Fertilizer": "युरिया + डीएपी + पोटॅश + एसएसपी", "Application Timing": "पुनर्लागवडीवेळी बेसल व एसएसपी; ३० व ४५ दिवसांनी युरियाचा हप्ता"},
        {"Crop": "सोयाबीन (MPKV मानक)", "Recommended Fertilizer": "डीएपी + पोटॅश + कमी युरिया", "Application Timing": "पेरणीवेळी पूर्ण डोस — सोयाबीन स्वतः नायट्रोजन तयार करते"},
        {"Crop": "गहू (MPKV मानक)", "Recommended Fertilizer": "डी.ए.पी. + युरिया + पोटॅश", "Application Timing": "पेरणीच्या वेळी डीएपी व पोटॅश; २१ व्या व ४२ व्या दिवशी युरिया"},
        {"Crop": "भात (MPKV मानक)", "Recommended Fertilizer": "डी.ए.पी. + युरिया + पोटॅश + झिंक", "Application Timing": "लागवडीवेळी डीएपी व पोटॅश; फूटवे फुटताना युरियाचे हप्ते"},
        {"Crop": "भाज्या (टमाटा/वांगी/मिरची)", "Recommended Fertilizer": "एन.पी.के. + युरिया + सूक्ष्मअन्नद्रव्ये", "Application Timing": "शुरुवातीला एनपीके; दर २-३ आठवड्यांनी युरिया"},
    ]
}

# ------------------------------------------------------------------------------
# SECTION 11: DYNAMIC PAGE ROUTING ENGINE
# ------------------------------------------------------------------------------

# PAGE 1: HOME MODULE
if menu == t["nav_home"]:
    st.markdown(f"""
        <div class="hero-card">
            <h1 class="hero-title">{t['home_hero_title']}</h1>
            <p class="hero-subtitle">{t['home_hero_sub']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    hero_path = os.path.join(os.path.dirname(__file__), "krishi_hero.jpg")
    if os.path.exists(hero_path):
        st.image(hero_path, use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-box"><div class="stat-val">15+</div><div class="stat-label">{t["stat_crops"]}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-box"><div class="stat-val">100%</div><div class="stat-label">{t["stat_dosing"]}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-box"><div class="stat-val">24/7</div><div class="stat-label">{t["stat_schemes"]}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-box"><div class="stat-val">4 Devs</div><div class="stat-label">{t["stat_support"]}</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(t["season_highlights"])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(f"**{t['kharif_title']}**\n\n{t['kharif_desc']}")
    with c2:
        st.success(f"**{t['rabi_title']}**\n\n{t['rabi_desc']}")
    with c3:
        st.warning(f"**{t['zaid_title']}**\n\n{t['zaid_desc']}")

# PAGE 2: FERTILIZER GUIDE MODULE
elif menu == t["nav_fert_guide"]:
    st.markdown(f"""
        <div class="hero-card">
            <h1 class="hero-title">{t['fert_hero_title']}</h1>
            <p class="hero-subtitle">{t['fert_hero_sub']}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### {t['common_ferts']}")
    
    cols = st.columns(3)
    ferts = fertilizers_multilingual[lang]
    for idx, f in enumerate(ferts):
        with cols[idx % 3]:
            st.markdown(f"""
                <div class="fert-card">
                    <span class="fert-badge">{f['tag']}</span>
                    <div class="fert-title">{f['icon']} {f['name']}</div>
                    <div class="fert-desc">{f['desc']}</div>
                    <div class="fert-note">{t['tip_prefix']} {f['note']}</div>
                </div>
                <br>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### {t['which_fert_title']}")
    
    df_table = pd.DataFrame(crop_table_multilingual[lang])
    search_crop = st.text_input(t["search_crop_ph"], "")
    
    if search_crop:
        filtered_df = df_table[df_table['Crop'].str.contains(search_crop, case=False, na=False)]
    else:
        filtered_df = df_table
        
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

# PAGE 3: TASK 2 - FERTILIZER DOSAGE CALCULATOR (MPKV RAHURI STANDARDS)
elif menu == t["nav_calculator"]:
    st.markdown(f"""
        <div class="hero-card">
            <h1 class="hero-title">{t['calc_hero_title']}</h1>
            <p class="hero-subtitle">{t['calc_hero_sub']}</p>
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader(t["enter_field_details"])
        
        # MPKV Regional Crops
        mpkv_crop_options = [
            "Sugarcane (ऊस / गन्ना)",
            "Onion (कांदा / प्याज)",
            "Soybean (सोयाबीन)",
            "Wheat (गहू / गेहूं)",
            "Rice / Paddy (भात / धान)",
            "Vegetables (भाज्या / सब्जियां - Tomato/Brinjal/Chilli)"
        ]
        crop_selected = st.selectbox(t["select_crop"], mpkv_crop_options)
        
        unit_opt = ["Acres / एकड़ / एकर", "Hectares / हेक्टेयर / हेक्टर", "Guntha / गुंठा"]
        area_unit = st.radio(t["field_unit"], unit_opt, horizontal=True)
        area_size = st.number_input(f"{t['enter_area']} ({area_unit.split('/')[0].strip()})", min_value=0.25, max_value=500.0, value=1.0, step=0.25)
        
        soil_fertility = st.select_slider(t["soil_fertility"], options=["Low / कम / कमी", "Medium / मध्यम", "High / उच्च / जास्त"], value="Medium / मध्यम")
        
        calc_btn = st.button(t["compute_btn"], type="primary", use_container_width=True)

    with c2:
        st.subheader(t["rec_quantity"])
        
        # Exact Field Unit Conversions:
        # 1 Hectare = 2.471 Acres
        # 1 Guntha = 0.025 Acres (40 Gunthas = 1 Acre)
        if "Hectares" in area_unit or "हेक्टेयर" in area_unit or "हेक्टर" in area_unit:
            acres = area_size * 2.471
        elif "Guntha" in area_unit or "गुंठा" in area_unit:
            acres = area_size * 0.025
        else:
            acres = area_size
            
        # Soil fertility adjustment multiplier (Low = +15%, High = -15%)
        mult = 1.15 if "Low" in soil_fertility or "कम" in soil_fertility or "कमी" in soil_fertility else (0.85 if "High" in soil_fertility or "उच्च" in soil_fertility else 1.0)
        
        # MPKV Rahuri RDF Base Rates (per acre):
        # 1. Sugarcane: ~4.0 Urea, 2.0 DAP, 1.5 Potash
        # 2. Onion: ~1.5 Urea, 1.0 DAP, 0.7 Potash, 1.0 SSP
        # 3. Soybean: ~0.5 Urea, 1.3 DAP, 0.4 Potash
        # 4. Wheat: ~1.5 Urea, 1.0 DAP, 0.7 Potash
        # 5. Rice: ~1.5 Urea, 1.0 DAP, 0.7 Potash, 10kg Zinc
        # 6. Vegetables: ~2.0 Urea, 1.1 DAP, 0.8 Potash
        
        extra_info = ""
        if "Sugarcane" in crop_selected:
            base_urea, base_dap, base_mop = 4.0, 2.0, 1.5
            extra_info = "🌾 **MPKV Note:** Apply 10 Tons Organic Compost (FYM) + 400kg Neem cake per acre."
        elif "Onion" in crop_selected:
            base_urea, base_dap, base_mop = 1.5, 1.0, 0.7
            extra_ssp = round(1.0 * acres * mult, 1)
            extra_info = f"🌱 **MPKV Sulphur Note:** Apply **{extra_ssp} Bags Single Super Phosphate (SSP)** for bulb pungent oil quality."
        elif "Soybean" in crop_selected:
            base_urea, base_dap, base_mop = 0.5, 1.3, 0.4
            extra_info = "🌱 **MPKV Note:** Low Urea required because Soybean rhizobium nodules fix atmospheric nitrogen."
        elif "Wheat" in crop_selected:
            base_urea, base_dap, base_mop = 1.5, 1.0, 0.7
            extra_info = "🌾 **MPKV Note:** Apply full DAP & Potash at sowing. Split Urea at 21 days (CRI stage) and 42 days."
        elif "Rice" in crop_selected:
            base_urea, base_dap, base_mop = 1.5, 1.0, 0.7
            extra_zinc = round(10.0 * acres, 1)
            extra_info = f"⚡ **MPKV Zinc Note:** Apply **{extra_zinc} kg Zinc Sulphate** at transplanting to prevent Khaira disease."
        else: # Vegetables
            base_urea, base_dap, base_mop = 2.0, 1.1, 0.8
            extra_info = "🥦 **MPKV Note:** Apply Micronutrient spray (Grade II) 25 days after transplanting."
            
        urea_bags = round(base_urea * acres * mult, 1)
        dap_bags = round(base_dap * acres * mult, 1)
        mop_bags = round(base_mop * acres * mult, 1)
        fym_tons = round(2.0 * acres, 1)
        
        st.success(f"**Field Area:** {area_size} {area_unit.split('/')[0].strip()} (**{round(acres,2)} Acres**)")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(t["urea_bags"], f"{urea_bags}")
            st.metric(t["dap_bags"], f"{dap_bags}")
        with res_col2:
            st.metric(t["mop_bags"], f"{mop_bags}")
            st.metric(t["fym_tons"], f"{fym_tons}")

        st.info(f"{t['sched_note']}\n\n{extra_info}")

# PAGE 4: COMPREHENSIVE CROP GUIDE MODULE
elif menu == t["nav_crop_guide"]:
    st.markdown(f"""
        <div class="hero-card">
            <h1 class="hero-title">{t['crop_hero_title']}</h1>
            <p class="hero-subtitle">{t['crop_hero_sub']}</p>
        </div>
    """, unsafe_allow_html=True)

    crops_info = [
        {"name": "Sugarcane / ऊस / गन्ना", "season": "Annual (12-18 Months)", "soil": "Deep Black Cotton / Heavy Clay Loam", "temp": "20°C - 38°C", "water": "High (18-24 Irrigations)", "yield": "45-60 Tons / Acre", "window": "Jan - Feb / Oct - Nov"},
        {"name": "Onion / कांदा / प्याज", "season": "Rabi / Late Kharif", "soil": "Friable Red Loam / Well-drained Silt", "temp": "15°C - 30°C", "water": "Medium (10-12 Irrigations)", "yield": "100-140 Quintals / Acre", "window": "Nov - Dec / Aug - Sept"},
        {"name": "Wheat / गेहूं / गहू", "season": "Rabi (Winter)", "soil": "Loam / Clay Loam", "temp": "15°C - 25°C", "water": "Medium (4-6 irrigations)", "yield": "18-22 Quintals / Acre", "window": "Oct - Nov"},
        {"name": "Rice Paddy / धान / भात", "season": "Kharif (Monsoon)", "soil": "Clay / Clay Loam", "temp": "20°C - 35°C", "water": "High (Standing water)", "yield": "20-25 Quintals / Acre", "window": "June - July"},
        {"name": "Cotton / कपास / कापूस", "season": "Kharif (Monsoon)", "soil": "Black Cotton Soil / Deep Loam", "temp": "21°C - 30°C", "water": "Medium", "yield": "8-12 Quintals / Acre", "window": "May - June"},
        {"name": "Soybean / सोयाबीन", "season": "Kharif (Monsoon)", "soil": "Well-drained Fertile Loam", "temp": "20°C - 32°C", "water": "Medium", "yield": "8-10 Quintals / Acre", "window": "June - July"},
    ]

    for crop in crops_info:
        with st.expander(f"🌾 {crop['name']} — {crop['season']}", expanded=True):
            col_a, col_b, col_c = st.columns(3)
            col_a.write(f"**{t['ideal_soil']}:** {crop['soil']}")
            col_a.write(f"**{t['temperature']}:** {crop['temp']}")
            col_b.write(f"**{t['water_req']}:** {crop['water']}")
            col_b.write(f"**{t['exp_yield']}:** {crop['yield']}")
            col_c.write(f"**{t['sowing_window']}:** {crop['window']}")

# PAGE 5: PEST & DISEASE MANAGEMENT MODULE
elif menu == t["nav_pest"]:
    st.markdown(f"""
        <div class="hero-card">
            <h1 class="hero-title">{t['pest_hero_title']}</h1>
            <p class="hero-subtitle">{t['pest_hero_sub']}</p>
        </div>
    """, unsafe_allow_html=True)

    pests = [
        {"name": "Stem Borer / तना छेदक / खोड किडा", "crops": "Rice, Sugarcane, Maize", "symptoms": "Dead hearts in young tillers, white heads at panicle stage.", "organic": "Neem Oil 10,000 ppm spray (5ml/L) + Trichogramma bio-cards.", "chemical": "Chlorantraniliprole 18.5% SC @ 60 ml/acre."},
        {"name": "Yellow Vein Mosaic / पीला मोजेक / पिवळा मोजेक", "crops": "Soybean, Okra, Pulses", "symptoms": "Yellow patches on leaves, stunted plant growth, low pod yield.", "organic": "Yellow sticky traps (15 traps/acre) to catch Whiteflies.", "chemical": "Thiamethoxam 25% WG @ 80g/acre for whitefly control."},
        {"name": "Thrips & Purple Blotch / फुलकिडे व करपा", "crops": "Onion, Garlic, Chilli", "symptoms": "Silver patches on leaves, purple sunken lesions on leaf blades.", "organic": "Blue Sticky Traps (20 traps/acre) + Pseudomonas fluorescens 5g/L.", "chemical": "Fipronil 5% SC @ 30ml/10L water or Tebuconazole 25.9% EC."}
    ]

    for p in pests:
        st.markdown(f"""
            <div style="background: white; padding: 1.25rem; border-radius: 1rem; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
                <h4 style="color: #991b1b; margin: 0;">🐛 {p['name']}</h4>
                <p style="font-size: 0.88rem; color: #475569; margin-top: 0.4rem;"><b>{t['affected_crops']}:</b> {p['crops']}</p>
                <p style="font-size: 0.88rem; color: #1e293b;"><b>{t['symptoms']}:</b> {p['symptoms']}</p>
                <div style="background: #f0fdf4; padding: 0.5rem 0.75rem; border-radius: 0.5rem; color: #166534; font-size: 0.85rem; margin-top: 0.5rem;">
                    {t['org_remedy']}: {p['organic']}
                </div>
                <div style="background: #fef2f2; padding: 0.5rem 0.75rem; border-radius: 0.5rem; color: #991b1b; font-size: 0.85rem; margin-top: 0.4rem;">
                    {t['chem_control']}: {p['chemical']}
                </div>
            </div>
        """, unsafe_allow_html=True)

# PAGE 6: TASK 1 - GOVT SCHEMES & MAHA SEVA KENDRA PROMOTION MODULE
elif menu == t["nav_schemes"]:
    st.markdown(f"""
        <div class="hero-card">
            <h1 class="hero-title">{t['schemes_hero_title']}</h1>
            <p class="hero-subtitle">{t['schemes_hero_sub']}</p>
        </div>
    """, unsafe_allow_html=True)

    # Prominent Maha Seva Kendra Banner
    st.markdown(f"""
        <div class="seva-banner">
            <h2 style="margin:0 0 0.4rem 0; font-size:1.6rem; color:#ffffff;">🏬 {t['seva_center_title']}</h2>
            <p style="margin:0 0 0.8rem 0; font-size:1.0rem; opacity:0.95;">{t['seva_center_sub']}</p>
            <div style="display:flex; flex-wrap:wrap; gap:1.5rem; background:rgba(255,255,255,0.12); padding:0.8rem 1.2rem; border-radius:0.5rem;">
                <div><b>👤 {t['operator']}:</b> Abhishek Khedekar (अभिषेक खेडेकर)</div>
                <div><b>📞 {t['contact_num']}:</b> <a href="tel:+919960870182" style="color:#fef08a; font-weight:bold;">+91 9960870182</a></div>
                <div><b>📍 {t['seva_address']}:</b> Shop No. 3, Trimurti Chowk, Near Relax Medical, Bharati Vidyapeeth, Pune - 411043</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Services Available at Maha Seva Kendra / Aaple Sarkar")
    
    # 4 Service Cards Grid
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"""
            <div class="seva-card">
                <h4 style="color:#1e3a8a; margin-top:0;">{t['certificates_title']}</h4>
                <ul>
                    <li><b>Income Certificate (उत्पन्न दाखला):</b> 1 Year, 3 Year certificates</li>
                    <li><b>Domicile Certificate (डोमासाईल):</b> Maharashtra Residence Proof</li>
                    <li><b>Non-Creamy Layer (नॉन क्रिमीलेअर):</b> OBC/SEBC reservation quota</li>
                    <li><b>Caste Certificate (जातीचा दाखला):</b> Category certification</li>
                    <li><b>Residence Certificate (रहिवासी दाखला):</b> Local residence proof</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="seva-card">
                <h4 style="color:#1e3a8a; margin-top:0;">{t['business_docs_title']}</h4>
                <ul>
                    <li><b>Udyam Aadhaar (उद्यम आधार):</b> MSME Business Registration</li>
                    <li><b>Food License (FSSAI):</b> Food business certification</li>
                    <li><b>Shop Act License (शॉप ऍक्ट):</b> Local shop registration</li>
                    <li><b>Digital 7/12 Smart Card (डिजिटल ७/१२):</b> Land record cards</li>
                    <li><b>Marriage & Life Certificates (हयातीचा दाखला):</b> Pensioner verification</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with sc2:
        st.markdown(f"""
            <div class="seva-card">
                <h4 style="color:#1e3a8a; margin-top:0;">{t['identity_docs_title']}</h4>
                <ul>
                    <li><b>PAN Card:</b> New PAN & Correction processing</li>
                    <li><b>Passport Services:</b> Fresh application & Renewal booking</li>
                    <li><b>Police Verification & Rent Agreement:</b> Tenant registration</li>
                    <li><b>Affidavit (प्रतिज्ञापत्र) & Gazette (राजपत्र):</b> Name change publication</li>
                    <li><b>Aadhaar Card Support:</b> Update & demographic correction assistance</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="seva-card">
                <h4 style="color:#1e3a8a; margin-top:0;">{t['financial_schemes_title']}</h4>
                <ul>
                    <li><b>PF Works (P.F. ची कामे):</b> EPF Withdrawal & UAN Activation</li>
                    <li><b>PMC Scheme Online Forms:</b> Pune Municipal Corporation welfare applications</li>
                    <li><b>PM-KISAN e-KYC:</b> Farmer Aadhaar linking & bank mapping</li>
                    <li><b>PM Fasal Bima Yojana:</b> Online crop insurance portal filing</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"**{t['call_to_action_title']}**\n\n{t['call_to_action_body']}")

    st.divider()
    st.markdown("### 🏛️ Major Flagship Agricultural Schemes")
    schemes = [
        {"title": "PM-KISAN Samman Nidhi (पीएम किसान सम्मान निधि)", "desc": "Financial benefit of ₹6,000 per year transferred directly to bank accounts of farmer families.", "eligibility": "All landholding farmer families with cultivable land."},
        {"title": "PM Fasal Bima Yojana - PMFBY (फसल बीमा योजना)", "desc": "Crop insurance cover against yield losses due to drought, floods, and natural pests.", "eligibility": "All farmers growing notified crops."},
        {"title": "Soil Health Card Scheme (मृदा स्वास्थ्य कार्ड)", "desc": "Customized soil test reports detailing NPK status and fertilizer recommendations.", "eligibility": "Free testing via local Agriculture Department."},
        {"title": "Kisan Credit Card - KCC (किसान क्रेडिट कार्ड)", "desc": "Short-term credit for crop cultivation at low subsidized 4% interest rate.", "eligibility": "Farmers, tenant farmers, and self-help groups."}
    ]

    for s in schemes:
        with st.expander(f"🏛️ {s['title']}", expanded=False):
            st.write(s['desc'])
            st.info(f"**{t['eligibility']}:** {s['eligibility']}")

# PAGE 7: WEATHER FORECAST MODULE
elif menu == t["nav_weather"]:
    st.markdown(f"""
        <div class="hero-card">
            <h1 class="hero-title">{t['weather_hero_title']}</h1>
            <p class="hero-subtitle">{t['weather_hero_sub']}</p>
        </div>
    """, unsafe_allow_html=True)

    selected_region = st.selectbox(
        t["select_region"],
        list(PUNE_REGIONS.keys()),
        index=0
    )
    coords = PUNE_REGIONS[selected_region]
    
    weather = fetch_pune_weather(coords["lat"], coords["lon"])
    st.info(f"📍 **{selected_region.split('/')[0].strip()}** Co-ordinates: `{coords['lat']}° N, {coords['lon']}° E`")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(t["past15_avg_temp"], f"{round(weather['temp_past15_avg'], 1)} °C")
    with c2:
        st.metric(t["past15_rain"], f"{round(weather['rain_past15_sum'], 1)} mm")
    with c3:
        st.metric(t["forecast7_rain"], f"{round(weather['rain_forecast_sum'], 1)} mm")
    with c4:
        st.metric(t["soil_moisture"], f"{round(weather['soil_moisture'], 1)} %")

    st.markdown("<br>", unsafe_allow_html=True)
    
    df_w = weather['df']
    
    st.subheader(t["temp_trend"])
    st.line_chart(df_w.set_index('date')[['temp_mean']])
    
    st.subheader(t["rain_trend"])
    st.line_chart(df_w.set_index('date')[['precipitation']])

# PAGE 8: TASK 3 - SMART AI PREDICTIONS MODULE (REFACTORED SOIL SELECTION + WEATHER)
elif menu == t["nav_ai_pred"]:
    st.markdown(f"""
        <div class="hero-card">
            <h1 class="hero-title">{t['ai_hero_title']}</h1>
            <p class="hero-subtitle">{t['ai_hero_sub']}</p>
        </div>
    """, unsafe_allow_html=True)

    # Taluka GPS Selector
    selected_ai_region = st.selectbox(
        t["select_region"],
        list(PUNE_REGIONS.keys()),
        index=0,
        key="ai_region_select"
    )
    coords_ai = PUNE_REGIONS[selected_ai_region]
    
    # Fetch live weather data for chosen Taluka coordinates
    weather = fetch_pune_weather(coords_ai["lat"], coords_ai["lon"])
    model_irrigation, model_disease = train_xgboost_models()

    col_in, col_res = st.columns([1, 1])

    # Left Column: Soil Type Dropdown Selector (Replaces raw numeric N, P, K, pH inputs)
    with col_in:
        st.subheader(t["soil_type_label"])
        
        selected_soil_type = st.selectbox(
            "Select Soil Type / जमिनीचा प्रकार निवडा",
            list(SOIL_TYPES_MAPPING.keys()),
            index=0
        )
        soil_meta = SOIL_TYPES_MAPPING[selected_soil_type]
        st.caption(f"ℹ️ {soil_meta['desc']}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        soil_moisture_input = st.slider("Adjust Top-Soil Moisture / मातीतील ओलावा (%)", 10.0, 90.0, float(weather['soil_moisture']), step=1.0)
        
        # Display Mapped Baseline Parameters
        st.markdown(f"""
            <div style="background:#f1f5f9; padding:0.8rem 1.0rem; border-radius:0.5rem; font-size:0.85rem;">
                <b>Mapped Soil Agronomic Parameters:</b><br>
                • Nitrogen (N): <code>{soil_meta['soil_N']} kg/ha</code> | • Phosphorus (P): <code>{soil_meta['soil_P']} kg/ha</code><br>
                • Potassium (K): <code>{soil_meta['soil_K']} kg/ha</code> | • Soil pH: <code>{soil_meta['soil_pH']}</code>
            </div>
        """, unsafe_allow_html=True)

        run_pred = st.button(t["predict_btn"], type="primary", use_container_width=True)

    # Right Column: Live Feature Vector Construction & XGBoost Model Inference
    with col_res:
        st.subheader("📊 AI Inference Results")
        
        # Formulate feature vector X_live combining mapped soil params + live weather parameters
        X_live = pd.DataFrame([{
            'temp_past15_avg': weather['temp_past15_avg'],
            'rain_past15_sum': weather['rain_past15_sum'],
            'temp_forecast_avg': weather['temp_forecast_avg'],
            'rain_forecast_sum': weather['rain_forecast_sum'],
            'soil_moisture': soil_moisture_input,
            'soil_N': soil_meta['soil_N'],
            'soil_P': soil_meta['soil_P'],
            'soil_K': soil_meta['soil_K'],
            'soil_pH': soil_meta['soil_pH']
        }])
        
        pred_irrigation = model_irrigation.predict(X_live)[0]
        pred_disease = model_disease.predict(X_live)[0]
        
        st.markdown(f"#### {t['irrigation_res']}")
        if pred_irrigation == 2:
            st.error("🔴 **HIGH IRRIGATION NEEDED!** High temperature forecast and low soil moisture detected. Water your crops within 24 hours.")
        elif pred_irrigation == 1:
            st.warning("🟡 **MODERATE IRRIGATION RECOMMENDED.** Soil moisture is decreasing. Plan light irrigation in 2-3 days.")
        else:
            st.success("🟢 **LOW / NO IRRIGATION NEEDED.** Soil moisture and rain forecast are adequate for optimal crop growth.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"#### {t['disease_res']}")
        if pred_disease == 1:
            st.error("⚠️ **HIGH FUNGAL DISEASE RISK DETECTED!** High recent rainfall and humidity create ideal conditions for fungal pathogens. Spray Neem Oil 10,000 ppm (5ml/L) or copper fungicide immediately.")
        else:
            st.success("✅ **LOW DISEASE RISK.** Weather and soil conditions are clean and safe for crop foliage.")

# PAGE 9: KRISHI AI ASSISTANT MODULE (OFFLINE CHATBOT)
elif menu == t["nav_chat"]:
    st.markdown(f"""
        <div class="hero-card">
            <h1 class="hero-title">{t['chat_hero_title']}</h1>
            <p class="hero-subtitle">{t['chat_hero_sub']}</p>
        </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": t["chat_welcome"]}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_prompt := st.chat_input(t["chat_input_ph"]):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        prompt_lower = user_prompt.lower()
        
        # Rule-Based Keyword Matcher
        if any(k in prompt_lower for k in ["fertilizer", "urea", "dap", "khad", "खत", "उर्वरक", "npk", "potash"]):
            if lang == "मराठी":
                response = "🌱 **MPKV खत मार्गदर्शन:**\n\n1. **डी.ए.पी. (DAP) व पोटॅश:** पेरणीच्या वेळी मुख्य (बेसल) डोस म्हणून द्या.\n2. **युरिया:** पेरणीच्या वेळी सर्व युरिया देऊ नका. तो पिकाच्या वाढीच्या टप्प्यावर २-३ समान हप्त्यांमध्ये द्या.\n3. **झिंक सल्फेट:** डीएपी सोबत थेट मिसळू नका."
            elif lang == "हिंदी":
                response = "🌱 **उर्वरक सलाह:**\n\n1. **डीएपी और पोटाश:** बुवाई के समय मुख्य (बेसल) खुराक के रूप में प्रयोग करें।\n2. **यूरिया:** बुवाई के समय पूरा यूरिया न डालें। इसे 2-3 बराबर भागों में फसल की बढ़वार पर दें।\n3. **जिंक सल्फेट:** डीएपी के साथ सीधे न मिलाएं।"
            else:
                response = "🌱 **Fertilizer Advisory:**\n\n1. **DAP & Potash:** Apply full dose as basal application during sowing.\n2. **Urea:** Avoid applying all Urea at sowing. Split into 2-3 equal doses during crop growth stages.\n3. **Zinc Sulphate:** Do not mix directly with DAP."

        elif any(k in prompt_lower for k in ["water", "irrigation", "paani", "rain", "पानी", "पाणी", "सिंचाई", "सिंचन", "शेततळे"]):
            weather = fetch_pune_weather()
            soil_m = round(weather['soil_moisture'], 1)
            rain_f = round(weather['rain_forecast_sum'], 1)
            if lang == "मराठी":
                response = f"💧 **सिंचन सल्ला (सध्याचा ओलावा: {soil_m}%):**\n\n- पुढील ७ दिवसांत **{rain_f} mm** पावसाचा अंदाज आहे.\n- जर मातीतील ओलावा ४०% पेक्षा कमी असेल, तर हलके पाणी द्या."
            elif lang == "हिंदी":
                response = f"💧 **सिंचाई सलाह (वर्तमान नमी: {soil_m}%):**\n\n- अगले 7 दिनों में **{rain_f} मिमी** बारिश का अनुमान है।\n- यदि मिट्टी की नमी 40% से कम है, तो शाम के समय सिंचाई करें।"
            else:
                response = f"💧 **Watering Advisory (Current Soil Moisture: {soil_m}%):**\n\n- Next 7 days rain forecast is **{rain_f} mm**.\n- If soil moisture is below 40%, irrigate during early morning or late evening."

        elif any(k in prompt_lower for k in ["pest", "disease", "yellow", "keeda", "fungus", "कीट", "रोग", "किडा", "कीड", "इल्ली"]):
            if lang == "मराठी":
                response = "🐛 **कीड व रोग नियंत्रण उपाय:**\n\n1. **सेंद्रिय उपाय:** कडुलिंब तेल (Neem Oil 10,000 ppm) ५ मिली प्रति लिटर पाण्यात मिसळून फवारा.\n2. **रसशोषक कीड (मावा):** पिवळे चिकट सापळे (१५ सापळे/एकरी) लावा.\n3. **रासायनिक औषध:** खोड किड्यासाठी क्लोरँट्रानिलीप्रोल (Chlorantraniliprole 18.5% SC @ 60ml/एकर)."
            elif lang == "हिंदी":
                response = "🐛 **कीट एवं रोग नियंत्रण उपचार:**\n\n1. **जैविक उपाय:** नीम का तेल (Neem Oil 10,000 ppm) 5 मिली प्रति लीटर पानी में मिलाकर स्प्रे करें।\n2. **रस चूसक कीट:** पीले चिपचिपे जाल (15 ट्रैप/एकड़) लगाएं।\n3. **रासायनिक दवा:** तना छेदक के लिए क्लोरैंट्रानिलिप्रोल (60 मिली/एकड़) का प्रयोग करें।"
            else:
                response = "🐛 **Pest & Disease Treatment Advisory:**\n\n1. **Organic Remedy:** Spray Neem Oil 10,000 ppm (5ml per Litre water).\n2. **Sucking Pests (Whitefly/Aphids):** Install Yellow Sticky Traps (15 traps/acre).\n3. **Chemical Option:** For Stem Borer, use Chlorantraniliprole 18.5% SC @ 60ml/acre."

        elif any(k in prompt_lower for k in ["scheme", "money", "pm kisan", "yojana", "योजना", "पैसे", "पैसा", "कर्ज", "seva", "दाखला", "certificate"]):
            if lang == "मराठी":
                response = "🏛️ **शासकीय योजना व महा सेवा केंद्र माहिती:**\n\n1. **दाखले व कागदपत्रे:** उत्पन्न दाखला, डोमासाईल, जातीचा दाखला काढण्यासाठी भारती विद्यापीठ, पुणे येथील **आपले सरकार सेवा केंद्रात (अभिषेक खेडेकर: +91 9960870182)** संपर्क साधा.\n2. **पीएम-किसान:** वर्षाला ₹६,००० बँक खात्यात मिळतात."
            elif lang == "हिंदी":
                response = "🏛️ **सरकारी योजना व महा सेवा केंद्र जानकारी:**\n\n1. **प्रमाण पत्र व दस्तावेज:** आय प्रमाण पत्र, डोमिसाइल, जाति प्रमाण पत्र के लिए भारती विद्यापीठ, पुणे में **आपले सरकार सेवा केंद्र (अभिषेक खेडेकर: +91 9960870182)** पर संपर्क करें।\n2. **पीएम-किसान:** ₹6,000 प्रति वर्ष बैंक खाते में ट्रांसफर होते हैं।"
            else:
                response = "🏛️ **Govt Schemes & Maha Seva Kendra Info:**\n\n1. **Official Certificates:** For Income, Domicile, or Caste certificates, contact **Aaple Sarkar Seva Kendra (Abhishek Khedekar: +91 9960870182)** at Bharati Vidyapeeth, Pune.\n2. **PM-KISAN:** Provides ₹6,000 per year directly to bank accounts."

        else:
            if lang == "मराठी":
                response = "🌾 **कृषी AI सहाय्यक:**\n\nमी तुम्हाला खालील विषयांवर मार्गदर्शन करू शकतो:\n- 🧪 **खत सल्ला** (MPKV प्रमाण)\n- 💧 **सिंचन सल्ला** (पाणी व्यवस्थापन)\n- 🐛 **कीड व रोग उपचार** (सेंद्रिय व रासायनिक औषधे)\n- 🏛️ **शासकीय योजना व दाखले** (महा सेवा केंद्र, पीएम-किसान)"
            elif lang == "हिंदी":
                response = "🌾 **कृषि AI सहायक:**\n\nमैं आपको निम्नलिखित विषयों पर मार्गदर्शन दे सकता हूँ:\n- 🧪 **उर्वरक सलाह** (MPKV प्रयोग)\n- 💧 **सिंचाई सलाह** (पानी प्रबंधन)\n- 🐛 **कीट-रोग उपचार** (जैविक व रासायनिक दवाएं)\n- 🏛️ **सरकारी योजनाएं व प्रमाण पत्र** (महा सेवा केंद्र, पीएम-किसान)"
            else:
                response = "🌾 **Krishi AI Assistant:**\n\nI can help you with the following topics:\n- 🧪 **Fertilizer Advisory** (MPKV usage)\n- 💧 **Irrigation & Weather** (Watering timing)\n- 🐛 **Pest & Disease Remedies** (Neem oil & chemical controls)\n- 🏛️ **Govt. Schemes & Seva Kendra** (Official certificates & forms)"

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# PAGE 10: TASK 4 - CONTACT & SUPPORT MODULE (REFACTORED WITH 2 DISTINCT CARDS)
elif menu == t["nav_contact"]:
    st.markdown(f"""
        <div class="hero-card">
            <h1 class="hero-title">{t['contact_hero_title']}</h1>
            <p class="hero-subtitle">{t['contact_hero_sub']}</p>
        </div>
    """, unsafe_allow_html=True)

    col_dev, col_seva = st.columns(2)

    # Card 1: Website Developers Team Card
    with col_dev:
        st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 1rem; border: 1px solid #cbd5e1; height: 100%;">
                <h3 style="color: #15803d; margin-top: 0;">💻 Website Developers & Technical Support</h3>
                <p style="font-size: 0.9rem; color: #475569;">Reach out to our core engineering team for application issues or feature updates:</p>
                <hr style="margin: 0.8rem 0;">
                <div style="margin-bottom: 0.8rem;">
                    <b>👤 Mriganka Bhattacharya</b><br>
                    📞 Phone: <a href="tel:+917709503474">+91 7709503474</a><br>
                    ✉️ Email: <code>simba.outlook@gmail.com</code>
                </div>
                <div style="margin-bottom: 0.8rem;">
                    <b>👤 Krushna Chavan</b><br>
                    📞 Phone: <a href="tel:+919356992287">+91 9356992287</a><br>
                    ✉️ Email: <code>chavankrushna524@gmail.com</code>
                </div>
                <div style="margin-bottom: 0.8rem;">
                    <b>👤 Digambar Mahale</b><br>
                    📞 Phone: <a href="tel:+917820873047">+91 7820873047</a><br>
                    ✉️ Email: <code>mahaledigambar650@gmail.com</code>
                </div>
                <div>
                    <b>👤 Nayan Tripathy</b><br>
                    📞 Phone: <a href="tel:+917780943490">+91 7780943490</a><br>
                    ✉️ Email: <code>nayantripathi16@gmail.com</code>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Card 2: Government Schemes & Maha Seva Kendra Desk Card
    with col_seva:
        st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 1rem; border: 1px solid #cbd5e1; height: 100%;">
                <h3 style="color: #1e3a8a; margin-top: 0;">🏬 Government Schemes & Maha Seva Kendra Desk</h3>
                <p style="font-size: 0.9rem; color: #475569;">For official certificate processing, PM-KISAN, 7/12 land records, and online government scheme applications:</p>
                <hr style="margin: 0.8rem 0;">
                <div style="font-size: 1.0rem; margin-bottom: 0.8rem;">
                    <b>👤 Operator: Abhishek Khedekar</b><br>
                    <span style="color: #1e40af; font-weight: 600;">Authorized Center:</span> Aaple Sarkar Seva Kendra / CSIC Maha Seva Kendra
                </div>
                <div style="margin-bottom: 0.8rem;">
                    📞 <b>Direct Phone:</b> <a href="tel:+919960870182" style="font-size: 1.1rem; font-weight: bold; color: #1e40af;">+91 9960870182</a>
                </div>
                <div style="margin-bottom: 0.8rem;">
                    📍 <b>Center Address:</b><br>
                    Shop No. 3, Trimurti Chowk, Near Relax Medical,<br>
                    Bharati Vidyapeeth, Pune - 411043<br>
                    <i>(शॉप नं ३, त्रिमूर्ती चौक, रिलॅक्स मेडिकल जवळ, भारती विद्यापीठ, पुणे - ४११०४३)</i>
                </div>
                <div style="background: #f0fdf4; padding: 0.6rem 0.8rem; border-radius: 0.5rem; color: #166534; font-size: 0.85rem;">
                    <b>Working Hours:</b> Mon - Sat (9:00 AM to 8:00 PM)
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Farmer Query Form
    st.subheader(t["send_query"])
    with st.form("contact_form"):
        name = st.text_input(t["farmer_name"])
        phone = st.text_input(t["mobile_num"])
        district = st.text_input(t["district"])
        topic = st.selectbox(t["query_topic"], ["Government Schemes & Certificates (Maha Seva Kendra)", "Fertilizer Recommendation (MPKV)", "Pest & Disease Help", "Technical App Issue", "Other"])
        message = st.text_area(t["message"])
        submit = st.form_submit_button(t["submit_btn"], type="primary")

        if submit:
            if name and phone:
                st.success(t["success_msg"].format(name=name, topic=topic, phone=phone))
            else:
                st.error(t["error_msg"])

# ------------------------------------------------------------------------------
# SECTION 12: PERSISTENT FOOTER COMPONENT
# ------------------------------------------------------------------------------
st.markdown(f"""
    <div class="footer-container">
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 1.5rem;">
            <div>
                <div class="footer-heading">🌾 {t['app_title']}</div>
                <p style="font-size: 0.85rem; max-width: 320px; opacity: 0.9;">{t['footer_tagline']}</p>
            </div>
            <div>
                <div class="footer-heading">🏬 Maha Seva Kendra Desk</div>
                <p style="font-size: 0.85rem; margin: 0;">Abhishek Khedekar (+91 9960870182)</p>
                <p style="font-size: 0.85rem; margin: 0;">Bharati Vidyapeeth, Pune - 411043</p>
            </div>
            <div>
                <div class="footer-heading">💻 Developer Team</div>
                <p style="font-size: 0.85rem; margin: 0;">Mriganka (+91 7709503474) | Krushna (+91 9356992287)</p>
                <p style="font-size: 0.85rem; margin: 0;">Digambar (+91 7820873047) | Nayan (+91 7780943490)</p>
            </div>
        </div>
        <hr style="border-color: rgba(255,255,255,0.15); margin: 1.5rem 0 1rem 0;">
        <p style="text-align: center; font-size: 0.78rem; opacity: 0.8;">© 2026 Krishi Mitra. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)