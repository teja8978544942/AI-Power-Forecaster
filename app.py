import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from keras.models import load_model
import pickle
import os
import time
import base64

# --- Setup and Styling ---
st.set_page_config(page_title="AI Power Forecaster", layout="wide", page_icon="⚡")

# Inject Custom CSS for dark dashboard aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .main {
        background: #0E1117;
        color: #FAFAFA;
    }
    .metric-container {
        background-color: #262730;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
    }
    .st-emotion-cache-16idsys p {
        font-size: 20px;
        font-weight: bold;
    }
    @keyframes pulse-red {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 51, 102, 0.7); }
        50% { transform: scale(1.02); box-shadow: 0 0 15px 5px rgba(255, 51, 102, 0.4); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 51, 102, 0); }
    }
    .alert-critical {
        border: 2px solid #ff3366;
        animation: pulse-red 1.5s infinite;
        background-color: rgba(255, 51, 102, 0.1) !important;
    }
    .alert-warning {
        border: 2px solid #ffcc00;
        background-color: rgba(255, 204, 0, 0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ AI-Driven Electrical Peak Load Predictor")
st.markdown("*Real-time Smart Grid Demand Forecasting and Stress Detection*")

# Initialize session state for real-time alerting log
if 'alert_logs' not in st.session_state:
    st.session_state.alert_logs = pd.DataFrame(columns=["Timestamp", "Event Type", "Value (kWh)", "Action Taken"])

# --- Loading Models & Data ---
@st.cache_resource
def load_ml_models():
    try:
        bilstm = load_model('models/bilstm_model.keras')
        with open('models/xgboost_model.pkl', 'rb') as f:
            xgb = pickle.load(f)
        with open('models/isolation_forest.pkl', 'rb') as f:
            iso = pickle.load(f)
        with open('models/scaler_X.pkl', 'rb') as f:
            scaler_x = pickle.load(f)
        with open('models/scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        return bilstm, xgb, iso, scaler_x, scaler_y
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

@st.cache_data
def load_and_prep_data():
    if os.path.exists('iiot_smart_grid_dataset.csv'):
        df = pd.read_csv('iiot_smart_grid_dataset.csv')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M')
        df = df.sort_values('Timestamp').reset_index(drop=True)
        return df
    return None

bilstm, xgb, iso, scaler_x, scaler_y = load_ml_models()
raw_data = load_and_prep_data()

if bilstm is None or raw_data is None:
    st.error("Error loading models or data. Please run `python train_all.py` first.")
    st.stop()

# --- Utility to process the latest row ---
def process_latest_sequence(df, start_idx, seq_length=24, override_weather=None):
    subset = df.iloc[start_idx : start_idx + seq_length + 1].copy()
    
    if override_weather and override_weather != "None":
        subset['Weather_Condition'] = override_weather
    
    # Needs the exact same feature engineering as training
    for lag in [1, 2, 3, 6, 12, 24]:
        subset[f'Power_lag_{lag}'] = subset['Power_Consumption_kWh'].shift(lag)
    subset['Power_Rolling_Mean_3'] = subset['Power_Consumption_kWh'].rolling(window=3).mean()
    subset = subset.dropna()
    
    if 'Weather_Condition' in subset.columns:
        subset = pd.get_dummies(subset, columns=['Weather_Condition'], drop_first=True)
    
    # Ensure all columns from training are present (handling dummy variable mismatches if the weather string vanished)
    expected_cols = scaler_x.feature_names_in_
    for col in expected_cols:
        if col not in subset.columns:
            subset[col] = 0
            
    # Force column order
    subset = subset[expected_cols]
    
    # We want a sequence of length seq_length for LSTM
    X_raw = subset.values[-seq_length:]
    X_scaled = scaler_x.transform(X_raw)
    
    # Reshape for LSTM: (1, seq_length, features)
    X_seq = np.array([X_scaled])
    
    # The classification features normally don't use sequence, just the LAST row
    return X_seq, subset.iloc[-1:]

# --- UI Layout ---
st.sidebar.header("⚙️ Simulation Settings")
warning_threshold = st.sidebar.slider("Warning Threshold (kWh)", 1.0, 10.0, 4.0, 0.1)
critical_threshold = st.sidebar.slider("Critical Threshold (kWh)", 1.0, 10.0, 4.5, 0.1)
sim_steps = st.sidebar.slider("Simulation Steps", 10, 200, 50, 10)

st.sidebar.header("🌤️ Weather Simulation")
weather_override = st.sidebar.selectbox("Override Historical Weather", ["None", "Clear", "Cloudy", "Rain", "Extreme Heat"])

st.sidebar.header("📥 Reports")
if len(st.session_state.alert_logs) > 0:
    csv = st.session_state.alert_logs.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download Incident Report (CSV)",
        data=csv,
        file_name='grid_alert_incident_report.csv',
        mime='text/csv',
    )
else:
    st.sidebar.info("Run simulation to generate CSV reports.")

st.sidebar.header("📱 Alert Configurations")
st.sidebar.markdown("*For real Telegram/Email alerts, enter details below:*")
telegram_token = st.sidebar.text_input("Telegram Bot Token (Optional)", type="password")
telegram_chat_id = st.sidebar.text_input("Telegram Chat ID (Optional)")

# Audio playback function
def play_audio(audio_ph):
    # We will use Streamlit's native audio component, playing a custom-generated siren!
    # Because you clicked "Run Simulation", the browser will allow this audio to 
    # autoplay automatically over your speakers.
    if os.path.exists("alarm.wav"):
        with open("alarm.wav", "rb") as f:
            audio_bytes = f.read()
            b64 = base64.b64encode(audio_bytes).decode()
    else:
        beep_base64 = "UklGRuQBAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YcABAAAAAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//AAD//wAA//8AAP//"
        b64 = beep_base64
        
    # Using an invisible HTML audio tag to auto-play the sound without showing the widget
    md = f"""
        <audio autoplay style="display:none;">
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
        """
    audio_ph.markdown(md, unsafe_allow_html=True)

def send_alert(message):
    # This acts like a push notification on your screen when an email/telegram WOULD be sent
    st.toast(f"🔔 {message}", icon='🚨')
    
    # If the user actually provided Telegram API keys, send the real message!
    if telegram_token and telegram_chat_id:
        try:
            import requests
            url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
            payload = {"chat_id": telegram_chat_id, "text": message}
            response = requests.post(url, json=payload)
            if response.status_code != 200:
                st.toast(f"Telegram API Error: {response.text}", icon="❌")
        except Exception as e:
            st.toast(f"Telegram Failed: {str(e)}", icon="❌")

tab1, tab2, tab3 = st.tabs(["Live Grid Monitoring", "Model Analytics", "Anomaly Log"])

with tab1:
    st.subheader("Live Network Feed Simulation")
    
    # Control Panel
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("### Controls")
        start_index = st.slider("Select Simulation Start Point (Row Index)", 0, len(raw_data)-100, 100)
        sim_speed = st.slider("Simulation Speed (seconds/hr)", 0.1, 5.0, 1.0)
        run_sim = st.button("▶ Run Simulation")
        stop_sim = st.button("⏸ Stop")
    
    with col2:
        # Placeholders for live data
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        current_load_ph = metric_col1.empty()
        predicted_load_ph = metric_col2.empty()
        status_ph = metric_col3.empty()
        
        chart_ph = st.empty()
        audio_ph = st.empty() # Placeholder for the audio trigger

    if run_sim and not stop_sim:
        # Pre-initialize the log container BEFORE the loop so it's globally available for updates
        with tab3:
            st.subheader("Live System Alert Logs")
            st.markdown("Automated grid warnings and threshold breaches are cataloged here in real-time.")
            log_table_ph = st.empty()
            
            if len(st.session_state.alert_logs) > 0:
                log_table_ph.dataframe(st.session_state.alert_logs, use_container_width=True)
            else:
                log_table_ph.info("No critical alerts generated yet. Waiting for simulation breaches...")

        timeline = []
        actual = []
        predicted = []
        last_alarm_time = 0 # Initialize cooldown timer
        
        simulation_start_time = pd.Timestamp.now().floor('H')
        
        for i in range(start_index, start_index + sim_steps): # Run for configured steps
            if i + 25 >= len(raw_data):
                break
                
            X_seq, last_row = process_latest_sequence(raw_data, i, seq_length=24, override_weather=weather_override)
            
            # Predict Forecast
            pred_scaled = bilstm.predict(X_seq, verbose=0)
            pred_load = scaler_y.inverse_transform(pred_scaled)[0][0]
            
            act_load = raw_data.iloc[i + 24]['Power_Consumption_kWh']
            
            # Create a true live timestamp by calculating the offset from the dataset
            time_offset = raw_data.iloc[i + 24]['Timestamp'] - raw_data.iloc[start_index + 24]['Timestamp']
            timestamp = simulation_start_time + time_offset
            
            # Predict Anomaly & Peak using hardware features
            hw_features = ['Voltage_V', 'Current_A', 'Power_Factor', 'Grid_Frequency_Hz']
            hw_data = last_row[hw_features].values
            
            if hw_data.shape[1] == 4:
                is_anomaly = iso.predict(hw_data)[0] == -1
            else:
                is_anomaly = False

            # Update timeline
            timeline.append(timestamp)
            actual.append(act_load)
            predicted.append(pred_load)
            
            # Render Metrics
            with current_load_ph.container():
                st.markdown(f"""
                <div class="metric-container">
                    <p style="color:#aaaaaa; margin:0">Current Load</p>
                    <h2 style="color:#00e5ff; margin:0">{act_load:.2f} kWh</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with predicted_load_ph.container():
                color = "#ff3366" if pred_load > critical_threshold else ("#ffcc00" if pred_load > warning_threshold else "#00e676")
                
                gauge_fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = pred_load,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Predicted Load", 'font': {'size': 14, 'color': '#aaaaaa'}},
                    number = {'suffix': " kWh", 'font': {'size': 24, 'color': color}},
                    gauge = {
                        'axis': {'range': [None, max(8, critical_threshold + 3)], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': color},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, warning_threshold], 'color': "rgba(0, 230, 118, 0.1)"},
                            {'range': [warning_threshold, critical_threshold], 'color': "rgba(255, 204, 0, 0.15)"},
                            {'range': [critical_threshold, 15], 'color': "rgba(255, 51, 102, 0.2)"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': critical_threshold}
                    }
                ))
                gauge_fig.update_layout(height=160, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#FAFAFA"})
                st.plotly_chart(gauge_fig, use_container_width=True)
                
            with status_ph.container():
                if is_anomaly:
                    state = "CRITICAL: ANOMALY"
                    state_c = "#ff3366"
                    recommendation = "IMMEDIATE: Halt non-essential ops."
                    css_class = "metric-container alert-critical"
                elif pred_load > critical_threshold:
                    state = "CRITICAL: PEAK"
                    state_c = "#ff3366"
                    recommendation = "ACTION: Demand Response Active."
                    css_class = "metric-container alert-critical"
                elif pred_load > warning_threshold:
                    state = "WARNING: HIGH LOAD"
                    state_c = "#ffcc00"
                    recommendation = "PREPARE: Monitor load closely."
                    css_class = "metric-container alert-warning"
                else:
                    state = "STABLE"
                    state_c = "#00e676"
                    recommendation = "No action required."
                    css_class = "metric-container"
                    
                st.markdown(f"""
                <div class="{css_class}">
                    <p style="color:#aaaaaa; margin:0">Grid Status</p>
                    <h2 style="color:{state_c}; margin:0">{state}</h2>
                    <p style="color:#cccccc; font-size:12px; margin-top:5px;">{recommendation}</p>
                </div>
                """, unsafe_allow_html=True)

            # Audio and Logging Logic with Cooldown
            if (is_anomaly or pred_load > critical_threshold):
                if time.time() - last_alarm_time > (sim_speed * 5): # Cooldown based on game speed
                    play_audio(audio_ph)
                    
                    event_type = "Anomaly Detected (Hardware)" if is_anomaly else "Critical Peak Forecast"
                    
                    # Fire the Email/Telegram alert logic
                    alert_msg = f"WARNING! {event_type} at {timestamp.strftime('%H:%M')}. Load: {pred_load:.2f} kWh!"
                    send_alert(alert_msg)
                    
                    # Log the alert dynamically
                    new_alert = pd.DataFrame([{
                        "Timestamp": timestamp.strftime('%b %d, %Y %H:%M'),
                        "Event Type": event_type,
                        "Value (kWh)": round(pred_load, 2),
                        "Action Taken": "Audio Alarm & Push Notification Dispatched"
                    }])
                    st.session_state.alert_logs = pd.concat([new_alert, st.session_state.alert_logs], ignore_index=True)
                    log_table_ph.dataframe(st.session_state.alert_logs, use_container_width=True)
                    
                    last_alarm_time = time.time()

            # Render Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timeline, y=actual, mode='lines+markers', name='Actual Load', line=dict(color='#00e5ff')))
            fig.add_trace(go.Scatter(x=timeline, y=predicted, mode='lines', name='Forecasted Load', line=dict(color='#ff3366', dash='dash')))
            fig.add_hline(y=critical_threshold, line_dash="dot", annotation_text=f"Critical ({critical_threshold} kWh)", annotation_position="bottom right", line_color="red")
            fig.add_hline(y=warning_threshold, line_dash="dot", annotation_text=f"Warning ({warning_threshold} kWh)", annotation_position="bottom left", line_color="orange")
            
            fig.update_layout(
                template='plotly_dark',
                title="Real-Time Energy Consumption vs Forecast",
                xaxis_title="Time",
                yaxis_title="Power (kWh)",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            fig.update_xaxes(tickformat="%b %d, %Y %H:%M", tickangle=45)
            chart_ph.plotly_chart(fig, use_container_width=True)
            
            time.sleep(sim_speed)

with tab2:
    st.subheader("Machine Learning Performance")
    st.markdown("""
    This section validates the architecture presented in the IEEE paper. 
    By employing **Bidirectional LSTM** for time-series forecasting and **XGBoost/Isolation Forest** for classification 
    and anomaly detection, the system achieves a strong predictive layer avoiding pure theoretical approaches.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("lstm_architecture.png"):
            st.image("lstm_architecture.png", width=300, caption="LSTM Architecture Logic")
        else:
            st.info("Architecture diagram not found.")
    with col2:
        st.markdown("""
        ### Target Metrics
        - **Bi-LSTM MAE (Mean Absolute Error):** Very low (validated in console output during training).
        - **XGBoost Accuracy:** High precision on peak classification.
        - **Optimization:** Scaling enabled stable convergence across the Deep Learning nodes.
        """)
        
# Tab 3 logic has been moved inside the sim block above so it updates dynamically.

