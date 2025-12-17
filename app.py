
import os
import sys
import pickle
import pandas as pd
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import open_meteo_client
from datetime import timedelta
import plotly
import plotly.graph_objects as go
import plotly.utils


# Initialize Flask
app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_DIR = 'models'
DATA_PATH = 'data/historical_data.csv'
PLOT_DIR = 'static/plots'

# --- GLOBAL ARTIFACTS ---
model = None
metadata = None
residual_std = 0.0

def load_artifacts():
    """Load ML artifacts safely."""
    global model, metadata, residual_std
    
    print("📦 Loading artifacts...")
    try:
        with open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        
        with open(os.path.join(MODEL_DIR, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
            
        res_path = os.path.join(MODEL_DIR, 'residual_std.pkl')
        if os.path.exists(res_path):
            with open(res_path, 'rb') as f:
                residual_std = pickle.load(f)
        
        print("✅ Artifacts loaded successfully.")
        return True
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        return False


def get_air_quality_category(pm25):
    """Determine category and color based on PM2.5 value."""
    if pm25 <= 50:
        return "Baik", "success"
    elif pm25 <= 100:
        return "Sedang", "warning"
    elif pm25 <= 150:
        return "Tidak Sehat", "orange"  # Custom class
    else:
        return "Sangat Tidak Sehat", "danger"

def calculate_risk_assessment(prediction, ci_width, last_actual):
    """
    Compute qualitative risk level and recommendation.
    
    Rules:
    - Base Level primarily on prediction value.
    - Adjusted if Trend is rapidly increasing (> 10 ug increase).
    """
    trend = prediction - last_actual
    
    # Base Level
    if prediction >= 150:
        level = "Kritis"
        color = "danger"
    elif prediction >= 100:
        level = "Tinggi"
        color = "orange"
    elif prediction >= 50:
        level = "Sedang"
        color = "warning"
    else:
        level = "Rendah"
        color = "success"
        
    # Recommendation Map
    recommendations = {
        "Kritis": "Hindari semua aktivitas di luar ruangan. Tutup jendela dan gunakan pembersih udara.",
        "Tinggi": "Kurangi aktivitas di luar ruangan. Kelompok rentan sebaiknya tetap di dalam ruangan. Gunakan masker jika harus keluar.",
        "Sedang": "Kelompok sensitif (lansia, anak-anak) sebaiknya membatasi aktivitas fisik di luar ruangan.",
        "Rendah": "Kualitas udara baik. Nikmati aktivitas di luar ruangan dengan normal."
    }
    
    rec_text = recommendations[level]
    
    # Confidence Indicator
    # Ratio of residual_std to prediction
    # Avoid division by zero
    safe_pred = max(prediction, 1.0)
    conf_ratio = (residual_std / safe_pred)
    
    if conf_ratio < 0.1:
        conf_level = "Tinggi"
        conf_desc = "Model sangat yakin dengan prakiraan ini."
        conf_badge = "success"
    elif conf_ratio < 0.25:
        conf_level = "Sedang"
        conf_desc = "Model memiliki keyakinan yang cukup."
        conf_badge = "primary"
    else:
        conf_level = "Rendah"
        conf_desc = "Prakiraan memiliki ketidakpastian tinggi. Harap waspada."
        conf_badge = "secondary"

    return {
        'level': level,
        'color': color,
        'recommendation': rec_text,
        'trend_val': trend,
        'trend_dir': "Rising" if trend > 5 else ("Falling" if trend < -5 else "Stable"),
        'conf_level': conf_level,
        'conf_desc': conf_desc,
        'conf_badge': conf_badge,
        'ci_width': ci_width
    }

def plot_forecast(history_df, forecasts):
    """Generate Matplotlib plot for forecast."""
    plt.figure(figsize=(10, 5))
    
    # Plot last 30 days history
    hist = history_df.iloc[-30:]
    plt.plot(hist['date'], hist['pm_duakomalima'], label='Historis', color='#2c3e50', linewidth=2)
    
    # Plot forecasts
    f_dates = [f['date_obj'] for f in forecasts]
    f_values = [f['value'] for f in forecasts]
    
    # Connect last historical to first forecast (dashed line)
    plt.plot([hist['date'].iloc[-1], f_dates[0]], 
             [hist['pm_duakomalima'].iloc[-1], f_values[0]], 
             linestyle='--', color='#e67e22', alpha=0.7)
    
    # Forecast points
    plt.plot(f_dates, f_values, linestyle='--', marker='o', color='#e67e22', label='Prakiraan', linewidth=2, markersize=8)
    
    # Styling
    plt.title("Tren dan Prakiraan PM2.5", fontsize=14, pad=15)
    plt.xlabel("Tanggal")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Format dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(os.path.join(PLOT_DIR, 'forecast_plot.png'), dpi=100)
    plt.close()

def generate_plotly_json(history_df, forecasts):
    """Generate Plotly JSON for interactive chart using Graph Objects."""
    # 1. Historical Data (Last 30 days)
    hist = history_df.iloc[-30:]
    
    fig = go.Figure()
    
    # Add Historical Trace
    fig.add_trace(go.Scatter(
        x=hist['date'],
        y=hist['pm_duakomalima'],
        mode='lines',
        name='Historis',
        line=dict(color='#2c3e50', width=3),
        hovertemplate='%{x|%d %b %Y}<br>PM2.5: %{y:.1f} µg/m³<extra></extra>'
    ))
    
    # 2. Forecast Data
    f_dates = [f['date_obj'] for f in forecasts]
    f_values = [f['value'] for f in forecasts]
    
    # Connect lines: Prepend last historical point to forecast
    connect_x = [hist['date'].iloc[-1]] + f_dates
    connect_y = [hist['pm_duakomalima'].iloc[-1]] + f_values
    
    # Add Forecast Trace
    fig.add_trace(go.Scatter(
        x=connect_x,
        y=connect_y,
        mode='lines+markers',
        name='Prakiraan',
        line=dict(color='#e67e22', width=3, dash='dash'),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='%{x|%d %b %Y}<br>Prediksi: %{y:.1f} µg/m³<extra></extra>'
    ))
    
    # 3. Add Confidence Intervals (if available in forecast objects)
    # Extract upper and lower bounds if they exist in forecasts
    if 'upper' in forecasts[0] and 'lower' in forecasts[0]:
        upper_bound = [hist['pm_duakomalima'].iloc[-1]] + [f['upper'] for f in forecasts]
        lower_bound = [hist['pm_duakomalima'].iloc[-1]] + [f['lower'] for f in forecasts]
        
        # Upper Bound (transparent line)
        fig.add_trace(go.Scatter(
            x=connect_x,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Lower Bound (filled to upper)
        fig.add_trace(go.Scatter(
            x=connect_x,
            y=lower_bound,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(230, 126, 34, 0.1)', # Transparent orange
            line=dict(width=0),
            name='Interval Kepercayaan (95%)',
            hoverinfo='skip'
        ))

    # 4. Improved Layout
    fig.update_layout(
        title=dict(
            text='<b>Tren Historis & Prakiraan Kualitas Udara (PM2.5)</b>',
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=18, family='Arial')
        ),
        xaxis=dict(
            title='Tanggal',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            tickformat='%d %b'
        ),
        yaxis=dict(
            title='Konsentrasi PM2.5 (µg/m³)',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=False
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode='x unified'
    )
    
    # Add shapes/zones for air quality categories (Optional visual guide)
    # Green (0-50), Yellow (50-100), Orange (100-150), Red (>150)
    # Using semi-transparent rectangles in background could be nice but might clutter.
    # We will stick to clean lines for now as per "friend's" implied simple import.
    
    return json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)

def plot_insights(history_df):
    """Generate insight plots."""
    # 1. Recent Actual vs Predicted (Simulated on history)
    # Since we don't have predictions for history in a convenient way without re-running,
    # we will run the model on the last 50 points of history.
    
    subset = history_df.iloc[-50:].copy()
    
    # Prepare features for subset
    # Need to generate time features
    subset['year'] = subset['date'].dt.year
    subset['day_of_week'] = subset['date'].dt.dayofweek
    subset['month'] = subset['date'].dt.month
    subset['day_of_year'] = subset['date'].dt.dayofyear
    subset['quarter'] = subset['date'].dt.quarter
    subset['is_weekend'] = subset['day_of_week'].isin([5, 6]).astype(int)
    subset['season'] = (subset['month'] % 12 + 3) // 3
    subset['is_rainy_season'] = ((subset['month'] >= 10) | (subset['month'] <= 3)).astype(int)
    
    # Get model features
    try:
         cols = model.feature_names_in_
         X = subset[cols]
         pred = model.predict(X)
         
         # Plot Actual vs Pred
         plt.figure(figsize=(10, 5))
         plt.plot(subset['date'], subset['pm_duakomalima'], label='Aktual', color='#2c3e50')
         plt.plot(subset['date'], pred, label='Prediksi Model', color='#27ae60', linestyle='--')
         plt.fill_between(subset['date'], pred - residual_std, pred + residual_std, color='#27ae60', alpha=0.1, label='Interval Kepercayaan')
         
         plt.title("Validasi Model: Aktual vs Prediksi (Data Terkini)", fontsize=14)
         plt.legend()
         plt.grid(True, alpha=0.3)
         plt.tight_layout()
         plt.savefig(os.path.join(PLOT_DIR, 'actual_vs_pred.png'), dpi=100)
         plt.close()
         
         # Plot Residuals
         residuals = subset['pm_duakomalima'] - pred
         plt.figure(figsize=(8, 5))
         plt.hist(residuals, bins=15, color='#3498db', edgecolor='white', alpha=0.7)
         plt.axvline(0, color='red', linestyle='dashed')
         plt.title("Distribusi Residual (Analisis Eror)", fontsize=14)
         plt.xlabel("Eror Prediksi")
         plt.ylabel("Frekuensi")
         plt.grid(True, alpha=0.3)
         plt.tight_layout()
         plt.savefig(os.path.join(PLOT_DIR, 'residuals.png'), dpi=100)
         plt.close()
         
    except Exception as e:
        print(f"Error generating insights: {e}")

def preprocess_and_load_data():
    """Load historical data and prepare it."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    # Create Date column
    # Assuming periode_data is YYYYMM (int or str) and tanggal is Day (int)
    # We need to handle potential format issues.
    
    def parse_date(row):
        try:
            p_str = str(row['periode_data'])
            year = int(p_str[:4])
            month = int(p_str[4:6])
            day = int(row['tanggal'])
            return pd.Timestamp(year=year, month=month, day=day)
        except:
            return pd.NaT

    df['date'] = df.apply(parse_date, axis=1)
    df = df.dropna(subset=['date']).sort_values('date')
    
    # Ensure numeric for pollutants
    pollutants = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 
                  'karbon_monoksida', 'ozon', 'nitrogen_dioksida', 'max']
    
    for col in pollutants:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Forward fill to handle missing values for smoothness in plots/context
    df[pollutants] = df[pollutants].ffill()

    # Aggregate by Date (Mean of all stations)
    # This fixes the issue where multiple stations create duplicate date entries
    df = df.groupby('date')[pollutants].mean().reset_index()
    df = df.sort_values('date')
    
    return df

def generate_forecast_features(last_df, days_ahead, scaling_factor=1.0):
    """
    Generate features for forecast with optional scaling for What-if analysis.
    scaling_factor: Multiplier for pollutant inputs (defaults to 1.0)
    """
    last_date = last_df['date'].iloc[-1]
    target_date = last_date + timedelta(days=days_ahead)
    
    # Calculate pollutant context (Last 7 days mean)
    context_window = last_df.iloc[-7:]
    pollutant_means = context_window[['pm_sepuluh', 'sulfur_dioksida', 'karbon_monoksida', 
                                      'ozon', 'nitrogen_dioksida', 'max']].mean()
    
    # Apply scaling
    pollutant_means = pollutant_means * scaling_factor
    
    # Construct row
    row = {}
    
    # Pollutants (Persistence/Mean Strategy)
    row['pm_sepuluh'] =             pollutant_means['pm_sepuluh']
    row['sulfur_dioksida'] =        pollutant_means['sulfur_dioksida']
    row['karbon_monoksida'] =       pollutant_means['karbon_monoksida']
    row['ozon'] =                   pollutant_means['ozon']
    row['nitrogen_dioksida'] =      pollutant_means['nitrogen_dioksida']
    row['max'] =                    pollutant_means['max']
    
    # Time Columns required by model
    # ['periode_data' 'bulan' 'year' 'day_of_week' 'month' 'day_of_year' 
    #  'quarter' 'is_weekend' 'season' 'is_rainy_season']
    
    row['periode_data'] = int(target_date.strftime('%Y%m'))
    row['bulan'] = target_date.month  # Raw 'bulan'
    row['year'] = target_date.year
    row['day_of_week'] = target_date.dayofweek
    row['month'] = target_date.month  # Feature 'month'
    row['day_of_year'] = target_date.dayofyear
    row['quarter'] = target_date.quarter
    row['is_weekend'] = 1 if target_date.dayofweek >= 5 else 0
    row['season'] = (target_date.month % 12 + 3) // 3
    row['is_rainy_season'] = 1 if (target_date.month >= 10 or target_date.month <= 3) else 0
    
    return pd.DataFrame([row]), target_date



# --- ROUTES ---

@app.route('/')
# --- ROUTES ---

@app.route('/')
def index():
    if model is None:
        if not load_artifacts():
            return "Error loading model artifacts. Check console."
            
    try:
        df = preprocess_and_load_data()
        
        # 1. Main Baseline Forecasts and Risk
        baseline_forecasts = []
        forecast_days = [1, 3, 7]
        last_actual = df['pm_duakomalima'].iloc[-1]
        
        for d in forecast_days:
            X_row, date_obj = generate_forecast_features(df, d, scaling_factor=1.0)
            
            if hasattr(model, 'feature_names_in_'):
                X_row = X_row[model.feature_names_in_]
                
            pred_val = float(model.predict(X_row)[0])
            cat, badge = get_air_quality_category(pred_val)
            
            baseline_forecasts.append({
                'label': f"H+{d}",
                'date': date_obj.strftime('%d %b %Y'),
                'date_obj': date_obj,
                'value': pred_val,
                'category': cat,
                'badge_color': badge,
                'lower': pred_val - residual_std,
                'upper': pred_val + residual_std
            })
            
        # Risk (H+1)
        h1_forecast = baseline_forecasts[0]
        risk_data = calculate_risk_assessment(
            prediction=h1_forecast['value'],
            ci_width=residual_std * 2,
            last_actual=last_actual
        )
        

        # 3. Scenario Simulator (New Feature)
        # Handle User Input safely
        scenario_input = request.args.get('scenario', 'normal')
        
        # Map input to scaling factor
        scenario_map = {
            'normal': {'scale': 1.0, 'label': 'Kondisi Normal'},
            'increase': {'scale': 1.1, 'label': 'Kenaikan Polusi (+10%)'},
            'decrease': {'scale': 0.9, 'label': 'Penurunan Polusi (-10%)'}
        }
        
        selected_sc = scenario_map.get(scenario_input, scenario_map['normal'])
        
        simulation_results = []
        for i, d in enumerate(forecast_days):
            # Calculate simulated forecast
            X_sim, _ = generate_forecast_features(df, d, scaling_factor=selected_sc['scale'])
            
            if hasattr(model, 'feature_names_in_'):
                X_sim = X_sim[model.feature_names_in_]
            
            sim_pred = float(model.predict(X_sim)[0])
            baseline_pred = baseline_forecasts[i]['value']
            delta = sim_pred - baseline_pred
            
            simulation_results.append({
                'day': f"H+{d}",
                'baseline': baseline_pred,
                'simulated': sim_pred,
                'delta': delta
            })
            
        simulation_data = {
            'active_scenario': scenario_input,
            'label': selected_sc['label'],
            'results': simulation_results,
            'safeguard': "Skenario ini adalah simulasi terbatas berdasarkan pola historis dan tidak merepresentasikan kondisi masa depan yang pasti."
        }
            
        # Generate plot (Baseline only to avoid clutter)
        plot_forecast(df, baseline_forecasts) # Legecy matplotlib restored for static image
        
        # Generate Plotly JSON
        plot_json_str = generate_plotly_json(df, baseline_forecasts)
        
        return render_template('index.html', 
                             forecasts=baseline_forecasts, 
                             risk=risk_data, 
                             sim=simulation_data,
                             plot_json=plot_json_str)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"System Error: {str(e)}"

@app.route('/insight')
def insight():
    if model is None:
        load_artifacts()
    
    try:
        df = preprocess_and_load_data()
        plot_insights(df)
        return render_template('insight.html', metadata=metadata)
    except Exception as e:
        return f"Insight Error: {str(e)}"

@app.route('/manual', methods=['GET', 'POST'])
def manual_predict():
    if model is None:
        load_artifacts()
        
    result = None
    error = None
    
    if request.method == 'POST':
        try:
            # 1. Get Form Data
            date_str = request.form.get('date')
            pm10 = float(request.form.get('pm_sepuluh'))
            so2 = float(request.form.get('sulfur_dioksida'))
            co = float(request.form.get('karbon_monoksida'))
            o3 = float(request.form.get('ozon'))
            no2 = float(request.form.get('nitrogen_dioksida'))
            max_val = float(request.form.get('max'))
            
            # 2. Process Date & Generate Features
            target_date = pd.to_datetime(date_str)
            
            row = {}
            # Pollutants
            row['pm_sepuluh'] = pm10
            row['sulfur_dioksida'] = so2
            row['karbon_monoksida'] = co
            row['ozon'] = o3
            row['nitrogen_dioksida'] = no2
            row['max'] = max_val
            
            # Time Features
            # ['periode_data' 'bulan' 'year' 'day_of_week' 'month' 'day_of_year' 
            #  'quarter' 'is_weekend' 'season' 'is_rainy_season']
            row['periode_data'] = int(target_date.strftime('%Y%m'))
            row['bulan'] = target_date.month
            row['year'] = target_date.year
            row['day_of_week'] = target_date.dayofweek
            row['month'] = target_date.month
            row['day_of_year'] = target_date.dayofyear
            row['quarter'] = target_date.quarter
            row['is_weekend'] = 1 if target_date.dayofweek >= 5 else 0
            row['season'] = (target_date.month % 12 + 3) // 3
            row['is_rainy_season'] = 1 if (target_date.month >= 10 or target_date.month <= 3) else 0
            
            # Create DataFrame
            X_input = pd.DataFrame([row])
            
            # Reorder columns to match model
            if hasattr(model, 'feature_names_in_'):
                X_input = X_input[model.feature_names_in_]
                
            # 3. Predict
            pred_val = float(model.predict(X_input)[0])
            
            # 4. Interpret Result
            cat, badge = get_air_quality_category(pred_val)
            
            # Get last actual for trend (optional, just use pred vs 0 or some default if unknown)
            # For manual prediction, we might not have 'last_actual' context easily available 
            # without querying the DB/CSV. We'll use the input PM10 as a proxy or 0 for trend calc 
            # if we want to reuse the function, or simplified logic.
            # Let's load historical to be safe for 'last_actual' context if possible, 
            # OR just calculate risk based on single point.
            
            risk_res = calculate_risk_assessment(pred_val, residual_std*2, pred_val) # Trend = 0
            
            result = {
                'value': pred_val,
                'category': cat,
                'badge_color': badge,
                'recommendation': risk_res['recommendation']
            }
            
        except Exception as e:
            error = f"Gagal melakukan prediksi: {str(e)}"
            
    return render_template('manual_predict.html', result=result, error=error)

@app.route('/api/live-data')
def get_live_data():
    """
    Fetch live air quality data for Jakarta using Open-Meteo Client.
    """
    data = open_meteo_client.fetch_jakarta_air_quality()
    
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'Gagal mengambil data dari Open-Meteo'}), 500

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    print("🚀 Starting Air Quality Warning System...")
    load_artifacts()
    
    # Get port from environment variable (default to 5000 for local dev)
    port = int(os.environ.get('PORT', 5000))
    
    # Debug mode should be False in production
    # In a real app, use an env var like FLASK_ENV or DEBUG
    is_debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    app.run(host='0.0.0.0', port=port, debug=is_debug)
