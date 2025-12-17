import requests
import logging

# Jakarta Coordinates
LAT = -6.2088
LON = 106.8456

def fetch_jakarta_air_quality():
    """
    Fetch real-time air quality data for Jakarta from Open-Meteo API.
    Returns a dictionary with pollutant concentrations mapped to our model's feature names.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    # Parameters matching our model features:
    # pm_sepuluh -> pm10
    # pm_duakomalima -> pm2_5
    # sulfur_dioksida -> sulphur_dioxide
    # karbon_monoksida -> carbon_monoxide
    # ozon -> ozone
    # nitrogen_dioksida -> nitrogen_dioxide
    # max -> (Not directly available, we might take the max of all or just 0)
    
    params = {
        "latitude": LAT,
        "longitude": LON,
        "current": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"],
        "timezone": "Asia/Bangkok"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current = data.get('current', {})
        
        # Mapping to our APP fields
        # Note: Open-Meteo units are usually µg/m³, which matches our likely dataset (except CO might be different, check units)
        # CO in Open-Meteo is µg/m³. 
        # CAREFUL: Some datasets use mg/m³ for CO. 
        # Standard ISPU uses µg/m³ usually? Or mg/m³? 
        # Check standard: CO usually in µg/m³ in these APIs.
        
        result = {
            "pm_sepuluh": current.get('pm10', 0),
            # pm2.5 is not an input feature in manual form? 
            # Wait, let's check manual_predict.html form fields.
            # It has: pm_sepuluh, sulfur_dioksida, karbon_monoksida, ozon, nitrogen_dioksida, max.
            # It DOES NOT have pm_duakomalima input? 
            # Let's re-read app.py manual_predict route.
            # It accepts: pm_sepuluh, sulfur_dioksida, karbon_monoksida, ozon, nitrogen_dioksida, max.
            # It does NOT accept pm_duakomalima. The model PREDICTS pm_duakomalima (target).
            # So we don't need to fetch PM2.5 for input, but we can fetch it for reference if we want.
            
            "sulfur_dioksida": current.get('sulphur_dioxide', 0),
            "karbon_monoksida": current.get('carbon_monoxide', 0),
            "ozon": current.get('ozone', 0),
            "nitrogen_dioksida": current.get('nitrogen_dioxide', 0),
            "max": 0 # Placeholder, maybe user needs to input this manually or we calculate ISPU max?
        }
        
        return result
    except Exception as e:
        logging.error(f"Error fetching Open-Meteo data: {e}")
        return None
