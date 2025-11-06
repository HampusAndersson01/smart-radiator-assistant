import requests
from datetime import datetime, timedelta, timezone

def get_weather(lat=59.5, lon=16):
    """Return current outside temp and +3h forecast using SMHI API.

    Returns (current, forecast_3h)
    
    SMHI API returns hourly forecasts with 't' (temperature in Celsius).
    Free, no API key required.
    """
    # Round coordinates to SMHI's grid (they use specific points)
    lat_rounded = round(lat, 6)
    lon_rounded = round(lon, 6)
    
    url = f"https://opendata-download-metfcst.smhi.se/api/category/pmp3g/version/2/geotype/point/lon/{lon_rounded}/lat/{lat_rounded}/data.json"
    
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        # SMHI returns timeSeries array with hourly forecasts
        time_series = data.get("timeSeries", [])
        if not time_series or len(time_series) == 0:
            print("SMHI API: No time series data")
            return None, None
        
        # Find current temp (first forecast point is usually now or very recent)
        current_temp = None
        forecast_3h_temp = None
        
        now = datetime.now(timezone.utc)
        target_3h = now + timedelta(hours=3)
        
        for i, forecast in enumerate(time_series):
            # Parse validTime: "2025-11-04T12:00:00Z"
            valid_time_str = forecast.get("validTime", "")
            if not valid_time_str:
                continue
            
            try:
                # Handle Z timezone indicator
                if valid_time_str.endswith("Z"):
                    valid_time = datetime.fromisoformat(valid_time_str.replace("Z", "+00:00"))
                else:
                    valid_time = datetime.fromisoformat(valid_time_str)
                
                # Make sure it's timezone-aware
                if valid_time.tzinfo is None:
                    valid_time = valid_time.replace(tzinfo=timezone.utc)
            except Exception as parse_error:
                print(f"Failed to parse time {valid_time_str}: {parse_error}")
                continue
            
            # Extract temperature from parameters
            params = forecast.get("parameters", [])
            temp = None
            for param in params:
                if param.get("name") == "t":  # 't' is temperature in Celsius
                    values = param.get("values", [])
                    if values and len(values) > 0:
                        temp = values[0]
                    break
            
            if temp is None:
                continue
            
            # First valid temp is current
            if current_temp is None:
                current_temp = temp
                print(f"SMHI: Current temp = {temp}°C at {valid_time}")
            
            # Find closest forecast to +3h
            if forecast_3h_temp is None and valid_time >= target_3h:
                forecast_3h_temp = temp
                print(f"SMHI: +3h forecast = {temp}°C at {valid_time}")
                break
        
        # Fallback if we didn't find +3h forecast
        if forecast_3h_temp is None and len(time_series) > 3:
            for param in time_series[min(3, len(time_series)-1)].get("parameters", []):
                if param.get("name") == "t":
                    values = param.get("values", [])
                    if values and len(values) > 0:
                        forecast_3h_temp = values[0]
                    break
        
        # If still no forecast, use current
        if forecast_3h_temp is None:
            forecast_3h_temp = current_temp
        
        print(f"SMHI: Returning current={current_temp}, forecast={forecast_3h_temp}")
        return current_temp, forecast_3h_temp
        
    except Exception as e:
        print(f"SMHI API error: {e}")
        import traceback
        traceback.print_exc()
        return None, None
