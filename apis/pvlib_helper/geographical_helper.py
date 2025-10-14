import numpy as np
import pandas as pd
import requests

from pvlib import iotools as pvgis

from common.enums import SurfaceType

class GeographicalHelper:

    @staticmethod
    def get_elevation_opentopo(lat, lon, dataset='eudem25m'):
        """
        Get elevation using OpenTopoData.
        Datasets: 'aster30m', 'srtm30m', 'eudem25m' (Europe only)
        """
        url = f"https://api.opentopodata.org/v1/{dataset}?locations={lat},{lon}"
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()['results'][0]
            elevation = result['elevation']
            
            # Check if we got a valid elevation
            if elevation is not None:
                return elevation
            else:
                url_fallback = f"https://api.opentopodata.org/v1/aster30m?locations={lat},{lon}"
                response_fallback = requests.get(url_fallback)
                if response_fallback.status_code == 200:
                    return response_fallback.json()['results'][0]['elevation']
        
        raise Exception(f"API error: {response.status_code}")
    
    @staticmethod
    def get_extreme_ambient_temperatures(lat, lon, altitude, surface_type=None, racking_type=None):

        tmy, meta = pvgis.get_pvgis_tmy(
            latitude=lat,
            longitude=lon,
            map_variables=True,
            url="https://re.jrc.ec.europa.eu/api/v5_3/"
        )
        solar_zenith = GeographicalHelper._calculate_solar_zenith(tmy.index, lat, lon)
        daytime_mask = solar_zenith < 85
        tamb = tmy.loc[daytime_mask, 'temp_air']

        hot_temp = float(np.percentile(tamb, 99))
        cold_temp = float(np.percentile(tamb, 1))

        if altitude and altitude > 0:
            hot_temp, cold_temp = GeographicalHelper._altitude_adjustment(
                hot_temp, cold_temp, altitude
            )

        hot_temp = max(hot_temp, GeographicalHelper._estimate_minimum_hot_temp(tmy, surface_type))
        cold_temp = min(cold_temp, GeographicalHelper._estimate_minimum_cold_temp(tmy, surface_type))
        
        return round(hot_temp, 1), round(cold_temp, 1)
    
    @staticmethod
    def _calculate_solar_zenith(datetime_index, lat, lon):
        """Calculate solar zenith angle for each timestamp"""
        # Simplified solar position calculation
        # In practice, use pvlib or similar for accurate solar position
        doy = datetime_index.dayofyear
        hour = datetime_index.hour + datetime_index.minute/60
        
        # Declination angle (degrees)
        declination = 23.45 * np.sin(np.radians(360 * (284 + doy) / 365))
        
        # Hour angle (degrees)
        hour_angle = 15 * (hour - 12)
        
        # Solar zenith angle (degrees)
        lat_rad = np.radians(lat)
        decl_rad = np.radians(declination)
        ha_rad = np.radians(hour_angle)
        
        cos_zenith = (np.sin(lat_rad) * np.sin(decl_rad) + 
                     np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad))
        
        zenith = np.degrees(np.arccos(np.clip(cos_zenith, -1, 1)))
        return zenith
    
    @staticmethod
    def _calculate_cell_temperature(tmy, solar_zenith, surface_type, racking_type, tilt_angle):
        """Improved cell temperature calculation using multiple models"""
        
        # Filter for daytime hours (solar zenith < 85 degrees)
        daytime_mask = solar_zenith < 85
        if daytime_mask.sum() == 0:
            daytime_mask = tmy['ghi'] > 50  # Fallback to GHI-based mask
        
        # NOCT-based model (IEC 61215)
        ghi = tmy.loc[daytime_mask, 'ghi']
        tamb = tmy.loc[daytime_mask, 'temp_air']
        wind_speed = np.maximum(tmy.loc[daytime_mask, 'wind_speed'], 0.5)  # Minimum wind speed
        
        # Base NOCT values (Nominal Operating Cell Temperature)
        if racking_type == 'open_rack':
            noct = 45
            u0, u1 = 25.0, 6.84  # Wind-dependent heat transfer coefficients
        elif racking_type == 'close_roof':
            noct = 48
            u0, u1 = 20.0, 4.0
        elif racking_type == 'insulated_back':
            noct = 50
            u0, u1 = 15.0, 2.5
        else:  # default open rack
            noct = 45
            u0, u1 = 25.0, 6.84
        
        # Surface albedo adjustment
        albedo = GeographicalHelper._get_albedo(surface_type)
        
        # Effective irradiance (considering albedo)
        poa_global = ghi * (1 + albedo * (1 - np.cos(np.radians(tilt_angle))) / 2)
        
        # Sandia Array Performance Model (King et al., 2004)
        wind_factor = u0 + u1 * wind_speed
        cell_temps = tamb + poa_global * np.exp(-3.473 - 0.0594 * wind_speed) + (
            poa_global / 1000 * (noct - 20)
        )
        
        # Alternative: Faiman model
        # cell_temps = tamb + poa_global / (u0 + u1 * wind_speed)
        
        # Ensure realistic bounds
        cell_temps = np.maximum(cell_temps, tamb - 5)  # Cell can't be much cooler than ambient
        cell_temps = np.minimum(cell_temps, 95)  # Practical upper limit
        
        # For nighttime, use ambient temperature
        full_series = pd.Series(index=tmy.index, data=tmy['temp_air'].values)
        full_series.loc[daytime_mask] = cell_temps
        
        return full_series

    @staticmethod
    def _get_albedo(surface_type):
        from pvlib.albedo import SURFACE_ALBEDOS
        if surface_type:
            return SURFACE_ALBEDOS[surface_type]
        return 0.2

    @staticmethod
    def _altitude_adjustment(hot_temp, cold_temp, altitude):
        """Proper altitude adjustment using environmental lapse rate"""
        # Standard environmental lapse rate: -6.5°C per 1000m
        lapse_rate = -6.5  # °C per 1000m
        
        adjustment = lapse_rate * (altitude / 1000)
        hot_temp += adjustment
        cold_temp += adjustment
        
        return hot_temp, cold_temp

    @staticmethod
    def _estimate_minimum_hot_temp(tmy, surface_type):
        """Provide conservative minimum hot temperature based on location and surface"""
        max_ambient = tmy['temp_air'].max()
        
        min_rise = SurfaceType.get_hot_temperature_delta(surface_type)
        
        return max_ambient + min_rise

    @staticmethod
    def _estimate_minimum_cold_temp(tmy, surface_type):
        """Provide conservative minimum hot temperature based on location and surface"""
        min_ambient = tmy['temp_air'].min()
        
        min_rise = SurfaceType.get_cold_temperature_delta(surface_type)
        
        return min_ambient + min_rise
