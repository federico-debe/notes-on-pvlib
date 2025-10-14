from common.enums import RackingType
from models.gisrecs.energy_plant_edit_info_container import EnergyPlantEditInfoContainer

import pandas as pd
import pvlib

class PVProductionHelper:
    _tz = 'Europe/Rome'
    _location = None

    def __init__(self) -> None:
        pass

    def get_peak_dc_production(self, payload: EnergyPlantEditInfoContainer, dcac_ratio=1.15):
        site = pvlib.location.Location(payload.latitude, payload.longitude, tz=self._tz, altitude=payload.altitude)
        times = pd.date_range(f"{2023}-01-01", f"{2023}-12-31 23:00", freq="1h", tz=self._tz)
        cs = site.get_clearsky(times, model="ineichen")
        temp_air = pd.Series(20.0, index=times)
        wind = pd.Series(1.0, index=times)
        ghi, dni, dhi = cs['ghi'], cs['dni'], cs['dhi']
        solpos = site.get_solarposition(times)

        u_c, u_v = RackingType.get_default_model_temperature_params(payload.mounting_place)
        
        pdc_arrays = []
        for array in payload.requested_pv_system_characteristics:


            poa = pvlib.irradiance.get_total_irradiance(
                surface_tilt=array.angle,
                surface_azimuth=array.aspect,
                dni=dni, ghi=ghi, dhi=dhi,
                solar_zenith=solpos['apparent_zenith'],
                solar_azimuth=solpos['azimuth'],
                albedo=0.2
            )
            tcell = pvlib.temperature.pvsyst_cell(poa["poa_global"], temp_air, 1, u_c=u_c, u_v=u_v)
            pdc_kw = pvlib.pvsystem.pvwatts_dc(
                effective_irradiance=poa["poa_global"],
                temp_cell=tcell,
                pdc0=1000.0 * array.peak_power,         # kWdc nameplate for this array
                gamma_pdc=-0.004
            ) / 1000.0
            pdc_arrays.append(pdc_kw)

        pdc_sum_kw = pd.concat(pdc_arrays, axis=1).sum(axis=1)
        return pdc_sum_kw.max() * dcac_ratio
