from pvlib import pvsystem

class DualAxisTrackerMount(pvsystem.AbstractMount):
    def get_orientation(self, solar_zenith, solar_azimuth):
        # clamp tilt to [0, 90]; keep azimuth valid; NaN at night
        tilt = solar_zenith.clip(lower=0.0, upper=90.0)
        azim = solar_azimuth.copy()
        tilt = tilt.where(solar_zenith < 90.0)          # NaN at night
        azim = azim.where(solar_zenith < 90.0)
        return {'surface_tilt': tilt, 'surface_azimuth': azim}
