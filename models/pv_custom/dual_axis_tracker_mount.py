from pvlib import pvsystem

class DualAxisTrackerMount(pvsystem.AbstractMount):
    def get_orientation(self, solar_zenith, solar_azimuth):
        return {'surface_tilt': solar_zenith, 'surface_azimuth': solar_azimuth}
