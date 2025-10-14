from typing import Optional
from pvlib.location import Location
from pvlib.pvsystem import retrieve_sam
from pydantic import BaseModel

from common.enums import RackingType
from models.simulatorrecs.pv_plant import InverterBasicInfo, ModuleBasicInfo


class PvCablingScheme(BaseModel):
    _module = None
    _inverter = None

    latitude: float
    longitude: float
    altitude: float
    hot_ambient_temperature: float
    cold_ambient_temperature: float
    peak_power: float
    mounting_place: RackingType
    module: ModuleBasicInfo
    inverter: InverterBasicInfo
    u_c: float
    u_v: float
    gcr: float
    b_0: float

    modules_per_string: Optional[int] = 1
    strings_per_inverter: Optional[int] = 1
    number_of_inverters: Optional[int] = 1



    def set_modules_per_string(self, n_series):
        self.modules_per_string = n_series

    def set_strings(self, n_strings):
        self.strings_per_inverter = n_strings

    def set_number_of_inverters(self):
        self.number_of_inverters = int(
            (self.peak_power * 1000.0) / 
            (self.modules_per_string * self.get_module_parameters['STC']) / 
            self.strings_per_inverter
        )

    @property
    def get_location(self):
        return Location(self.latitude, self.longitude, tz="Europe/Rome", altitude=self.altitude)

    @property
    def get_module_parameters(self):
        if self._module is None:
            cecm = retrieve_sam('cecmod')
            module = cecm[self.module.name]
            module['K'] = module.get('K', self.b_0)
            self._module = module
        return self._module

    @property
    def get_inverter_parameters(self):
        if self._inverter is None:
            ceci = retrieve_sam('cecinverter')
            inverter = ceci[self.inverter.name]
            self._inverter = inverter
        return self._inverter

    @property
    def get_temperature_model_parameters(self):
        if self._module is None:
            self._module = self.get_module_parameters
        eta_m = float(self._module['I_mp_ref'] * self._module['V_mp_ref']) / (1000.0 * float(self._module['A_c']))
        u_c, u_v = RackingType.get_default_model_temperature_params(self.mounting_place)
        if self.u_c is None or self.u_c == 0:
            self.u_c = u_c
        if self.u_v is None or self.u_v == 0:
            self.u_v = u_v
        temperature_model_parameters = dict(u_c=self.u_c, u_v=self.u_v, eta_m=eta_m, alpha_absorption=0.9)
        return temperature_model_parameters

    @property
    def effective_peak_power_dc(self):
        return self.number_of_inverters * self.modules_per_string * self.strings_per_inverter * self.get_module_parameters['STC'] / 1000

    @property
    def effective_peak_power_ac(self):
        return self.number_of_inverters * self.get_inverter_parameters['Paco'] / 1000


    @property
    def dcac_ratio(self) -> float:
        # Rooftop: ~1.05–1.25
        # Ground-mount/utility: ~1.10–1.40 (hot/sunny sites often higher)
        # Modules rarely operate at STC; heat and irradiance reduce DC output. A DC/AC >1.0 keeps the inverter better loaded and accepts occasional clipping.
        return self.effective_peak_power_dc / self.effective_peak_power_ac