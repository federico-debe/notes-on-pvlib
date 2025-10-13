from typing import List, Optional
from pydantic import BaseModel

from common.enums import RackingType, SurfaceType, TechChoice, TrackingType

from pvlib.pvsystem import retrieve_sam


class InverterBasicInfo(BaseModel):
    name: str
    nominal_power: float

class ModuleBasicInfo(BaseModel):
    name: str
    nominal_power: float


class TechComponentEditInfoContainer(BaseModel):
    gcr: Optional[float] = 0.35
    u_c: Optional[float] = None
    u_v: Optional[float] = None
    b_0: Optional[float] = 0.05
    hot_ambient_temperature: Optional[float] = 40
    cold_ambient_temperature: Optional[float] = -5

    module: Optional[ModuleBasicInfo] = None
    inverter: Optional[InverterBasicInfo] = None

    modules: List[ModuleBasicInfo]
    inverters: List[InverterBasicInfo]

class PVPlant(BaseModel):
    '''class representing properties of a photovoltaic plant'''

    _module = None
    _inverter = None

    uri: str
    pod: str
    pod_uri: str
    kwp: float
    longitude: float
    latitude: float
    altitude: Optional[float] = 0 # meters
    angle: float # degrees above horizontal (0 = horizontal axis, 90 = vertical axis)
    aspect: float # degrees from North, clockwise (0 = N, 90 = E, 180 = S, 270 = W)
    max_angle: Optional[float] = None
    tech_choice: TechChoice
    tracking_type: TrackingType
    mounting_place: RackingType
    percentage_loss: float
    yearly_loss: float
    surface_type: Optional[SurfaceType] = SurfaceType.concrete

    module_name: Optional[str] = ''
    inverter_name: Optional[str] = ''
    inevrter_id: Optional[str] = ''
    module_names: Optional[List[ModuleBasicInfo]] = []
    inverter_names: Optional[List[InverterBasicInfo]] = []
    strings: Optional[int] = 1
    modules_per_string: Optional[int] = 1
    number_of_inverters: Optional[int] = 1
    backtrack: Optional[bool] = None
    gcr: Optional[float] = 0.35
    b0: Optional[float] = 0.05
    u_v: Optional[float] = None
    u_c: Optional[float] = None

    maximum_area: Optional[float] = 1000 # square meters

    @property
    def get_mount(self):
        return TrackingType.get_mount(
            tracking_type=self.tracking_type,
            racking_model=self.mounting_place,
            tilt=self.angle,
            azimuth=self.aspect,
            max_angle=self.max_angle,
            gcr=self.gcr,
            backtrack=self.backtrack
        )

    @property
    def get_module_parameters(self):
        if self._module is None:
            cecm = retrieve_sam('cecmod')
            module = cecm[self.module_name]
            module['K'] = module.get('K', self.b0)
            self._module = module
        return self._module

    @property
    def get_inverter_parameters(self):
        if self._inverter is None:
            ceci = retrieve_sam('cecinverter')
            inverter = ceci[self.inverter_name]
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

    def set_modules(self, module_names):
        self.module_names = module_names

    def set_inverters(self, inverter_names):
        self.inverter_names = inverter_names

    def set_modules_per_string(self, n_series):
        self.modules_per_string = n_series

    def set_strings(self, n_strings):
        self.strings = n_strings

    def set_number_of_inverters(self):
        self.number_of_inverters = int((self.kwp * 1000.0) / (self.modules_per_string * self.get_module_parameters['STC'])) / self.strings

    @property
    def effective_peak_power_dc(self):
        return self.number_of_inverters * self.modules_per_string * self.strings * self.get_module_parameters['STC'] / 1000

    @property
    def effective_peak_power_ac(self):
        return self.number_of_inverters * self.get_inverter_parameters['Paco'] / 1000


    @property
    def dcac_ratio(self) -> float:
        # Rooftop: ~1.05–1.25
        # Ground-mount/utility: ~1.10–1.40 (hot/sunny sites often higher)
        # Modules rarely operate at STC; heat and irradiance reduce DC output. A DC/AC >1.0 keeps the inverter better loaded and accepts occasional clipping.
        return self.effective_peak_power_dc / self.effective_peak_power_ac
