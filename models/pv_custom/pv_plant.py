from typing import List, Optional
from pydantic import BaseModel

from models.pv_custom.enums import RackingType, SurfaceType, TechChoice, TrackingType

from pvlib.pvsystem import retrieve_sam


class InverterBasicInfo(BaseModel):
    name: str
    nominal_power: float

class ModuleBasicInfo(BaseModel):
    name: str
    nominal_power: float

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

    @property
    def get_tech_choice_description(self):
        return TechChoice.get_tech_choice_descritpion(self.tech_choice)

    def set_altitude(self, altitude):
        if altitude is not None:
            self.altitude = altitude
        else:
            self.altitude = 0.0

    def set_modules(self, module_names):
        self.module_names = module_names

    def set_inverters(self, inverter_names):
        self.inverter_names = inverter_names
