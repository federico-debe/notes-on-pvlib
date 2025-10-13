from pydantic import BaseModel
from common.enums import EditStatus, EnergyPlantType, RackingType, SurfaceType, TechChoice, TrackingType
from datetime import date
from typing import List, Optional

from models.gisrecs.aggregate_basic_info import AggregateBasicInfo
# from models.gisrecs.datetime_slot_info import DatetimeSlotInfo
from models.gisrecs.pod_basic_info import BatteryBasicInfo, EnergyPlantBasicInfo, PoDBasicInfo
from models.gisrecs.prices_search_info_container import TechnologyPriceInfoContainer


class EnergyPlantEditInfoContainer(BaseModel):
    uri: str = ''
    project_uri: str
    selected_aggregate: AggregateBasicInfo
    selected_pod: PoDBasicInfo
    capex_cost: float = 0.
    opex_cost: float = 0.
    capital_contribution: float = 0.
    plant_type: EnergyPlantType = EnergyPlantType.NONE
    # grid_connection_type: GridConnectionType
    peak_power: float = 0.
    square_meters: float = 0.
    angle: float = 0.
    aspect: float = 0.
    max_angle: Optional[float] = 90 # only if single-axis tracking
    backtrack: Optional[float] = False # only if freestanding
    material: TechChoice = TechChoice.Mono_c_Si
    percentage_loss: float = 14.
    yearly_loss: float = 0.7
    battery_loss: float = 100.
    tracking_type: TrackingType = TrackingType.fixed
    mounting_place: RackingType = RackingType.semi_integrated
    surface_type: Optional[SurfaceType] = SurfaceType.concrete
    latitude: float
    longitude: float
    altitude: Optional[float] = 0
    # maintenance_periods: Optional[List[DatetimeSlotInfo]] = [DatetimeSlotInfo()]
    battery: Optional[BatteryBasicInfo] = None
    existing_energy_plants: Optional[List[EnergyPlantBasicInfo]] = []
    pv_uris_with_battery_connection: Optional[List[str]] = []
    implementation_date: Optional[date] = None
    selected_prices: Optional[TechnologyPriceInfoContainer] = None
    prices: Optional[list[TechnologyPriceInfoContainer]] = []
    simulation_precision: int
    operation_status: EditStatus = EditStatus.NONE


    @property
    def get_tech_choice_description(self):
        return TechChoice.get_tech_choice_descritpion(self.material)

    def set_altitude(self, altitude):
        if altitude is not None:
            self.altitude = altitude
        else:
            self.altitude = 0.0

class EnergyPlantEditInfoContainers(BaseModel):
    energy_plant_edit_info_containers: List[EnergyPlantEditInfoContainer] = []
    