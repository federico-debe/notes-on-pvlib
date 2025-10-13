from typing import Optional
from pydantic import BaseModel

from common.enums import EnergyPlantType, PVBatteryConnectionType

class EnergyPlantBasicInfo(BaseModel):
    plant_uri: str
    peak_power: float
    plant_type: EnergyPlantType = EnergyPlantType.NONE
    percentage_loss: Optional[float]
    battery_loss: Optional[float]
    pv_battery_connection_type: Optional[PVBatteryConnectionType] = PVBatteryConnectionType.NONE

class BatteryBasicInfo(BaseModel):
    battery_uri: str
    nominal_capacity: float
    percentage_loss: Optional[float]
    
class PoDBasicInfo(BaseModel):
    uri: str
    code: str
    pod_type: Optional[int]

class AggregateSimulationPoDInfo(PoDBasicInfo):
    name: str
    industry_code: str = ''
    industry_code_description: str = ''
    type: str = ''
    industry_code_uri: str = ''
    has_battery: bool
    selected: bool = True
    considered_uc: bool
    considered_up: bool
    considered_up_battery: bool
