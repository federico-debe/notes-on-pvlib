from pydantic import BaseModel

from common.enums import RackingType, SurfaceType, TechChoice


class TechComponentsCriteria(BaseModel):
    requested_dc_kwp: float
    latitude: float
    longitude: float
    altitude: float
    material: TechChoice
    mounting_place: RackingType
    surface_type: SurfaceType
    angle: float
