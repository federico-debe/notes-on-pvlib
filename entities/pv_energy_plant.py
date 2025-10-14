from pydantic import BaseModel


class PVEnergyPlant(BaseModel):
    id: str
    kwp: float
    tilt: float
    azimuth: float
    material: int
    # ...
    