from typing import Optional
from pydantic import BaseModel


class Inverter(BaseModel):
    id: str
    power: Optional[float] = 0.
    cec_name: Optional[str] = ''
    capex: float
    opex: float
