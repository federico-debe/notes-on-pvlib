from typing import List, Optional
from pydantic import BaseModel
from common.enums import PriceType, SortOption


class PricesSearchInfo(BaseModel):
    name: str
    price_type: PriceType
    price_status: int
    count: int

class TechnologyPriceDetail(BaseModel):
    capex_price_kwh: float
    opex_price_kwh: float
    inverter_cost: Optional[float]
    technology_type: int
    minimum_power: float
    maximum_power: float
    minimum_nominal_capacity: Optional[float] = None
    maximum_nominal_capacity: Optional[float] = None

class TechnologyPriceInfoContainer(PricesSearchInfo):
    price_details: List[TechnologyPriceDetail]

class PricesSearchInfoContainer(BaseModel):
    prices: List[PricesSearchInfo]
    count: int

class PriceSearchCriteria(BaseModel):
    search_text: str
    name: str
    price_type: Optional[PriceType] = None
    sort_field: Optional[SortOption] = SortOption.NAME_ASC
