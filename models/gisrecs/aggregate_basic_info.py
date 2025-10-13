from pydantic import BaseModel


class AggregateBasicInfo(BaseModel):
    uri: str
    name: str