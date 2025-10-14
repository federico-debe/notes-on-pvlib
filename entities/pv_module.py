from pydantic import BaseModel


class PVModule(BaseModel):
    id: str
    gcr: float
    b_0: float
    u_c: float
    u_v: float
    cec_name: str

    @property
    def manifacturer(self):
        return self.cec_name
