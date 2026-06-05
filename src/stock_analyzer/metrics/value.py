import math
from datetime import date
from pydantic import BaseModel, field_validator


class MetricValue(BaseModel):
    key: str
    label: str
    value: float | None
    unit: str
    category: str
    as_of: date

    @field_validator("value")
    @classmethod
    def _nan_to_none(cls, v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    def display(self) -> str:
        if self.value is None:
            return "N/A"
        return f"{self.value:.2f}"
