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
        if self.unit == "$":
            v = self.value
            for thresh, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
                if abs(v) >= thresh:
                    return f"${v / thresh:.2f}{suf}"
            return f"${v:,.2f}"
        if self.unit == "%":
            return f"{self.value:.2f}%"
        if self.unit == "x":
            return f"{self.value:.2f}x"
        return f"{self.value:,.2f}"
