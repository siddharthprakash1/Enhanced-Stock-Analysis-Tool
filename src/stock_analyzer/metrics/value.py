from datetime import date
from pydantic import BaseModel


class MetricValue(BaseModel):
    key: str
    label: str
    value: float | None
    unit: str
    category: str
    as_of: date

    def display(self) -> str:
        if self.value is None:
            return "N/A"
        return f"{self.value:.2f}"
