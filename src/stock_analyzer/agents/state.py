from typing import TypedDict, Literal
from pydantic import BaseModel


class AnalystFinding(BaseModel):
    summary: str
    key_points: list[str]
    outlook: Literal["bullish", "bearish", "neutral"]
    rationale: str
    cited_metrics: list[str]


class ReportSection(BaseModel):
    id: str
    prose: str
    referenced_metrics: list[str]
    charts: list[str]


class ReportDraft(BaseModel):
    sections: list[ReportSection]
    recommendation: Literal["buy", "hold", "sell"]
    confidence: Literal["low", "medium", "high"]


class AnalysisState(TypedDict, total=False):
    symbol: str
    metrics_block: str
    fundamental: AnalystFinding
    technical: AnalystFinding
    risk: AnalystFinding
    valuation: AnalystFinding
    report: ReportDraft
    claims: list
    verdicts: list
    revisions: int
    verdict_feedback: str | None
    audit: dict
