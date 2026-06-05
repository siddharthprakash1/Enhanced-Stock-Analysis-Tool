from typing import Literal
from pydantic import BaseModel


class Claim(BaseModel):
    id: str
    text: str
    metric_key: str | None = None
    claimed_value: float | None = None
    claim_type: Literal["numeric", "directional", "categorical", "qualitative"]
    source_section: str


class Verdict(BaseModel):
    claim_id: str
    status: Literal["supported", "contradicted", "unsupported"]
    expected_value: float | None = None
    claimed_value: float | None = None
    delta: float | None = None
    rationale: str
    correction: str | None = None


class ClaimList(BaseModel):
    claims: list[Claim]


class VerificationAudit(BaseModel):
    total_claims: int
    supported: int
    contradicted: int
    unsupported: int
    corrections_applied: list[Verdict]
    residual_unverified: list[Verdict]
