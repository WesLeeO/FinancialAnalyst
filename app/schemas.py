from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    metric: str = Field(description="exact key from tool output, e.g. forward_pe, rsi_14")
    value: float = Field(description="the exact number from the tool output")
    meaning: str = Field(description="short interpretation, e.g. 'cheap vs. growth'")


class AdviceReport(BaseModel):
    ticker: str = Field(description="The stock ticker symbol")
    recommendation: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"] = Field(description="Overall rating")
    rationale: str = Field(description="2-4 sentences explaining the recommendation with the most important evidence")
    evidences: list[Evidence] = Field(min_length=3, max_length=8, description="3 to 8 measured facts from the tool output that support the recommendation")
    risks: list[str] = Field(min_length=1, description="Concrete risks found in the data or news, not generic ones")
    confidence_level: float = Field(ge=0.0, le=1.0, description="Confidence in the recommendation (0 to 1); lower when signals conflict or data is missing")
    disclaimer: str = Field(default="Educational analysis based on public data, not personal financial advice.", description="Leave as default")
