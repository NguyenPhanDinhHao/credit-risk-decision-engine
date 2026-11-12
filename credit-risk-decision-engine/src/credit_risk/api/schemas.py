"""Pydantic request/response schemas."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApplicantInput(BaseModel):
    """Core applicant fields (Home Credit application schema). Every field is optional: absent fields are
    treated as missing and imputed with training statistics, which degrades prediction quality.
    Additional Home Credit columns may be supplied and are passed to the model as-is (unknown ones are ignored)."""

    model_config = ConfigDict(extra="allow")

    NAME_CONTRACT_TYPE: Literal["Cash loans", "Revolving loans"] | None = None
    FLAG_OWN_CAR: Literal["Y", "N"] | None = None
    FLAG_OWN_REALTY: Literal["Y", "N"] | None = None
    CNT_CHILDREN: int | None = Field(None, ge=0, le=25)
    CNT_FAM_MEMBERS: float | None = Field(None, ge=1, le=40)
    AMT_INCOME_TOTAL: float | None = Field(None, gt=0, le=1e9)
    AMT_CREDIT: float | None = Field(None, gt=0, le=1e9)
    AMT_ANNUITY: float | None = Field(None, gt=0, le=1e8)
    AMT_GOODS_PRICE: float | None = Field(None, gt=0, le=1e9)
    NAME_INCOME_TYPE: str | None = Field(None, max_length=64)
    NAME_EDUCATION_TYPE: str | None = Field(None, max_length=64)
    NAME_HOUSING_TYPE: str | None = Field(None, max_length=64)
    OCCUPATION_TYPE: str | None = Field(None, max_length=64)
    ORGANIZATION_TYPE: str | None = Field(None, max_length=64)
    DAYS_EMPLOYED: float | None = Field(None, le=365243, description="Negative days before application; 365243 = not employed placeholder")
    DAYS_REGISTRATION: float | None = Field(None, le=0)
    DAYS_ID_PUBLISH: float | None = Field(None, le=0)
    DAYS_LAST_PHONE_CHANGE: float | None = Field(None, le=0)
    REGION_RATING_CLIENT: int | None = Field(None, ge=1, le=3)
    EXT_SOURCE_1: float | None = Field(None, ge=0, le=1)
    EXT_SOURCE_2: float | None = Field(None, ge=0, le=1)
    EXT_SOURCE_3: float | None = Field(None, ge=0, le=1)

    @field_validator("*", mode="before")
    @classmethod
    def _scalars_only(cls, v: Any) -> Any:
        if isinstance(v, (dict, list, tuple, set)):
            raise ValueError("nested values are not allowed")
        return v

    def to_record(self) -> dict[str, Any]:
        return self.model_dump()


class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]


class PredictionResponse(BaseModel):
    probability_of_default: float = Field(..., ge=0, le=1)
    risk_category: Literal["Low Risk", "Medium Risk", "High Risk"]
    model_version: str
    top_factors: list[FeatureContribution] | None = None
    disclaimer: str


class BatchRequest(BaseModel):
    applicants: list[ApplicantInput] = Field(..., min_length=1)


class BatchPrediction(BaseModel):
    index: int
    probability_of_default: float
    risk_category: str


class BatchResponse(BaseModel):
    count: int
    model_version: str
    predictions: list[BatchPrediction]
    disclaimer: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    data_source: str | None
    trained_at: str | None
    calibration_method: str | None
    n_features_after_preprocessing: int | None
    risk_bands: dict[str, float]
    excluded_sensitive_features: list[str]
    test_metrics: dict[str, float]
    disclaimer: str
