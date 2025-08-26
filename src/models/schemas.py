from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum


class MetricType(str, Enum):
    """Available metric types."""

    BERT_SCORE = "bert_score"
    BLEU = "bleu_score"
    ROUGE = "rouge_score"


class TextPair(BaseModel):
    """A pair of reference and candidate texts for evaluation."""

    reference: str = Field(..., description="The reference/ground truth text")
    candidate: str = Field(..., description="The candidate text to evaluate")


class MetricsRequest(BaseModel):
    """Request model for metrics evaluation."""

    text_pairs: List[TextPair] = Field(
        ...,
        min_length=1,
        description="List of text pairs to evaluate",
    )
    metrics: List[MetricType] = Field(
        ...,
        min_length=1,
        description="List of metrics to calculate",
    )
    batch_size: Optional[int] = Field(
        default=32, description="Batch size for processing optimization"
    )


class MetricResult(BaseModel):
    """Result for a single metric on a text pair."""

    metric_name: str
    score: float
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TextPairResult(BaseModel):
    """Results for all metrics on a single text pair."""

    pair_index: int
    reference: str
    candidate: str
    metrics: List[MetricResult]


class MetricsResponse(BaseModel):
    """Response model for metrics evaluation."""

    success: bool
    message: str
    results: List[TextPairResult]
    processing_time_seconds: float
