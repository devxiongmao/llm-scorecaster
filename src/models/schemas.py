from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field


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


class AsyncJobResponse(BaseModel):
    """Response model for async job submission."""

    job_id: str = Field(..., description="Unique identifier for the submitted job")
    status: str = Field(..., description="Current status of the job")
    message: str = Field(..., description="Human-readable status message")
    estimated_completion_time: Optional[float] = Field(
        None, description="Estimated completion time in seconds"
    )


class JobStatusResponse(BaseModel):
    """Response model for job status queries."""

    job_id: str = Field(..., description="Unique identifier for the job")
    status: str = Field(..., description="Current status of the job")
    message: str = Field(..., description="Human-readable status message")
    progress: Optional[int] = Field(
        default=None, ge=0, le=100, description="Progress percentage (0-100)"
    )
    total_pairs: Optional[int] = Field(
        default=None, description="Total number of text pairs being processed"
    )
    total_metrics: Optional[int] = Field(
        default=None, description="Total number of metrics being computed"
    )
    completed: Optional[bool] = Field(
        default=False, description="Whether the job has completed successfully"
    )
    failed: Optional[bool] = Field(
        default=False, description="Whether the job has failed"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if the job failed"
    )
    started_at: Optional[str] = Field(
        default=None, description="ISO timestamp when the job started processing"
    )
    completed_at: Optional[str] = Field(
        default=None, description="ISO timestamp when the job completed"
    )
