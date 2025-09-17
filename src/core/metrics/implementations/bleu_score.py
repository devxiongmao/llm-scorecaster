"""BLEU Score metric implementation."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import logging
from src.core.metrics.base import BaseMetric
from src.models.schemas import MetricType, MetricResult, TextPair

logger = logging.getLogger(__name__)


@dataclass
class BleuConfig:
    """Configuration for BLEU metric.
    max_n: Maximum n-gram order to consider (default: 4 for BLEU-4)
    smooth_method: Smoothing method ('exp', 'floor', 'add-k', 'none')
    smooth_value: Smoothing value for add-k method
    tokenize: Tokenization method ('13a', 'intl', 'zh', 'ja-mecab', 'none')
    lowercase: Whether to lowercase the input
    """

    max_n: int = 4
    smooth_method: str = "exp"
    smooth_value: float = 0.0
    tokenize: str = "13a"
    lowercase: bool = False


class BleuMetric(BaseMetric):
    """
    BLEU Score metric implementation using SacreBLEU.

    BLEU (Bilingual Evaluation Understudy) measures the quality of text
    by comparing n-grams in the candidate text to n-grams in the reference text.
    Uses SacreBLEU for standardized, reproducible BLEU computation.
    """

    def __init__(self, config: Optional[BleuConfig] = None):
        """
        Initialize BLEU metric with SacreBLEU.

        Args:
            config: BLEU configuration object
        """
        super().__init__()
        self.config = config or BleuConfig()
        self._sacrebleu_loaded = False
        self.sacrebleu = None

    @property
    def name(self) -> str:
        return "bleu_score"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.BLEU

    @property
    def description(self) -> str:
        return f"BLEU Score: N-gram based evaluation (BLEU-{self.config.max_n}) using SacreBLEU"

    @property
    def requires_model_download(self) -> bool:
        return False  # SacreBLEU doesn't require model downloads

    def _ensure_sacrebleu(self) -> None:
        """Ensure SacreBLEU is available."""
        if self._sacrebleu_loaded:
            return

        try:
            import sacrebleu  # pylint: disable=import-outside-toplevel

            self.sacrebleu = sacrebleu
            self._sacrebleu_loaded = True
            logger.info("SacreBLEU loaded successfully")

        except ImportError as e:
            raise ImportError(
                "sacrebleu package is required for BLEU Score metric. "
                "Install with: poetry install --extras 'bleu"
            ) from e
        except Exception as e:
            logger.error("Failed to load SacreBLEU: %s", e)
            raise RuntimeError(f"Failed to initialize BLEU Score: {e}") from e

    def compute_single(self, reference: str, candidate: str) -> MetricResult:
        """
        Compute BLEU Score for a single text pair.

        Args:
            reference: The reference/ground truth text
            candidate: The candidate text to evaluate

        Returns:
            MetricResult: BLEU Score result with overall score and n-gram precisions
        """
        self._ensure_sacrebleu()

        try:
            assert self.sacrebleu is not None
            bleu = self.sacrebleu.BLEU(
                max_ngram_order=self.config.max_n,
                smooth_method=self.config.smooth_method,
                smooth_value=self.config.smooth_value,
                tokenize=self.config.tokenize,
                lowercase=self.config.lowercase,
            )

            # Compute BLEU score
            # SacreBLEU expects references as a list (for multiple references)
            bleu_score = bleu.sentence_score(candidate, [reference])

            # Extract individual n-gram precisions
            individual_scores = {}
            for n in range(1, self.config.max_n + 1):
                if n <= len(bleu_score.precisions):
                    individual_scores[f"bleu_{n}"] = round(
                        bleu_score.precisions[n - 1], 4
                    )
                else:
                    individual_scores[f"bleu_{n}"] = 0.0

            return MetricResult(
                metric_name=self.name,
                score=round(
                    bleu_score.score / 100.0, 4
                ),  # SacreBLEU returns 0-100, normalize to 0-1
                details={
                    "bleu_score": round(bleu_score.score / 100.0, 4),
                    "bleu_score_100": round(
                        bleu_score.score, 2
                    ),  # Keep original 0-100 scale too
                    "max_n": self.config.max_n,
                    **individual_scores,
                    "brevity_penalty": round(bleu_score.bp, 4),
                    "length_ratio": round(
                        bleu_score.sys_len / max(bleu_score.ref_len, 1), 4
                    ),
                    "reference_length": bleu_score.ref_len,
                    "candidate_length": bleu_score.sys_len,
                    "tokenization": self.config.tokenize,
                    "smoothing": self.config.smooth_method,
                },
            )

        except Exception as e:
            logger.error("Error computing BLEU Score: %s", e)
            return MetricResult(metric_name=self.name, score=0.0, error=str(e))

    def compute_batch(
        self, text_pairs: List[TextPair], batch_size: int = 32
    ) -> List[MetricResult]:
        """
        Compute BLEU Score for multiple text pairs.

        SacreBLEU processes pairs individually, but we can optimize by
        creating the BLEU object once and reusing it.

        Args:
            text_pairs: List of text pairs to evaluate
            batch_size: Batch size (not used for BLEU but kept for interface consistency)

        Returns:
            List[MetricResult]: List of BLEU Score results
        """
        self._ensure_sacrebleu()

        if not text_pairs:
            return []

        results = []
        self._notify_start(len(text_pairs))

        try:
            assert self.sacrebleu is not None
            bleu = self.sacrebleu.BLEU(
                max_ngram_order=self.config.max_n,
                smooth_method=self.config.smooth_method,
                smooth_value=self.config.smooth_value,
                tokenize=self.config.tokenize,
                lowercase=self.config.lowercase,
            )

            for i, pair in enumerate(text_pairs):
                try:
                    # Compute BLEU score for this pair
                    bleu_score = bleu.sentence_score(pair.candidate, [pair.reference])

                    # Extract individual n-gram precisions
                    individual_scores = {}
                    for n in range(1, self.config.max_n + 1):
                        if n <= len(bleu_score.precisions):
                            individual_scores[f"bleu_{n}"] = round(
                                bleu_score.precisions[n - 1], 4
                            )
                        else:
                            individual_scores[f"bleu_{n}"] = 0.0

                    result = MetricResult(
                        metric_name=self.name,
                        score=round(bleu_score.score / 100.0, 4),  # Normalize to 0-1
                        details={
                            "bleu_score": round(bleu_score.score / 100.0, 4),
                            "bleu_score_100": round(bleu_score.score, 2),
                            "max_n": self.config.max_n,
                            **individual_scores,
                            "brevity_penalty": round(bleu_score.bp, 4),
                            "length_ratio": round(
                                bleu_score.sys_len / max(bleu_score.ref_len, 1), 4
                            ),
                            "reference_length": bleu_score.ref_len,
                            "candidate_length": bleu_score.sys_len,
                            "tokenization": self.config.tokenize,
                            "smoothing": self.config.smooth_method,
                        },
                    )
                    results.append(result)
                    self._notify_pair_processed(i, result)

                except Exception as e:
                    logger.error("Error processing pair %d: %s", i, e)
                    error_result = MetricResult(
                        metric_name=self.name, score=0.0, error=str(e)
                    )
                    results.append(error_result)
                    self._notify_pair_processed(i, error_result)

            self._notify_complete(results)
            return results

        except Exception as e:
            self._notify_error(e)
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the BLEU configuration."""
        return {
            "status": "loaded" if self._sacrebleu_loaded else "not_loaded",
            "library": "sacrebleu",
            "max_n": self.config.max_n,
            "smooth_method": self.config.smooth_method,
            "smooth_value": self.config.smooth_value,
            "tokenize": self.config.tokenize,
            "lowercase": self.config.lowercase,
            "requires_download": False,
        }

    def configure(self, config: Optional[Union[BleuConfig, dict]] = None) -> None:
        """
        Update BLEU configuration. Will take effect on next computation.
        Allows you to update the settings at runtime without recreating the metric.

        Args:
            config: BLEU configuration object or dictionary with configuration parameters
        """
        if config is None:
            return

        # Handle dictionary input by converting to BleuConfig
        if isinstance(config, dict):
            try:
                config = BleuConfig(**config)
            except TypeError as e:
                raise ValueError(f"Invalid BLEU configuration parameters: {e}") from e
            except Exception as e:
                raise ValueError(f"Failed to create BLEU configuration: {e}") from e

        if config.max_n is not None:
            self.config.max_n = config.max_n
        if config.smooth_method is not None:
            self.config.smooth_method = config.smooth_method
        if config.smooth_value is not None:
            self.config.smooth_value = config.smooth_value
        if config.tokenize is not None:
            self.config.tokenize = config.tokenize
        if config.lowercase is not None:
            self.config.lowercase = config.lowercase

        logger.info(
            "BLEU configuration updated: max_n=%s, tokenize=%s, smooth_method=%s",
            self.config.max_n,
            self.config.tokenize,
            self.config.smooth_method,
        )
