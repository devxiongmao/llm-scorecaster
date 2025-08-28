from typing import List, Dict, Any, Set, Optional
import logging
from src.core.metrics.base import BaseMetric
from src.models.schemas import MetricType, MetricResult, TextPair

logger = logging.getLogger(__name__)


class RougeMetric(BaseMetric):
    """
    ROUGE metric implementation using rouge-score library.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics
    for evaluating automatic summarization and machine translation. It measures
    the overlap of n-grams, word sequences, and word pairs between candidate
    and reference texts.
    """

    def __init__(
        self, rouge_types: Optional[List[str]] = None, use_stemmer: bool = True
    ):
        """
        Initialize ROUGE metric.

        Args:
            rouge_types: List of ROUGE variants to compute. Defaults to ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
                        - rouge1: Unigram overlap
                        - rouge2: Bigram overlap
                        - rougeL: Longest Common Subsequence (LCS)
                        - rougeLsum: LCS applied to summary-level (sentence-split)
            use_stemmer: Whether to use Porter stemmer for preprocessing
        """
        super().__init__()

        # Set default ROUGE types if not provided
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        # Validate rouge_types
        self._validate_rouge_types(rouge_types)

        self.rouge_types = rouge_types
        self.use_stemmer = use_stemmer
        self._scorer = None
        self._rouge_score_loaded = False

    @property
    def name(self) -> str:
        return "rouge_score"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.ROUGE

    @property
    def description(self) -> str:
        types_str = ", ".join(self.rouge_types)
        stemmer_str = "with stemmer" if self.use_stemmer else "without stemmer"
        return f"ROUGE Score: N-gram overlap evaluation ({types_str}) {stemmer_str}"

    @property
    def requires_model_download(self) -> bool:
        # ROUGE doesn't require model downloads, but may need NLTK data for stemming
        return self.use_stemmer

    def _validate_rouge_types(self, rouge_types: List[str]) -> None:
        """Validate ROUGE type configuration."""
        valid_types = self.get_supported_rouge_types()
        invalid_types = set(rouge_types) - valid_types
        if invalid_types or len(rouge_types) == 0:
            raise ValueError(
                f"Invalid ROUGE types: {invalid_types}. "
                f"Valid types: {sorted(valid_types)}"
            )

    def _ensure_rouge_score(self) -> None:
        """Ensure rouge-score library is available and scorer is loaded."""
        if self._rouge_score_loaded:
            return

        try:
            from rouge_score import rouge_scorer # pylint: disable=import-outside-toplevel

            # Initialize ROUGE scorer with specified types and stemmer option
            self._scorer = rouge_scorer.RougeScorer(
                rouge_types=self.rouge_types, use_stemmer=self.use_stemmer
            )
            self._rouge_score_loaded = True
            logger.info(
                "ROUGE scorer loaded successfully with types: %s", self.rouge_types
            )

        except ImportError:
            raise ImportError(
                "rouge-score package is required for ROUGE metric. "
                "Install with: pip install rouge-score"
            )
        except Exception as e:
            logger.error("Failed to load ROUGE scorer: %s", e)
            raise RuntimeError(f"Failed to initialize ROUGE scorer: {e}")

    def _process_rouge_scores(self, reference: str, candidate: str) -> MetricResult:
        """Process ROUGE scores for a reference and candidate text."""
        assert self._scorer is not None

        scores = self._scorer.score(reference, candidate)

        # Extract detailed scores
        details = {}
        primary_f1_score = 0.0

        for rouge_type in self.rouge_types:
            if rouge_type in scores:
                rouge_score = scores[rouge_type]

                # Round scores for consistency
                precision = round(rouge_score.precision, 4)
                recall = round(rouge_score.recall, 4)
                f1 = round(rouge_score.fmeasure, 4)

                details[rouge_type] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

                # Use ROUGE-L F1 as primary score, fallback to first available F1
                if rouge_type == "rougeL":
                    primary_f1_score = f1
                elif primary_f1_score == 0.0:
                    primary_f1_score = f1

        # Add configuration info to details
        details.update(
            {
                "rouge_types": self.rouge_types,
                "use_stemmer": self.use_stemmer,
                "library": "rouge-score",
            }
        )

        return MetricResult(
            metric_name=self.name,
            score=primary_f1_score,
            details=details,
        )

    def _process_rouge_pair(self, pair: TextPair) -> MetricResult:
        """Process a single text pair and return ROUGE result."""
        return self._process_rouge_scores(pair.reference, pair.candidate)

    def compute_single(self, reference: str, candidate: str) -> MetricResult:
        """
        Compute ROUGE scores for a single text pair.

        Args:
            reference: The reference/ground truth text
            candidate: The candidate text to evaluate

        Returns:
            MetricResult: ROUGE result with scores for all configured variants
        """
        self._ensure_rouge_score()
        assert self._scorer is not None

        try:
            return self._process_rouge_scores(reference, candidate)
        except Exception as e:
            logger.error("Error computing ROUGE scores: %s", e)
            return self._create_error_result(e)

    def _create_error_result(self, error: Exception) -> MetricResult:
        """Create a MetricResult for errors."""
        return MetricResult(metric_name=self.name, score=0.0, error=str(error))

    def compute_batch(
        self, text_pairs: List[TextPair], batch_size: int = 32
    ) -> List[MetricResult]:
        """
        Compute ROUGE scores for multiple text pairs.

        ROUGE doesn't have native batch processing like BERT Score,
        but we can still optimize by processing in chunks and reusing
        the scorer instance.

        Args:
            text_pairs: List of text pairs to evaluate
            batch_size: Size of batches for processing (used for progress reporting)

        Returns:
            List[MetricResult]: List of ROUGE results
        """
        self._ensure_rouge_score()
        assert self._scorer is not None

        if not text_pairs:
            return []

        results = []
        self._notify_start(len(text_pairs))

        try:
            # Process in batches for progress reporting and memory management
            for i in range(0, len(text_pairs), batch_size):
                batch = text_pairs[i : i + batch_size]

                try:
                    # Process each pair in the batch
                    for j, pair in enumerate(batch):
                        try:
                            result = self._process_rouge_pair(pair)
                            results.append(result)
                            self._notify_pair_processed(i + j, result)

                        except Exception as e:
                            logger.error("Error processing pair %d: %s", i + j, e)
                            error_result = self._create_error_result(e)
                            results.append(error_result)
                            self._notify_pair_processed(i + j, error_result)

                except Exception as e:
                    logger.error(
                        "Error processing batch %d: %s", i // batch_size + 1, e
                    )

                    # Create error results for any remaining pairs in the batch
                    for j, pair in enumerate(batch):
                        if i + j >= len(results):  # Only add if not already processed
                            error_result = self._create_error_result(e)
                            results.append(error_result)
                            self._notify_pair_processed(i + j, error_result)

            self._notify_complete(results)
            return results

        except Exception as e:
            self._notify_error(e)
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded scorer."""
        return {
            "status": "loaded" if self._rouge_score_loaded else "not_loaded",
            "library": "rouge-score",
            "rouge_types": self.rouge_types,
            "use_stemmer": self.use_stemmer,
            "requires_download": self.use_stemmer,  # Only for NLTK stemmer data
            "supported_types": sorted(self.get_supported_rouge_types()),
        }

    def configure(
        self,
        rouge_types: Optional[List[str]] = None,
        use_stemmer: Optional[bool] = None,
    ) -> None:
        """
        Update ROUGE configuration. Will take effect on next computation.
        Forces reinitialization of the scorer with new settings.

        Args:
            rouge_types: List of ROUGE variants to compute
            use_stemmer: Whether to use Porter stemmer for preprocessing
        """
        config_changed = False

        if rouge_types is not None:
            self._validate_rouge_types(rouge_types)
            if rouge_types != self.rouge_types:
                self.rouge_types = rouge_types
                config_changed = True

        if use_stemmer is not None:
            if use_stemmer != self.use_stemmer:
                self.use_stemmer = use_stemmer
                config_changed = True

        # Force reinitialization if config changed
        if config_changed:
            self._rouge_score_loaded = False
            self._scorer = None
            logger.info(
                "ROUGE configuration updated: rouge_types=%s, use_stemmer=%s",
                self.rouge_types,
                self.use_stemmer,
            )

    def get_supported_rouge_types(self) -> Set[str]:
        """Get the set of supported ROUGE types."""
        return {"rouge1", "rouge2", "rougeL", "rougeLsum"}

    def __str__(self) -> str:
        types_str = ",".join(self.rouge_types)
        return f"{self.__class__.__name__}({self.name}, types=[{types_str}])"
