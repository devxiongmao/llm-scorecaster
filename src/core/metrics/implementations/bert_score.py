from typing import List, Dict, Any
import logging
from src.core.metrics.base import BaseMetric
from src.models.schemas import MetricType, MetricResult, TextPair

logger = logging.getLogger(__name__)


class BertScoreMetric(BaseMetric):
    """
    BERT Score metric implementation.

    BERT Score leverages pre-trained contextual embeddings from BERT
    and matches words in candidate and reference sentences by cosine similarity.
    """

    def __init__(self):
        super().__init__()
        self._scorer = None
        self._model_loaded = False

    @property
    def name(self) -> str:
        return "bert_score"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.BERT_SCORE

    @property
    def description(self) -> str:
        return "BERT Score: Contextual embeddings-based evaluation using BERT"

    @property
    def requires_model_download(self) -> bool:
        return True

    def _load_model(self) -> None:
        """Load the BERT Score model lazily."""
        if self._model_loaded:
            return

        try:
            from bert_score import BERTScorer  # pylint: disable=import-outside-toplevel

            # Initialize with default model (roberta-large)
            # You can customize this based on your needs
            self._scorer = BERTScorer(
                model_type="roberta-large", lang="en", rescale_with_baseline=True
            )
            self._model_loaded = True
            logger.info("BERT Score model loaded successfully")

        except ImportError as e:
            raise ImportError(
                "bert-score package is required for BERT Score metric. "
                "Install with: pip install bert-score"
            ) from e
        except Exception as e:
            logger.error("Failed to load BERT Score model: %s", e)
            raise RuntimeError(f"Failed to initialize BERT Score: {e}") from e

    def _create_bert_score_result(
        self, precision: float, recall: float, f1: float
    ) -> MetricResult:
        """Create a MetricResult from BERT Score components."""
        return MetricResult(
            metric_name=self.name,
            score=f1,  # Use F1 as the primary score
            details={
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            },
        )

    def _create_error_result(self, error: Exception) -> MetricResult:
        """Create a MetricResult for errors."""
        return MetricResult(metric_name=self.name, score=0.0, error=str(error))

    def _process_bert_scores(
        self, candidates: List[str], references: List[str]
    ) -> List[MetricResult]:
        """Process BERT scores for lists of candidates and references."""
        # Compute BERT Score for the batch/single pair
        assert self._scorer is not None

        precision, recall, f1 = self._scorer.score(candidates, references)

        # Create results for each pair
        results = []
        for i in range(len(candidates)):
            precision_score = float(precision[i])
            recall_score = float(recall[i])
            f1_score = float(f1[i])

            result = self._create_bert_score_result(
                precision_score, recall_score, f1_score
            )
            results.append(result)

        return results

    def compute_single(self, reference: str, candidate: str) -> MetricResult:
        """
        Compute BERT Score for a single text pair.

        Args:
            reference: The reference/ground truth text
            candidate: The candidate text to evaluate

        Returns:
            MetricResult: BERT Score result with precision, recall, and F1
        """
        self._load_model()
        assert self._scorer is not None

        try:
            results = self._process_bert_scores([candidate], [reference])
            return results[0]  # Return the single result
        except Exception as e:
            logger.error("Error computing BERT Score: %s", e)
            return self._create_error_result(e)

    def compute_batch(
        self, text_pairs: List[TextPair], batch_size: int = 32
    ) -> List[MetricResult]:
        """
        Compute BERT Score for multiple text pairs efficiently.

        Overrides the default implementation to use batch processing
        for better performance with BERT Score.

        Args:
            text_pairs: List of text pairs to evaluate
            batch_size: Number of pairs to process in each batch

        Returns:
            List[MetricResult]: List of BERT Score results
        """
        self._load_model()
        assert self._scorer is not None

        if not text_pairs:
            return []

        results = []
        self._notify_start(len(text_pairs))

        try:
            # Process in batches for memory efficiency
            for i in range(0, len(text_pairs), batch_size):
                batch = text_pairs[i : i + batch_size]

                try:
                    # Prepare batch data
                    candidates = [pair.candidate for pair in batch]
                    references = [pair.reference for pair in batch]

                    # Process the batch using helper method
                    batch_results = self._process_bert_scores(candidates, references)

                    # Add results and notify observers
                    for j, result in enumerate(batch_results):
                        results.append(result)
                        self._notify_pair_processed(i + j, result)

                except Exception as e:
                    logger.error(
                        "Error processing batch %d: %s", i // batch_size + 1, e
                    )

                    # Create error results for the batch
                    for j, pair in enumerate(batch):
                        error_result = self._create_error_result(e)
                        results.append(error_result)
                        self._notify_pair_processed(i + j, error_result)

            self._notify_complete(results)
            return results

        except Exception as e:
            self._notify_error(e)
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._model_loaded:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "model_type": "roberta-large",
            "language": "en",
            "rescale_with_baseline": True,
        }
