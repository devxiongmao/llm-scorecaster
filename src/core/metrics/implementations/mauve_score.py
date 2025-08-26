from typing import List, Dict, Any, Optional
from src.core.metrics.base import BaseMetric
from src.models.schemas import MetricType, MetricResult, TextPair
import logging

logger = logging.getLogger(__name__)


class MauveMetric(BaseMetric):
    """
    MAUVE metric implementation for evaluating text generation quality.

    MAUVE (Measuring the Gap Between Neural Text and Human Text) is a measure 
    that evaluates the statistical gap between two text distributions using 
    Kullback-Leibler divergences in a quantized embedding space. Unlike other 
    metrics that work on individual text pairs, MAUVE is designed to compare 
    distributions and requires a large number of samples (typically 1000+) 
    to produce meaningful results.
    
    Note: MAUVE works on collections of texts rather than individual pairs.
    For single pair evaluation, it returns the same score for all pairs in a batch.
    """

    def __init__(
        self,
        model_id: str = "gpt2-large",
        max_text_length: int = 1024,
        device_id: int = 0,
        verbose: bool = False,
        num_buckets: Optional[int] = None,
        batch_size: int = 8,
    ):
        """
        Initialize MAUVE metric.

        Args:
            model_id: Model to use for feature encoding (default: gpt2-large)
            max_text_length: Maximum text length for encoding  
            device_id: GPU device ID to use (use -1 for CPU)
            verbose: Whether to show progress messages
            num_buckets: Number of clusters for quantization (default: auto)
            batch_size: Batch size for feature encoding
        """
        super().__init__()
        
        self.model_id = model_id
        self.max_text_length = max_text_length
        self.device_id = device_id
        self.verbose = verbose
        self.num_buckets = num_buckets
        self.encoding_batch_size = batch_size
        self._mauve_loaded = False
        self._mauve = None
        
        # Cache for batch-level MAUVE computation
        self._cached_score = None
        self._cached_batch_size = 0

    @property
    def name(self) -> str:
        return "mauve"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.MAUVE

    @property
    def description(self) -> str:
        return f"MAUVE: Statistical gap measurement between text distributions using {self.model_id}"

    @property
    def requires_model_download(self) -> bool:
        return True

    def _validate_configuration(self) -> None:
        """Validate MAUVE configuration."""
        if self.max_text_length <= 0:
            raise ValueError("max_text_length must be positive")
        
        if self.encoding_batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def _ensure_mauve(self) -> None:
        """Ensure MAUVE library is available and ready."""
        if self._mauve_loaded:
            return

        try:
            import mauve
            self._mauve = mauve
            self._mauve_loaded = True
            logger.info(f"MAUVE library loaded successfully with model: {self.model_id}")

        except ImportError:
            raise ImportError(
                "mauve-text package is required for MAUVE metric. "
                "Install with: pip install mauve-text"
            )
        except Exception as e:
            logger.error(f"Failed to load MAUVE library: {e}")
            raise RuntimeError(f"Failed to initialize MAUVE: {e}")

    def compute_single(self, reference: str, candidate: str) -> MetricResult:
        """
        Compute MAUVE for a single text pair.
        
        Note: MAUVE is designed for distribution comparison, not individual pairs.
        This method treats the single pair as two single-item distributions,
        which may not produce meaningful results. Use compute_batch for proper
        MAUVE evaluation with multiple samples.

        Args:
            reference: The reference/ground truth text
            candidate: The candidate text to evaluate

        Returns:
            MetricResult: MAUVE result (may not be meaningful for single pairs)
        """
        self._ensure_mauve()
        
        # Type assertion since _ensure_mauve ensures _mauve is not None
        assert self._mauve is not None

        logger.warning(
            "MAUVE is designed for comparing distributions of texts, not single pairs. "
            "Results from single pairs may not be meaningful. Consider using compute_batch "
            "with multiple samples for proper MAUVE evaluation."
        )

        try:
            # Compute MAUVE with single texts (not ideal, but supported)
            out = self._mauve.compute_mauve(
                p_text=[reference],
                q_text=[candidate],
                device_id=self.device_id,
                max_text_length=self.max_text_length,
                verbose=self.verbose,
                batch_size=self.encoding_batch_size,
                num_buckets=self.num_buckets,
            )

            score = round(float(out.mauve), 4)

            # Create minimal details as requested
            details = {
                "model": self.model_id,
                "library": "mauve-text",
            }

            return MetricResult(
                metric_name=self.name,
                score=score,
                details=details,
            )

        except Exception as e:
            logger.error(f"Error computing MAUVE: {e}")
            return MetricResult(metric_name=self.name, score=0.0, error=str(e))

    def compute_batch(
        self, text_pairs: List[TextPair], batch_size: int = 32
    ) -> List[MetricResult]:
        """
        Compute MAUVE for multiple text pairs.
        
        This is the preferred method for MAUVE evaluation as it compares the 
        distribution of reference texts to the distribution of candidate texts.
        All pairs receive the same MAUVE score since MAUVE evaluates the 
        distributional similarity rather than individual pair similarities.

        Args:
            text_pairs: List of text pairs to evaluate
            batch_size: Ignored for MAUVE (uses encoding_batch_size instead)

        Returns:
            List[MetricResult]: List of MAUVE results (all with the same score)
        """
        self._ensure_mauve()
        
        # Type assertion since _ensure_mauve ensures _mauve is not None
        assert self._mauve is not None

        if not text_pairs:
            return []

        self._notify_start(len(text_pairs))
        
        try:
            # Check if we need to recompute (different batch size or first time)
            if self._cached_score is None or len(text_pairs) != self._cached_batch_size:
                
                # Extract reference and candidate texts
                p_text = [pair.reference for pair in text_pairs]  # Reference (human)
                q_text = [pair.candidate for pair in text_pairs]  # Candidate (model)

                logger.info(f"Computing MAUVE for {len(text_pairs)} text pairs")
                
                if len(text_pairs) < 100:
                    logger.warning(
                        f"MAUVE works best with at least a few hundred samples. "
                        f"You have {len(text_pairs)} pairs. Results may be less reliable."
                    )

                # Compute MAUVE between the two distributions
                out = self._mauve.compute_mauve(
                    p_text=p_text,
                    q_text=q_text,
                    device_id=self.device_id,
                    max_text_length=self.max_text_length,
                    verbose=self.verbose,
                    batch_size=self.encoding_batch_size,
                    num_buckets=self.num_buckets,
                )

                # Cache the result
                self._cached_score = round(float(out.mauve), 4)
                self._cached_batch_size = len(text_pairs)
                
                logger.info(f"MAUVE computation completed. Score: {self._cached_score}")

            # Create result for all pairs (same score since MAUVE is distributional)
            details = {
                "model": self.model_id,
                "library": "mauve-text",
            }

            results = []
            for i in range(len(text_pairs)):
                result = MetricResult(
                    metric_name=self.name,
                    score=self._cached_score,
                    details=details,
                )
                results.append(result)
                
                # Notify observers
                self._notify_pair_processed(i, result)

            self._notify_complete(results)
            return results

        except Exception as e:
            logger.error(f"Error computing MAUVE batch: {e}")
            self._notify_error(e)
            
            # Return error results for all pairs
            error_results = []
            for i in range(len(text_pairs)):
                error_result = MetricResult(
                    metric_name=self.name, score=0.0, error=str(e)
                )
                error_results.append(error_result)
                self._notify_pair_processed(i, error_result)
            
            return error_results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the MAUVE configuration."""
        return {
            "status": "loaded" if self._mauve_loaded else "not_loaded",
            "library": "mauve-text",
            "model": self.model_id,
            "max_text_length": self.max_text_length,
            "device_id": self.device_id,
            "encoding_batch_size": self.encoding_batch_size,
            "num_buckets": self.num_buckets or "auto",
            "requires_download": True,
            "supported_models": self.get_supported_models(),
        }

    def configure(
        self,
        model_id: Optional[str] = None,
        max_text_length: Optional[int] = None,
        device_id: Optional[int] = None,
        verbose: Optional[bool] = None,
        num_buckets: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Update MAUVE configuration. Will take effect on next computation.
        Clears any cached results when configuration changes.

        Args:
            model_id: Model to use for feature encoding
            max_text_length: Maximum text length for encoding
            device_id: GPU device ID to use
            verbose: Whether to show progress messages
            num_buckets: Number of clusters for quantization
            batch_size: Batch size for feature encoding
        """
        config_changed = False

        if model_id is not None and model_id != self.model_id:
            self.model_id = model_id
            config_changed = True

        if max_text_length is not None:
            self.max_text_length = max_text_length
            config_changed = True

        if device_id is not None and device_id != self.device_id:
            self.device_id = device_id
            config_changed = True

        if verbose is not None:
            self.verbose = verbose

        if num_buckets is not None:
            self.num_buckets = num_buckets
            config_changed = True

        if batch_size is not None:
            self.encoding_batch_size = batch_size

        # Clear cache if configuration changed
        if config_changed:
            self._cached_score = None
            self._cached_batch_size = 0
            logger.info(
                f"MAUVE configuration updated: model={self.model_id}, "
                f"max_length={self.max_text_length}, device={self.device_id}, "
                f"num_buckets={self.num_buckets}"
            )

    def get_supported_models(self) -> List[str]:
        """Get the list of commonly supported models for MAUVE."""
        return [
            "gpt2",
            "gpt2-medium", 
            "gpt2-large",
            "gpt2-xl",
            "distilgpt2",
            # MAUVE can work with other HuggingFace models too
        ]
        
    def clear_cache(self) -> None:
        """Clear cached MAUVE scores. Useful when processing different datasets."""
        self._cached_score = None
        self._cached_batch_size = 0
        logger.info("MAUVE cache cleared")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, model={self.model_id})"