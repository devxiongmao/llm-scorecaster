from typing import List, Dict, Any, Optional
from src.core.metrics.base import BaseMetric
from src.models.schemas import MetricType, MetricResult, TextPair
import logging

logger = logging.getLogger(__name__)


class AlignScoreMetric(BaseMetric):
    """
    AlignScore metric implementation for evaluating text alignment.

    AlignScore is a metric that evaluates the factual consistency and semantic
    alignment between generated text and reference text using pre-trained 
    language models. It produces scores between 0 and 1, where higher scores
    indicate better alignment.
    """

    def __init__(
        self, 
        model: str = "kaist-ai/align-base",
        batch_size: int = 32,
        device: Optional[str] = None,
        max_length: int = 1024
    ):
        """
        Initialize AlignScore metric.

        Args:
            model: Model name/path for AlignScore evaluation
            batch_size: Default batch size for processing
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length for tokenization
        """
        super().__init__()
        
        self.model_name = model
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self._scorer = None
        self._alignscore_loaded = False
        
        # Validate model name
        self._validate_model_name(model)

    @property
    def name(self) -> str:
        return "alignscore"

    @property
    def metric_type(self) -> MetricType:
        return MetricType.ALIGN_SCORE

    @property
    def description(self) -> str:
        return f"AlignScore: Factual consistency and semantic alignment evaluation using {self.model_name}"

    @property
    def requires_model_download(self) -> bool:
        return True

    def _validate_model_name(self, model: str) -> None:
        """Validate model name configuration."""
        supported_models = self.get_supported_models()
        if model not in supported_models and not self._is_valid_model_path(model):
            logger.warning(
                f"Model '{model}' not in known supported models: {supported_models}. "
                f"Will attempt to load anyway."
            )

    def _is_valid_model_path(self, model: str) -> bool:
        """Check if model path seems valid (basic validation)."""
        return "/" in model or model.startswith("./") or model.startswith("../")

    def _ensure_alignscore(self) -> None:
        """Ensure AlignScore library is available and model is loaded."""
        if self._alignscore_loaded:
            return

        try:
            from alignscore import AlignScore
            
            # Initialize AlignScore model
            self._scorer = AlignScore(
                model=self.model_name,
                batch_size=self.batch_size,
                device=self.device,
                ckpt_path=None,  # Use default for HuggingFace models
                evaluation_mode="nli_sp"  # Default evaluation mode
            )
            self._alignscore_loaded = True
            logger.info(f"AlignScore model loaded successfully: {self.model_name}")

        except ImportError:
            raise ImportError(
                "alignscore package is required for AlignScore metric. "
                "Install with: pip install alignscore"
            )
        except Exception as e:
            logger.error(f"Failed to load AlignScore model: {e}")
            raise RuntimeError(f"Failed to initialize AlignScore model: {e}")

    def compute_single(self, reference: str, candidate: str) -> MetricResult:
        """
        Compute AlignScore for a single text pair.

        Args:
            reference: The reference/ground truth text
            candidate: The candidate text to evaluate

        Returns:
            MetricResult: AlignScore result
        """
        self._ensure_alignscore()
        
        # Type assertion since _ensure_alignscore ensures _scorer is not None
        assert self._scorer is not None

        try:
            # Compute AlignScore
            # Note: AlignScore typically takes (claim, evidence) where claim is candidate
            # and evidence is reference for factual consistency evaluation
            score = self._scorer.score(
                contexts=[reference], 
                claims=[candidate]
            )[0]  # Get first (and only) score

            # Round score for consistency
            score = round(float(score), 4)

            # Create minimal details as requested
            details = {
                "model": self.model_name,
                "library": "alignscore",
            }

            return MetricResult(
                metric_name=self.name,
                score=score,
                details=details,
            )

        except Exception as e:
            logger.error(f"Error computing AlignScore: {e}")
            return MetricResult(metric_name=self.name, score=0.0, error=str(e))

    def compute_batch(
        self, text_pairs: List[TextPair], batch_size: int = 32
    ) -> List[MetricResult]:
        """
        Compute AlignScore for multiple text pairs with optimized batching.

        AlignScore supports native batch processing, so this implementation
        takes advantage of that for better efficiency.

        Args:
            text_pairs: List of text pairs to evaluate
            batch_size: Size of batches for processing

        Returns:
            List[MetricResult]: List of AlignScore results
        """
        self._ensure_alignscore()
        
        # Type assertion since _ensure_alignscore ensures _scorer is not None
        assert self._scorer is not None

        if not text_pairs:
            return []

        results = []
        self._notify_start(len(text_pairs))
        
        try:
            # Process in batches for memory management and progress reporting
            for i in range(0, len(text_pairs), batch_size):
                batch = text_pairs[i : i + batch_size]
                
                try:
                    # Prepare batch data
                    contexts = [pair.reference for pair in batch]
                    claims = [pair.candidate for pair in batch]
                    
                    # Compute batch scores
                    batch_scores = self._scorer.score(
                        contexts=contexts,
                        claims=claims
                    )
                    
                    # Process results for this batch
                    for j, (pair, score) in enumerate(zip(batch, batch_scores)):
                        try:
                            # Round score for consistency
                            score = round(float(score), 4)
                            
                            # Create minimal details as requested
                            details = {
                                "model": self.model_name,
                                "library": "alignscore",
                            }

                            result = MetricResult(
                                metric_name=self.name,
                                score=score,
                                details=details,
                            )
                            results.append(result)
                            
                            # Notify observers
                            self._notify_pair_processed(i + j, result)
                            
                        except Exception as e:
                            logger.error(f"Error processing result for pair {i + j}: {e}")
                            error_result = MetricResult(
                                metric_name=self.name, score=0.0, error=str(e)
                            )
                            results.append(error_result)
                            self._notify_pair_processed(i + j, error_result)

                except Exception as e:
                    logger.error(f"Error processing batch {i // batch_size + 1}: {e}")
                    
                    # Create error results for this batch
                    for j, pair in enumerate(batch):
                        if i + j >= len(results):  # Only add if not already processed
                            error_result = MetricResult(
                                metric_name=self.name, score=0.0, error=str(e)
                            )
                            results.append(error_result)
                            self._notify_pair_processed(i + j, error_result)

            self._notify_complete(results)
            return results

        except Exception as e:
            self._notify_error(e)
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "status": "loaded" if self._alignscore_loaded else "not_loaded",
            "library": "alignscore",
            "model": self.model_name,
            "batch_size": self.batch_size,
            "device": self.device,
            "max_length": self.max_length,
            "requires_download": True,
            "supported_models": sorted(self.get_supported_models()),
        }

    def configure(
        self,
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> None:
        """
        Update AlignScore configuration. Will take effect on next computation.
        Forces reinitialization if model changes.

        Args:
            model: Model name/path for AlignScore evaluation
            batch_size: Default batch size for processing
            device: Device to run model on
            max_length: Maximum sequence length for tokenization
        """
        config_changed = False

        if model is not None and model != self.model_name:
            self._validate_model_name(model)
            self.model_name = model
            config_changed = True

        if batch_size is not None:
            self.batch_size = batch_size

        if device is not None:
            if device != self.device:
                self.device = device
                config_changed = True

        if max_length is not None:
            self.max_length = max_length

        # Force reinitialization if model or device changed
        if config_changed:
            self._alignscore_loaded = False
            self._scorer = None
            logger.info(
                f"AlignScore configuration updated: model={self.model_name}, "
                f"device={self.device}, batch_size={self.batch_size}, "
                f"max_length={self.max_length}"
            )

    def get_supported_models(self) -> List[str]:
        """Get the list of known supported AlignScore models."""
        return [
            "kaist-ai/align-base",
            "kaist-ai/align-large",
            "princeton-nlp/align-base",
            "princeton-nlp/align-large",
        ]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, model={self.model_name})"