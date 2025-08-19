# BERT Score... and BLEU... and ROUGE.... Oh my!

Within this folder lies all code related to specific metrics calculation. 


## Using the Metric Observer

```python
from src.core.metrics.base import MetricObserver
from src.core.metrics.registry import metric_registry
from src.models.schemas import TextPair

class ProgressTracker(MetricObserver):
    def on_metric_start(self, metric_name: str, total_pairs: int):
        print(f"Starting {metric_name} for {total_pairs} pairs")
    
    def on_pair_processed(self, metric_name: str, pair_index: int, result):
        print(f"{metric_name}: processed pair {pair_index}")
    
    def on_metric_complete(self, metric_name: str, results):
        print(f"{metric_name}: completed with {len(results)} results")
    
    def on_metric_error(self, metric_name: str, error):
        print(f"{metric_name}: error occurred: {error}")

# Use it:
bert_metric = metric_registry.get_metric("bert_score")
tracker = ProgressTracker()
bert_metric.add_observer(tracker)

# Now when you compute, you'll see progress updates
result = bert_metric.compute_batch([
  TextPair(candidate="reference text", reference="reference text"),
  TextPair(candidate="this be a test text", reference="wooooooooo text"),
])
```



## Validation and Error Handling

```python
# Validate metrics before processing
valid_metrics, invalid_metrics = metric_registry.validate_metrics(["bert_score", "invalid_metric"])

if invalid_metrics:
    raise HTTPException(
        status_code=400,
        detail=f"Unknown metrics: {invalid_metrics}"
    )

# Get metric information
metric_info = metric_registry.get_metric_info("bert_score")
print(metric_info)
# {'name': 'bert_score', 'type': 'bert_score', 'description': '...', 'requires_model_download': True}
```

## To Add a New Metric

To add a new metric, just create a file in implementations/:

```python
# src/core/metrics/implementations/bleu.py
from src.core.metrics.base import BaseMetric
from src.models.schemas import MetricType, MetricResult

class BleuMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "bleu"
    
    @property 
    def metric_type(self) -> MetricType:
        return MetricType.BLEU
    
    def compute_single(self, reference: str, candidate: str) -> MetricResult:
        # Implementation here
        pass
```

The registry will automatically discover it on the next discover_metrics() call!

- Zero configuration: Just create metric files, they're automatically discovered
- Type safety: Everything uses your Pydantic models
- Extensible: Add new metrics without changing existing code
- Error resilient: Failed metrics don't break the whole request
- Progress tracking: Monitor long-running computations