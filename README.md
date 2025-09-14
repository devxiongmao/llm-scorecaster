# LLM-Scorecaster

An open-source REST API for evaluating Large Language Model (LLM) responses using various metrics like BERT Score, BLEU, ROUGE, and more. This tool provides both synchronous and asynchronous processing capabilities for comprehensive LLM evaluation.

## Features

- **Multiple Metrics**: Support for BERT Score, BLEU (multiple N-grams), ROUGE (1, 2, L and L-Sum) to name a few
- **Synchronous API**: Real-time evaluation for immediate feedback
- **Asynchronous API**: Batch processing for large-scale evaluation
- **Simple Authentication**: API key-based authentication
- **Extensible Architecture**: Easy to add new metrics
- **Fast & Lightweight**: Built with FastAPI for high performance
- **Auto-Documentation**: Interactive API docs with Swagger UI
- **Modular Dependencies**: Install only the metrics you need

## Architecture

- **FastAPI**: Modern, fast web framework with automatic API documentation
- **Redis**: Message broker and temporary result storage for async processing
- **Celery**: Distributed task processing (for async workflows)
- **Pydantic**: Data validation and serialization
- **No Database Required**: Simplified architecture using Redis for temporary storage

## Quick Start

### Prerequisites

- Python 3.12+
- Redis server (for async processing)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/devxiongmao/llm-scorecaster.git
   cd llm-scorecaster
   ```

2. **Install dependencies:**

   The project uses Poetry for dependency management with optional extras. You can install only what you need:

   ```bash
   # Core installation (API server without metric libraries)
   poetry install

   # Install with specific metrics
   poetry install --extras "bert"        # BERT Score only
   poetry install --extras "bleu"        # BLEU Score only  
   poetry install --extras "rouge"       # ROUGE Score only

   # Install multiple metrics
   poetry install --extras "bert bleu"   # BERT + BLEU
   poetry install --extras "bert rouge"  # BERT + ROUGE

   # Install all NLP metrics at once
   poetry install --extras "nlp-metrics"

   # Install everything including development tools
   poetry install --extras "all"

   # Development installation
   poetry install --extras "dev"         # All dev tools
   poetry install --extras "test"        # Testing tools only
   poetry install --extras "lint"        # Linting tools only
   ```

   **Available Installation Options:**

   | Extra | Dependencies | Use Case |
   |-------|-------------|----------|
   | `bert` | bert-score | BERT Score metric only |
   | `bleu` | sacrebleu | BLEU Score metric only |
   | `rouge` | rouge-score | ROUGE Score metric only |
   | `nlp-metrics` | All metric libraries | All NLP evaluation metrics |
   | `test` | pytest, pytest-asyncio | Testing framework |
   | `lint` | black, pyright, pylint | Code quality tools |
   | `dev` | All test + lint tools | Full development setup |
   | `all` | Everything | Complete installation |

   **Using Make commands:**
   ```bash
   # Install everything (equivalent to poetry install --extras "all")
   make init

   # Core installation only
   make install
   ```

3. **Set up environment variables:**

   ```bash
   API_KEY=your-secret-api-key-here
   REDIS_URL=redis://localhost:6379

   # Or, if using docker
   REDIS_URL=redis://redis:6379
   ```

4. **Start Using Docker:**

   ```bash
   make docker-dev

5. **Start Without Docker:**
   ```bash
   # Ensure redis is running
   make redis-start
   
   # In one terminal
   make dev

   # In another terminal
   make worker
   ```

The API will be available at `http://localhost:8000`

## Installation Notes

### Lightweight Installation

For production environments where you only need specific metrics, use targeted installations:

```bash
# Minimal BERT-only setup
poetry install --extras "bert"

# BLEU + ROUGE without BERT (saves ~1GB of model downloads)
poetry install --extras "bleu rouge"
```

### Handling Missing Dependencies

If you try to use a metric without installing its dependencies, you'll get a helpful error message:

```json
{
  "error": "BERT Score not available. Install with: poetry install --extras 'bert'"
}
```

### Development Setup

For contributors and developers:

```bash
# Full development environment
poetry install --extras "dev"

# Or install everything
poetry install --extras "all"
```

## Configuration

### Environment Variables

| Variable      | Default                  | Description                       |
| ------------- | ------------------------ | --------------------------------- |
| `API_KEY`     | _required_               | Authentication key for API access |
| `REDIS_URL`   | `redis://localhost:6379` | Redis connection string           |
| `ENVIRONMENT` | `development`            | Application environment           |
| `MAX_TIMEOUT` | `30`                     | Max timeout for webhook requests  |
| `MAX_RETRIES` | `3`                      | Max retries for webhook requests  |

### API Documentation

Once running, visit:

- **Interactive docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/health

## Usage

### Synchronous Evaluation

Use the synchronous endpoint for real-time metric calculation:

```bash
curl -X POST "http://localhost:8000/api/v1/metrics/evaluate" \
  -H "Authorization: Bearer your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "text_pairs": [
      {
        "reference": "The cat sat on the mat",
        "candidate": "A cat was sitting on a mat"
      },
      {
        "reference": "Hello world, how are you?",
        "candidate": "Hi there world, how are you doing?"
      }
    ],
    "metrics": ["bert_score", "bleu_score", "rouge_score"],
    "batch_size": 32
  }'
```

### Asynchronous Evaluation

Users also have the option of using an async version of the API. The async implementation offers webhook support for automatic posting of results. If your application doesn't support webhooks, a polling option has also been created to check the status of the results (simply omit the webhook_url param in the below request to use this version). Notice the change in URL.

```bash
curl -X POST "http://localhost:8000/api/v1/async/evaluate" \
  -H "Authorization: Bearer your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "text_pairs": [
      {
        "reference": "The cat sat on the mat",
        "candidate": "A cat was sitting on a mat"
      },
      {
        "reference": "Hello world, how are you?",
        "candidate": "Hi there world, how are you doing?"
      }
    ],
    "metrics": ["bert_score", "bleu_score", "rouge_score"],
    "batch_size": 32,
    "webhook_url": "http://localhost:3000/test-llm",
  }'
```

The response from this request is:

```json
{
  "job_id":"b43339ba-35a9-4d15-9700-e0cd85f0b001",
  "status":"PENDING",
  "message":"Job queued successfully. Results will be sent to webhook URL: http://localhost:3000/test-llm",
  "estimated_completion_time":3.0
}
```

### Managing Async Jobs

Taking the returned `job_id` from a `/api/v1/async/evaluate` request, users can query a status endpoint for the status of their results. 

```bash
curl -X GET "http://localhot:8000/api/v1/jobs/status/b43339ba-35a9-4d15-9700-e0cd85f0b001" \
  -H "Authorization: Bearer your-secret-api-key-here"
```

Once ready, results can be requested via:

```bash
curl -X GET "http://localhot:8000/api/v1/jobs/results/b43339ba-35a9-4d15-9700-e0cd85f0b001" \
  -H "Authorization: Bearer your-secret-api-key-here"
```

Users can also delete an active job if so desired:

```bash
curl -X DELETE "http://localhot:8000/api/v1/jobs/b43339ba-35a9-4d15-9700-e0cd85f0b001" \
  -H "Authorization: Bearer your-secret-api-key-here"
```

Users can also query for a list of active jobs:

```bash
curl -X GET "http://localhot:8000/api/v1/jobs/" \
  -H "Authorization: Bearer your-secret-api-key-here"
```

### Results Response Format

```json
{
  "success": true,
  "message": "Successfully calculated 3 metrics for 2 text pairs",
  "results": [
    {
      "pair_index": 0,
      "reference": "The cat sat on the mat",
      "candidate": "A cat was sitting on a mat",
      "metrics": [
        {
          "metric_name": "bert_score",
          "score": 0.6999493837356567,
          "details": {
            "precision": 0.6577,
            "recall": 0.7418,
            "f1": 0.6999
          },
          "error": null
        },
        {
          "metric_name": "bleu_score",
          "score": 0.0864,
          "details": {
            "bleu_score": 0.0864,
            "bleu_score_100": 8.64,
            "max_n": 4,
            "bleu_1": 42.8571,
            "bleu_2": 8.3333,
            "bleu_3": 5,
            "bleu_4": 3.125,
            "brevity_penalty": 1,
            "length_ratio": 1.1667,
            "reference_length": 6,
            "candidate_length": 7,
            "tokenization": "13a",
            "smoothing": "exp"
          },
          "error": null
        },
        {
          "metric_name": "rouge_score",
          "score": 0.4615,
          "details": {
            "rouge1": {
              "precision": 0.4286,
              "recall": 0.5,
              "f1": 0.4615
            },
            "rouge2": {
              "precision": 0,
              "recall": 0,
              "f1": 0
            },
            "rougeL": {
              "precision": 0.4286,
              "recall": 0.5,
              "f1": 0.4615
            },
            "rougeLsum": {
              "precision": 0.4286,
              "recall": 0.5,
              "f1": 0.4615
            },
            "rouge_types": [
              "rouge1",
              "rouge2",
              "rougeL",
              "rougeLsum"
            ],
            "use_stemmer": true,
            "library": "rouge-score"
          },
          "error": null
        }
      ]
    },
    {
      "pair_index": 1,
      "reference": "Hello world, how are you?",
      "candidate": "Hi there world, how are you doing?",
      "metrics": [
        {
          "metric_name": "bert_score",
          "score": 0.646856427192688,
          "details": {
            "precision": 0.5851,
            "recall": 0.7089,
            "f1": 0.6469
          },
          "error": null
        },
        {
          "metric_name": "bleu_score",
          "score": 0.4671,
          "details": {
            "bleu_score": 0.4671,
            "bleu_score_100": 46.71,
            "max_n": 4,
            "bleu_1": 66.6667,
            "bleu_2": 50,
            "bleu_3": 42.8571,
            "bleu_4": 33.3333,
            "brevity_penalty": 1,
            "length_ratio": 1.2857,
            "reference_length": 7,
            "candidate_length": 9,
            "tokenization": "13a",
            "smoothing": "exp"
          },
          "error": null
        },
        {
          "metric_name": "rouge_score",
          "score": 0.6667,
          "details": {
            "rouge1": {
              "precision": 0.5714,
              "recall": 0.8,
              "f1": 0.6667
            },
            "rouge2": {
              "precision": 0.5,
              "recall": 0.75,
              "f1": 0.6
            },
            "rougeL": {
              "precision": 0.5714,
              "recall": 0.8,
              "f1": 0.6667
            },
            "rougeLsum": {
              "precision": 0.5714,
              "recall": 0.8,
              "f1": 0.6667
            },
            "rouge_types": [
              "rouge1",
              "rouge2",
              "rougeL",
              "rougeLsum"
            ],
            "use_stemmer": true,
            "library": "rouge-score"
          },
          "error": null
        }
      ]
    }
  ],
  "processing_time_seconds": 4.017
}
```

### Available Metrics

Within the `src/core/metrics` folder lies all code related to specific metrics calculation. 

- `bert_score`: Contextual embeddings-based evaluation
- `bleu_score`: Bilingual Evaluation Understudy score (multipl N-grams supported)
- `rouge_score`: Recall-Oriented Understudy for Gisting Evaluation (Longest Common Subsequence). ROUGE-1 (unigram overlap), ROUGE-2 (bigram overlap), ROUGE-L and ROUGE-LSUM are supported

## Using a Metric Observer

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

# Modify the src/core/computation.py file to set your observers.
# Like so:
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
- Type safety: Everything uses Pydantic models
- Extensible: Add new metrics without changing existing code
- Error resilient: Failed metrics don't break the whole request
- Progress tracking: Monitor long-running computations

## Development Status

- 🟢 **Complete**: Synchronous API
- 🟢 **Complete**: BERT, BLEU and ROUGE metric implementation 
- 🟢 **Complete**: Asynchronous API, Celery workers 
- 🟢 **Complete**: Webhook support, post your results back when ready 
- 🟢 **Complete**: Dockerize the app
- 🟡 **In-Progress**: Metrics Router for live configuration and discovery

## Contributing

We welcome contributions! Please submit a pull request or open an issue if you have suggestions.

### Development Setup

For contributors:

```bash
# Clone and setup development environment
git clone https://github.com/devxiongmao/llm-scorecaster.git
cd llm-scorecaster

# Install with all development dependencies
poetry install --extras "dev"

# Or install everything
poetry install --extras "all"
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Support

- **Documentation**: http://localhost:8000/docs (when running locally)
- **Issues**: [GitHub Issues](https://github.com/devxiongmao/llm-scorecaster/issues)
