# LLM-Scorecaster

An open-source REST API for evaluating Large Language Model (LLM) responses using various metrics like BERT Score, BLEU, ROUGE, and AlignScore. This tool provides both synchronous (real-time) and asynchronous (batch) processing capabilities for comprehensive LLM evaluation.

## Features

- **Multiple Metrics**: Support for BERT Score, BLEU, ROUGE-L, ROUGE-1, ROUGE-2, and AlignScore
- **Synchronous API**: Real-time evaluation for immediate feedback
- **Asynchronous API**: Batch processing for large-scale evaluation (coming soon)
- **Simple Authentication**: API key-based authentication
- **Extensible Architecture**: Easy to add new metrics
- **Fast & Lightweight**: Built with FastAPI for high performance
- **Auto-Documentation**: Interactive API docs with Swagger UI

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

   ```bash
   # Using makefile
   make init

   # Or to solely install dependancies
   make install
   ```

3. **Set up environment variables:**

   ```bash
   API_KEY=your-secret-api-key-here
   REDIS_URL=redis://localhost:6379
   ```

4. **Start Redis (required for async processing):**

   ```bash
   # Using Docker
   docker run -d -p 6379:6379 redis:alpine

   # Or install locally (macOS)
   brew install redis
   brew services start redis
   ```

5. **Run the API server:**
   ```bash
   make dev
   ```

The API will be available at `http://localhost:8000`

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
    "metrics": ["bert_score", "bleu", "rouge_l"],
    "batch_size": 32
  }'
```

### Response Format

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
          "score": 0.851,
          "details": {
            "precision": 0.867,
            "recall": 0.835,
            "f1": 0.851
          }
        },
        {
          "metric_name": "bleu",
          "score": 0.423,
          "details": null
        },
        {
          "metric_name": "rouge_l",
          "score": 0.678,
          "details": null
        }
      ]
    }
  ],
  "processing_time_seconds": 1.234
}
```

### Available Metrics

- `bert_score`: Contextual embeddings-based evaluation
- `bleu`: Bilingual Evaluation Understudy score
- `rouge_l`: Recall-Oriented Understudy for Gisting Evaluation (Longest Common Subsequence)
- `rouge_1`: ROUGE-1 (unigram overlap)
- `rouge_2`: ROUGE-2 (bigram overlap)
- `align_score`: Alignment-based evaluation score

## Configuration

### Environment Variables

| Variable      | Default                  | Description                       |
| ------------- | ------------------------ | --------------------------------- |
| `API_KEY`     | _required_               | Authentication key for API access |
| `REDIS_URL`   | `redis://localhost:6379` | Redis connection string           |
| `ENVIRONMENT` | `development`            | Application environment           |
| `DEBUG`       | `false`                  | Enable debug mode                 |

## Development Status

🟢 **Ready**: Synchronous API with placeholder metrics  
🟡 **In Progress**: Actual metric implementations  
🔴 **Planned**: Asynchronous API, Celery workers

## Contributing

We welcome contributions! Please submit a pull request or open an issue if you have suggestions.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Support

- **Documentation**: http://localhost:8000/docs (when running locally)
- **Issues**: [GitHub Issues](link-to-your-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)
