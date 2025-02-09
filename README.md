# LLM-Scorecaster

## Overview

**LLM-Scorecaster** is a system for evaluating and streaming LLM-generated responses using BERTScore and other user defined metrics. It takes in the candidate response from an LLM and the user-persisted response, calculates the BERTScore, and emits the results to a data warehouse of choice.

## Features

- 🧠 **BERTScore Calculation** – Compares LLM-generated responses with user-persisted responses for similarity.
- 📊 **Data Warehouse Integration** – Sends evaluation results to a configurable data warehouse.
- ⚡ **Scalable & Modular** – Designed for flexibility in handling different LLM outputs and storage backends.

## Use Cases

- **LLM Performance Monitoring** – Track response quality over time.
- **User Feedback Analysis** – Compare persisted responses with generated ones.
- **Data-Driven Fine-Tuning** – Use scores to improve model outputs.

## Installation

To get started, clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/LLM-Scorecaster.git
cd LLM-Scorecaster
poetry install
```

## Usage

Run the system with:

```bash
python main.py 
```

### Configuration

Modify `config.yaml` to specify:
- Data warehouse connection details
- LLM response source
- Scoring parameters

## Contributing

We welcome contributions! Please submit a pull request or open an issue if you have suggestions.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

