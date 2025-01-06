
# FinanceFlowAI

FinanceFlowAI is an open-source project designed to train, evaluate, and deploy small language models (SLMs) for personal finance question-answering tasks. It empowers users with tailored financial advice to better manage their spending, savings, and investments.

---

## Features

- **Interactive Financial Advisor**: Engage with the AI to receive personalized financial advice.
- **Training and Fine-Tuning**: Use preloaded datasets or customize your own for fine-tuning the model.
- **Performance Evaluation**: Assess model accuracy with metrics like Precision, Recall, and F1-score.
- **Docker Integration**: Seamlessly deploy using containerized environments.

---

## Installation

### Prerequisites
- Python 3.10+
- [Docker](https://www.docker.com/) (optional for containerized environments)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/finance-flow-ai.git
   cd finance-flow-ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

4. (Optional) Build the Docker image:
   ```bash
   docker build -t finance-flow-ai .
   ```

---

## Usage

### Run the Interactive Advisor
```bash
python src/main.py
```

### Train the Model
```bash
python src/train.py --config configs/training_config.yaml
```

### Evaluate the Model
```bash
python src/evaluate.py --data datasets/sample_data.csv
```

---

## Project Structure

```
finance-flow-ai/
├── datasets/            # Data for training and evaluation
├── docs/                # Documentation files
├── notebooks/           # Jupyter Notebooks for prototyping
├── src/                 # Source code for the project
├── tests/               # Unit and integration tests
├── .pre-commit-config.yaml  # Pre-commit hook configuration
├── .gitignore           # Git ignore rules
├── Dockerfile           # Docker image setup
├── Makefile             # Automation tasks
├── README.md            # Project overview and instructions
└── pyproject.toml       # Project dependencies and configuration
```

---

## Contributing

We welcome contributions! Please read our [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before getting started.

---

## License

This project is licensed under the [MIT License](LICENSE).

