
from peft import LoraConfig, TaskType

TRANSFORMERS_MODEL_PRETRAINED = "HuggingFaceTB/FinanceLM2-1.7B-Instruct"
DATA_AUGMENTATION_MODEL = "HuggingFaceTB/FinanceLM2-360M-Instruct"
INFERENCE_MODEL = "financeflowai/FinanceFlowAI-1.7B-Instruct"
MODEL_NAME = "FinanceFlowAI-1.7B-Instruct"
HF_REPO_ID = "FinanceFlowAI-1.7B-Instruct"

TRAINING_CONFIG = {
    "name": "financeflowai/finance_data_final",
    "split": "train",
    "max_samples": 10000,
    "max_length": 2048,
    "num_train_epochs": 3,
    "warmup_ratio": 0.2,
    "batch_size": 8,
    "lr": 3e-4,
    "accumulation_steps": 1,
    "weight_decay": 2e-2,
}
DATASETS_CONFIG = {
    "finance_transactions": {
        "name": "datasets/finance_transactions",
        "text_column": "Transaction",
        "response_column": "Category",
        "split": "train",
        "max_samples": 4500,
        "max_length": 512,
    },
    "user_financial_data": {
        "name": "user/financial-data",
        "text_column": "Input",
        "response_column": "Output",
        "split": "train",
        "max_samples": 3500,
        "max_length": 512,
    },
    "global_finance_data": {
        "name": "global/finance-data",
        "text_column": "Input",
        "response_column": "Output",
        "split": "train",
        "max_samples": 3500,
        "max_length": 512,
    },
}

EVALUATION_DATASET = {
    "name": "Amod/financial_advisory_conversations",
    "text_column": "Transaction",
    "response_column": "Category",
    "max_samples": 500,
}

FINAL_DATASETS = {
    "datasets/finance_transactions": "financeflowai/finance_transactions",
    "datasets/finance_data_initial_v2": "financeflowai/finance_data_initial_v2",
    "datasets/finance_data_final_v2": "financeflowai/finance_data_final_v2",
}
DPO_DATASETS = {
    "datasets/finance_data_dpo/finance_clean": "financeflowai/finance_data_dpo"
}

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

FINANCE_INSTRUCTION = "You are a knowledgeable and supportive financial assistant, providing accurate and personalized advice to users about their financial queries."
