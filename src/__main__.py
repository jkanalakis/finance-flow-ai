
import argparse
from src.ui import start_ui

def main():
    parser = argparse.ArgumentParser(description="FinanceFlowAI - Interactive Financial Advisor")
    parser.add_argument("--mode", type=str, default="interactive", choices=["interactive", "train", "evaluate"], 
                        help="Choose the mode of operation: interactive, train, or evaluate.")
    args = parser.parse_args()

    if args.mode == "interactive":
        start_ui()
    elif args.mode == "train":
        from src.train import train_model
        train_model()
    elif args.mode == "evaluate":
        from src.evaluate import evaluate_model
        evaluate_model()

if __name__ == "__main__":
    main()
