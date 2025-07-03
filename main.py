import argparse
from .core import mats

def main():
    parser = argparse.ArgumentParser(description="MATS: Merged AI Tools System")
    subparsers = parser.add_subparsers(dest="command")

    # Load model from HF
    parser_model = subparsers.add_parser("load-model")
    parser_model.add_argument("model_id", help="Model ID from Hugging Face")
    parser_model.add_argument("--task", help="Task type like text-classification", default=None)

    # Download dataset
    parser_data = subparsers.add_parser("dataset")
    parser_data.add_argument("id", help="Dataset ID")
    parser_data.add_argument("--source", help="Source (kaggle|huggingface)", default="kaggle")

    args = parser.parse_args()

    if args.command == "load-model":
        mats.load_model(args.model_id, task=args.task)
    elif args.command == "dataset":
        mats.dataset(args.id, source=args.source)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()