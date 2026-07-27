from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.data.synthetic import write_dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate CERPT synthetic state-transition data")
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_dataset(args.output_dir, args.train_size, args.validation_size, args.test_size, args.seed)
