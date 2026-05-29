import argparse
import rng_control
import mnist_train
import pandas as pd
import json
from pathlib import Path

BOLD = "\033[1m"
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"


def build_parser():
    m_parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    m_parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                          help='input batch size for training (default: 64)')
    m_parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                          help='input batch size for testing (default: 1000)')
    m_parser.add_argument('--epochs', type=int, default=4, metavar='N',
                          help='number of epochs to train (default: 4)')
    m_parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                          help='learning rate (default: 1.0)')
    m_parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                          help='Learning rate step gamma (default: 0.7)')
    m_parser.add_argument('--no-accel', action='store_true',
                          help='disables accelerator')
    m_parser.add_argument('--seeds', type=int, nargs='+', default=[42], metavar='S',
                          help='random seeds (default: [42])')
    m_parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                          help='how many batches to wait before logging training status')
    return m_parser

def save_results_to_json(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def load_results_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    seeds = args.seeds
    prng_engines = ["MT19937", "PCG", "PHILOX", "LCG"]

    results = {
        "MT19937": {},
        "PCG": {},
        "PHILOX": {},
        "LCG": {}
    }

    for prng_engine in prng_engines:
        print("\n   =====================================   ")
        print(f"         Testing: {GREEN}{prng_engine}{RESET}\n")

        for seed in seeds:
            print(f"\nTesting seed: {YELLOW}{seed}{RESET}\n")

            rng_control.change_rng_type(prng_engine, seed)
            trainer = mnist_train.BaseMNISTTrainer(args)
            losses, test_losses, accuracies = trainer.run()

            if seed not in results[prng_engine]:
                results[prng_engine][seed] = {}

            results[prng_engine][seed]["losses"] = losses
            results[prng_engine][seed]["test_losses"] = test_losses
            results[prng_engine][seed]["accuracies"] = accuracies

            loss_path = Path(f"results/{prng_engine}/{seed}_loss.csv")
            eval_path = Path(f"results/{prng_engine}/{seed}_evaluation.csv")
            loss_path.parent.mkdir(parents=True, exist_ok=True)
            eval_path.parent.mkdir(parents=True, exist_ok=True)

            pd.DataFrame({
                "step": list(range(len(losses))),
                "train_loss": losses
            }).to_csv(loss_path, index=False)

            pd.DataFrame({
                "epoch": list(range(len(test_losses))),
                "test_loss": test_losses,
                "accuracy": accuracies
            }).to_csv(eval_path, index=False)

    results_path = Path(f"results/results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_to_json(results, results_path)