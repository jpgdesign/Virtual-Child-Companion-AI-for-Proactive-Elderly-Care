from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from R_data import RewardCalculator
from dueling_dqn import DuelingDQNAgent
from tabular_q_learning import TabularQLearningAgent


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


ALGORITHM_ALIASES = {
    "dqn": "dqn",
    "dueling_dqn": "dqn",
    "q": "q_learning",
    "q_learning": "q_learning",
    "qlearning": "q_learning",
    "tabular_q": "q_learning",
}


@dataclass
class TrainerConfig:
    algorithm: str = "dqn"
    state_dim: int = 5
    action_dim: int = 12
    test_ratio: float = 0.2
    random_seed: int = 42


def normalize_algorithm(name: str) -> str:
    normalized = ALGORITHM_ALIASES.get(name.lower())
    if not normalized:
        raise ValueError(f"Unsupported algorithm: {name}")
    return normalized


class IntegratedRLTrainer:
    def __init__(self, output_dir: str = "outputs", algorithm: str = "dqn") -> None:
        self.config = TrainerConfig(algorithm=normalize_algorithm(algorithm))
        self.output_dir = Path(output_dir)
        self.models_dir = self.output_dir / "models"
        self.reports_dir = self.output_dir / "reports"
        self.matrices_dir = self.output_dir / "matrices"

        for directory in [self.output_dir, self.models_dir, self.reports_dir, self.matrices_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.agent = self._build_agent()
        logger.info("Initialized RL trainer with algorithm=%s", self.config.algorithm)

    def _build_agent(self) -> DuelingDQNAgent | TabularQLearningAgent:
        if self.config.algorithm == "dqn":
            return DuelingDQNAgent(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                learning_rate=0.001,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.15,
                epsilon_decay=0.999,
                target_update=50,
                memory_capacity=10_000,
                batch_size=64,
            )

        return TabularQLearningAgent(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            alpha=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.15,
            epsilon_decay=0.999,
            memory_capacity=10_000,
            batch_size=64,
        )

    def run_complete_pipeline(
        self,
        dialogue_files: Sequence[str],
        epochs: int = 300,
        batch_size: int = 64,
        save_interval: int = 25,
    ) -> Dict[str, Any]:
        records = self.load_training_records(dialogue_files)
        train_records, test_records = self.train_test_split(records, self.config.test_ratio, self.config.random_seed)
        logger.info("Loaded %s records -> train=%s test=%s", len(records), len(train_records), len(test_records))

        training_results = self.train_agent(
            train_records=train_records,
            test_records=test_records,
            epochs=epochs,
            batch_size=batch_size,
            save_interval=save_interval,
        )

        q_matrix = self.extract_q_matrix(training_results["final_model_path"])
        r_matrix_stats = self.extract_r_matrix_stats(records)
        matrices_path = self.save_matrix_data(q_matrix, r_matrix_stats)
        report_path = self.generate_summary_report(training_results, q_matrix, r_matrix_stats, matrices_path)

        return {
            "algorithm": self.config.algorithm,
            "training": training_results,
            "q_matrix": q_matrix,
            "r_matrix_stats": r_matrix_stats,
            "matrices_file_path": str(matrices_path),
            "report_path": str(report_path),
        }

    def load_training_records(self, dialogue_files: Sequence[str]) -> List[Dict[str, Any]]:
        calculator = RewardCalculator()
        records: List[Dict[str, Any]] = []

        for file_path in dialogue_files:
            path = Path(file_path)
            payload = json.loads(path.read_text(encoding="utf-8"))

            if "data" in payload:
                raw_records = payload["data"]
            elif "state_action_data" in payload:
                raw_records = calculator.calculate_all_rewards(payload["state_action_data"])
            elif isinstance(payload, list):
                raw_records = payload
            else:
                raise ValueError(f"Unsupported input format: {path}")

            for record in raw_records:
                records.append(self.normalize_record(record))

        if not records:
            raise ValueError("No training records were loaded.")

        return records

    def normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        state = [int(value) for value in record["state"]]
        next_state = [int(value) for value in record["next_state"]]
        action = record["action"]
        action_idx = int(np.argmax(action)) if isinstance(action, list) else int(action)
        one_hot_action = [0.0] * self.config.action_dim
        one_hot_action[action_idx] = 1.0

        return {
            "state": state,
            "action": one_hot_action,
            "action_idx": action_idx,
            "reward": float(record["reward"]),
            "next_state": next_state,
            "done": bool(record.get("done", record.get("reward_requirements", {}).get("is_terminal", False))),
        }

    def train_test_split(
        self,
        records: Sequence[Dict[str, Any]],
        test_ratio: float,
        seed: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        split_index = max(1, int(len(shuffled) * (1 - test_ratio)))
        train_records = shuffled[:split_index]
        test_records = shuffled[split_index:] or shuffled[-1:]
        return train_records, test_records

    def seed_replay_buffer(self, train_records: Sequence[Dict[str, Any]]) -> None:
        for record in train_records:
            self.agent.memory.push(
                record["state"],
                record["action"],
                record["reward"],
                record["next_state"],
                record["done"],
            )

    def train_agent(
        self,
        train_records: Sequence[Dict[str, Any]],
        test_records: Sequence[Dict[str, Any]],
        epochs: int,
        batch_size: int,
        save_interval: int,
    ) -> Dict[str, Any]:
        self.seed_replay_buffer(train_records)

        losses: List[float] = []
        train_rewards: List[float] = []
        test_accuracies: List[float] = []
        epsilon_values: List[float] = []
        best_test_accuracy = -1.0
        best_model_path = self.models_dir / f"best_{self.config.algorithm}_model"
        final_model_path = self.models_dir / f"final_{self.config.algorithm}_model"

        model_suffix = ".pt" if self.config.algorithm == "dqn" else ".json"
        best_model_path = best_model_path.with_suffix(model_suffix)
        final_model_path = final_model_path.with_suffix(model_suffix)

        for epoch in range(1, epochs + 1):
            loss = self.agent.train(batch_size)
            losses.append(float(loss))
            train_rewards.append(float(np.mean([record["reward"] for record in train_records])))

            if epoch % save_interval == 0 or epoch == epochs:
                accuracy = self.evaluate_on_test_data(test_records)
                test_accuracies.append(accuracy)
                epsilon_values.append(float(self.agent.epsilon))

                if accuracy >= best_test_accuracy:
                    best_test_accuracy = accuracy
                    self.agent.save_model(str(best_model_path))

                logger.info(
                    "[%s] epoch=%s/%s loss=%.4f test_acc=%.3f epsilon=%.4f",
                    self.config.algorithm,
                    epoch,
                    epochs,
                    loss,
                    accuracy,
                    self.agent.epsilon,
                )

        self.agent.save_model(str(final_model_path))
        return {
            "algorithm": self.config.algorithm,
            "losses": losses,
            "train_rewards": train_rewards,
            "test_accuracies": test_accuracies,
            "epsilon_values": epsilon_values,
            "best_test_accuracy": best_test_accuracy,
            "final_model_path": str(final_model_path),
            "best_model_path": str(best_model_path),
            "total_epochs_trained": epochs,
            "final_evaluation": test_accuracies[-1] if test_accuracies else 0.0,
        }

    def evaluate_on_test_data(self, test_records: Sequence[Dict[str, Any]]) -> float:
        if not test_records:
            return 0.0

        correct_actions = 0
        for record in test_records:
            predicted_action = self.agent.select_action(record["state"], training=False)
            predicted_idx = int(np.argmax(predicted_action))
            if predicted_idx == record["action_idx"]:
                correct_actions += 1
        return correct_actions / len(test_records)

    def extract_r_matrix_stats(self, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        rewards = [record["reward"] for record in records]
        state_action_rewards: Dict[tuple[tuple[int, ...], int], List[float]] = {}

        for record in records:
            key = (tuple(record["state"]), record["action_idx"])
            state_action_rewards.setdefault(key, []).append(record["reward"])

        avg_rewards = {str(key): float(np.mean(values)) for key, values in state_action_rewards.items()}
        return {
            "total_records": len(records),
            "unique_state_action_pairs": len(state_action_rewards),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "state_action_avg_rewards": avg_rewards,
        }

    def extract_q_matrix(self, model_path: str | None = None) -> np.ndarray:
        if model_path:
            self.agent.load_model(model_path)

        rows = []
        for index in range(2 ** self.config.state_dim):
            state = self.index_to_state(index)
            rows.append(self.agent.get_q_values(state))
        return np.asarray(rows, dtype=np.float32)

    def index_to_state(self, index: int) -> List[int]:
        return [(index >> shift) & 1 for shift in range(self.config.state_dim - 1, -1, -1)]

    def save_matrix_data(self, q_matrix: np.ndarray, r_matrix_stats: Dict[str, Any]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        payload = {
            "timestamp": timestamp,
            "algorithm": self.config.algorithm,
            "q_matrix": {
                "shape": list(q_matrix.shape),
                "data": q_matrix.tolist(),
                "stats": {
                    "min": float(q_matrix.min()),
                    "max": float(q_matrix.max()),
                    "mean": float(q_matrix.mean()),
                    "std": float(q_matrix.std()),
                },
            },
            "r_matrix_stats": r_matrix_stats,
        }
        output_path = self.matrices_dir / f"{self.config.algorithm}_matrices_{timestamp}.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    def generate_summary_report(
        self,
        training_results: Dict[str, Any],
        q_matrix: np.ndarray,
        r_matrix_stats: Dict[str, Any],
        matrices_path: Path,
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"{self.config.algorithm}_training_report_{timestamp}.md"
        report = f"""# RL Training Report

- Algorithm: `{self.config.algorithm}`
- Generated at: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`
- Epochs: `{training_results['total_epochs_trained']}`
- Final evaluation accuracy: `{training_results['final_evaluation']:.3f}`
- Best evaluation accuracy: `{training_results['best_test_accuracy']:.3f}`
- Final model: `{training_results['final_model_path']}`
- Matrix dump: `{matrices_path}`

## Reward Statistics

- Total records: `{r_matrix_stats['total_records']}`
- Unique state-action pairs: `{r_matrix_stats['unique_state_action_pairs']}`
- Mean reward: `{r_matrix_stats['mean_reward']:.3f}`
- Reward range: `[{r_matrix_stats['min_reward']:.3f}, {r_matrix_stats['max_reward']:.3f}]`

## Q Matrix

- Shape: `{tuple(q_matrix.shape)}`
- Mean: `{q_matrix.mean():.3f}`
- Min: `{q_matrix.min():.3f}`
- Max: `{q_matrix.max():.3f}`
"""
        report_path.write_text(report, encoding="utf-8")
        return report_path


IntegratedDQNTrainer = IntegratedRLTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL policy with DQN or Q-learning.")
    parser.add_argument(
        "--algorithm",
        default=os.getenv("RL_ALGORITHM", "dqn"),
        help="dqn (default) or q_learning",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["rl_data_20250721_142929.json"],
        help="Input RL data files.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = IntegratedRLTrainer(output_dir=args.output_dir, algorithm=args.algorithm)
    results = trainer.run_complete_pipeline(
        dialogue_files=args.input,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
    )

    print("\n" + "=" * 60)
    print(f"Algorithm: {results['algorithm']}")
    print(f"Final evaluation: {results['training']['final_evaluation']:.3f}")
    print(f"Best evaluation: {results['training']['best_test_accuracy']:.3f}")
    print(f"Report: {results['report_path']}")
    print(f"Matrices: {results['matrices_file_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
